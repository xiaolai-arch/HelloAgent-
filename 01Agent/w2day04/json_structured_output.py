# -*- coding: utf-8 -*-
"""JSON 结构化输出四种机制对比：纯 Prompt / JSON Mode / Schema 约束 Prompt / Function Calling

对应任务：
  1. 理解 JSON Mode 和 Function Calling 格式的区别（报告第一节 + 本实验的实测数据）
  2. 写代码让 LLM 输出结构化 JSON（抽取：人名 / 日期 / 事件）
  3. 写第 2 套结构化 Prompt 模板（带 JSON Schema 约束），见 prompts/schema_constrained.md
     （第 1 套是 w2day01 的 LangGPT 模板）

实验设计（控制变量法）：
  - 固定：模型、temperature=0.2、max_tokens=2000、3 段文本、同一条基础指令
  - 变量：结构化输出的"约束机制"，四种方法：
      plain            纯 Prompt 约定：只在提示词里说"以 JSON 格式输出"
      json_mode        JSON Mode：API 加 response_format={"type":"json_object"}
      schema_prompt    Schema 约束 Prompt：第 2 套模板，把 JSON Schema 写进提示词
      function_calling Function Calling：Schema 挂在 tools 上，强制走工具调用
  - 任务：从文本中抽取 人名 / 日期 / 事件 + 一句话摘要，输出 JSON
  - 每种方法跑完全部 3 段文本，共 12 次调用

评分（100 分制，规则透明、可复现）：
  - JSON 可解析      40 分：裸输出直接 json.loads 成功 40 分；
                     需要修复（去 ``` 围栏 / 截取 {…} 子串）才成功 25 分；失败 0 分
  - Schema 校验通过  30 分：required 齐全、类型正确、无多余字段；每个错误扣 10 分
  - 内容准确         30 分：人名/日期/事件三类 F1 的平均 × 30（日期必须保留原文写法）

产出：
  - 终端打印每次输出与汇总
  - json_structured_report.md   实验报告（含 JSON Mode vs Function Calling 对比）
  - json_structured_result.json 原始数据

配置同 first_llm_call.py：../../vision_config.json 或环境变量。
"""
import sys
import io
import os
import re
import json
import datetime
import urllib.request
import urllib.error

# Windows 终端默认 GBK，强制 UTF-8 输出，避免中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
TEXTS_PATH = os.path.join(BASE_DIR, "texts.json")
REPORT_PATH = os.path.join(BASE_DIR, "json_structured_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "json_structured_result.json")

# ---- 配置加载（与 first_llm_call.py 一致：配置文件优先，环境变量兜底）----
_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
MODEL = _cfg.get("model", "mimo-v2.6-flash")

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，或设置环境变量 "
             "COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")

TEMPERATURE = 0.2
MAX_TOKENS = 2000

# ---- 目标 JSON Schema：抽取 人名 / 日期 / 事件 + 摘要 ----
# 四种方法共用这一份 Schema，才叫控制变量。字段名用英文，因为它就是
# 给下游代码看的契约——本实验恰好要验证：不钉死契约时，模型会不会
# 按指令的语言自作主张用中文键名（实测：会）。
# additionalProperties=False 用来抓模型的"自由发挥"：多输出的每个字段
# 都会被 validate 记一条错误、扣 10 分。
SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "string"},
                   "description": "文本中出现的人名，保留原文"},
        "dates": {"type": "array", "items": {"type": "string"},
                  "description": "文本中出现的日期/时间，保留原文写法"},
        "events": {"type": "array", "items": {"type": "string"},
                   "description": "文本中发生的事件，简短短语"},
        "summary": {"type": "string", "description": "一句话摘要"}
    },
    "required": ["people", "dates", "events", "summary"],
    "additionalProperties": False
}

# ---- Function Calling：同一个 Schema 挂在 tools 上 ----
# 关键：parameters 引用的就是上面那份 SCHEMA——两种机制的差异只在
# Schema 放的位置（Prompt 里 vs API 协议里），Schema 内容必须完全一致。
TOOL_NAME = "save_extraction"
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "保存从文本中抽取的结构化信息",
        "parameters": SCHEMA,
    },
}]

# 四种方法：key → 展示名
METHODS = [
    ("plain", "纯 Prompt 约定"),
    ("json_mode", "JSON Mode"),
    ("schema_prompt", "Schema 约束 Prompt"),
    ("function_calling", "Function Calling"),
]


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def render(template, mapping):
    """{{KEY}} 占位替换，不用 str.format，避免 JSON 花括号冲突。"""
    for key, value in mapping.items():
        template = template.replace("{{" + key + "}}", value)
    return template


TEXTS = json.loads(load_text(TEXTS_PATH))
PLAIN_TPL = load_text(os.path.join(PROMPT_DIR, "plain_json.md"))
SCHEMA_TPL = load_text(os.path.join(PROMPT_DIR, "schema_constrained.md"))
SCHEMA_TEXT = json.dumps(SCHEMA, ensure_ascii=False, indent=2)


def build_prompt(method, text):
    """四种方法都用同一条基础指令；schema_prompt 换成第 2 套模板。

    刻意保持指令是中文：这会诱导模型用中文键名输出——正好用来实测
    "没有任何机制钉死字段名时，模型会不会老老实实按 Schema 来"。
    """
    if method == "schema_prompt":
        return render(SCHEMA_TPL, {"SCHEMA": SCHEMA_TEXT, "TEXT": text})
    return render(PLAIN_TPL, {"TEXT": text})


def build_extra(method):
    """API 层的差异全部收敛在这里：这就是四种方法在本实验里的唯一不同。

    plain            返回空 dict，请求体和裸调用完全一样（对照组）
    json_mode        只多一个 response_format 字段 → 约束"输出形态"
    function_calling 多 tools + tool_choice → 约束"输出意图"；
                     tool_choice 强制指定函数名，不让模型自己挑
    schema_prompt    也返回空 dict——它的武器全在 Prompt 里
    """
    if method == "json_mode":
        return {"response_format": {"type": "json_object"}}
    if method == "function_calling":
        return {"tools": TOOLS,
                "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}}}
    return {}


def chat(messages, extra=None):
    """OpenAI 兼容 POST /chat/completions，返回 (resp, error_msg)。

    extra 是机制差异的注入点（response_format / tools / tool_choice），
    原样合入请求体——这里不做任何机制特判，特判全在 build_extra。
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + API_KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ask(method, text):
    """发一次请求，统一取出 JSON 字符串（content 或 tool_calls 的 arguments）。

    两种机制的返回位置不同：JSON Mode 在 message.content，
    Function Calling 在 message.tool_calls[0].function.arguments。
    这里统一抽成 raw 返回，后面的解析/打分代码就不用关心机制差异。
    """
    prompt = build_prompt(method, text)
    messages = [{"role": "user", "content": prompt}]

    resp, err = chat(messages, build_extra(method))
    note = ""
    if err and method == "function_calling":
        # 部分网关支持 tools 但不支持强制 tool_choice。
        # 降级为 tool_choice="auto" 再试一次；两种错误都保留在返回值里，
        # 报告要如实展示"换一个端点，机制可能就用不了"这个现实。
        fallback = {"tools": TOOLS, "tool_choice": "auto"}
        resp, err2 = chat(messages, fallback)
        if err2:
            return {"ok": False, "error": f"强制 tool_choice：{err} | 降级 auto：{err2}"}
        note = f"强制 tool_choice 失败（{err[:120]}），已降级 tool_choice=auto"

    if err:
        return {"ok": False, "error": err}

    choice = resp["choices"][0]
    message = choice.get("message") or {}
    raw = message.get("content") or ""
    source = "content"
    calls = message.get("tool_calls") or []
    if calls:
        raw = calls[0].get("function", {}).get("arguments") or ""
        source = "tool_calls"

    return {
        "ok": True,
        "raw": raw,
        "source": source,
        "finish_reason": choice.get("finish_reason", "?"),
        "usage": resp.get("usage", {}),
        "note": note,
        "prompt_chars": len(prompt),
    }


# ---- 三级修复解析：原文 → 去 Markdown 围栏 → 截取最外层 {…} ----
# 修复到哪一级，直接决定 parse_score：raw=40，任何修复=25，失败=0。
# 真实管线里确实需要这些兜底，但实验要量化的是"裸输出的可靠度"，
# 所以每一次修复都必须记录进 repairs，让报告能如实展示。
def parse_json_robust(raw):
    """三级修复解析，返回 (解析结果或 None, 修复步骤列表, 解析阶段)。"""
    repairs = []
    text = (raw or "").strip()
    if text.startswith("```"):
        # 第一级：模型无视"不要围栏"的要求，输出被 ```json 包裹
        repairs.append("去掉 Markdown 围栏")
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        # repairs 为空走到这里 = 裸输出直接可解析（最理想，得满分）
        return json.loads(text), repairs, ("raw" if not repairs else "fenced")
    except Exception:
        pass
    # 第二级：JSON 前后混了说明文字，截取第一个 { 到最后一个 } 之间
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        repairs.append("截取最外层 {…} 子串")
        try:
            return json.loads(text[i:j + 1]), repairs, "extracted"
        except Exception:
            pass
    # 三级都失败：把"解析失败"也记进 repairs，打分函数据此给 0 分
    return None, repairs + ["解析失败"], "failed"


# ---- 极简 JSON Schema 校验（覆盖 type / required / items / additionalProperties）----
# 不引第三方库（jsonschema），stdlib 手写一个够用的子集，实验代码保持零依赖。
# 坑：Python 里 bool 是 int 的子类，isinstance(True, int) 为 True，
# 校验 number/integer 时必须显式把 bool 排除掉，否则 True 会被当成数字放行。
_TYPE_MAP = {"string": str, "array": list, "object": dict,
             "number": (int, float), "integer": int, "boolean": bool}


def validate(data, schema, path="$"):
    """递归校验 data 是否符合 schema，返回错误列表（空列表 = 通过）。"""
    errors = []
    want = schema.get("type")
    if want:
        # 第一层：类型检查。类型不对就不用再往下递归了，直接返回
        py = _TYPE_MAP.get(want)
        ok = isinstance(data, py) and not (want in ("number", "integer")
                                           and isinstance(data, bool))
        if not ok:
            errors.append(f"{path}: 期望 {want}，实际 {type(data).__name__}")
            return errors
    if isinstance(data, dict) and want == "object":
        props = schema.get("properties", {})
        # required：字段必须存在，缺一个记一条错误（打分时每条扣 10）
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: 缺少 required 字段 {key}")
        # additionalProperties=False：抓模型的"自由发挥"。本实验里
        # 模型输出的中文键名（人名/日期/事件）就是在这里全部现形的
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in props:
                    errors.append(f"{path}: Schema 之外的多余字段 {key}")
        # 存在的字段递归往下查；path 一路拼下去，错误信息能直接定位字段
        for key, sub in props.items():
            if key in data:
                errors.extend(validate(data[key], sub, f"{path}.{key}"))
    # 数组：逐个元素按 items 的 Schema 递归校验
    if isinstance(data, list) and "items" in schema:
        for idx, item in enumerate(data):
            errors.extend(validate(item, schema["items"], f"{path}[{idx}]"))
    return errors


# ---- 内容准确度：人名/日期双向包含匹配，事件按备选关键词组匹配 ----
# 为什么不用精确匹配：模型可能输出"李明（创始人）"、可能多带引号空格，
# 精确匹配会把对的判成错的；归一化后"双向包含"更接近人工判断。
# 代价：比精确匹配宽松——这一点已如实写进报告的"实验局限"。
_PUNCT = "[\\s“”‘’\"'。，、；：;；（）()\\[\\]【】《》<>·—…？！?!.，]"


def norm(s):
    """归一化：去掉空白和标点，让"李明。"和"李明"可以比较。"""
    return re.sub(_PUNCT, "", str(s))


def _pair_match(a, b):
    """双向包含：归一化后 a 含 b 或 b 含 a 都算命中。

    抽取粒度可粗可细：期望"2025年3月"、输出"项目2025年3月启动"也算对。
    """
    na, nb = norm(a), norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def _f1(expected, got):
    """计算单类实体的 precision / recall / f1。

    people/dates：期望值是字符串列表，双向包含匹配；
    events：期望值是备选关键词组的列表——事件没有唯一正确写法，
            抽成"举办秋季发布会"或"秋季发布会"都该算对，
            任一备选关键词命中即可。
    expected 为空的约定：模型也没抽 → 满分；模型多抽了 → 0 分（编造要罚）。
    """
    if not expected:
        f1 = 1.0 if not got else 0.0
        return {"precision": f1, "recall": 1.0, "f1": f1}
    if isinstance(expected[0], list):  # 事件：备选关键词组，任一命中即可
        used = [False] * len(got)
        me = 0
        for alts in expected:
            for i, g in enumerate(got):
                if not used[i] and any(norm(x) and norm(x) in norm(g) for x in alts):
                    me += 1
                    used[i] = True
                    break
        mg = sum(1 for g in got
                 if any(any(norm(x) and norm(x) in norm(g) for x in alts)
                        for alts in expected))
    else:
        me = sum(1 for e in expected if any(_pair_match(e, g) for g in got))
        mg = sum(1 for g in got if any(_pair_match(e, g) for e in expected))
    precision = mg / len(got) if got else 0.0
    recall = me / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3)}


def evaluate(raw, expected):
    """打分：JSON 可解析 40 + Schema 校验 30 + 内容准确 30，满分 100。

    三部分故意分开记：JSON Mode 的典型表现是 40 + 0 + 0——语法过了、
    字段全乱、内容其实还在但键名对不上。分开记分才能看清每层机制
    到底"管什么、不管什么"，这正是本次实验要回答的问题。
    """
    data, repairs, stage = parse_json_robust(raw)
    # data 原样存进 metrics：报告结论要用它现场举证"模型实际输出了哪些键"
    m = {"parse_stage": stage, "repairs": repairs, "data": data}

    # 第一部分：可解析性。裸输出成功才给满分，修复后成功只给 25
    m["parse_score"] = {"raw": 40, "fenced": 25, "extracted": 25, "failed": 0}[stage]

    # 第二部分：Schema 合规。解析失败就无从校验，直接记一条错误
    errors = validate(data, SCHEMA) if data is not None else ["JSON 解析失败，无法校验"]
    m["schema_errors"] = errors
    m["schema_score"] = max(0, 30 - 10 * len(errors))

    # 第三部分：内容准确。解析失败时三类 F1 全空、内容分 0（失败级联）
    if data is None:
        m.update({"f1_people": {}, "f1_dates": {}, "f1_events": {}, "avg_f1": 0.0})
    else:
        # 三类实体分别和 texts.json 里的期望值对齐，再取平均
        f1s = {
            "people": _f1(expected["people"], data.get("people") or []),
            "dates": _f1(expected["dates"], data.get("dates") or []),
            "events": _f1(expected["events"], data.get("events") or []),
        }
        m["f1_people"], m["f1_dates"], m["f1_events"] = \
            f1s["people"], f1s["dates"], f1s["events"]
        m["avg_f1"] = round(sum(v["f1"] for v in f1s.values()) / 3, 3)

    m["content_score"] = round(m["avg_f1"] * 30)
    m["score"] = m["parse_score"] + m["schema_score"] + m["content_score"]
    return m


def run_experiment():
    """跑完全部 文本 × 方法 组合，边跑边打印，返回完整结果列表。

    失败的调用也记进 results（ok=False）：报告要统计 API 错误率，
    "网关不支持某个参数"本身就是实验结论的一部分，不能静默吞掉。
    """
    results = []
    for t in TEXTS:
        for key, name in METHODS:
            print(f"\n{'=' * 62}")
            print(f"[{t['id']} · {name}] {t['type']}")
            print("=" * 62)
            r = ask(key, t["text"])
            if not r["ok"]:
                print(f"  调用失败：{r['error']}")
                results.append({
                    "text_id": t["id"], "text_type": t["type"], "method": key,
                    "method_name": name, "ok": False, "error": r["error"],
                    "raw": "", "metrics": {}, "usage": {},
                })
                continue

            m = evaluate(r["raw"], t["expected"])
            u = r["usage"]
            think = u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            print(r["raw"] or "(空输出)")
            err_txt = f" Schema错误={m['schema_errors']}" if m["schema_errors"] else ""
            print(f"\n  [judge] 解析={m['parse_stage']}({m['parse_score']}) "
                  f"Schema={m['schema_score']}/30{err_txt}")
            print(f"  [judge] F1 人名={m['f1_people'].get('f1', '-')} "
                  f"日期={m['f1_dates'].get('f1', '-')} "
                  f"事件={m['f1_events'].get('f1', '-')} "
                  f"总分={m['score']}/100")
            print(f"  [usage] prompt={u.get('prompt_tokens', '?')} "
                  f"completion={u.get('completion_tokens', '?')} "
                  f"(思考 {think}) finish_reason={r['finish_reason']!r} "
                  f"source={r['source']}")
            if r["note"]:
                print(f"  [note] {r['note']}")

            results.append({
                "text_id": t["id"], "text_type": t["type"], "method": key,
                "method_name": name, "ok": True,
                "raw": r["raw"], "source": r["source"],
                "finish_reason": r["finish_reason"], "usage": u,
                "think_tokens": think, "note": r["note"],
                "prompt_chars": r["prompt_chars"],
                "metrics": m,
                "expected": t["expected"],
            })
    return results


def aggregate(results):
    """按方法汇总：裸解析/需修复/失败次数、Schema 通过数、三类 F1、平均分。

    汇总口径和打分规则一一对应，报告 4.1 的表格直接消费这里的行；
    API 错误单独挂在 api_errors 字段，结论里据此提示"机制可能不被支持"。
    """
    rows = []
    for key, name in METHODS:
        rs = [r for r in results if r["method"] == key and r.get("ok")]
        failed = [r for r in results if r["method"] == key and not r.get("ok")]
        n = len(rs)
        row = {"method": key, "method_name": name, "n_ok": n,
               "n_api_error": len(failed)}
        if failed:
            row["api_errors"] = [r["error"][:200] for r in failed]
        if n:
            stages = [r["metrics"]["parse_stage"] for r in rs]
            row.update({
                "parse_raw": stages.count("raw"),
                "parse_fenced": stages.count("fenced"),
                "parse_extracted": stages.count("extracted"),
                "parse_failed": stages.count("failed"),
                "schema_pass": sum(1 for r in rs if not r["metrics"]["schema_errors"]),
                "avg_f1_people": round(sum(r["metrics"]["f1_people"].get("f1", 0) for r in rs) / n, 3),
                "avg_f1_dates": round(sum(r["metrics"]["f1_dates"].get("f1", 0) for r in rs) / n, 3),
                "avg_f1_events": round(sum(r["metrics"]["f1_events"].get("f1", 0) for r in rs) / n, 3),
                "avg_score": round(sum(r["metrics"]["score"] for r in rs) / n, 1),
                "avg_prompt": round(sum(r["usage"].get("prompt_tokens", 0) for r in rs) / n),
                "avg_completion": round(sum(r["usage"].get("completion_tokens", 0) for r in rs) / n),
                "tool_call_rate": round(sum(1 for r in rs if r.get("source") == "tool_calls") / n, 2),
            })
        rows.append(row)
    return rows


def fmt_bool(v):
    return "✅" if v else "❌"


def clean_cell(s, limit=80):
    """表格单元格安全化：去换行、去竖线、截断。"""
    s = str(s).replace("\n", " ").replace("|", "/").strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def build_report(results, aggs):
    """拼装 Markdown 报告：概念对比(静态) + 实验结果(动态) + 结论(半动态)。

    第一节是手写的概念梳理，与实验数据无关；四、五节全部由数据生成。
    结论里凡涉及"模型实际输出了什么"的判断，都从 results 现场取证
    （_actual_keys / _structure_desc），不写死——temperature=0.2 下
    模型行为仍有波动，写死了就会和 result.json 对不上。
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []  # 报告按行拼装，最后 "\n".join 一次性写出
    a = L.append

    a("# JSON 结构化输出机制对比实验报告（JSON Mode vs Function Calling）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 模型：`{MODEL}`")
    a(f"- 采样参数：temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}")
    a(f"- 文本数：{len(TEXTS)} × 方法数：{len(METHODS)} = "
      f"{len(TEXTS) * len(METHODS)} 次调用")
    a("- 任务：抽取 人名 / 日期 / 事件 + 一句话摘要")
    a("")
    a("## 一、JSON Mode 和 Function Calling 的区别")
    a("")
    a("### 1.1 一句话区分")
    a("")
    a("- **JSON Mode**：约束**输出的形态**——保证 `content` 是一段合法 JSON。")
    a("  至于字段叫什么、类型对不对、内容准不准，API 层不管。")
    a("- **Function Calling**：约束**输出的意图**——模型以“我要调用某个函数”的形式回答，")
    a("  产出 函数名 + 参数 JSON，参数结构由挂在 `tools` 上的 JSON Schema 描述。")
    a("  模型并不真的执行函数，执行方是你的代码。")
    a("")
    a("### 1.2 格式对比")
    a("")
    a("| 维度 | JSON Mode | Function Calling |")
    a("|---|---|---|")
    a('| 开启方式 | `response_format: {"type": "json_object"}` | `tools: [...]` + `tool_choice` |')
    a("| Schema 写在哪 | API 层不带 Schema，只能写在 Prompt 里（严格模式除外） "
      "| `tools[0].function.parameters`，直接传 JSON Schema |")
    a("| 模型返回位置 | `choices[0].message.content` "
      "| `choices[0].message.tool_calls[0].function.arguments` |")
    a("| 返回的内容 | 一段 JSON 文本 | 函数名 + 参数 JSON 文本 |")
    a("| finish_reason | `stop` | `tool_calls` |")
    a("| 保证什么 | 语法合法（是 JSON） | 参数按 Schema 生成、函数名来自候选列表 |")
    a("| 不保证什么 | 字段齐全、类型正确、内容准确 | 参数值真实准确（模型可能幻觉） |")
    a("| 对话能否继续 | 通常就此结束 "
      "| 可把执行结果以 `role=tool` 回传继续对话 → 多轮工具循环 |")
    a("| 典型场景 | 信息抽取、格式转换、结构化写作 "
      "| Agent 调工具：搜索、下单、查库、执行代码 |")
    a("| 额外成本 | 几乎为零 | tools 定义计入 prompt tokens |")
    a("")
    a("### 1.3 请求 / 响应长什么样")
    a("")
    a("两种机制的请求体只有一小块不同（本实验脚本里全部收敛在 `build_extra()`）：")
    a("")
    a("```json")
    a("// JSON Mode：加一个字段")
    a('{')
    a('  "model": "...",')
    a('  "messages": [{"role": "user", "content": "……"}],')
    a('  "response_format": {"type": "json_object"}')
    a('}')
    a("")
    a("// Function Calling：把 Schema 挂到 tools 上，并强制调用")
    a('{')
    a('  "model": "...",')
    a('  "messages": [{"role": "user", "content": "……"}],')
    a('  "tools": [{')
    a('    "type": "function",')
    a('    "function": {')
    a('      "name": "save_extraction",')
    a('      "description": "保存从文本中抽取的结构化信息",')
    a('      "parameters": { ...同一段 JSON Schema... }')
    a("    }")
    a("  }],")
    a('  "tool_choice": {"type": "function", "function": {"name": "save_extraction"}}')
    a("}")
    a("```")
    a("")
    a("响应里的 JSON 藏的位置不同：")
    a("")
    a("```json")
    a('// JSON Mode 响应：JSON 字符串藏在 content 里，finish_reason=stop')
    a('{')
    a('  "choices": [{')
    a('    "finish_reason": "stop",')
    a('    "message": {')
    a('      "role": "assistant",')
    a('      "content": "{\\"people\\": [\\"李明\\"], \\"dates\\": [...], ...}"')
    a("    }")
    a("  }]")
    a("}")
    a("")
    a('// Function Calling 响应：JSON 字符串藏在 arguments 里，finish_reason=tool_calls')
    a('{')
    a('  "choices": [{')
    a('    "finish_reason": "tool_calls",')
    a('    "message": {')
    a('      "role": "assistant",')
    a('      "content": null,')
    a('      "tool_calls": [{')
    a('        "id": "call_1",')
    a('        "type": "function",')
    a('        "function": {')
    a('          "name": "save_extraction",')
    a('          "arguments": "{\\"people\\": [\\"李明\\"], \\"dates\\": [...], ...}"')
    a("        }")
    a("      }]")
    a("    }")
    a("  }]")
    a("}")
    a("```")
    a("")
    a("关键观察：**两者最终都产出一段 JSON 字符串**，区别只在——")
    a("")
    a("1. 位置不同：`content` vs `tool_calls[0].function.arguments`；")
    a("2. 语义不同：一段数据 vs 一次函数调用；")
    a("3. Schema 的位置不同：Mode 不带 Schema（只能写进 Prompt），FC 的 Schema 是 API 请求体的一部分。")
    a("")
    a("### 1.4 两者的关系与演进")
    a("")
    a("- Function Calling 的对话循环：模型给 `arguments` → 你的代码执行真正的函数 → "
      "把结果以 `role: \"tool\"` 消息发回 → 模型基于结果继续生成。JSON Mode 没有这个循环，")
    a("  它只是一次性输出。")
    a("- 可以把 FC 理解为 JSON Mode 的“意图版”：都产出 JSON，但 FC 把“要什么字段、什么类型”")
    a("  升级成了 API 协议的一部分（`parameters` 就是 JSON Schema）。")
    a("- 新一代 API 把两者合体：OpenAI Structured Outputs（`response_format: json_schema`）")
    a("  和 Anthropic 的 strict tool use，都是“Schema 进协议 + 受限解码”，保证 100% 语法合规。")
    a("")
    a("## 二、实验设计")
    a("")
    a("- **控制变量**：同一个模型、同 3 段文本、temperature=0.2、max_tokens=2000；")
    a("  plain / json_mode / function_calling 三种方法用**同一条基础指令**，只改 API 层参数。")
    a("- **自变量**：结构化输出的约束机制。Schema 本身对四种方法完全相同。")
    a("")
    a("| 方法 | Prompt | API 层参数 | 想验证什么 |")
    a("|---|---|---|---|")
    a("| plain | 只说“以 JSON 格式输出” | 无 | 最裸的口头约定有多不可靠 |")
    a("| json_mode | 同 plain | `response_format=json_object` | Mode 保证可解析，但保证 Schema 吗 |")
    a("| schema_prompt | 第 2 套模板（Schema 写进 Prompt） | 无 | Prompt 层给 Schema 能做到什么程度 |")
    a("| function_calling | 同 plain | `tools` + 强制 `tool_choice` | 协议层给 Schema 字段是不是最稳 |")
    a("")
    a("评分规则（100 分）：")
    a("")
    a("- JSON 可解析 40 分：裸输出直接 `json.loads` 成功 40；需要修复（去围栏/截取子串）25；失败 0")
    a("- Schema 校验 30 分：required 齐全、类型正确、无多余字段，每个错误扣 10")
    a("- 内容准确 30 分：人名/日期/事件三类 F1 平均 × 30；人名日期按“双向包含”匹配，")
    a("  事件按备选关键词命中；日期必须保留原文写法（写成 2026-09-15 会判不匹配）")
    a("")
    a("### 测试文本与期望值")
    a("")
    for t in TEXTS:
        e = t["expected"]
        events = "；".join("/".join(alts) for alts in e["events"])
        a(f"- **{t['id']}（{t['type']}）**：{t['text'][:40]}……")
        a(f"  - 人名：{'、'.join(e['people'])}｜日期：{'、'.join(e['dates'])}｜事件关键词：{events}")
    a("")
    a("## 三、Prompt 模板与 JSON Schema 全文")
    a("")
    a("### 3.1 基础指令（plain / json_mode / function_calling 共用）")
    a("")
    a("```text")
    a(PLAIN_TPL)
    a("```")
    a("")
    a("### 3.2 第 2 套结构化 Prompt 模板（Schema 约束，schema_prompt 方法使用）")
    a("")
    a("第 1 套是 w2day01 的 LangGPT 结构化模板（管“人设与回答风格”）；")
    a("这一套管“机器可校验的输出格式”：把 JSON Schema 直接写进 Prompt，")
    a("并用硬性约束条款钉死解析细节。`{{SCHEMA}}`、`{{TEXT}}` 是占位符，脚本运行时替换，")
    a("换一个 Schema 就能复用到任何抽取任务：")
    a("")
    a("````markdown")
    a(SCHEMA_TPL)
    a("````")
    a("")
    a("### 3.3 目标 JSON Schema（四种方法完全相同）")
    a("")
    a("```json")
    a(SCHEMA_TEXT)
    a("```")
    a("")
    a("## 四、实验结果")
    a("")
    a("### 4.1 汇总（按方法）")
    a("")
    a("| 方法 | 成功 | 裸解析 | 需修复 | 解析失败 | Schema通过 | 人名F1 | 日期F1 | 事件F1 | 平均分 | 平均prompt | 平均completion | tool_calls占比 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in aggs:
        if not x.get("n_ok"):
            a(f"| {x['method_name']} | ❌ 调用失败({x['n_api_error']}) | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        a(f"| {x['method_name']} | {x['n_ok']}/3 | {x['parse_raw']} | "
          f"{x['parse_fenced'] + x['parse_extracted']} | {x['parse_failed']} | "
          f"{x['schema_pass']}/3 | {x['avg_f1_people']:.2f} | {x['avg_f1_dates']:.2f} | "
          f"{x['avg_f1_events']:.2f} | {x['avg_score']:.1f} | {x['avg_prompt']} | "
          f"{x['avg_completion']} | {x['tool_call_rate']:.0%} |")
    a("")
    failed_runs = [r for r in results if not r.get("ok")]
    if failed_runs:
        a("调用失败的请求（网关不支持对应参数，或超时）：")
        a("")
        for r in failed_runs:
            err = clean_cell(r.get("error", ""), 120)
            a(f"- {r['text_id']} · {r['method_name']}：{err}")
        a("")
    a("### 4.2 逐条结果")
    a("")
    a("| 文本 | 方法 | JSON来自 | 解析 | 解析分 | Schema | 内容分 | 总分 | completion | 思考 | finish_reason |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if not r.get("ok"):
            a(f"| {r['text_id']} | {r['method_name']} | - | ❌ 调用失败 | - | - | - | - | - | - | - |")
            continue
        m, u = r["metrics"], r["usage"]
        schema_cell = fmt_bool(not m["schema_errors"])
        if m["schema_errors"]:
            schema_cell += f"({m['schema_score']})"
        a(f"| {r['text_id']} | {r['method_name']} | {r['source']} | "
          f"{m['parse_stage']} | {m['parse_score']} | {schema_cell} | "
          f"{m['content_score']} | **{m['score']}** | "
          f"{u.get('completion_tokens', '?')} | {r.get('think_tokens', 0)} | "
          f"{r['finish_reason']} |")
    a("")
    a("### 4.3 同一段文本（T1）四种方法的原始输出对比")
    a("")
    for key, name in METHODS:
        r = next((x for x in results if x["text_id"] == "T1" and x["method"] == key), None)
        a(f"**{name}**" + (f"（{r['source']}，{r['finish_reason']}）" if r and r.get("ok") else "（调用失败）"))
        a("")
        if r and r.get("ok"):
            raw = r["raw"]
            shown = raw[:500] + ("\n……（截断，全文见 result.json）" if len(raw) > 500 else "")
            a("```json")
            a(shown or "(空输出)")
            a("```")
        else:
            a("```text")
            a(clean_cell(r.get("error", ""), 200) if r else "无记录")
            a("```")
        a("")
    a("### 4.4 Schema 校验错误明细")
    a("")
    err_runs = [r for r in results if r.get("ok") and r["metrics"]["schema_errors"]]
    if not err_runs:
        a("本轮所有成功请求全部通过 Schema 校验。")
    else:
        for r in err_runs:
            errs = "；".join(r["metrics"]["schema_errors"])
            a(f"- {r['text_id']} · {r['method_name']}：{clean_cell(errs, 200)}")
    a("")
    notes = [r for r in results if r.get("ok") and r.get("note")]
    if notes:
        a("### 4.5 降级重试记录")
        a("")
        for r in notes:
            a(f"- {r['text_id']} · {r['method_name']}：{clean_cell(r['note'], 200)}")
        a("")
    a("## 五、结论与讨论")
    a("")

    def _actual_keys(method):
        """收集某方法各次输出里实际出现的顶层键组合，用于结论举证。"""
        sigs = []
        for r in results:
            if r["method"] == method and r.get("ok"):
                data = r["metrics"].get("data")
                if isinstance(data, dict) and data:
                    sig = "/".join(str(k) for k in data.keys())
                    if sig not in sigs:
                        sigs.append(sig)
        return sigs

    def _sigs_phrase(sigs, n):
        """把"实际出现的顶层键组合"说成人话：全一样，还是几套写法并存。"""
        if len(sigs) == 1:
            return f"{n} 次输出的顶层键全是 {sigs[0]}"
        return f"顶层键出现 {len(sigs)} 套写法：{'；'.join(sigs)}"

    def _structure_desc(method):
        """描述各次输出里 events 字段是嵌套对象还是扁平字符串。"""
        descs = []
        for r in results:
            if r["method"] == method and r.get("ok"):
                data = r["metrics"].get("data") or {}
                ev = data.get("事件") or data.get("events") or []
                kind = "嵌套对象" if ev and isinstance(ev[0], dict) else "扁平字符串"
                descs.append(f"{r['text_id']} 事件为{kind}")
        return "，".join(descs)

    ok_aggs = [x for x in aggs if x.get("n_ok")]
    if ok_aggs:
        best = max(ok_aggs, key=lambda x: x["avg_score"])
        a(f"- **平均分最高**：{best['method_name']}（{best['avg_score']:.1f}/100）。")
    for x in aggs:
        key = x["method"]
        if not x.get("n_ok"):
            a(f"- **{x['method_name']} 全部调用失败**：当前网关/模型不支持该参数"
              f"（{clean_cell(x.get('api_errors', [''])[0], 100)}）。"
              f"这本身就是真实世界里的一大差异：JSON Mode 和 Function Calling "
              f"都不是 OpenAI 兼容协议的强制项，换一个端点就可能用不了。")
            continue
        if key == "plain":
            fixed = x["parse_fenced"] + x["parse_extracted"]
            if fixed > 0:
                a(f"- **纯 Prompt 约定 {fixed}/3 次需要修复**：口头要求“输出 JSON”挡不住"
                  f"模型加 ``` 围栏或加说明文字，这就是 JSON Mode 存在的理由。")
            sigs = _actual_keys("plain")
            if x["schema_pass"] < x["n_ok"] and sigs:
                a(f"- **纯 Prompt 约定的字段名完全失控**：{_sigs_phrase(sigs, x['n_ok'])}；"
                  f"{_structure_desc('plain')}。F1 全为 0 不是因为内容抽错——"
                  f"内容其实都在——而是字段名和结构每次都不一样，"
                  f"下游代码没有稳定契约可写。Schema 的价值正在这里："
                  f"**不提高模型智商，只钉死契约。**")
        if key == "json_mode":
            if x["parse_failed"] == 0:
                a(f"- **JSON Mode 全部裸解析通过（{x['parse_raw']}/3 raw）**："
                  f"它兑现了“语法合法”的承诺。")
                sigs = _actual_keys("json_mode")
                if x["schema_pass"] < x["n_ok"] and sigs:
                    a(f"  但{_sigs_phrase(sigs, x['n_ok'])}——内容本身没错，"
                      f"却和 Schema 声明的 people/dates/events/summary 对不上，"
                      f"下游代码拿到手照样崩。原因很直白：指令是中文的，"
                      f"又没有任何机制钉死字段名，模型就按中文写键了。")
                    a(f"  **这就是 JSON Mode 和 Schema 的分工：Mode 保证“是 JSON”，"
                      f"Schema 才保证“是哪个 JSON”。**")
                else:
                    a(f"  Schema 通过 {x['schema_pass']}/3，日期 F1 "
                      f"{x['avg_f1_dates']:.2f}：语法合法 ≠ 内容合规，"
                      f"字段类型和原文格式仍要自己校验。")
        if key == "schema_prompt":
            a(f"- **Schema 约束 Prompt（第 2 套模板）**：Schema 通过 {x['schema_pass']}/3，"
              f"平均分 {x['avg_score']:.1f}。它不依赖网关支持任何特殊参数，"
              f"可移植性最好，代价是每条请求多花 Schema 文本的 prompt tokens"
              f"（平均 prompt {x['avg_prompt']}）。")
        if key == "function_calling":
            a(f"- **Function Calling**：{x['n_ok']}/3 成功，"
              f"JSON 来自 tool_calls 的比例 {x['tool_call_rate']:.0%}，"
              f"Schema 通过 {x['schema_pass']}/3。Schema 进了协议层，"
              f"但参数值仍可能偏离原文（幻觉），内容分不会自动满分。")
    a("")
    a("### 怎么选")
    a("")
    a("1. **只要数据，不要动作**（抽取、翻译、格式转换）：JSON Mode 或 Schema 约束 Prompt "
      "就够了；生产上再配 jsonschema/pydantic 校验 + 失败重试，因为 Mode 管语法不管内容。")
    a("2. **要动作**（Agent 调搜索、下单、执行代码）：必须 Function Calling——")
    a("   它的 `role=tool` 回传循环是 Agent 执行闭环的基础。可以说：")
    a("   **Function Calling 是 Agent 的手，JSON Mode 是 Agent 的嘴。**")
    a("3. **网关不支持 API 层机制时**：第 2 套 Schema 约束模板是纯 Prompt 的平替，"
      "任何 OpenAI 兼容端点都能用。")
    a("4. **三者可以叠加**：JSON Mode + Prompt 里写 Schema + 客户端校验重试，"
      "是很多生产管线的实际形态；structured outputs（json_schema 严格模式）"
      "则是把这条路做进 API 的终态。")
    a("")
    a("### 实验局限")
    a("")
    a("- 每种方法只有 3 段文本、各跑 1 次，样本小；temperature=0.2 下仍可能有波动。")
    a("- 判分里的 F1 用“双向包含/关键词命中”近似匹配，比精确匹配宽松，"
      "但比人工评判严格，规则全部公开可复现。")
    a("- 事件没有唯一正确答案，期望值是我们预设的关键词组，模型换一种概括方式会被扣分。")
    a("- 若网关对 json_object / tools 的支持程度与原生 OpenAI 不同，结论不能直接推广。")
    a("")
    a("## 六、复现方式")
    a("")
    a("```bash")
    a("python 01Agent/w2day04/json_structured_output.py")
    a("```")
    a("")
    a("配置读取顺序：`vision_config.json` 优先，环境变量兜底；密钥不会写进脚本或报告。")
    a("")
    return "\n".join(L)


def main():
    """完整流程：跑实验 → 按方法汇总 → 落盘原始数据 → 生成报告。"""
    print(f"[config] base_url={BASE_URL}  model={MODEL}")
    results = run_experiment()
    aggs = aggregate(results)

    print("\n" + "=" * 62)
    print("汇总（按方法）")
    print("=" * 62)
    for x in aggs:
        if not x.get("n_ok"):
            print(f"  {x['method_name']:<22} 调用失败 {x['n_api_error']}/3")
            continue
        print(f"  {x['method_name']:<22} 平均分={x['avg_score']:>5}  "
              f"裸解析={x['parse_raw']}/3  Schema通过={x['schema_pass']}/3  "
              f"F1(人/日/事)={x['avg_f1_people']:.2f}/{x['avg_f1_dates']:.2f}/"
              f"{x['avg_f1_events']:.2f}")

    # 原始数据先落盘（含每次 raw 输出和解析结果）：
    # 报告随时可以重新生成，但调用证据丢了就没了
    payload = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "schema": SCHEMA,
        },
        "results": results,
        "aggregate": aggs,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(results, aggs))

    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 原始数据：{RESULT_PATH}")


if __name__ == "__main__":
    main()
