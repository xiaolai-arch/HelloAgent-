# -*- coding: utf-8 -*-
"""Zero-shot / Few-shot / Chain-of-Thought 三种 Prompt 策略对比实验

对应任务：
  1. 分别写 Zero-shot、Few-shot、Chain-of-Thought 三种 Prompt
  2. 用同一组题目、同一个模型、同一个温度，对比输出质量
  3. 把实验报告写到 w2day02

实验设计（控制变量法）：
  - 固定：模型、temperature=0.2、max_tokens=2000、4 道题、只发 user 消息
  - 变量：Prompt 策略（三种模板放在 prompts/ 目录）
  - 题目：4 道集合应用题（含 1 道“只参加一门”语义陷阱题），答案唯一，便于自动判分
  - 每种策略跑完全部 4 题，共 12 次调用

质量评分（100 分制，规则透明、可复现）：
  - 答案正确        60 分
  - 有推理步骤      20 分（至少 2 个步骤标记）
  - 有集合关系/公式 10 分（提到交集、并集、容斥、重叠等）
  - 格式规范        10 分（最后出现“答案：X”）
这套评分特意把“推理过程”计入质量：数学题不仅要答对，还要能说明怎么得到。

产出：
  - 终端打印每次输出与汇总
  - prompt_strategies_report.md  实验报告（含三种 Prompt 全文、逐题结果、完整输出）
  - prompt_strategies_result.json 原始数据

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

# __file__ 是当前脚本路径；统一用绝对路径拼资源，避免从不同目录运行时找不到模板。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions.json")
REPORT_PATH = os.path.join(BASE_DIR, "prompt_strategies_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "prompt_strategies_result.json")

# ---- 配置加载（与 first_llm_call.py 一致：配置文件优先，环境变量兜底）----
# 密钥只放在被 .gitignore 忽略的 vision_config.json 或环境变量里，脚本本身可安全提交。
_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

# .rstrip("/") 防止 base_url 末尾多一个 /，拼出 ".../v1//chat/completions"。
BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
# 配置文件没写 model 时的兜底；当前网关使用 mimo-v2.6-flash。
MODEL = _cfg.get("model", "mimo-v2.6-flash")

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，或设置环境变量 "
             "COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")

# 数学题要的是稳定复现，不是创意：0.2 让采样更接近概率最高的路径，
# 三种策略必须用同一温度，否则正确率差异可能来自随机性而不是 Prompt。
TEMPERATURE = 0.2
# 2000 是“思考 + 可见回答”的总预算；步骤写得长或模型思考久时可能撞到上限，
# 所以报告里要同时看 finish_reason，不能只看答案对不对。
MAX_TOKENS = 2000

# 三种策略：名称 → 模板文件。
# 列表顺序决定终端打印/报告里策略的展示顺序；运行顺序是“题目外层、策略内层”。
STRATEGIES = [
    ("Zero-shot", "zero_shot.md"),
    ("Few-shot", "few_shot.md"),
    ("Chain-of-Thought", "cot.md"),
]


def load_text(path):
    # 统一 UTF-8 读取，并去掉文件末尾换行，避免模板末尾多出空白影响 token 统计。
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def render_prompt(template, question):
    """模板统一用 {{QUESTION}} 占位，避免 str.format 与正文花括号冲突。"""
    # 这里故意不用 str.format：模板里可能有 JSON、代码块或数学花括号，
    # str.format 会把它们当占位符而报错；简单 replace 只替换我们约定的标记。
    return template.replace("{{QUESTION}}", question)


# 题目在启动时一次性载入；expected 是标准答案，用于自动判分。
QUESTIONS = json.loads(load_text(QUESTIONS_PATH))
# 三种模板也只读一次，避免 12 次调用反复读磁盘；键是策略名，值是模板正文。
PROMPTS = {name: load_text(os.path.join(PROMPT_DIR, fname))
           for name, fname in STRATEGIES}


def chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """OpenAI 兼容 POST /chat/completions，返回 (resp, error_msg)。"""
    # 只发四个核心字段：模型、消息数组、温度、生成上限。
    # 本实验不使用 system，messages 里只有一条 user，确保差异只来自三种 Prompt 模板。
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # HTTP body 是字节流；显式 UTF-8 才能安全传中文题目和中文回答。
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",  # OpenAI 兼容网关的标准端点
        data=body,                       # 带 data 即自动变为 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 密钥放请求头，不进 URL
                 "Content-Type": "application/json"})   # 告诉网关 body 是 JSON
    try:
        # 思考型模型可能先推理十几秒；180s 给足余量，避免误判为网络失败。
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        # 错误体只截前 500 字符，足够定位 401/429/400；不中断整个实验。
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ask(prompt):
    """单条 user 消息发一次请求，只取用户可见的 content。"""
    # 关键控制变量：这里不传 system。三种策略的差异只能来自 prompt 本身，
    # 否则 system 里的额外指令会污染“Zero-shot vs Few-shot vs CoT”的对比。
    resp, err = chat([{"role": "user", "content": prompt}])
    if err:
        return {"ok": False, "error": err, "content": "", "usage": {}}

    choice = resp["choices"][0]
    return {
        "ok": True,
        # 只取 content，不取 reasoning_content：思维链是内部草稿，
        # 混进判分文本可能让“答案提取”误抓中间数字。
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason", "?"),
        # 原样保留 usage，后面要对比三种策略的 token 成本。
        "usage": resp.get("usage", {}),
        "response_id": resp.get("id"),
    }


# ---- 自动判分 ---------------------------------------------------------------
def extract_answer(text):
    """优先取“答案：X”，否则取文本里最后一个数字。"""
    # 优先匹配明确的“答案：99”“答案是 99”等写法；允许负号和小数，
    # 因为同一套判分逻辑以后可能复用到其他应用题。
    # 用 findall 而不是 search：模型可能先引用示例答案，最后才给自己的答案，取最后一个更稳。
    m = re.findall(r"答案\s*[:：]\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m[-1])
    # 兜底：没有“答案：”标记时，取全文最后一个数字。
    # 这是启发式——如果最后一句是“所以不是 30”，可能误判；报告里会同时展示原文供人工复核。
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return None


def evaluate(text, expected):
    # 汉字数只统计 \u4e00-\u9fff，排除标点、英文和数字，衡量“解释含量”。
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    answer = extract_answer(text)
    # 浮点数比较留 1e-6 容差，避免 66.0 与 66 被当成不同答案。
    correct = answer is not None and abs(answer - expected) < 1e-6

    # 步骤标记用四种信号互相兜底，因为不同策略的排版差异很大：
    #   1) 编号行：1. / 1、 / 1)
    #   2) 中文“第一步/第二步”
    #   3) 3 个以上 Markdown 小标题（CoT 常用）
    #   4) “首先/然后/因此/所以/最后”等连接词至少 2 个
    # 任一信号满足就算“展示了步骤”，不要求固定格式。
    numbered = len(re.findall(r"(?m)^\s*(?:#{1,6}\s*)?\d+[.、)]", text))
    cn_steps = len(re.findall(r"第[一二三四五]步", text))
    headings = len(re.findall(r"(?m)^#{2,6}\s+\S+", text))
    connectors = sum(k in text for k in ["首先", "然后", "因此", "所以", "最后"])
    has_steps = cn_steps >= 2 or numbered >= 2 or headings >= 3 \
                or connectors >= 2

    # 不要求模型写出标准公式符号（∩/∪），只要出现集合关系关键词就算“有公式意识”。
    # 这是关键词代理，不能区分“正确使用”和“提到但用错”，所以答案正确性仍占 60 分。
    has_formula = any(k in text for k in
                      ["交集", "并集", "容斥", "至少", "重叠", "减去", "相加",
                       "两者都", "三种都", "两项都", "两科都", "至少一项", "至少一门"])

    # 允许“答案：至少参加一门的有 99 人”这种自然写法，不再强制“答案：99”。
    # 只看最后 300 个字符：避免把开头示例里的“答案：X”误当成最终答案。
    format_ok = bool(re.search(r"答案\s*[:：][^\n]{0,30}\d", text[-300:]))
    # 权重设计：正确性 60（最重要），步骤 20（过程可解释），公式 10，格式 10。
    # 这样“答对但没过程”最高 80 分，而“过程完整但算错”最多 40 分，正确性始终优先。
    score = (60 if correct else 0) + (20 if has_steps else 0) \
            + (10 if has_formula else 0) + (10 if format_ok else 0)

    return {
        "answer": answer,
        "correct": correct,
        "has_steps": has_steps,
        "has_formula": has_formula,
        "format_ok": format_ok,
        "score": score,
        "chinese_chars": chinese_chars,
        # 去掉空白后的字符数，包含英文/数字；用于观察回答长度，不参与评分。
        "answer_chars": len(re.sub(r"\s", "", text)),
    }


def run_experiment():
    results = []
    # 题目放外层、策略放内层：同一道题的三种策略连续跑，
    # 报告里也能按题目分组展示，方便横向比较“同一题谁做得更好”。
    for q in QUESTIONS:
        for name, _ in STRATEGIES:
            # 每个策略模板都用同一道题替换 {{QUESTION}}，保证输入逐字相同。
            prompt = render_prompt(PROMPTS[name], q["question"])
            print(f"\n{'=' * 62}")
            print(f"[{q['id']} · {name}] {q['type']}")
            print("=" * 62)
            r = ask(prompt)
            if not r["ok"]:
                print(f"  调用失败：{r['error']}")
                # 失败也保留一条完整记录（含题号/策略），否则汇总时 n 会悄悄变小。
                results.append({
                    "question_id": q["id"], "question_type": q["type"],
                    "strategy": name, "ok": False, "error": r["error"],
                    "content": "", "usage": {}, "metrics": {},
                })
                continue

            # 用标准答案 expected 判分；metrics 里同时保留答案、步骤、公式、格式四个维度。
            m = evaluate(r["content"], q["expected"])
            u = r["usage"]
            # 思考型模型的 completion_tokens 里包含 reasoning_tokens；
            # 单独拆出来，才能判断 CoT 的额外成本是花在“写步骤”还是“内部思考”。
            think = u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            print(r["content"])
            print(f"\n  [judge] 答案={m['answer']} 正确={m['correct']} "
                  f"得分={m['score']}/100 步骤={m['has_steps']} "
                  f"公式={m['has_formula']} 格式={m['format_ok']}")
            print(f"  [usage] prompt={u.get('prompt_tokens', '?')} "
                  f"completion={u.get('completion_tokens', '?')} "
                  f"(思考 {think}) finish_reason={r['finish_reason']!r}")

            results.append({
                "question_id": q["id"],
                "question_type": q["type"],
                "question": q["question"],
                "expected": q["expected"],
                "strategy": name,
                "ok": True,
                "content": r["content"],
                "finish_reason": r["finish_reason"],
                "usage": u,
                "think_tokens": think,
                "metrics": m,
                # 记录替换后的完整 prompt 长度（含题目/示例），用于观察模板本身的体积。
                "prompt_chars": len(prompt),
            })
    return results


def aggregate(results):
    """按策略汇总：正确率、平均分、平均 token、格式/步骤合规率。"""
    rows = []
    for name, _ in STRATEGIES:
        # 只聚合成功调用；分母用 n（成功次数），避免失败记录把正确率拉低。
        rs = [r for r in results if r["strategy"] == name and r.get("ok")]
        n = len(rs)
        if n == 0:
            # 某策略全部失败时保留一条 n=0 的记录，报告里能看出“不是没跑，而是全挂了”。
            rows.append({"strategy": name, "n": 0})
            continue
        rows.append({
            "strategy": name,
            "n": n,
            "correct": sum(1 for r in rs if r["metrics"]["correct"]),
            "accuracy": sum(1 for r in rs if r["metrics"]["correct"]) / n,
            "avg_score": sum(r["metrics"]["score"] for r in rs) / n,
            # token 分开统计：prompt 反映模板/示例的固定成本，
            # completion 反映推理/步骤的生成成本，total 才是账单口径。
            "avg_completion": sum(r["usage"].get("completion_tokens", 0) for r in rs) / n,
            "avg_prompt": sum(r["usage"].get("prompt_tokens", 0) for r in rs) / n,
            "avg_total": sum(r["usage"].get("total_tokens", 0) for r in rs) / n,
            "avg_chinese": sum(r["metrics"]["chinese_chars"] for r in rs) / n,
            "format_rate": sum(1 for r in rs if r["metrics"]["format_ok"]) / n,
            "steps_rate": sum(1 for r in rs if r["metrics"]["has_steps"]) / n,
        })
    return rows


def fmt_bool(v):
    # 报告里用 emoji 表示布尔值，比 True/False 更易扫读。
    return "✅" if v else "❌"


def build_summary_table(results):
    # 逐题明细：一行 = 一道题 × 一种策略；失败行保留，避免“少一行”被忽略。
    header = ("| 题号 | 策略 | 答案 | 正确 | 得分 | 步骤 | 公式 | 格式 | "
              "中文字数 | completion | 思考 | finish_reason |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        if not r.get("ok"):
            rows.append(f"| {r['question_id']} | {r['strategy']} | - | ❌ | 0 | - | - | - | - | - | - | 失败 |")
            continue
        m, u = r["metrics"], r["usage"]
        rows.append(
            f"| {r['question_id']} | {r['strategy']} | {m['answer']} | "
            f"{fmt_bool(m['correct'])} | {m['score']} | "
            f"{fmt_bool(m['has_steps'])} | {fmt_bool(m['has_formula'])} | "
            f"{fmt_bool(m['format_ok'])} | {m['chinese_chars']} | "
            f"{u.get('completion_tokens', '?')} | {r.get('think_tokens', 0)} | "
            f"{r['finish_reason']} |")
    return "\n".join(rows)


def build_aggregate_table(aggs):
    # 策略级汇总：先看正确率/平均分，再看 token 成本，避免“只追求高分不看成本”。
    header = ("| 策略 | 正确率 | 平均分 | 平均 prompt | 平均 completion | 平均 total | "
              "平均中文字数 | 格式规范率 | 步骤展示率 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for a in aggs:
        if a["n"] == 0:
            continue
        rows.append(
            f"| {a['strategy']} | {a['correct']}/{a['n']} "
            f"({a['accuracy']:.0%}) | {a['avg_score']:.1f} | "
            f"{a['avg_prompt']:.0f} | {a['avg_completion']:.0f} | "
            f"{a['avg_total']:.0f} | "
            f"{a['avg_chinese']:.0f} | {a['format_rate']:.0%} | "
            f"{a['steps_rate']:.0%} |")
    return "\n".join(rows)


def build_conclusion(aggs):
    # 结论不写死“谁最好”，而是从聚合结果里现算：
    # 平均分最高、正确率最高、总 token 最省，三个维度可能不是同一种策略。
    valid = [a for a in aggs if a["n"] > 0]
    if not valid:
        return "本次实验全部调用失败，无法得出结论。"
    best_score = max(valid, key=lambda a: a["avg_score"])
    best_acc = max(valid, key=lambda a: a["accuracy"])
    cheapest = min(valid, key=lambda a: a["avg_total"])
    # 如果三种策略正确率完全相同，就合并成一句话，避免报告里重复三遍。
    acc_text = (
        f"- **正确率**：三种策略相同（{best_acc['correct']}/{best_acc['n']}，"
        f"{best_acc['accuracy']:.0%}）。\n"
        if all(a["accuracy"] == best_acc["accuracy"] for a in valid)
        else f"- **正确率最高**：{best_acc['strategy']}（{best_acc['correct']}/{best_acc['n']}，"
             f"{best_acc['accuracy']:.0%}）。\n"
    )
    return (
        f"- **平均分最高**：{best_score['strategy']}（{best_score['avg_score']:.1f}/100）。\n"
        + acc_text
        + f"- **总 token 最省**：{cheapest['strategy']}（平均 {cheapest['avg_total']:.0f} total tokens，"
          f"其中 prompt {cheapest['avg_prompt']:.0f} + completion {cheapest['avg_completion']:.0f}）。\n"
        + f"- 注意：本次每种策略只有 {valid[0]['n']} 道题、每道题只跑 1 次，"
          "样本很小，结论应理解为“这次实验中的趋势”，而不是普遍规律。"
    )


def write_report(results, aggs):
    # 报告由 results/aggs 自动生成；重跑脚本会覆盖旧报告，但结构保持一致。
    os.makedirs(BASE_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 报告顺序：目的 → 设计 → Prompt 全文 → 汇总 → 逐题原文 → 结论。
    # 先给方法和模板，读者才能判断结果是否可比；再给全文，避免“只给分数不给证据”。
    lines = [
        "# Zero-shot / Few-shot / Chain-of-Thought 对比实验报告",
        "",
        f"- 生成时间：{now}",
        f"- 模型：`{MODEL}`",
        f"- 采样参数：temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        f"- 题目数：{len(QUESTIONS)}（4 道集合应用题，含 1 道“只参加一门”语义陷阱题）",
        f"- 每种策略：跑完全部题目，共 {len(results)} 次调用",
        "",
        "## 一、实验目的",
        "",
        "比较三种常见 Prompt 策略在**有唯一答案的多步数学题**上的表现：",
        "",
        "| 策略 | 核心做法 | 预期优势 |",
        "|---|---|---|",
        "| Zero-shot | 只给问题，不给示例、不提示步骤 | 最省 token，检验模型原始能力 |",
        "| Few-shot | 给 2 个“问题 → 答案”示例，再问新题 | 用示例锁定输出格式和解题模式 |",
        "| Chain-of-Thought | 要求逐步思考、写公式、最后给答案 | 把推理过程显式化，降低多步计算错误 |",
        "",
        "## 二、实验设计",
        "",
        "- **控制变量**：同一个模型、同一组题、temperature=0.2、max_tokens=2000、"
        "都只发一条 user 消息（不使用 system）。",
        "- **自变量**：Prompt 策略。三种模板见 `prompts/` 目录，脚本会替换 `{{QUESTION}}`。",
        "- **判分规则**（100 分）：答案正确 60 分；有至少 2 个步骤标记 20 分；"
        "提到集合关系/公式 10 分；最后出现“答案：X” 10 分。",
        "- **答案提取**：优先取“答案：X”，否则取文本最后一个数字。",
        "",
        "### 测试题",
        "",
    ]

    for q in QUESTIONS:
        lines.append(f"- **{q['id']}（{q['type']}，答案 {q['expected']}）**：{q['question']}")

    lines += [
        "",
        "## 三、三种 Prompt 全文",
        "",
        "### 3.1 Zero-shot",
        "",
        "```text",
        PROMPTS["Zero-shot"],
        "```",
        "",
        "### 3.2 Few-shot",
        "",
        "```text",
        PROMPTS["Few-shot"],
        "```",
        "",
        "### 3.3 Chain-of-Thought",
        "",
        "```text",
        PROMPTS["Chain-of-Thought"],
        "```",
        "",
        "> 说明：Few-shot 的示例只给“问题 → 答案”，不展示推理过程。"
        "这是为了把“示例”与“显式推理”分开，避免 Few-shot 悄悄变成 Few-shot CoT。",
        "",
        "## 四、结果总览",
        "",
        build_aggregate_table(aggs),
        "",
        "### 逐题明细",
        "",
        build_summary_table(results),
        "",
        "## 五、完整输出",
        "",
    ]

    # 按题目分组贴出三种策略的完整回答：自动分数只是索引，
    # 真正判断“推理是否严谨”还要回看原文。
    for q in QUESTIONS:
        q_results = [r for r in results if r["question_id"] == q["id"]]
        lines.append(f"### {q['id']} · {q['type']}")
        lines.append("")
        lines.append(f"> 题目：{q['question']}")
        lines.append(f"> 标准答案：{q['expected']}")
        lines.append("")
        for r in q_results:
            lines.append(f"#### {r['strategy']}")
            lines.append("")
            if not r.get("ok"):
                lines.append(f"调用失败：`{r['error']}`")
            else:
                lines.append(r["content"])
                lines.append("")
                m, u = r["metrics"], r["usage"]
                lines.append(
                    f"> 判定：答案 `{m['answer']}`，{'正确' if m['correct'] else '错误'}，"
                    f"得分 `{m['score']}/100`；completion=`{u.get('completion_tokens', '?')}`"
                    f"（思考 `{r.get('think_tokens', 0)}`），"
                    f"finish_reason=`{r['finish_reason']}`")
            lines.append("")

    lines += [
        "## 六、结论与讨论",
        "",
        build_conclusion(aggs),
        "",
        "### 怎么解读这组结果",
        "",
        "1. **Correctness 优先**：数学题先看答案对不对，再看过程。过程再漂亮，答案错了也是 0 分。",
        "2. **CoT 的收益与成本**：CoT 通常会增加 completion tokens；如果模型本身已经是思考型模型，"
        "显式 CoT 可能和内部思维链重复，收益不一定抵得过成本。",
        "3. **Few-shot 的真正作用**：示例最擅长锁定格式、语气、字段和解题套路；"
        "它不保证模型真的理解规则，遇到分布外题目可能照样错。",
        "4. **Zero-shot 不是“差”**：简单任务上它最便宜、最快；复杂任务上才需要示例或推理引导。",
        "",
        "### 实验局限",
        "",
        "- 每种策略只有 4 道题、每题 1 次，样本小，且同一模型在不同时间的输出可能波动。",
        "- 题目都是集合/百分比应用题，结论不能直接推广到写作、代码、抽取等任务。",
        "- Few-shot 示例只给答案，若改成“示例 + 推理”，结果可能显著不同（那就接近 Few-shot CoT）。",
        "- 自动判分只看最终数字和结构特征，不能判断推理是否真正严谨。",
        "",
        "## 七、复现方式",
        "",
        "```bash",
        "python 01Agent/w2day02/prompt_strategies.py",
        "```",
        "",
        "配置读取顺序：`vision_config.json` 优先，环境变量兜底；"
        "密钥不会写进脚本或报告。",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")
    # 提前算好总调用次数，跑之前就能确认“3 策略 × 4 题 = 12 次”，避免漏跑。
    print(f"[setup ] 策略={len(STRATEGIES)} 题目={len(QUESTIONS)} "
          f"共 {len(STRATEGIES) * len(QUESTIONS)} 次调用 | "
          f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS}")

    results = run_experiment()
    # 先聚合再打印：终端和报告共用同一份 aggs，避免两处算法不一致。
    aggs = aggregate(results)

    print("\n" + "=" * 62)
    print("汇总对比")
    print("=" * 62)
    print(build_aggregate_table(aggs))
    print("\n" + build_conclusion(aggs))

    os.makedirs(BASE_DIR, exist_ok=True)
    # JSON 保存题目、三种模板全文、逐次结果和聚合值：
    # 报告是给人看的摘要，JSON 是以后换模型/加题时做二次对比的原始数据。
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "questions": QUESTIONS,
            "prompts": PROMPTS,
            "results": results,
            "aggregates": aggs,
        }, f, ensure_ascii=False, indent=2)

    write_report(results, aggs)
    print(f"\n[output] 报告已写入 {REPORT_PATH}")
    print(f"[output] 原始数据已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
