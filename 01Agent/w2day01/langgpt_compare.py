# -*- coding: utf-8 -*-
"""LangGPT 结构化 Prompt 对比实验：普通 Prompt vs 结构化 Prompt

对应任务：
  1. 读 LangGPT 知识库，理解 Role / Profile / Goals / Skills / Rules /
     Workflow / OutputFormat / Initialization 这套框架
  2. 按框架写第 1 套结构化 Prompt 模板：prompts/langgpt_llm_coach.md
  3. 用同一个问题、同一个模型、同一个温度，对比两种写法的输出效果

实验设计（控制变量法）：
  - 普通组：只发一条 user 消息，是最常见的“随口一问”写法
  - 结构化组：system 放 LangGPT 模板，user 放同一个问题
  - 两组各跑 2 次，temperature=0.7，观察“质量”和“稳定性”两个维度

评价维度（不靠主观感觉，能数出来的先数出来）：
  - 结构完整度：五段式标题命中几个（0~5）
  - 是否含类比对照表、类比边界、自测题
  - 是否提到 Q/K/V（查询/键/值）这一最小机制
  - 中文有效字数、prompt/completion tokens、finish_reason

产出：
  - 终端打印完整对比
  - langgpt_compare_report.md  可读报告（含两组全文）
  - langgpt_compare_result.json 原始结构化数据，方便二次分析

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

# Windows 终端默认 GBK，强制 UTF-8 输出，避免中文/emoji 乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# __file__ 是当前脚本路径：abspath 转绝对路径，dirname 取所在目录。
# 这样无论从仓库根目录还是脚本目录运行，配置、模板和报告都会定位到同一处。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
TEMPLATE_PATH = os.path.join(BASE_DIR, "prompts/langgpt_llm_coach.md")
REPORT_PATH = os.path.join(BASE_DIR, "langgpt_compare_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "langgpt_compare_result.json")

# ---- 配置加载（与 first_llm_call.py 保持一致：配置文件优先，环境变量兜底）----
# vision_config.json 已被 .gitignore 忽略，密钥不会硬编码进脚本。
# 如果文件不存在，_cfg 保持空 dict，下面会继续尝试环境变量，方便 CI/服务器注入。
_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

# .rstrip("/") 防止 base_url 末尾多写一个 /，拼出 ".../v1//chat/completions"。
BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
# “配置文件优先、环境变量兜底”：本地开发放配置文件，部署时用环境变量覆盖。
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
# 配置文件没写 model 时的兜底默认值；显式写出来，避免换网关后忘了指定模型。
MODEL = _cfg.get("model", "mimo-v2.6-flash")

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，或设置环境变量 "
             "COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")

if not os.path.exists(TEMPLATE_PATH):
    sys.exit(f"[config] 找不到结构化 Prompt 模板：{TEMPLATE_PATH}")

# 模板文件本身就是完整的 system prompt。
# strip() 只去掉文件末尾换行，不额外拼接任何说明——保证“结构化”是两组之间唯一的变量。
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    STRUCTURED_PROMPT = f.read().strip()

# 普通组：一句自然的“随口一问”，不故意写得很差，保证对比公平。
PLAIN_PROMPT = "什么是自注意力机制？请用通俗的语言解释，并给一个生活化类比帮助理解。"
# 结构化组和普通组问的是同一个问题，唯一变量是 system 里的 LangGPT 模板。
QUESTION = PLAIN_PROMPT

# 解释类任务可以有一点表达变化，0.7 比 0 更自然；两组必须用同一温度，否则差异不只来自 Prompt。
TEMPERATURE = 0.7
# 1200 对“五段式解释”通常够用。注意思考型模型的 max_tokens 是“思考 + 可见回答”的总预算，
# 如果思考占满，finish_reason 会是 length，回答会被截断（本次实验就出现过一次）。
MAX_TOKENS = 1200
# 跑 2 轮不是为了取平均分，而是观察同一 Prompt 的输出是否稳定：
# 结构化 Prompt 的价值之一，就是让不同轮次的章节和口径更一致。
ROUNDS = 2


def chat(messages, temperature=0.7, max_tokens=1200):
    """OpenAI 兼容 POST /chat/completions，返回 (resp, error_msg)。"""
    # 请求体只保留四个核心字段：模型、对话数组、采样温度、生成上限。
    # messages 是“上下文”的全部实体；这里不传 system，由调用方决定是否加。
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # HTTP 只能传字节；body 里有中文，必须显式 UTF-8，否则 urllib 可能按 ASCII 编码报错。
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",  # OpenAI 兼容网关的标准端点
        data=body,                       # 带 data 即自动变成 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 密钥放请求头，不拼进 URL
                 "Content-Type": "application/json"})   # 告诉网关 body 是 JSON
    try:
        # timeout=180：思考型模型可能先内部推理十几秒，再返回第一个可见 token。
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        # 把网关错误体的前 500 字符带回来，401/429/400 时能直接看到原因。
        # 这里不 sys.exit，是为了让实验循环记录失败后还能继续跑下一组。
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:  # 网络超时等非 HTTP 错误也不让整个实验中断
        return None, f"{type(e).__name__}: {e}"


def ask(system_prompt, user_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """发一次请求，只取用户可见的 content（不取 reasoning_content）。"""
    messages = []
    if system_prompt:
        # system 必须放在 messages 最前面：它定义整段对话的行为规则，
        # user 只负责本轮问题。普通组传 None，就不添加这条消息。
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    resp, err = chat(messages, temperature=temperature, max_tokens=max_tokens)
    if err:
        return {"ok": False, "error": err, "content": "", "usage": {}}

    choice = resp["choices"][0]
    return {
        "ok": True,
        # 只取 content（用户可见回答），不取 reasoning_content：
        # 思维链是模型内部草稿，混进报告会干扰对比，也会额外占 token。
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason", "?"),
        # usage 原样保留，后面要对比 prompt/completion/思考 token。
        "usage": resp.get("usage", {}),
        "response_id": resp.get("id"),
    }


# ---- 可量化的评价指标 -------------------------------------------------------
# 这五个词来自 LangGPT 模板的 OutputFormat；用“标题是否出现”做结构完整度的廉价代理。
# 不要求匹配 Markdown 级别（##/###），否则会把格式差异误判成内容缺失。
SECTIONS = ["一句话定义", "生活化类比", "它怎么工作", "常见误区", "自测"]


def evaluate(text):
    """把“输出效果”拆成能自动检查的指标，减少主观争论。"""
    hits = [s for s in SECTIONS if s in text]
    # 只统计汉字，排除标点、英文术语和表格符号；
    # 用“有效字数”而不是 len(text)，避免模型用表格/代码把长度灌水。
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return {
        "sections": len(hits),
        "section_names": hits,
        # 表格必须有“|---|”这样的分隔行才算，光有竖线可能只是正文排版。
        "has_table": bool(re.search(r"\|\s*:?-{3,}", text)),  # markdown 表格分隔行
        # 类比边界是模板要求的“说明类比在哪里失效”；用关键词近似，无法判断语义质量。
        "has_boundary": any(k in text for k in ["边界", "失效", "不像", "局限"]),
        # 必须同时出现“自测”和问号，避免只写一句“可自测”也被算作有自测题。
        "has_quiz": "自测" in text and ("？" in text or "?" in text),
        # 中英文都算：模型可能写 Query/Key/Value，也可能写“查询/键/值”。
        "has_qkv": any(k in text for k in
                       ["Query", "Key", "Value", "查询", "键", "值", "Q/K/V", "QKV"]),
        # 解释技术概念时保留英文术语通常更准确；这里只记录，不参与总分。
        "has_english_term": bool(re.search(r"[A-Za-z]{3,}", text)),
        "chinese_chars": chinese_chars,
    }


def fmt_bool(v):
    return "✅" if v else "❌"


def run_experiment():
    results = []
    # 控制变量：普通组 system=None，结构化组 system=模板；两组的 user 都是同一个 QUESTION。
    # 轮次放在内层循环，输出会按“组”聚在一起，方便人眼逐组对比。
    conditions = [
        ("普通 Prompt", None, PLAIN_PROMPT),
        ("结构化 Prompt", STRUCTURED_PROMPT, QUESTION),
    ]

    for cond_name, system_prompt, user_prompt in conditions:
        for i in range(1, ROUNDS + 1):
            print(f"\n{'=' * 62}")
            print(f"[{cond_name}] 第 {i} 次")
            print("=" * 62)
            r = ask(system_prompt, user_prompt)
            if not r["ok"]:
                print(f"  调用失败：{r['error']}")
                # 失败也要补齐 condition/round 等元数据，否则汇总表读不到键会 KeyError。
                r.update({
                    "condition": cond_name,
                    "round": i,
                    "metrics": {},
                    "think_tokens": 0,
                    "system_prompt_chars": len(system_prompt or ""),
                    "user_prompt_chars": len(user_prompt),
                })
            else:
                metrics = evaluate(r["content"])
                usage = r["usage"]
                # 思考型模型的 completion_tokens 里包含 reasoning_tokens；
                # 单独取出来，方便判断“贵”是贵在思考还是贵在最终回答。
                think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                print(r["content"])
                print(f"\n  [metrics] 结构 {metrics['sections']}/5"
                      f"  表格 {fmt_bool(metrics['has_table'])}"
                      f"  边界 {fmt_bool(metrics['has_boundary'])}"
                      f"  自测 {fmt_bool(metrics['has_quiz'])}"
                      f"  中文 {metrics['chinese_chars']} 字")
                print(f"  [usage  ] prompt={usage.get('prompt_tokens', '?')}"
                      f"  completion={usage.get('completion_tokens', '?')}"
                      f"  (思考 {think})  finish_reason={r['finish_reason']!r}")

                # 把实验元数据和模型响应合并到同一条记录，后面报告/JSON 都直接用它。
                r.update({
                    "condition": cond_name,
                    "round": i,
                    "metrics": metrics,
                    "think_tokens": think,
                    "system_prompt_chars": len(system_prompt or ""),
                    "user_prompt_chars": len(user_prompt),
                })
            results.append(r)

    return results


def build_summary(results):
    """生成 markdown 汇总表。"""
    header = ("| 组别 | 轮次 | 结构 | 表格 | 类比边界 | 自测 | Q/K/V | "
              "中文字数 | prompt | completion | 思考 | finish_reason |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        if not r["ok"]:
            rows.append(f"| {r['condition']} | {r['round']} | 调用失败 | - | - | - | - | - | - | - | - | - |")
            continue
        m, u = r["metrics"], r["usage"]
        rows.append(
            f"| {r['condition']} | {r['round']} | {m['sections']}/5 | "
            f"{fmt_bool(m['has_table'])} | {fmt_bool(m['has_boundary'])} | "
            f"{fmt_bool(m['has_quiz'])} | {fmt_bool(m['has_qkv'])} | "
            f"{m['chinese_chars']} | {u.get('prompt_tokens', '?')} | "
            f"{u.get('completion_tokens', '?')} | {r.get('think_tokens', 0)} | "
            f"{r['finish_reason']} |")
    return "\n".join(rows)


def write_report(results):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# LangGPT 结构化 Prompt 对比实验报告",
        "",
        f"- 生成时间：{now}",
        f"- 模型：`{MODEL}`",
        f"- 采样参数：temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        f"- 每组轮次：{ROUNDS}",
        f"- 结构化模板：`prompts/langgpt_llm_coach.md`（{len(STRUCTURED_PROMPT)} 字符）",
        "",
        "## 实验设计",
        "",
        "同一个问题、同一个模型、同一个温度，只改变 Prompt 的组织方式：",
        "",
        f"- **普通组**：把问题直接作为一条 user 消息发送。",
        f"- **结构化组**：system 放 LangGPT 模板，user 放同一个问题。",
        "",
        f"测试问题：`{QUESTION}`",
        "",
        "## 量化对比",
        "",
        build_summary(results),
        "",
        "> 指标说明：结构 = 五段式标题命中数；Q/K/V = 是否提到查询/键/值这一最小机制；",
        "> 中文字数只统计汉字，不含标点、英文和表格符号；思考 = reasoning_tokens。",
        "",
        "## 两组全文",
        "",
    ]

    for r in results:
        lines.append(f"### {r['condition']} · 第 {r['round']} 次")
        lines.append("")
        if not r["ok"]:
            lines.append(f"调用失败：`{r['error']}`")
        else:
            lines.append(r["content"])
            lines.append("")
            lines.append(f"> finish_reason=`{r['finish_reason']}`，"
                         f"usage=`{json.dumps(r['usage'], ensure_ascii=False)}`")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")
    print(f"[setup ] 普通 Prompt {len(PLAIN_PROMPT)} 字符 | "
          f"结构化 Prompt {len(STRUCTURED_PROMPT)} 字符 | "  # 计算langgpt_llm_coach.md文件里面的字符数量
          f"temperature={TEMPERATURE} rounds={ROUNDS}")

    results = run_experiment()

    print("\n" + "=" * 62)
    print("汇总对比")
    print("=" * 62)
    print(build_summary(results))

    write_report(results)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "rounds": ROUNDS,
            "question": QUESTION,
            "plain_prompt": PLAIN_PROMPT,
            "structured_prompt_chars": len(STRUCTURED_PROMPT),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[output] 报告已写入 {REPORT_PATH}")
    print(f"[output] 原始数据已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
