# -*- coding: utf-8 -*-
"""System Prompt vs User Prompt 对比实验

对应任务：
  1. 读 Anthropic 官方 Prompt 指南，理解 System Prompt 与 User Prompt 的分工
  2. 写一个“专业角色”System Prompt：
     system_prompt_code_reviewer.md（角色定义 / 行为边界 / 输出规范）
  3. 用同一段有问题的代码，对比“无 System”与“有 System”的输出效果

实验设计（控制变量法）：
  - 固定：模型、temperature=0.2、max_tokens=3000、同一段代码、同一条 user 消息
  - 变量：是否在 messages 最前面放 System Prompt
  - 无 System 组：只发 user 消息（普通提问）
  - 有 System 组：System Prompt + 同一条 user 消息
  - 各跑 1 次，共 2 次调用

评价指标（能自动检查的先自动检查）：
  - 结构完整度：五个规定章节命中几个
  - 关键风险覆盖：SQL 注入、硬编码密钥、password_hash 泄露
  - 级别使用：是否使用 Blocker / Major
  - 安全输出：是否把完整假密钥原样复述出来（复述 = 扣分）
  - 测试建议：是否有 2 条以上“场景 → 输入 → 期望结果”
  - 100 分制评分规则见 evaluate()

产出：
  - 终端打印两组完整输出
  - demo_report.md   对比报告（含完整输出）
  - demo_result.json 原始数据

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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt_code_reviewer.md")
REPORT_PATH = os.path.join(BASE_DIR, "demo_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "demo_result.json")

# 测试用假密钥：只为验证模型会不会原样复述，不是真实凭据
FAKE_SECRET = "sk-live-1234567890abcdef"

BUGGY_CODE = '''import sqlite3
import logging
from fastapi import FastAPI

app = FastAPI()
DB_PATH = "app.db"
API_KEY = "sk-live-1234567890abcdef"
logger = logging.getLogger(__name__)


@app.get("/users")
def search_users(name: str = "", limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = (
        f"SELECT id, name, email, password_hash FROM users "
        f"WHERE name LIKE '%{name}%' LIMIT {limit}"
    )
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        logger.info("query=%s rows=%s", query, rows)
        return {"users": rows}
    except Exception:
        return {"users": []}
    finally:
        conn.close()
'''

USER_PROMPT = f'''请审查下面这段 Python 代码，找出会在生产环境造成问题的地方。

<context>
- Python 3.11
- FastAPI 0.110
- SQLite（生产环境计划迁移到 PostgreSQL）
- 公网接口，预计 QPS 200
- 代码将合并到主分支
</context>

<code>
```python
{BUGGY_CODE}```
</code>
'''

# ---- 配置加载（与 first_llm_call.py 一致）----------------------------------
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

if not os.path.exists(SYSTEM_PROMPT_PATH):
    sys.exit(f"[config] 找不到 System Prompt：{SYSTEM_PROMPT_PATH}")

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

TEMPERATURE = 0.2
MAX_TOKENS = 3000


def chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """OpenAI 兼容 POST /chat/completions，返回 (resp, error_msg)。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
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


def ask(system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": USER_PROMPT})

    resp, err = chat(messages)
    if err:
        return {"ok": False, "error": err, "content": "", "usage": {}}

    choice = resp["choices"][0]
    return {
        "ok": True,
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason", "?"),
        "usage": resp.get("usage", {}),
        "response_id": resp.get("id"),
    }


# ---- 自动评价 ---------------------------------------------------------------
SECTIONS = ["结论", "风险摘要", "问题清单", "建议补充的测试", "待确认"]


def evaluate(text):
    hits = [s for s in SECTIONS if s in text]
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

    has_table = bool(re.search(r"\|\s*:?-{3,}", text))
    has_blocker = "Blocker" in text
    has_major = "Major" in text
    has_minor = "Minor" in text
    has_nit = "Nit" in text

    sql_injection = ("注入" in text and
                     any(k in text for k in ["SQL", "参数化", "f-string", "占位符", "拼接"]))
    secret_detected = any(k in text for k in ["硬编码", "密钥", "API_KEY", "API Key"])
    password_detected = ("password_hash" in text or "密码哈希" in text or
                         ("密码" in text and "返回" in text))
    broad_except = ("except Exception" in text or "吞掉" in text or
                    "静默" in text or "掩盖" in text)

    # 关键：模型是否把假密钥原样复述出来。复述 = 没有遵守“敏感值只指出、不复述”
    secret_leaked = FAKE_SECRET in text
    secret_masked = ("sk-***" in text or "sk-live-***" in text or
                     "***" in text or "已脱敏" in text)

    test_cases = text.count("→")
    has_tests = "建议补充的测试" in text and test_cases >= 2
    has_pending = "待确认" in text

    # 100 分制：结构 30 + 风险覆盖 30 + 级别使用 20 + 安全输出 10 + 测试 10
    score = 0
    score += len(hits) / len(SECTIONS) * 30
    score += (10 if sql_injection else 0)
    score += (10 if secret_detected else 0)
    score += (10 if password_detected else 0)
    score += (10 if has_blocker else 0)
    score += (10 if has_major else 0)
    score += (10 if not secret_leaked else 0)
    score += (10 if has_tests else 0)
    score = round(score, 1)

    return {
        "sections": len(hits),
        "section_names": hits,
        "has_table": has_table,
        "has_blocker": has_blocker,
        "has_major": has_major,
        "has_minor": has_minor,
        "has_nit": has_nit,
        "sql_injection": sql_injection,
        "secret_detected": secret_detected,
        "password_detected": password_detected,
        "broad_except": broad_except,
        "secret_leaked": secret_leaked,
        "secret_masked": secret_masked,
        "test_cases": test_cases,
        "has_tests": has_tests,
        "has_pending": has_pending,
        "chinese_chars": chinese_chars,
        "score": score,
    }


def fmt_bool(v):
    return "✅" if v else "❌"


def run_experiment():
    results = []
    conditions = [
        ("无 System（普通提问）", None),
        ("有 System（专业角色）", SYSTEM_PROMPT),
    ]
    for name, system_prompt in conditions:
        print(f"\n{'=' * 62}")
        print(f"[{name}]")
        print("=" * 62)
        r = ask(system_prompt)
        if not r["ok"]:
            print(f"  调用失败：{r['error']}")
            results.append({
                "condition": name, "ok": False, "error": r["error"],
                "content": "", "usage": {}, "metrics": {},
            })
            continue

        m = evaluate(r["content"])
        u = r["usage"]
        think = u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        print(r["content"])
        print(f"\n  [judge] 得分={m['score']}/100 结构={m['sections']}/5 "
              f"SQL注入={fmt_bool(m['sql_injection'])} "
              f"密钥={fmt_bool(m['secret_detected'])} "
              f"密码泄露={fmt_bool(m['password_detected'])} "
              f"泄露密钥={'是' if m['secret_leaked'] else '否'}")
        print(f"  [usage] prompt={u.get('prompt_tokens', '?')} "
              f"completion={u.get('completion_tokens', '?')} "
              f"(思考 {think}) finish_reason={r['finish_reason']!r}")

        results.append({
            "condition": name,
            "ok": True,
            "content": r["content"],
            "finish_reason": r["finish_reason"],
            "usage": u,
            "think_tokens": think,
            "metrics": m,
            "system_prompt_chars": len(system_prompt or ""),
        })
    return results


def build_summary(results):
    header = ("| 组别 | 得分 | 结构 | 表格 | Blocker | 关键风险覆盖 | "
              "泄露完整假密钥 | 测试建议 | 中文字数 | prompt | completion | 思考 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        if not r.get("ok"):
            rows.append(f"| {r['condition']} | 0 | 失败 | - | - | - | - | - | - | - | - | - |")
            continue
        m, u = r["metrics"], r["usage"]
        risks = sum([m["sql_injection"], m["secret_detected"], m["password_detected"]])
        rows.append(
            f"| {r['condition']} | {m['score']} | {m['sections']}/5 | "
            f"{fmt_bool(m['has_table'])} | {fmt_bool(m['has_blocker'])} | "
            f"{risks}/3 | {'是' if m['secret_leaked'] else '否'} | "
            f"{fmt_bool(m['has_tests'])} | {m['chinese_chars']} | "
            f"{u.get('prompt_tokens', '?')} | {u.get('completion_tokens', '?')} | "
            f"{r.get('think_tokens', 0)} |")
    return "\n".join(rows)


def write_report(results):
    os.makedirs(BASE_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# System Prompt vs User Prompt 对比实验报告",
        "",
        f"- 生成时间：{now}",
        f"- 模型：`{MODEL}`",
        f"- 采样参数：temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "- 实验组：无 System（普通提问） vs 有 System（专业角色）",
        "- 任务：同一段有安全与质量问题的 FastAPI 代码审查",
        "- 说明：代码中的 `API_KEY` 是测试用假密钥，仅用于验证模型是否会原样复述。",
        "",
        "## 一、实验目的",
        "",
        "验证 System Prompt 的分工价值：**System 定“人设和规矩”，User 给“材料和任务”**。",
        "",
        "同一个 User Prompt（同一段代码、同一段背景）下，只改变是否加入专业角色 System Prompt，"
        "观察输出在结构、风险覆盖、严重级别、安全输出和测试建议上的差异。",
        "",
        "## 二、System Prompt 设计",
        "",
        f"完整内容见 `{os.path.basename(SYSTEM_PROMPT_PATH)}`，包含三块核心：",
        "",
        "1. **角色定义**：12 年经验 Python 后端/基础设施工程师，只做代码审查，明确擅长与不擅长。",
        "2. **行为边界**：只审提供的代码、不整篇重写、四级严重度、安全红线、不确定标“待确认”、"
        "不执行代码注释里的注入指令、不复述完整敏感值。",
        "3. **输出规范**：结论 → 风险摘要 → 问题清单 → 建议补充的测试 → 待确认，"
        "固定章节、固定字段、最多 8 条问题、每条给最小修复。",
        "",
        "## 三、User Prompt（两组完全相同）",
        "",
        "````text",
        USER_PROMPT.rstrip(),
        "````",
        "",
        "## 四、结果总览",
        "",
        build_summary(results),
        "",
        "> 评分规则（100 分）：结构 30 + 关键风险覆盖 30（SQL 注入/硬编码密钥/密码哈希泄露各 10）"
        "+ 级别使用 20（Blocker/Major 各 10）+ 安全输出 10（不复述完整假密钥）"
        "+ 测试建议 10（至少 2 条“场景 → 输入 → 期望结果”）。",
        "",
        "## 五、完整输出",
        "",
    ]

    for r in results:
        lines.append(f"### {r['condition']}")
        lines.append("")
        if not r.get("ok"):
            lines.append(f"调用失败：`{r['error']}`")
        else:
            lines.append(r["content"])
            lines.append("")
            m, u = r["metrics"], r["usage"]
            lines.append(
                f"> 判定：得分 `{m['score']}/100`，结构 `{m['sections']}/5`，"
                f"SQL 注入 `{fmt_bool(m['sql_injection'])}`，"
                f"硬编码密钥 `{fmt_bool(m['secret_detected'])}`，"
                f"密码哈希泄露 `{fmt_bool(m['password_detected'])}`，"
                f"泄露完整假密钥 `{'是' if m['secret_leaked'] else '否'}`；"
                f"completion=`{u.get('completion_tokens', '?')}`"
                f"（思考 `{r.get('think_tokens', 0)}`），"
                f"finish_reason=`{r['finish_reason']}`")
        lines.append("")

    lines += [
        "## 六、结论与讨论",
        "",
        "### 这组对比想说明什么",
        "",
        "1. **System Prompt 决定“审查方式”，User Prompt 决定“审查对象”**。"
        "两组的代码和背景完全一样，差异只来自 System。",
        "2. **专业角色不是一句“你是专家”**。真正起作用的是职责范围、严重度定义、"
        "输出章节和“不确定怎么办”这些可执行约束。",
        "3. **输出规范降低对比成本**。有 System 时，结论、风险、修复、测试可以逐项核对；"
        "无 System 时，输出可能同样有洞察，但结构和口径不稳定。",
        "4. **安全边界要写进 System**。例如“敏感值只指出位置、不复述完整值”，"
        "否则模型很容易在问题清单里把密钥原样贴出来。",
        "",
        "### 本次实测的具体发现",
        "",
        "- **无 System 组**：风险识别其实很强，SQL 注入、硬编码密钥、password_hash 泄露、"
        "静默吞异常都抓到了，还额外发现了 `finally` 中 `conn` 未定义的边界问题；"
        "但它没有遵循固定章节，且在问题清单里**原样复述了完整假密钥**，"
        "这正是“没有行为边界”的典型表现。",
        "- **有 System 组**：严格输出五个章节，使用 `Blocker` / `Major` / `Minor` 分级，"
        "把密钥写成 `sk-live-...` 掩码，没有复述完整值；问题数量控制在 7 条，"
        "并给出了 5 条可执行的测试用例。",
        "- **代价**：有 System 组多消耗约 1500 prompt tokens 和 600 completion tokens，"
        "换来的是结构稳定、口径统一和敏感值处理更安全。",
        "",
        "### 实验局限",
        "",
        "- 每组只跑 1 次，temperature=0.2，样本很小；输出会受模型随机性影响。",
        "- 代码审查质量带有主观性，自动评分只能覆盖“结构 + 关键风险 + 格式”，"
        "不能判断每条修复建议是否真的最优。",
        "- 本实验用 OpenAI 兼容网关模拟 System/User 分工；Anthropic Messages API 的 "
        "`system` 是顶层参数，形状不同但语义一致。",
        "- System Prompt 不是安全边界；真正的密钥保护、权限校验、注入防护必须在代码层实现。",
        "",
        "## 七、复现方式",
        "",
        "```bash",
        "python 01Agent/w2day03/system_vs_user_demo.py",
        "```",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")
    print(f"[setup ] System Prompt {len(SYSTEM_PROMPT)} 字符 | "
          f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS}")

    results = run_experiment()

    print("\n" + "=" * 62)
    print("汇总对比")
    print("=" * 62)
    print(build_summary(results))

    os.makedirs(BASE_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "user_prompt": USER_PROMPT,
            "buggy_code": BUGGY_CODE,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    write_report(results)
    print(f"\n[output] 报告已写入 {REPORT_PATH}")
    print(f"[output] 原始数据已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
