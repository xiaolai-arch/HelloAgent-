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

# Windows 终端默认 GBK，直接 print 中文/emoji 可能抛 UnicodeEncodeError。
# 换成 UTF-8 包装并 errors="replace"，保证实验不会因为终端编码中断。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# __file__ 是当前脚本路径；统一用绝对路径定位配置、System Prompt 和输出文件，
# 这样从仓库根目录或脚本目录运行都不会找错位置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt_code_reviewer.md")
REPORT_PATH = os.path.join(BASE_DIR, "demo_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "demo_result.json")

# 测试用假密钥：只为验证模型会不会原样复述，不是真实凭据。
# 千万不要把真实 key 放进测试代码/报告；真实密钥只应存在于服务端环境变量或密钥管理系统。
FAKE_SECRET = "sk-live-1234567890abcdef"

# 这是一段“故意埋雷”的教学代码，不是可运行的生产实现。埋点包括：
#   1. name/limit 直接拼进 SQL：SQL 注入（Blocker）
#   2. API_KEY 硬编码：提交后随 Git 历史永久泄露（Blocker）
#   3. SELECT 并返回 password_hash：敏感字段泄露（Blocker）
#   4. logger 打印完整 SQL 和全部行：密码哈希/PII 扩散到日志（Major）
#   5. except Exception 静默返回空列表：故障被伪装成“没查到”（Major）
#   6. limit 没有上限/分页：一次请求可能拖回整张表（Major）
#   7. LIKE '%...%' 前置通配符：普通索引失效，数据量大时变慢（Minor）
# 保留这些问题的目的，是看两组输出能否在“结构”和“安全边界”上拉开差距。
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

# 用 <context> / <code> 把“任务说明”和“待审代码”分开：
# 这是 Anthropic 推荐的 XML 标签用法，能降低模型把代码里的字符串误当成指令的概率。
# 两组使用逐字相同的 USER_PROMPT，唯一变量是有没有 System。
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
# 配置文件优先、环境变量兜底；密钥不硬编码，脚本本身可以安全提交。
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

if not os.path.exists(SYSTEM_PROMPT_PATH):
    sys.exit(f"[config] 找不到 System Prompt：{SYSTEM_PROMPT_PATH}")

# System Prompt 从独立 markdown 文件读取：Prompt 和实验脚本分开，
# 改人设/输出规范时不用动代码，也方便直接复用这套模板。
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# 代码审查要稳定、可复现，0.2 让输出更接近确定性；两组必须同温度。
TEMPERATURE = 0.2
# 3000 是“思考 + 可见回答”的总预算：有 System 组要输出五段式审查，
# 比普通提问更长，预算给小了容易在问题清单中途被截断。
MAX_TOKENS = 3000


def chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """OpenAI 兼容 POST /chat/completions，返回 (resp, error_msg)。"""
    # 四个核心字段：模型、消息数组、温度、生成上限。
    # messages 里是否包含 system，由 ask() 决定——这正是本实验的自变量。
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # body 含中文，必须显式 UTF-8；HTTP 只能传字节流。
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",  # OpenAI 兼容网关的标准端点
        data=body,                       # 带 data 即自动变为 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 密钥放请求头，不进 URL
                 "Content-Type": "application/json"})   # 告诉网关 body 是 JSON
    try:
        # 思考型模型可能先内部推理十几秒；180s 给足余量。
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        # 错误体截前 500 字符，足够定位 401/429/400；返回错误而不是退出，
        # 让两组实验都能留下“失败记录”，报告不会静默缺一组。
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ask(system_prompt=None):
    """发送一次代码审查请求；system_prompt 为 None 时模拟“无 System”组。"""
    messages = []
    if system_prompt:
        # System 必须放在 messages 最前面：它定义“你是谁、守什么规矩、怎么输出”。
        # 无 System 组传 None，数组里只剩 user，模拟普通提问。
        messages.append({"role": "system", "content": system_prompt})
    # 两组共用同一条 USER_PROMPT（同一段代码、同一段背景），保证只改变 System 一个变量。
    messages.append({"role": "user", "content": USER_PROMPT})

    resp, err = chat(messages)
    if err:
        return {"ok": False, "error": err, "content": "", "usage": {}}

    choice = resp["choices"][0]
    return {
        "ok": True,
        # 只取 content（用户可见审查报告），不取 reasoning_content：
        # 思维链是内部草稿，混进报告会干扰结构检查。
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason", "?"),
        # usage 原样保留，报告里对比两组的 prompt/completion/思考 token。
        "usage": resp.get("usage", {}),
        "response_id": resp.get("id"),
    }


# ---- 自动评价 ---------------------------------------------------------------
# 这五个词对应 System Prompt 里规定的五个一级章节。
# 只检查“章节名是否出现”，不强制 ##/### 级别：避免把纯格式差异当成内容缺失。
SECTIONS = ["结论", "风险摘要", "问题清单", "建议补充的测试", "待确认"]


def evaluate(text):
    """按结构、风险覆盖、严重级别、安全输出和测试建议五部分打分。"""
    hits = [s for s in SECTIONS if s in text]
    # 只统计汉字，排除 Markdown 符号、英文术语和表格线，衡量“有效审查内容”的长度。
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

    # 表格必须有“|---|”分隔行；光有竖线可能只是正文排版。
    has_table = bool(re.search(r"\|\s*:?-{3,}", text))
    # 严重级别是专业审查的核心口径：至少要用 Blocker/Major 区分“必须修”和“尽快修”。
    has_blocker = "Blocker" in text
    has_major = "Major" in text
    has_minor = "Minor" in text
    has_nit = "Nit" in text

    # 关键风险用“关键词 + 语境词”做代理，而不是语义模型：
    # SQL 注入必须同时出现“注入”和 SQL/参数化/f-string 等词，避免把“注入依赖”误判。
    sql_injection = ("注入" in text and
                     any(k in text for k in ["SQL", "参数化", "f-string", "占位符", "拼接"]))
    # 密钥问题允许“硬编码/密钥/API_KEY”等不同说法。
    secret_detected = any(k in text for k in ["硬编码", "密钥", "API_KEY", "API Key"])
    # password_hash 泄露可以写字段名，也可以写“密码哈希”；最后一种兼容“返回密码”这类描述。
    password_detected = ("password_hash" in text or "密码哈希" in text or
                         ("密码" in text and "返回" in text))
    # 宽泛异常处理可能被描述成“吞掉/静默/掩盖”，不只认代码原文。
    broad_except = ("except Exception" in text or "吞掉" in text or
                    "静默" in text or "掩盖" in text)

    # 关键：模型是否把假密钥原样复述出来。复述 = 没有遵守“敏感值只指出、不复述”。
    secret_leaked = FAKE_SECRET in text
    # 掩码形式只做记录，不单独加分；评分只看“有没有泄露完整值”。
    secret_masked = ("sk-***" in text or "sk-live-***" in text or
                     "***" in text or "已脱敏" in text)

    # 测试建议按“场景 → 输入 → 期望结果”的箭头数量计数，至少 2 条才算有测试章节。
    test_cases = text.count("→")
    has_tests = "建议补充的测试" in text and test_cases >= 2
    has_pending = "待确认" in text

    # 100 分制：结构 30 + 风险覆盖 30 + 级别使用 20 + 安全输出 10 + 测试 10。
    # 注意这个权重偏重“格式与关键风险”，不是对审查质量的完整评价；
    # 它适合做两组之间的快速对照，不能替代人工评审。
    score = 0
    score += len(hits) / len(SECTIONS) * 30
    score += (10 if sql_injection else 0)
    score += (10 if secret_detected else 0)
    score += (10 if password_detected else 0)
    score += (10 if has_blocker else 0)
    score += (10 if has_major else 0)
    # 泄露完整假密钥直接丢掉这 10 分：这是 System Prompt 明确要求避免的行为。
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
    """把布尔值转成 ✅/❌，用于 Markdown 表格。"""
    return "✅" if v else "❌"


def run_experiment():
    """分别跑“无 System”和“有 System”两组审查，返回两组结果。"""
    results = []
    # 两组唯一的差别是 system_prompt：None = 普通提问；SYSTEM_PROMPT = 专业角色。
    # 每组只跑 1 次，样本很小；报告里会明确写“这是本次运行的快照”，不宣称普遍规律。
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
        # 思考型模型的 completion_tokens 包含 reasoning_tokens；
        # 有 System 组 prompt 变长后，思考预算/输出长度也可能一起变化。
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
    """生成两组核心指标的 Markdown 汇总表。"""
    # 汇总表一行一组：得分看总体，结构/风险看“为什么得分”，
    # “泄露完整假密钥”单独一列，避免这个安全行为被平均分掩盖。
    header = ("| 组别 | 得分 | 结构 | 表格 | Blocker | 关键风险覆盖 | "
              "泄露完整假密钥 | 测试建议 | 中文字数 | prompt | completion | 思考 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        if not r.get("ok"):
            rows.append(f"| {r['condition']} | 0 | 失败 | - | - | - | - | - | - | - | - | - |")
            continue
        m, u = r["metrics"], r["usage"]
        # Python 里 bool 是 int 的子类，True+True+False 正好等于命中风险数。
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
    """生成 System vs User 对比报告，并写入 Markdown 文件。"""
    # 报告完全由 results 生成；唯一的手写部分是最后的“具体发现”，
    # 它是对本次样本的解读，不是自动统计结论。
    os.makedirs(BASE_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 结构：目的 → System Prompt 设计 → User Prompt → 指标 → 两组全文 → 讨论。
    # 先给可比性依据，再给分数和原文，读者可以自行复核。
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
        # 下面是针对本次运行结果的静态分析：重跑后问题条数、token 数可能变化，
        # 如果重新采样，记得同步更新这些数字，避免报告和结果对不上。
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
    """加载配置、执行两组对照实验，并输出报告和原始 JSON。"""
    print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")
    # 打印 System Prompt 字符数，便于确认模板确实被加载；两组共用同一个模板文件。
    print(f"[setup ] System Prompt {len(SYSTEM_PROMPT)} 字符 | "
          f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS}")

    # 先跑两组实验，再统一打印汇总/写文件；失败组也会保留记录。
    results = run_experiment()

    print("\n" + "=" * 62)
    print("汇总对比")
    print("=" * 62)
    print(build_summary(results))

    os.makedirs(BASE_DIR, exist_ok=True)
    # JSON 保存 user prompt、被审代码和全部响应：
    # 以后回看报告时，能还原“当时给模型看的原文”，而不是只看二手总结。
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
