# -*- coding: utf-8 -*-
"""Prompt A/B/C 测试：普通 Prompt vs 角色 Prompt vs 完整结构化 Prompt

对应任务：
  1. 写第 3 套结构化 Prompt 模板：prompts/technical_design_reviewer.md
  2. 设计 A/B 测试：同一份技术方案、同一个模型、同一组参数，只改变 Prompt
  3. 对比输出质量、结构、风险覆盖和 token 成本

实验设计（控制变量法）：
  - 固定：模型、temperature=0.3、max_tokens=3200、同一份技术方案、同一条 user 消息、每组跑 2 次
  - 自变量：Prompt 组织方式
      A 普通 Prompt：只有 user 消息，直接说“评审这份方案”
      B 角色 Prompt：system 只给“12 年经验架构师”的角色和一句要求，不给输出规范
      C 结构化 Prompt：system 使用第 3 套模板（角色 / 目标 / 边界 / 维度 / 输出规范 / 流程）
  - 轮次：A → B → C → A → B → C，降低时间漂移的影响

评分（100 分，规则透明、可复现）：
  - 结构完整度 20：五个规定章节各 4 分
  - 关键风险覆盖 40：10 个埋点风险各 4 分
  - 决策明确度 15：明确结论 5 + 理由 5 + 至少 3 条上线前动作 5
  - 可执行性 15：优先级 5 + 修复列 5 + 验收/验证列 5
  - 待确认与假设 10：有章节 5 + 至少 3 条 5

说明：
  - 这套评分偏向“可检查的交付物”，结构化 Prompt 天然占便宜；因此报告同时给出
    风险覆盖、token 成本和逐条输出，避免只看总分。
  - 自动评分不能判断修复建议是否真的最优，结论应结合人工阅读。

产出：
  - 终端打印每组输出与汇总
  - prompt_ab_test_report.md   数据报告（含三组 Prompt 全文、逐轮结果、完整输出）
  - prompt_ab_test_result.json 原始数据（可离线复算）

配置同前：../../vision_config.json 优先，环境变量兜底。
"""
import sys
import io
import os
import re
import json
import time
import statistics
import datetime
import urllib.request
import urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
DESIGN_DOC_PATH = os.path.join(BASE_DIR, "design_doc.md")
TASK_PATH = os.path.join(PROMPT_DIR, "ab_task.md")
REPORT_PATH = os.path.join(BASE_DIR, "prompt_ab_test_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "prompt_ab_test_result.json")

RUNS = 2
TEMPERATURE = 0.3
MAX_TOKENS = 3200
SCORING_VERSION = 2


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


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

DESIGN_DOC = load_text(DESIGN_DOC_PATH)
TASK = load_text(TASK_PATH).replace("{{DESIGN_DOC}}", DESIGN_DOC)

# 默认用 A/B 报告对应的 v1.0 模板，保证结果可复现；
# 想测当前 v1.1 模板时，设置 AB_TEMPLATE=prompts/technical_design_reviewer.md
_DEFAULT_TEMPLATE = os.path.join(PROMPT_DIR, "technical_design_reviewer_v1.0.md")
_env_template = os.environ.get("AB_TEMPLATE", "").strip()
if _env_template:
    STRUCTURED_PROMPT_PATH = (_env_template if os.path.isabs(_env_template)
                              else os.path.join(BASE_DIR, _env_template))
else:
    STRUCTURED_PROMPT_PATH = _DEFAULT_TEMPLATE

STRUCTURED_PROMPT = load_text(STRUCTURED_PROMPT_PATH)
ROLE_PROMPT = load_text(os.path.join(PROMPT_DIR, "ab_B_role_only.md"))

GROUPS = [
    {
        "key": "A",
        "name": "A · 普通 Prompt",
        "system": None,
        "desc": "只有 user 消息，直接要求评审",
    },
    {
        "key": "B",
        "name": "B · 角色 Prompt",
        "system": ROLE_PROMPT,
        "desc": "只有“12 年经验架构师”的角色设定，没有输出规范",
    },
    {
        "key": "C",
        "name": "C · 结构化 Prompt",
        "system": STRUCTURED_PROMPT,
        "desc": "第 3 套模板：角色 + 目标 + 边界 + 维度 + 输出规范 + 流程",
    },
]

# ---- 关键风险埋点 -----------------------------------------------------------
# 每个风险给一组可观察的关键词/正则；命中任意一个记 4 分。
# 这些词对应 design_doc.md 中有意留下的设计缺陷。
RISKS = [
    {
        "id": "R1",
        "name": "并发超卖 / 名额扣减非原子",
        "patterns": [
            r"超卖", r"先查后插", r"check[- ]?then", r"非原子", r"原子性",
            r"竞态", r"race condition", r"行锁", r"悲观锁", r"乐观锁",
            r"事务.*(名额|报名|插入|扣减)", r"UPDATE.*capacity",
            r"唯一约束.*(名额|报名)",
        ],
    },
    {
        "id": "R2",
        "name": "幂等缺失 / 重复报名",
        "patterns": [
            r"幂等", r"重复报名", r"重复提交", r"重复插入", r"重复请求",
            r"idempot",
            r"唯一(索引|约束)[^\n]{0,40}(phone|手机号|重复报名|报名记录)",
            r"(phone|手机号)[^\n]{0,40}唯一(索引|约束)",
        ],
    },
    {
        "id": "R3",
        "name": "PII 明文存储 / 日志和接口泄露",
        "patterns": [
            r"明文", r"脱敏", r"加密存储", r"全量返回", r"最小必要",
            r"个人信息保护", r"\bPII\b", r"数据泄露",
            r"日志.*(手机号|身份证|请求体|敏感)",
            r"(手机号|身份证).*(明文|泄露|日志|返回)",
        ],
    },
    {
        "id": "R4",
        "name": "管理后台固定口令放前端 / 无鉴权",
        "patterns": [
            r"admin123", r"固定口令", r"前端.*(口令|密码|密钥|secret)",
            r"服务端鉴权", r"管理后台.*(鉴权|认证|权限)", r"越权",
            r"未授权", r"\bRBAC\b", r"鉴权",
        ],
    },
    {
        "id": "R5",
        "name": "短信同步调用阻塞 / 无超时重试降级",
        "patterns": [
            r"同步.*(短信|调用)", r"短信.*(同步|阻塞|超时|重试|降级|异步|队列|熔断)",
            r"阻塞.*(请求|响应|主链路)", r"异步.*短信", r"短信.*队列",
            r"短信.*失败", r"短信轰炸",
        ],
    },
    {
        "id": "R6",
        "name": "SQLite 单机 / 多 worker / 定时任务同进程",
        "patterns": [
            r"SQLite", r"多 ?worker", r"多进程", r"写锁", r"单机",
            r"扩展性", r"APScheduler.*(重复|多|单进程|worker)",
            r"4 ?workers", r"2 ?核 ?4G", r"容量不足", r"数据库.*瓶颈",
        ],
    },
    {
        "id": "R7",
        "name": "列表全表返回 / 缺索引 / 前端分页",
        "patterns": [
            r"全表", r"索引", r"分页", r"\bLIMIT\b", r"游标",
            r"一次性.*(加载|返回)", r"N\+1", r"慢查询", r"内存.*(爆|压力)",
        ],
    },
    {
        "id": "R8",
        "name": "时间存本地字符串 / 无时区",
        "patterns": [
            r"时区", r"\bUTC\b", r"本地时间", r"timestamp", r"timestamptz",
            r"datetime", r"时间字符串", r"排序.*时间", r"跨时区",
        ],
    },
    {
        "id": "R9",
        "name": "无限流 / 无验证码 / 无风控",
        "patterns": [
            r"限流", r"验证码", r"风控", r"刷单", r"短信轰炸", r"频控",
            r"防刷", r"人机校验",
        ],
    },
    {
        "id": "R10",
        "name": "备份同盘 / 无监控告警 / 回滚不可靠",
        "patterns": [
            r"备份", r"监控", r"告警", r"回滚", r"灰度", r"同盘",
            r"异地", r"可观测", r"发布.*(回滚|灰度)", r"单点",
        ],
    },
]

SECTION_PATTERNS = {
    "结论": r"(?m)^#{1,4}[ \t]*[^\n]*(评审结论|结论|决定|verdict)",
    "风险清单": r"(?m)^#{1,4}[ \t]*[^\n]*(风险|问题清单)",
    "修改建议": r"(?m)^#{1,4}[ \t]*[^\n]*(修改建议|修复建议|上线前|行动项|建议)",
    "验收标准": r"(?m)^#{1,4}[ \t]*[^\n]*(验收|完成标准|definition of done)",
    "待确认": r"(?m)^#{1,4}[ \t]*[^\n]*(待确认|假设|开放问题|未知)",
}


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
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_messages(group):
    messages = []
    if group["system"]:
        messages.append({"role": "system", "content": group["system"]})
    messages.append({"role": "user", "content": TASK})
    return messages


def ask(group):
    started = time.perf_counter()
    resp, err = chat(build_messages(group))
    elapsed = round(time.perf_counter() - started, 2)
    if err:
        return {"ok": False, "error": err, "content": "", "usage": {},
                "duration": elapsed}

    choice = resp["choices"][0]
    usage = resp.get("usage", {}) or {}
    think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    return {
        "ok": True,
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason", "?"),
        "usage": usage,
        "think_tokens": think,
        "response_id": resp.get("id"),
        "duration": elapsed,
    }


# ---- 自动评分 ---------------------------------------------------------------
DECISION_LABELS = [
    ("No-Go", r"\bNo[- ]?Go\b|不通过|不建议[^\n]{0,20}上线|不能[^\n]{0,20}上线|"
              r"无法[^\n]{0,20}上线|不可上线|拒绝上线|暂缓上线|延期上线"),
    ("Conditional Go", r"\bConditional[- ]?Go\b|有条件通过|有条件上线|附条件"),
    ("Go", r"(?<!No-)(?<!Conditional-)\bGo\b|可以上线|同意上线|建议上线"),
]


def detect_decision(text):
    """优先读“结论：…”这一行，避免把选项菜单当成最终结论。"""
    m = re.search(r"结论[^\n]{0,12}?[:：][ \t]*([^\n]+)", text, re.I)
    if m:
        segment = m.group(1)
        for label, pat in DECISION_LABELS:
            if re.search(pat, segment, re.I):
                return label

    # 退而求其次：找只包含一个决策标签的行，跳过“Go / Conditional Go / No-Go”这种菜单行
    for line in text[:3000].splitlines():
        found = list(dict.fromkeys(
            label for label, pat in DECISION_LABELS if re.search(pat, line, re.I)))
        if len(found) == 1:
            return found[0]

    # 最后按优先级在开头范围内兜底
    head = text[:3000]
    for label, pat in DECISION_LABELS:
        if re.search(pat, head, re.I):
            return label
    return None


def extract_section(text, heading_pattern):
    """提取标题匹配的章节，遇到同级或更高级标题时停止；忽略代码块里的 # 行。"""
    lines = text.splitlines(keepends=True)
    in_fence = False
    start = None
    level = 0

    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        hm = re.match(r"^(#{1,4})[ \t]*([^\n]*)", line)
        if not hm:
            continue

        if start is None:
            if re.search(heading_pattern, hm.group(2), re.I):
                start = i + 1
                level = len(hm.group(1))
        elif len(hm.group(1)) <= level:
            return "".join(lines[start:i])

    return "".join(lines[start:]) if start is not None else ""


def count_items(section):
    if not section:
        return 0
    numbered = len(re.findall(r"(?m)^\s*(?:\*\*)?\d+\.\s+", section))
    bullets = len(re.findall(r"(?m)^\s*[-*]\s+", section))
    table_rows = [line for line in section.splitlines()
                  if line.strip().startswith("|") and "---" not in line]
    table_items = max(0, len(table_rows) - 1)  # 去掉表头
    return max(numbered, bullets, table_items)


def table_header_text(text):
    """只取每个 Markdown 表格的表头行，避免把数据单元格里的“验证”当表头。"""
    lines = text.splitlines()
    headers = []
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        if i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            headers.append(line)
    return " ".join(headers)


def evaluate(text):
    sections_hit = [name for name, pat in SECTION_PATTERNS.items()
                    if re.search(pat, text, re.I)]
    risk_hits = [r["id"] for r in RISKS
                 if any(re.search(p, text, re.I) for p in r["patterns"])]

    has_table = bool(re.search(r"\|\s*:?-{3,}", text))
    decision = detect_decision(text)
    rationale = bool(re.search(r"一句话理由|理由[:：]|因为|由于|前提|条件", text[:2500]))

    must_section = extract_section(
        text, r"上线前必须完成|必须完成|必须修复|必须改|修改建议|修复建议|行动项|上线前")
    must_fix_count = count_items(must_section)
    must_fix = must_fix_count >= 3

    table_headers = table_header_text(text)
    fix_col = bool(re.search(r"修复|修改|建议|动作|措施", table_headers))
    verify_col = bool(re.search(r"验收|验证|完成标准", table_headers))
    priority = (len(re.findall(r"\bP[0-2]\b", text)) >= 2 or
                ("高优先级" in text and "中优先级" in text))

    pending_section = extract_section(text, r"待确认|假设|开放问题|未知")
    pending_items = count_items(pending_section)
    has_pending = "待确认" in sections_hit
    pending_enough = pending_items >= 3 and "无" not in pending_section[:40]

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

    structure_score = len(sections_hit) * 4                 # 20
    risk_score = len(risk_hits) * 4                         # 40
    decision_score = (5 if decision else 0) + (5 if rationale else 0) + (5 if must_fix else 0)
    action_score = (5 if priority else 0) + (5 if fix_col else 0) + (5 if verify_col else 0)
    pending_score = (5 if has_pending else 0) + (5 if pending_enough else 0)
    total = structure_score + risk_score + decision_score + action_score + pending_score

    return {
        "sections": len(sections_hit),
        "section_names": sections_hit,
        "has_table": has_table,
        "risk_hits": risk_hits,
        "risk_count": len(risk_hits),
        "decision": decision,
        "rationale": rationale,
        "must_fix_count": must_fix_count,
        "must_fix": must_fix,
        "priority": priority,
        "fix_col": fix_col,
        "verify_col": verify_col,
        "pending_items": pending_items,
        "pending_enough": pending_enough,
        "chinese_chars": chinese_chars,
        "structure_score": structure_score,
        "risk_score": risk_score,
        "decision_score": decision_score,
        "action_score": action_score,
        "pending_score": pending_score,
        "score": total,
    }


def fmt_bool(v):
    return "✅" if v else "❌"


def mean(values):
    return round(statistics.mean(values), 2) if values else 0


def group_results(results, key):
    return [r for r in results if r["group_key"] == key and r.get("ok")]


def build_summary(results):
    lines = [
        "| 组别 | 平均分 | 结构 /20 | 风险 /40 | 决策 /15 | 可执行 /15 | 待确认 /10 | 中文字数 | prompt | completion | total | 平均用时(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in GROUPS:
        rs = group_results(results, g["key"])
        if not rs:
            lines.append(f"| {g['name']} | 失败 | - | - | - | - | - | - | - | - | - | - |")
            continue
        m = [r["metrics"] for r in rs]
        u = [r["usage"] for r in rs]
        lines.append(
            f"| {g['name']} | {mean([x['score'] for x in m])} | "
            f"{mean([x['structure_score'] for x in m])} | "
            f"{mean([x['risk_score'] for x in m])} | "
            f"{mean([x['decision_score'] for x in m])} | "
            f"{mean([x['action_score'] for x in m])} | "
            f"{mean([x['pending_score'] for x in m])} | "
            f"{mean([x['chinese_chars'] for x in m])} | "
            f"{mean([x.get('prompt_tokens', 0) for x in u])} | "
            f"{mean([x.get('completion_tokens', 0) for x in u])} | "
            f"{mean([x.get('total_tokens', 0) for x in u])} | "
            f"{mean([r['duration'] for r in rs])} |"
        )
    return "\n".join(lines)


def build_per_run(results):
    lines = [
        "| 轮次 | 组别 | 得分 | 结构 | 风险命中 | 结论 | 中文字数 | completion | 用时(s) | finish_reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r.get("ok"):
            lines.append(f"| {r['run']} | {r['group_name']} | 失败 | - | - | - | - | - | {r['duration']} | {r.get('error', '')} |")
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['run']} | {r['group_name']} | {m['score']} | {m['sections']}/5 | "
            f"{m['risk_count']}/10 | {m['decision'] or '未给出'} | {m['chinese_chars']} | "
            f"{r['usage'].get('completion_tokens', '?')} | {r['duration']} | {r['finish_reason']} |"
        )
    return "\n".join(lines)


def build_risk_matrix(results):
    lines = [
        "| 埋点风险 | A · 普通 | B · 角色 | C · 结构化 |",
        "|---|---|---|---|",
    ]
    for risk in RISKS:
        cells = []
        for g in GROUPS:
            rs = group_results(results, g["key"])
            hit = sum(1 for r in rs if risk["id"] in r["metrics"]["risk_hits"])
            cells.append(f"{hit}/{len(rs)}" if rs else "-")
        lines.append(f"| {risk['id']} {risk['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_token_table(results):
    lines = [
        "| 组别 | 平均 prompt | 平均 completion | 平均 total | 平均分 | 每 1k total tokens 得分 |",
        "|---|---|---|---|---|---|",
    ]
    for g in GROUPS:
        rs = group_results(results, g["key"])
        if not rs:
            continue
        avg_prompt = mean([r["usage"].get("prompt_tokens", 0) for r in rs])
        avg_completion = mean([r["usage"].get("completion_tokens", 0) for r in rs])
        avg_total = mean([r["usage"].get("total_tokens", 0) for r in rs])
        avg_score = mean([r["metrics"]["score"] for r in rs])
        per_1k = round(avg_score / avg_total * 1000, 2) if avg_total else 0
        lines.append(f"| {g['name']} | {avg_prompt} | {avg_completion} | {avg_total} | {avg_score} | {per_1k} |")
    return "\n".join(lines)


def build_fast_view(results):
    """用数据生成一段描述性速览，不做因果断言。"""
    summary = []
    for g in GROUPS:
        rs = group_results(results, g["key"])
        if rs:
            summary.append({
                "name": g["name"],
                "score": mean([r["metrics"]["score"] for r in rs]),
                "risk": mean([r["metrics"]["risk_count"] for r in rs]),
                "tokens": mean([r["usage"].get("total_tokens", 0) for r in rs]),
                "min": min(r["metrics"]["score"] for r in rs),
                "max": max(r["metrics"]["score"] for r in rs),
            })
    if not summary:
        return "- 本次没有成功结果。"
    best = max(summary, key=lambda x: x["score"])
    best_risk = max(summary, key=lambda x: x["risk"])
    cheapest = min(summary, key=lambda x: x["tokens"])
    return "\n".join([
        f"- 平均分最高：**{best['name']}**（{best['score']}）。",
        f"- 平均风险覆盖最多：**{best_risk['name']}**（{best_risk['risk']}/10）。",
        f"- 平均 token 成本最低：**{cheapest['name']}**（{cheapest['tokens']} tokens）。",
        "- 逐组分数区间： " + "；".join(
            f"{x['name']} {x['min']}~{x['max']}" for x in summary),
        "- 以上是本次运行的描述性统计，不构成因果结论；解释与局限见 `ab_test_conclusions.md`。",
    ])


def write_report(results, task=None, role_prompt=None, structured_prompt=None,
                 structured_prompt_path=None):
    task = task if task is not None else TASK
    role_prompt = role_prompt if role_prompt is not None else ROLE_PROMPT
    structured_prompt = structured_prompt if structured_prompt is not None else STRUCTURED_PROMPT
    structured_prompt_path = (structured_prompt_path if structured_prompt_path is not None
                              else STRUCTURED_PROMPT_PATH)
    if not os.path.isabs(structured_prompt_path):
        structured_prompt_path = os.path.join(BASE_DIR, structured_prompt_path)
    template_label = os.path.relpath(structured_prompt_path, BASE_DIR).replace(os.sep, "/")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Prompt A/B/C 测试报告：普通 vs 角色 vs 结构化",
        "",
        f"- 生成时间：{now}",
        f"- 模型：`{MODEL}`",
        f"- 采样参数：temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        f"- 评分口径：v{SCORING_VERSION}（首轮后离线复算，未重新调用模型）",
        f"- 每组轮次：{RUNS}",
        "- 任务：评审同一份《活动报名系统技术方案 v0.1》，判断 2 周后能否上线",
        f"- 结构化模板：`{template_label}`（第 3 套结构化 Prompt）",
        "- 说明：4.4 节保存的是本次 A/B 实际使用的 C 组模板全文；当前模板文件可能已迭代，以 4.4 节为准。",
        "",
        "## 一、实验目的",
        "",
        "验证三个问题：",
        "",
        "1. 只加一句“你是资深架构师”，输出会比普通提问更好吗？",
        "2. 加上角色、目标、边界、评审维度、输出规范和自检流程后，输出会稳定多少？",
        "3. 结构化带来的结构/决策/可执行性提升，值不值它多花的 token？",
        "",
        "## 二、实验设计",
        "",
        "### 2.1 控制变量",
        "",
        "- 同一个模型、同一组采样参数、同一份技术方案、同一条 user 消息。",
        "- 三组都要求“判断能否在 2 周后上线，指出主要风险，并给出修改建议”。",
        "- 每组跑 2 次，按 A → B → C → A → B → C 轮转，降低时间漂移影响。",
        "",
        "### 2.2 实验组",
        "",
        "| 组别 | System Prompt | 变量说明 |",
        "|---|---|---|",
        "| A · 普通 Prompt | 无 | 只有 user 消息，直接要求评审 |",
        "| B · 角色 Prompt | 一句话角色 + 一句要求 | 有“资深架构师”人设，但没有输出规范、评审维度和流程 |",
        "| C · 结构化 Prompt | 第 3 套完整模板 | 角色 + 目标 + 边界 + 维度 + 输出规范 + 工作流程 + 自检 |",
        "",
        "### 2.3 评分规则（100 分）",
        "",
        "| 维度 | 分值 | 判定方式 |",
        "|---|---|---|",
        "| 结构完整度 | 20 | 五个规定章节（结论/风险清单/修改建议/验收标准/待确认）各 4 分 |",
        "| 关键风险覆盖 | 40 | 10 个埋点风险各 4 分，命中关键词即得分 |",
        "| 决策明确度 | 15 | 明确 Go/No-Go 5 + 有理由 5 + 至少 3 条上线前动作 5 |",
        "| 可执行性 | 15 | 使用 P0/P1/P2 5 + 有修复列 5 + 有验收/验证列 5 |",
        "| 待确认与假设 | 10 | 有该章节 5 + 至少 3 条 5 |",
        "",
        "> 注意：这套评分偏向“可检查的交付物”，C 组天然占便宜。因此下面同时报告风险覆盖、token 成本和完整输出，避免只看总分。",
        "",
        "### 2.4 评分口径修正（v2）",
        "",
        "首轮运行后发现三处测量问题，已离线复算，没有重新调用模型：",
        "",
        "1. **结论识别**：原正则只认“不建议上线”，漏掉了“不建议按当前方案在 2 周后上线”“不能按 2 周窗口上线”等自然表达。",
        "2. **R2 幂等误判**：原先把“名额唯一约束”也算作防重复报名；v2 要求出现 phone/手机号/重复报名/幂等等语义。",
        "3. **表头识别**：原先把表格数据行里的“验证”误当表头；v2 只检查真正的表头行。",
        "",
        "修正只影响自动判分，不影响模型输出；完整原始输出仍在第七节。",
        "",
        "## 三、被评审的技术方案",
        "",
        "完整内容见 `design_doc.md`。方案中有意埋了 10 类问题：",
        "",
        "1. 先查后插的名额扣减，存在并发超卖。",
        "2. 没有幂等键或唯一约束，重复提交会重复报名。",
        "3. 手机号、身份证号明文存储，日志打印完整请求体，列表接口全量返回。",
        "4. 管理后台固定口令 `admin123` 写在前端环境变量。",
        "5. 同步调用短信服务，没有超时、重试、降级。",
        "6. SQLite + 单机 2C4G + 4 workers + APScheduler 同进程。",
        "7. 列表全表返回、`event_id` 无索引、前端分页。",
        "8. 时间用本地字符串存储，没有时区。",
        "9. 没有验证码、限流、风控。",
        "10. 备份到同一块磁盘、没有监控告警、回滚方案不可靠。",
        "",
        "## 四、三组 Prompt 全文",
        "",
        "### 4.1 公共 User Prompt（三组完全相同）",
        "",
        "````text",
        task,
        "````",
        "",
        "### 4.2 A 组：普通 Prompt",
        "",
        "无 System Prompt，只发送上面的 User Prompt。",
        "",
        "### 4.3 B 组：角色 Prompt",
        "",
        "````text",
        role_prompt,
        "````",
        "",
        "### 4.4 C 组：结构化 Prompt（第 3 套模板）",
        "",
        "````text",
        structured_prompt,
        "````",
        "",
        "## 五、结果总览",
        "",
        build_summary(results),
        "",
        "### 5.1 逐轮明细",
        "",
        build_per_run(results),
        "",
        "### 5.2 风险覆盖矩阵",
        "",
        build_risk_matrix(results),
        "",
        "### 5.3 Token 成本",
        "",
        build_token_table(results),
        "",
        "## 六、数据速览",
        "",
        build_fast_view(results),
        "",
        "## 七、完整输出",
        "",
    ]

    for r in results:
        lines.append(f"### 第 {r['run']} 轮 · {r['group_name']}")
        lines.append("")
        if not r.get("ok"):
            lines.append(f"调用失败：`{r['error']}`")
            lines.append("")
            continue
        m, u = r["metrics"], r["usage"]
        lines.append(r["content"])
        lines.append("")
        lines.append(
            f"> 判定：得分 `{m['score']}/100`，结构 `{m['sections']}/5`，"
            f"风险 `{m['risk_count']}/10`，结论 `{m['decision'] or '未给出'}`，"
            f"中文字数 `{m['chinese_chars']}`；"
            f"prompt=`{u.get('prompt_tokens', '?')}`，"
            f"completion=`{u.get('completion_tokens', '?')}`"
            f"（思考 `{r.get('think_tokens', 0)}`），"
            f"finish_reason=`{r['finish_reason']}`"
        )
        lines.append("")

    lines += [
        "## 八、复现方式",
        "",
        "```bash",
        "python 01Agent/w2day05/prompt_ab_test.py",
        "```",
        "",
        "配置读取顺序：`vision_config.json` 优先，环境变量兜底；密钥不会写进脚本或报告。",
        "",
        "> 本报告由脚本生成，重跑会覆盖。人工分析与结论见 `ab_test_conclusions.md`。",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def rescore():
    """离线复算：读取已有 JSON 的模型输出，不重新调用 API。"""
    if not os.path.exists(RESULT_PATH):
        sys.exit(f"[rescore] 找不到 {RESULT_PATH}，请先完整跑一次实验")

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for r in data.get("results", []):
        if r.get("ok"):
            r["metrics"] = evaluate(r.get("content", ""))

    data["rescored_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    data["scoring_version"] = SCORING_VERSION
    # 同步当前评分口径使用的风险定义，避免 JSON 里的 risks 与 metrics 不一致。
    data["risks"] = RISKS
    if not data.get("structured_prompt_path"):
        data["structured_prompt_path"] = os.path.relpath(STRUCTURED_PROMPT_PATH, BASE_DIR)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    results = data["results"]
    prompts = data.get("prompts") or {}
    write_report(
        results,
        task=data.get("task"),
        role_prompt=prompts.get("B"),
        structured_prompt=prompts.get("C"),
        structured_prompt_path=data.get("structured_prompt_path"),
    )
    print(build_summary(results))
    print(f"\n[rescore] 已按评分口径 v{SCORING_VERSION} 复算，未调用模型")
    print(f"[output] 报告已写入 {REPORT_PATH}")
    print(f"[output] 原始数据已写入 {RESULT_PATH}")


def main():
    if "--rescore" in sys.argv:
        rescore()
        return

    print(f"[config] 网关: {BASE_URL}  模型: {MODEL}")
    print(f"[setup ] 方案 {len(DESIGN_DOC)} 字符 | 模板 {len(STRUCTURED_PROMPT)} 字符 "
          f"({os.path.relpath(STRUCTURED_PROMPT_PATH, BASE_DIR)}) | "
          f"temperature={TEMPERATURE} max_tokens={MAX_TOKENS} runs={RUNS}")

    results = []
    for run in range(1, RUNS + 1):
        for g in GROUPS:
            print(f"\n{'=' * 62}")
            print(f"[第 {run} 轮] {g['name']}")
            print("=" * 62)
            r = ask(g)
            record = {
                "run": run,
                "group_key": g["key"],
                "group_name": g["name"],
                "system_prompt_chars": len(g["system"] or ""),
                "duration": r["duration"],
                "ok": r["ok"],
                "error": r.get("error"),
                "content": r.get("content", ""),
                "usage": r.get("usage", {}),
                "think_tokens": r.get("think_tokens", 0),
                "finish_reason": r.get("finish_reason"),
                "metrics": evaluate(r.get("content", "")) if r["ok"] else {},
            }
            results.append(record)
            if not r["ok"]:
                print(f"  调用失败：{r['error']}")
                continue
            m = record["metrics"]
            print(r["content"])
            print(f"\n  [judge] 得分={m['score']}/100 结构={m['sections']}/5 "
                  f"风险={m['risk_count']}/10 结论={m['decision'] or '未给出'} "
                  f"中文字数={m['chinese_chars']}")
            print(f"  [usage] prompt={r['usage'].get('prompt_tokens', '?')} "
                  f"completion={r['usage'].get('completion_tokens', '?')} "
                  f"(思考 {r['think_tokens']}) finish_reason={r.get('finish_reason')!r} "
                  f"用时={r['duration']}s")

    print("\n" + "=" * 62)
    print("汇总对比")
    print("=" * 62)
    print(build_summary(results))

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "runs": RUNS,
            "scoring_version": SCORING_VERSION,
            "task": TASK,
            "design_doc": DESIGN_DOC,
            "structured_prompt_path": os.path.relpath(STRUCTURED_PROMPT_PATH, BASE_DIR),
            "prompts": {
                "A": None,
                "B": ROLE_PROMPT,
                "C": STRUCTURED_PROMPT,
            },
            "risks": RISKS,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    write_report(results)
    print(f"\n[output] 报告已写入 {REPORT_PATH}")
    print(f"[output] 原始数据已写入 {RESULT_PATH}")


if __name__ == "__main__":
    main()
