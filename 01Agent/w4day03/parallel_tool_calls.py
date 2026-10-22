# -*- coding: utf-8 -*-
"""W4 Day3：并行工具调用（Parallel Function Calling）与 tool_choice 机制

对应任务：
  1. 实现多 Tool 并行调用（Parallel Function Calling）
  2. 理解 Tool Choice 机制（auto / none / required / 强制指定函数名）
  3. 对比 GPT-4o / DeepSeek / Claude 的 FC 实现差异（报告静态梳理节）

实验设计：
  幕1 并行调用：问"北京、上海、广州哪边现在最热？"
      → 预期模型在【同一条】assistant 消息里返回 3 个 tool_calls；
        客户端用线程池并发执行、一次性回传 3 条 role=tool 消息。
        并行的收益：工具是 IO 时总延迟 = max(各工具) 而不是 sum。
  幕2 tool_choice 四档对照（同一个天气问题）：
      auto      模型自己决定（预期：调用）
      none      禁止工具（预期：stop 直答；观察它怎么处理"不知道的天气"）
      required  必须调一个（预期：tool_calls）
      强制函数名 强制调 get_weather（用一个无关问题，观察"被迫调用"）
  幕3 数据结构观察：3 个 tool_calls 的 id 顺序、arguments 的批内独立性。

产出：parallel_tool_calls_report.md + parallel_tool_calls_result.json
依赖：仅标准库；LLM 网关同 W1/W2。
"""
import sys
import io
import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "parallel_tool_calls_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "parallel_tool_calls_result.json")

import ragkit  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市当前天气（模拟数据源）",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string",
                                               "description": "城市名"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算四则运算表达式",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string",
                                                     "description": "如 3*7+2"}},
                       "required": ["expression"]}}},
]

WEATHER_DB = {"北京": {"celsius": 9, "cond": "晴"},
              "上海": {"celsius": 17, "cond": "多云"},
              "广州": {"celsius": 26, "cond": "阵雨"}}


def exec_tool(name, args):
    """工具的真实执行方。返回 (结果, 耗时秒)——耗时用于演示并行的收益。"""
    import time
    t0 = time.time()
    time.sleep(0.3)  # 模拟网络 IO：每个工具 300ms，并行收益一目了然
    if name == "get_weather":
        w = WEATHER_DB.get(args.get("city"))
        return (w or {"error": f"暂无 {args.get('city')} 数据"},
                round(time.time() - t0, 2))
    if name == "calculate":
        import re
        expr = args.get("expression", "")
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return ({"error": "非法表达式"}, round(time.time() - t0, 2))
        try:
            return (eval(expr), round(time.time() - t0, 2))  # noqa: S307
        except Exception as e:
            return ({"error": str(e)}, round(time.time() - t0, 2))
    return ({"error": f"未知工具 {name}"}, round(time.time() - t0, 2))


def chat_with_tools(messages, tool_choice="auto"):
    extra = {"tools": TOOLS}
    if tool_choice:
        extra["tool_choice"] = tool_choice
    return ragkit.chat(messages, temperature=0.0, max_tokens=1000, extra=extra)


def execute_calls(calls, parallel=True):
    """执行一批 tool_calls。parallel=True 用线程池并发（并行 FC 的关键收益）。"""
    def run(c):
        args = json.loads(c["function"]["arguments"])
        result, dur = exec_tool(c["function"]["name"], args)
        return c, args, result, dur

    if parallel and len(calls) > 1:
        with ThreadPoolExecutor(max_workers=len(calls)) as ex:
            out = list(ex.map(run, calls))
        wall = max(d for *_, d in out)          # 并行：墙钟 = 最慢的那个
        serial = sum(d for *_, d in out)        # 串行对照
    else:
        out = [run(c) for c in calls]
        wall = sum(d for *_, d in out)
        serial = wall
    return out, round(wall, 2), round(serial, 2)


def main():
    print(f"[model] {ragkit.MODEL}")
    results = {}

    print()
    print("=" * 62)
    print("幕 1：并行工具调用 —— 一条消息 3 个 tool_calls")
    print("=" * 62)
    messages = [{"role": "user",
                 "content": "北京、上海、广州现在分别多少度？哪个最热？"}]
    resp, err = chat_with_tools(messages)
    if err:
        sys.exit(f"[error] {err}")
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print(f"  finish_reason={resp['choices'][0]['finish_reason']}  "
          f"tool_calls 数量={len(calls)}")
    for c in calls:
        print(f"    - {c['function']['name']}({c['function']['arguments']})  "
              f"id={c.get('id')}")

    executed, wall, serial = execute_calls(calls, parallel=True)
    for c, args, result, dur in executed:
        print(f"    执行 {args} → {result}  ({dur}s)")
    print(f"  [并行收益] 墙钟 {wall}s vs 串行累计 {serial}s "
          f"（每个工具模拟 0.3s IO）")

    # 全部回传：一条 assistant(tool_calls) + N 条 role=tool
    messages.append(msg)
    for c, args, result, _ in executed:
        messages.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": json.dumps(result, ensure_ascii=False)})
    resp2, _ = chat_with_tools(messages)
    final = resp2["choices"][0]["message"]["content"]
    print(f"  [final] finish_reason={resp2['choices'][0]['finish_reason']}")
    print(f"  {final[:200]}")

    results["parallel"] = {
        "question": "北京、上海、广州现在分别多少度？哪个最热？",
        "n_tool_calls": len(calls),
        "calls": [{"name": c["function"]["name"],
                   "args": c["function"]["arguments"],
                   "id": c.get("id")} for c in calls],
        "wall_sec": wall, "serial_sum_sec": serial,
        "final_answer": final,
        "finish_reasons": [resp["choices"][0]["finish_reason"],
                           resp2["choices"][0]["finish_reason"]],
    }

    print()
    print("=" * 62)
    print("幕 2：tool_choice 四档对照（同一个天气问题 + 一个无关问题）")
    print("=" * 62)
    choices = [
        ("auto", "北京多少度？", "auto"),
        ("none", "北京多少度？", None),
        ("required", "北京多少度？", "required"),
        ("强制-get_weather", "用一句话解释注意力机制（与天气无关）",
         {"type": "function", "function": {"name": "get_weather"}}),
    ]
    tc_results = []
    for label, question, choice in choices:
        resp, err = chat_with_tools([{"role": "user", "content": question}],
                                    tool_choice=choice)
        if err:
            tc_results.append({"label": label, "error": err})
            print(f"  [{label}] 请求失败：{err[:80]}")
            continue
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        fr = resp["choices"][0]["finish_reason"]
        content_head = (msg.get("content") or "")[:80].replace("\n", " ")
        calls_head = "、".join(f"{c['function']['name']}({c['function']['arguments']})"
                               for c in calls) or "无"
        print(f"  [{label:<18}] finish_reason={fr:<10} calls={calls_head}")
        print(f"  {'':<20} content: {content_head}")
        tc_results.append({"label": label, "question": question,
                           "tool_choice": choice if isinstance(choice, str) else "forced",
                           "finish_reason": fr,
                           "calls": [{"name": c["function"]["name"],
                                      "args": c["function"]["arguments"]}
                                     for c in calls],
                           "content_head": (msg.get("content") or "")[:200],
                           "usage": resp.get("usage", {})})
    results["tool_choice"] = tc_results

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": {"generated_at": datetime.datetime.now().isoformat(),
                            "model": ragkit.MODEL, "tools": TOOLS,
                            "mock_io_sec": 0.3},
                   **results}, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(results))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(r):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = r["parallel"]
    L = []
    a = L.append
    a("# 并行工具调用与 tool_choice 实验报告（W4 Day3）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 模型：`{ragkit.MODEL}`（temperature=0）")
    a(f"- 工具：get_weather（模拟 0.3s IO）/ calculate")
    a("")
    a("## 一、并行 Function Calling")
    a("")
    a(f"- 问题：{p['question']}")
    a(f"- 模型在一条 assistant 消息里返回 **{p['n_tool_calls']} 个 tool_calls**："
      f"{[c['args'] for c in p['calls']]}")
    a(f"- 线程池并发执行：墙钟 **{p['wall_sec']}s**，若串行累计要 "
      f"**{p['serial_sum_sec']}s**——每个工具 0.3s 模拟 IO，"
      "并行的收益就是把 sum 变成 max。")
    a(f"- 最终回答（finish_reason={p['finish_reasons'][1]}）：{p['final_answer'][:120]}")
    a("")
    a("并行的三个要点：")
    a("")
    a("1. **模型侧**：一次响应给出多个 tool_calls（各自带独立 id 和 arguments），")
    a("   调用之间没有隐藏依赖时才会并行——有依赖的（先查 A 再用 A 查 B）模型")
    a("   通常会拆成多轮。")
    a("2. **客户端侧**：线程池/asyncio 并发执行，然后把 N 条结果按")
    a("   `role=tool + tool_call_id` 一次性全部回传，不要查一条回一条。")
    a("3. **什么时候能并行**：工具是只读查询（天气/搜索/查库）时放心并行；")
    a("   写操作（下单/转账）要慎重——并发执行绕过了顺序保证。")
    a("")
    a("## 二、tool_choice 四档对照（实测）")
    a("")
    a("| 请求的 tool_choice | 问题 | finish_reason | 实际调用 |")
    a("|---|---|---|---|")
    for x in r["tool_choice"]:
        if x.get("error"):
            a(f"| {x['label']} | - | 请求失败 | {x['error'][:50]} |")
            continue
        calls = "、".join(f"{c['name']}({c['args']})" for c in x["calls"]) or "无（直接 stop）"
        a(f"| `{x['tool_choice']}` | {x['question'][:22]} | `{x['finish_reason']}` | {calls} |")
    a("")
    a("### 逐档解读（对照本机实测结果）")
    a("")
    by_label = {x["label"]: x for x in r["tool_choice"] if not x.get("error")}
    auto_x = by_label.get("auto")
    if auto_x:
        called = bool(auto_x["calls"])
        a(f"- **auto**：{'按预期调用了工具' if called else '没有调用工具'}——"
          "auto 是默认档：模型按\"回答问题需不需要\"自主决定，FC 的本意"
          "正是\"可以调\"而非\"必须调\"。")
    none_x = by_label.get("none")
    if none_x:
        if none_x["calls"]:
            a(f"- **none（实测异常）**：请求了 tool_choice=none，模型仍然返回了 "
              f"{[c['name'] for c in none_x['calls']]}——本网关没有落实 none 档。"
              "协议里的可选字段不一定被每个端点实现，**每个字段都要实测**，"
              "不能只信文档。业务上需要\"绝不允许调工具\"时，最稳的做法是"
              "请求里干脆不带 tools 字段。")
        else:
            a("- **none**：按预期禁用了工具，模型只靠参数知识作答。")
    req_x = by_label.get("required")
    if req_x:
        if req_x["calls"]:
            a(f"- **required**：按预期强制调用（{[c['name'] for c in req_x['calls']]}）。"
              "适合\"必须经过工具\"的业务（如查库后再答）。")
        else:
            a("- **required（实测异常）**：请求了 required 但模型直接 stop——"
              "本网关未落实 required 档。")
    forced_x = by_label.get("强制-get_weather")
    if forced_x:
        if forced_x["calls"]:
            a(f"- **强制指定函数名**：无关问题（注意力机制）也被迫调用了 "
              f"{[c['args'] for c in forced_x['calls']]}。这演示了强制档只约束"
              "\"调不调\"，不约束\"参数对不对\"——city 是模型编的一个值，"
              "参数幻觉在任何档位都存在，要靠 Schema 校验和业务校验兜底。")
        else:
            a(f"- **强制指定函数名（实测异常）**：finish_reason="
              f"`{forced_x['finish_reason']}`、无 tool_calls——本网关忽略了强制档，"
              "模型按普通 auto 语义直接回答了。若业务依赖\"必须调某个函数\""
              "（如强制保存表单），在这类端点上要加客户端兜底校验。")
    a("")
    a("## 三、GPT-4o / DeepSeek / Claude 的 FC 实现差异（静态梳理）")
    a("")
    a("| 维度 | OpenAI / DeepSeek（兼容系） | Claude（Anthropic 系） |")
    a("|---|---|---|")
    a("| 声明方式 | 请求体 `tools` 数组，function 内嵌 parameters | 请求体 `tools`，`input_schema` 字段 |")
    a("| 模型输出 | `message.tool_calls[]`（function.name/arguments） | `content[]` 里的 `tool_use` 块（id/name/input） |")
    a("| 结果回传 | `role: \"tool\"` 消息 + tool_call_id | `role: \"user\"` 消息内 `tool_result` 块 |")
    a("| 强制调用 | `tool_choice: required` 或指定函数名 | `tool_choice: {\"type\": \"tool\", \"name\": ...}` |")
    a("| 并行控制 | 模型自行决定（parallel_tool_calls 参数可关） | 默认并行，disable_parallel_tool_use 可关 |")
    a("| 思想 | 完全一致：模型产意图、代码执行、结果回传、循环到 stop ||")
    a("")
    a("记忆锚点：**OpenAI 系的工具结果是一条独立消息，Claude 系的工具结果")
    a("塞在 user 消息的块里**——做网关适配时最容易踩的差异。")
    a("")
    a("## 四、复现方式")
    a("")
    a("```bash")
    a("python 01Agent/w4day03/parallel_tool_calls.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
