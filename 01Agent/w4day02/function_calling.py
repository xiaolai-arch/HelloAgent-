# -*- coding: utf-8 -*-
"""W4 Day2：Function Calling 完整闭环 —— 协议层的工具调用

对应任务：
  1. 读 OpenAI/DeepSeek Function Calling 官方文档
  2. 写一个 Tool 定义（JSON Schema），如 get_weather(location, unit)
  3. 跑通 FC 调用流程：定义工具 → 模型决策 → 执行工具 → 结果回传 → 最终回答

与 Day1 手写 ReAct 的对照（同一条工具链，两种协议）：
    Day1：工具协议写在 system prompt 里，模型输出结构化文本，代码正则解析
    Day2：工具 Schema 挂在请求体 tools 字段上，模型输出 tool_calls 对象，
          字段名/参数类型由协议保证，无需解析容错
  两种方式的模型决策地位完全相同：模型都只是"产出调用意图"，
  真正执行工具的永远是你的代码。

实验（3 幕）：
  幕1 单工具循环：天气问答走完 FC 四步，打印每一跳的请求/响应解剖
  幕2 需要否（auto 决策）：问一个不需要工具的问题，观察模型直接回答
      （finish_reason=stop，无 tool_calls）——"工具列表在但不必用"
  幕3 多轮 FC：天气答完后追问"那上海呢"，观察模型在第 3 轮再次发起调用
      ——FC 的对话历史里 tool 消息是正式公民

天气数据为本地模拟（确定值），实验关注协议形状而非数据真伪。

产出：function_calling_report.md + function_calling_result.json
依赖：仅标准库；LLM 网关同 W1/W2。
"""
import sys
import io
import os
import json
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "function_calling_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "function_calling_result.json")

import ragkit  # noqa: E402

# ---- 幕 1/3 的工具定义：Schema 挂在 tools 上，进 API 协议层 ----------------

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市当前天气（模拟数据源）",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "温度单位，默认 celsius"},
            },
            "required": ["city"],
        },
    },
}]

# 模拟天气源：确定值，实验可复现
WEATHER_DB = {"北京": {"celsius": 9, "cond": "晴", "wind": "西北风 3 级"},
              "上海": {"celsius": 17, "cond": "多云", "wind": "东风 2 级"},
              "广州": {"celsius": 26, "cond": "阵雨", "wind": "南风 2 级"}}


def execute_get_weather(args):
    """真实执行方。模型给的 arguments 是 JSON 字符串，解析后查模拟库。"""
    city = args.get("city", "")
    unit = args.get("unit", "celsius")
    if city not in WEATHER_DB:
        return {"error": f"暂无 {city} 的天气数据"}
    w = WEATHER_DB[city]
    temp = w["celsius"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
    return {"city": city, "temperature": temp, "unit": unit,
            "condition": w["cond"], "wind": w["wind"]}


def fc_turn(messages, use_tools=True):
    """发一跳请求，返回 (resp, err)。use_tools=False 时摘掉 tools 字段。"""
    extra = {"tools": TOOLS} if use_tools else None
    return ragkit.chat(messages, temperature=0.0, max_tokens=1000, extra=extra)


def run_loop(question, trace, max_hops=4):
    """FC 主循环（与 Day1 ReAct 循环逐行对照的协议版）。"""
    messages = [{"role": "user", "content": question}]
    hops = []
    for hop in range(1, max_hops + 1):
        resp, err = fc_turn(messages)
        if err:
            hops.append({"hop": hop, "error": err})
            return f"[API 错误] {err}", hops
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        hop_info = {"hop": hop,
                    "finish_reason": resp["choices"][0]["finish_reason"],
                    "content": msg.get("content"),
                    "tool_calls": [{"name": c["function"]["name"],
                                    "arguments": c["function"]["arguments"],
                                    "id": c.get("id")}
                                   for c in calls],
                    "usage": resp.get("usage", {})}
        if not calls:
            hops.append(hop_info)
            trace.append(hops)
            return msg["content"], hops

        # 关键一跳：assistant 的 tool_calls 消息必须原样回填历史，
        # 然后把执行结果按 (role=tool, tool_call_id) 逐条追加
        messages.append(msg)
        for c in calls:
            args = json.loads(c["function"]["arguments"])
            result = execute_get_weather(args)
            hop_info["executed"] = hop_info.get("executed", []) + [
                {"tool": c["function"]["name"], "args": args, "result": result}]
            print(f"    hop{hop} 模型请求调用 {c['function']['name']}"
                  f"({c['function']['arguments']})  finish_reason="
                  f"{resp['choices'][0]['finish_reason']}")
            print(f"    hop{hop} 代码执行 → {result}")
            messages.append({"role": "tool",
                             "tool_call_id": c.get("id"),
                             "content": json.dumps(result, ensure_ascii=False)})
        hops.append(hop_info)
    trace.append(hops)
    return "[达到跳数上限]", hops


def main():
    print(f"[model] {ragkit.MODEL}")
    all_traces = []

    print()
    print("=" * 62)
    print("幕 1：单工具 FC 循环（问题 → tool_calls → 执行 → 最终回答）")
    print("=" * 62)
    trace = []
    ans, hops = run_loop("北京今天天气怎么样？适合穿什么？", trace)
    for h in hops:
        print(f"  hop{h['hop']}: finish_reason={h['finish_reason']} "
              f"tool_calls={[c['name'] for c in h['tool_calls']]} "
              f"content={str(h['content'])[:60] if h['content'] else None}")
    print(f"  [final] {ans}")
    all_traces.append({"id": "S1", "question": "北京今天天气怎么样？适合穿什么？",
                       "final_answer": ans, "hops": trace[0]})

    print()
    print("=" * 62)
    print("幕 2：不需要工具的问题（auto 决策：工具在，但不必用）")
    print("=" * 62)
    resp, err = fc_turn([{"role": "user", "content": "用一句话解释什么是注意力机制"}])
    msg = resp["choices"][0]["message"]
    print(f"  finish_reason={resp['choices'][0]['finish_reason']}  "
          f"tool_calls={msg.get('tool_calls')}")
    print(f"  content={msg['content'][:100]}")
    all_traces.append({"id": "S2", "question": "用一句话解释什么是注意力机制",
                       "finish_reason": resp["choices"][0]["finish_reason"],
                       "tool_calls": bool(msg.get("tool_calls")),
                       "content": msg["content"]})

    print()
    print("=" * 62)
    print("幕 3：多轮 FC（追问触发第二次工具调用，历史里 tool 消息是正式公民）")
    print("=" * 62)
    messages = [{"role": "user", "content": "北京今天多少度？"}]
    dialog = []
    for turn_q in ["北京今天多少度？", "那上海呢？"]:
        messages.append({"role": "user", "content": turn_q})
        resp, err = fc_turn(messages)
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print(f"  [user] {turn_q}")
        print(f"  [model] finish_reason={resp['choices'][0]['finish_reason']} "
              f"calls={[c['function']['arguments'] for c in calls]}")
        dialog.append({"user": turn_q,
                       "finish_reason": resp["choices"][0]["finish_reason"],
                       "calls": [c["function"]["arguments"] for c in calls]})
        if calls:
            messages.append(msg)
            for c in calls:
                result = execute_get_weather(json.loads(c["function"]["arguments"]))
                print(f"  [exec ] {result}")
                messages.append({"role": "tool", "tool_call_id": c.get("id"),
                                 "content": json.dumps(result, ensure_ascii=False)})
            resp, _ = fc_turn(messages)
            msg = resp["choices"][0]["message"]
            print(f"  [model] {msg['content'][:80]}")
            dialog[-1]["final"] = msg["content"]
        else:
            print(f"  [model] {str(msg['content'])[:80]}")
            dialog[-1]["final"] = msg["content"]
        if msg["content"]:
            messages.append({"role": "assistant", "content": msg["content"]})
    all_traces.append({"id": "S3", "dialog": dialog})

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": {"generated_at": datetime.datetime.now().isoformat(),
                            "model": ragkit.MODEL, "tools": TOOLS},
                   "traces": all_traces}, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(all_traces))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(traces):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s1 = traces[0]
    s2 = traces[1]
    L = []
    a = L.append
    a("# Function Calling 完整闭环实验报告（W4 Day2）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 模型：`{ragkit.MODEL}`（temperature=0）")
    a("- 工具：`get_weather(city, unit)`（模拟数据源，确定值可复现）")
    a("")
    a("## 一、FC 四步流程（与 Day1 手写 ReAct 对照）")
    a("")
    a("```text")
    a("①定义工具        请求体 tools 字段挂 JSON Schema")
    a("      │")
    a("      ▼")
    a("②模型决策        返回 message.tool_calls（finish_reason=tool_calls）")
    a("      │          模型只产出\"调用意图\"，不执行任何函数")
    a("      ▼")
    a("③代码执行        你的代码解析 arguments → 查天气/算数/查库")
    a("      │")
    a("      ▼")
    a("④结果回传        追加 role=tool 消息（带 tool_call_id）→ 再请求")
    a("                 模型基于结果生成最终回答（finish_reason=stop）")
    a("```")
    a("")
    a("| 维度 | Day1 手写 ReAct（文本协议） | Day2 Function Calling（API 协议） |")
    a("|---|---|---|")
    a("| 协议位置 | system prompt 里的格式约定 | 请求体 tools 字段的 JSON Schema |")
    a("| 模型输出 | 结构化文本（Thought/Action/...） | 结构化对象（tool_calls[].function） |")
    a("| 解析 | 正则 + 格式容错（Day1 出现格式违规） | json.loads arguments，字段由协议保证 |")
    a("| 参数校验 | 自己写 | Schema 声明类型/枚举/required |")
    a("| 决策自由度 | 相同：模型都可以选择不调用 | 相同（tool_choice=auto 时） |")
    a("| 本质 | 相同：模型给意图，代码干活，结果回上下文 ||")
    a("")
    a("## 二、工具定义全文（进 API 协议层的那份 JSON）")
    a("")
    a("```json")
    a(json.dumps(TOOLS, ensure_ascii=False, indent=2))
    a("```")
    a("")
    a("## 三、运行轨迹")
    a("")
    a(f"### 幕 1：{s1['question']}")
    a("")
    for h in s1["hops"]:
        a(f"- **hop{h['hop']}** finish_reason=`{h['finish_reason']}`")
        for c in h["tool_calls"]:
            a(f"  - 模型请求：`{c['name']}({c['arguments']})`（call id: {c['id']}）")
        for e in h.get("executed", []):
            a(f"  - 代码执行：{e['args']} → {e['result']}")
        if h["content"] and h["finish_reason"] == "stop":
            a(f"  - 最终回答：{h['content'][:160]}")
    a("")
    a(f"### 幕 2：不需要工具的问题")
    a("")
    a(f"- 问题：{s2['question']}")
    a(f"- 结果：finish_reason=`{s2['finish_reason']}`，"
      f"tool_calls={'有' if s2['tool_calls'] else '无'}——工具列表在场，"
      "模型按 auto 语义自行决定不用。FC 不是\"必须调工具\"，"
      "而是\"可以调工具\"。")
    a("")
    a(f"### 幕 3：多轮 FC（追问）")
    a("")
    for d in traces[2]["dialog"]:
        a(f"- **user**: {d['user']}")
        a(f"  - finish_reason=`{d['finish_reason']}`，calls={d['calls']}")
        a(f"  - 回答：{str(d.get('final'))[:100]}")
    a("")
    a("## 四、观察与结论")
    a("")
    hops1 = s1["hops"]
    fr = [h["finish_reason"] for h in hops1]
    u_first = hops1[0]["usage"]
    u_last = hops1[-1]["usage"]
    a(f"1. **finish_reason 是循环的开关**：幕 1 的轨迹是 {fr}——")
    a("   tool_calls 表示\"等我执行完再回来\"，stop 表示\"说完了\"。")
    a("   Agent 主循环就是盯着这个字段决定继续还是退出。")
    a(f"2. **工具定义有 token 成本**：幕 1 第一跳 prompt="
      f"{u_first.get('prompt_tokens')}（含 tools Schema），最终跳="
      f"{u_last.get('prompt_tokens')}（历史又加上了 tool 结果）。")
    a("   工具越多 Schema 越肥，这正是 W2 上下文工程\"预算\"视角的具体案例。")
    a("3. **tool 消息的配对规则**：assistant 的 tool_calls 必须原样回填，")
    a("   执行结果用 role=tool + tool_call_id 逐条对应——错配会被网关拒绝，")
    a("   这是 FC 历史管理最常见的坑。")
    a("4. **多轮时模型在第 3 轮能再次发起调用**：历史里的 tool 消息不是"
      "一次性脚手架，而是对话的正式公民；模型可以基于前几轮的工具结果")
    a("   决定继续查新城市。")
    a("")
    a("## 五、复现方式")
    a("")
    a("```bash")
    a("python 01Agent/w4day02/function_calling.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
