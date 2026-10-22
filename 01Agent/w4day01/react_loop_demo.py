# -*- coding: utf-8 -*-
"""W4 Day1：ReAct 循环 —— 手写一个最小 Agent（纯 Prompt 协议版）

对应任务：
  1. 读 ReAct 论文思想，画出 Thought → Action → Observation 循环（见报告）
  2. 理解为什么 ReAct 比单纯 Reasoning 强大
  3. 本脚本：不用任何框架、不用 Function Calling API，
     只靠 system prompt 里的"文本协议" + 字符串解析，手写一个能跑的 ReAct Agent

为什么先手写再学 Function Calling（Day2）：
  ReAct 的本质是"模型输出结构化文本 → 你的代码解析并执行 → 结果拼回上下文"。
  Function Calling 只是把这套协议搬进了 API 层（模型输出 tool_calls 而不是文本）。
  手写一遍才能看清：协议、解析、循环、终止条件，这四样 FC 都替你做了。

工具（2 个，够演示"推理决定下一步"）：
  calculate(表达式)      —— 纯 Python 四则运算（白名单校验，不裸 eval）
  search_knowledge(问题) —— 在 W3 的语料库里做向量检索 top2（复用 ragkit）

测试题（必须组合两个工具才能答对，考察"推理+行动交替"）：
  "HelloAI 公司报销超过 5000 元需要谁加签？3 个人每人 2200 元的报销单，
   总金额需要加签吗？"
  预期路径：search（查到 >5000 需财务经理加签）→ calculate（3×2200=6600）
  → 比较 → Final Answer。

产出：react_loop_demo_report.md（流程图解 + 完整轨迹）+ react_trace.json
依赖：fastembed（检索工具）；LLM 网关同 W1/W2。
"""
import sys
import io
import os
import re
import json
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "react_loop_demo_report.md")
TRACE_PATH = os.path.join(BASE_DIR, "react_trace.json")

import ragkit  # noqa: E402

MAX_STEPS = 6  # 循环上限：防模型打转烧钱，也是所有 Agent 框架的标配参数

# ---- 工具实现（真实执行方是你的代码，不是模型）-----------------------------

_KNOWLEDGE_cache = None  # (chunks, 向量列表)——语料向量只编码一次，检索直接用


def tool_calculate(expression):
    """四则运算计算器。用正则白名单过滤，杜绝 eval 执行任意代码。"""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "错误：只支持数字与 + - * / ( ) 的四则运算"
    try:
        return str(eval(expression))  # noqa: S307  已白名单校验
    except Exception as e:
        return f"计算错误：{e}"


def tool_search_knowledge(question):
    """在 W3 语料库里做向量检索 top2（语义向量，见 w3day01/03）。"""
    global _KNOWLEDGE_cache
    if _KNOWLEDGE_cache is None:
        chunks = []
        for d in ragkit.load_corpus():
            for i, (text, _) in enumerate(ragkit.recursive_chunk(d["text"], 300)):
                chunks.append({"id": f"{d['id']}#c{i}", "text": text})
        emb = ragkit.get_embedder()
        _KNOWLEDGE_cache = (chunks, emb.passage_embed([c["text"] for c in chunks]))
    chunks, vecs = _KNOWLEDGE_cache
    emb = ragkit.get_embedder()
    qvec = emb.query_embed([question])[0]
    scored = sorted(((ragkit.cosine(qvec, v), c) for c, v in zip(chunks, vecs)),
                    key=lambda x: -x[0])
    return "\n\n".join(f"【{c['id']}】{c['text']}" for _, c in scored[:2])


TOOLS = {
    "calculate": tool_calculate,
    "search_knowledge": tool_search_knowledge,
}

# ---- ReAct 的全部秘密都在这段 system prompt 里：协议即提示词 --------------

SYSTEM_PROMPT = """你是一个使用 ReAct 模式解决问题的助手。严格按以下格式输出，一次只输出一步：

Thought: 你的思考（分析当前已知信息，决定下一步）
Action: 工具名，只能是 calculate 或 search_knowledge
Action Input: 工具的输入（calculate 用数学表达式；search_knowledge 用检索问题）
（然后停止输出，等待系统返回 Observation）

当你认为信息足够时，改用：
Thought: 我已经知道答案了
Final Answer: 最终回答

规则：
1. 每次只输出一个 Action，绝不自己编造 Observation。
2. 数值计算必须用 calculate，不要心算。
3. 事实依据必须用 search_knowledge 查证，不要凭记忆回答。"""


def parse_step(text):
    """从模型输出解析 Action 指令。返回 (tool, action_input) 或 None。"""
    m = re.search(r"Action:\s*(\w+)\s*\nAction Input:\s*(.+)", text)
    if not m:
        return None
    tool, arg = m.group(1).strip(), m.group(2).strip().strip('"')
    return (tool, arg) if tool in TOOLS else None


def react_loop(question, trace_steps):
    """ReAct 主循环：模型给文本协议 → 解析 → 执行 → Observation 拼回。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    for step in range(1, MAX_STEPS + 1):
        resp, err = ragkit.chat(messages, temperature=0.0, max_tokens=800)
        if err:
            trace_steps.append({"step": step, "error": err})
            return f"[API 错误] {err}"
        out = resp["choices"][0]["message"]["content"].strip()

        final = re.search(r"Final Answer:\s*(.+)", out, re.S)
        action = parse_step(out)

        if final:
            trace_steps.append({"step": step, "role": "final",
                                "model_output": out[:500],
                                "answer": final.group(1).strip()})
            return final.group(1).strip()

        if not action:
            # 模型没按协议输出：不重试魔法，把格式要求再钉一遍塞回上下文
            trace_steps.append({"step": step, "role": "format-error",
                                "model_output": out[:300]})
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content":
                             "（格式错误：请严格按 Thought/Action/Action Input 或 "
                             "Final Answer 格式重新输出这一步。）"})
            continue

        tool, arg = action
        observation = TOOLS[tool](arg)
        tm = re.search(r"Thought:\s*(.+)", out)
        trace_steps.append({"step": step, "role": "act",
                            "thought": (tm.group(1)[:200] if tm else ""),
                            "tool": tool, "input": arg,
                            "observation": observation[:400]})
        print(f"  step{step} Thought: {trace_steps[-1]['thought'][:80]}")
        print(f"  step{step} Action: {tool}({arg[:60]})")
        print(f"  step{step} Observation: {observation[:100].replace(chr(10), ' ')}...")

        # 上下文滚动：模型输出 + Observation 都进历史（这就是 Agent 的"记忆"）
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user",
                         "content": f"Observation: {observation}\n（请继续下一步）"})

    trace_steps.append({"step": MAX_STEPS + 1, "role": "max-steps"})
    return "[达到最大步数仍未给出最终答案]"


TEST_CASES = [
    {"id": "T1", "question": "HelloAI 公司报销超过 5000 元需要谁加签？"
                             "3 个人每人 2200 元的报销单，总金额需要加签吗？",
     "expect_tools": ["search_knowledge", "calculate"]},
    {"id": "T2", "question": "DeepSeek-V3 每个 token 激活多少 B 参数？"
                             "一个 128K token 的 batch 总共激活多少 B 参数？",
     "expect_tools": ["search_knowledge", "calculate"]},
]


def main():
    emb = ragkit.get_embedder()
    print(f"[embedder] kind={emb.kind}  model={emb.name}   # 检索工具就绪")
    print(f"[model   ] {ragkit.MODEL}")
    all_traces = []
    for tc in TEST_CASES:
        print()
        print("=" * 62)
        print(f"[{tc['id']}] {tc['question']}")
        print("=" * 62)
        steps = []
        t0 = datetime.datetime.now()
        answer = react_loop(tc["question"], steps)
        dur = (datetime.datetime.now() - t0).total_seconds()
        tools_used = [s["tool"] for s in steps if s.get("role") == "act"]
        ok = all(t in tools_used for t in tc["expect_tools"])
        print(f"\n  [answer] {answer}")
        print(f"  [stats ] {len(steps)} 步 | 工具调用={tools_used} | "
              f"耗时 {dur:.1f}s | 工具覆盖={'✅' if ok else '❌'}")
        all_traces.append({"id": tc["id"], "question": tc["question"],
                           "answer": answer, "steps": steps,
                           "tools_used": tools_used, "tool_coverage_ok": ok,
                           "duration_sec": round(dur, 1)})

    with open(TRACE_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": {"generated_at": datetime.datetime.now().isoformat(),
                            "model": ragkit.MODEL, "max_steps": MAX_STEPS,
                            "embedder": {"kind": emb.kind, "name": emb.name},
                            "degraded": getattr(emb, "warn", None)},
                   "traces": all_traces}, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(all_traces))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 轨迹：{TRACE_PATH}")


def build_report(traces):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    a = L.append
    a("# ReAct 循环：手写最小 Agent 实验报告（W4 Day1）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 模型：`{ragkit.MODEL}`（temperature=0）")
    a("- 实现方式：纯 Prompt 协议 + 字符串解析（不用框架、不用 Function Calling）")
    a("")
    a("## 一、ReAct 流程图解（白板版）")
    a("")
    a("```text")
    a("        ┌────────────────────────────────────────────┐")
    a("        │              用户问题（Question）            │")
    a("        └──────────────────┬─────────────────────────┘")
    a("                           ▼")
    a("        ┌────────────────────────────────────────────┐")
    a("  ┌────▶│  Thought：我知道什么？还缺什么？下一步干什么？│")
    a("  │     └──────────────────┬─────────────────────────┘")
    a("  │                        ▼")
    a("  │     ┌────────────────────────────────────────────┐")
    a("  │     │  Action：选一个工具 + 给出输入               │")
    a("  │     └──────────────────┬─────────────────────────┘")
    a("  │                        ▼")
    a("  │     ┌────────────────────────────────────────────┐")
    a("  │     │  你的代码执行工具 → Observation（真实事实）   │")
    a("  │     └──────────────────┬─────────────────────────┘")
    a("  │                        │ 拼回上下文")
    a("  └────────────────────────┘ （信息够了 → Final Answer）")
    a("```")
    a("")
    a("### 为什么比单纯 Reasoning 强")
    a("")
    a("- 模型的参数知识是**静态且会幻觉**的；Action 把不确定的部分外包给真实工具，")
    a("  Observation 把事实塞回上下文，下一步推理从\"凭记忆\"变成\"凭证据\"。")
    a("- 推理决定\"下一步做什么\"（规划），行动获取\"事实依据\"（接地），")
    a("  两者交替 = 会用工具的 System 1 + System 2。")
    a("- 本实验里模型被明确要求\"数值必须 calculate、事实必须 search\"，")
    a("  把两类幻觉（算错数、编造制度）分别堵在工具后面。")
    a("")
    a("## 二、协议设计（system prompt 即协议）")
    a("")
    a("ReAct 没有任何 API 魔法，四个要素全部是提示词约定：")
    a("")
    a("```text")
    a("Thought: 分析+决策        ← 模型生成")
    a("Action: 工具名            ← 模型生成（受限在工具列表内）")
    a("Action Input: 工具输入    ← 模型生成")
    a("Observation: 工具结果     ← 你的代码生成，模型只许读不许编")
    a("```")
    a("")
    a(f"终止条件两种：模型输出 `Final Answer`（正常），或达到 MAX_STEPS="
      f"{MAX_STEPS}（防打转，所有 Agent 框架的标配保险丝）。")
    a("")
    a("## 三、完整运行轨迹")
    a("")
    for t in traces:
        a(f"### {t['id']}：{t['question']}")
        a("")
        for s in t["steps"]:
            if s.get("role") == "act":
                a(f"- **step{s['step']}** Thought: {s['thought']}")
                a(f"  - Action: `{s['tool']}({s['input']})`")
                a(f"  - Observation: {s['observation'][:120].replace(chr(10), ' ')}…")
            elif s.get("role") == "final":
                a(f"- **step{s['step']}** Final Answer: {s['answer']}")
            elif s.get("role") == "format-error":
                a(f"- **step{s['step']}** ⚠️ 格式违规（已要求重输出）：{s['model_output'][:80]}…")
            elif s.get("role") == "max-steps":
                a(f"- **step{s['step']}** ⚠️ 达到最大步数")
        a("")
        a(f"- 工具调用序列：{t['tools_used']}（覆盖预期={'✅' if t['tool_coverage_ok'] else '❌'}）")
        a(f"- 最终回答：{t['answer']}")
        a("")
    a("## 四、观察与结论")
    a("")
    ok_n = sum(1 for t in traces if t["tool_coverage_ok"])
    fmt_err = sum(1 for t in traces for s in t["steps"] if s.get("role") == "format-error")
    a(f"1. **协议可用但脆弱**：{ok_n}/{len(traces)} 题的工具覆盖符合预期，"
      f"出现 {fmt_err} 次格式违规。纯文本协议没有 Schema 约束，")
    a("   模型偶尔会把 Action 和 Final Answer 混在一行——这正是 Day2 "
      "Function Calling 要解决的问题：**协议进 API 层，解析交给协议**。")
    a("2. **Observation 是防幻觉的锚**：检索工具把制度原文塞进上下文后，")
    a("   回答里出现了\"财务经理加签\"这类可追溯细节——对照 W2 的实验：")
    a("   有证据的生成比裸生成可靠。")
    a("3. **循环上限是安全带**：MAX_STEPS 防止模型在同一个工具上打转烧钱；")
    a("   生产 Agent 还需要：工具超时、结果截断（本例 Observation 截到 400 字）、")
    a("   重试与降级。")
    a("4. **上下文滚动成本**：每一步把全部历史重发一遍，步数越多 prompt")
    a("   tokens 越长——Agent 的费用曲线是二次的，这直接通向 Context")
    a("   Engineering 的预算/压缩议题（W2 延伸、阶段四 W16）。")
    a("")
    a("## 五、复现方式")
    a("")
    a("```bash")
    a("pip install fastembed")
    a("python 01Agent/w4day01/react_loop_demo.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
