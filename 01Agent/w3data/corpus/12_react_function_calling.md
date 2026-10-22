# ReAct 与 Function Calling

## ReAct：推理与行动交替

ReAct（Reasoning + Acting）是 Agent 最基础的运行框架，核心是一个循环：

```text
Thought（我想做什么/为什么） → Action（调用某个工具） → Observation（工具返回结果） → 再 Thought …… 直到 Final Answer
```

为什么 ReAct 比单纯的 Reasoning（直接让模型想到底）更强：模型的知识是静态的、会幻觉；Action 让它把"不确定的部分"外包给真实工具——搜索、计算器、数据库——Observation 把真实世界的事实塞回上下文，下一步推理就有据可依。推理决定"下一步干什么"，行动获取"事实依据"，两者交替形成闭环。

## Function Calling：模型怎么"伸手"

Function Calling 的完整流程四步：

1. **定义工具**：请求体的 `tools` 参数里给出工具名、描述、参数 JSON Schema。
2. **模型决策**：模型看到用户问题和工具列表，判断要不要调用、调哪个、参数是什么，以 `tool_calls` 形式返回（finish_reason=tool_calls）。模型并不执行函数，只产出"调用意图"。
3. **代码执行**：你的代码解析 arguments，执行真实的函数（查天气、算数、查库）。
4. **结果回传**：把执行结果以 `role=tool` 消息追加进 messages 再发一次请求，模型基于结果生成最终回答。

这个"模型决策 → 代码执行 → 结果回传"的循环就是 Agent 的执行闭环：Function Calling 是 Agent 的手，JSON Mode 只是 Agent 的嘴。

## tool_choice 与并行调用

- `tool_choice: "auto"`：模型自己决定调不调（默认）。
- `tool_choice: "none"`：禁止调用工具，纯文本回答。
- `tool_choice: "required"`：必须调用某个工具。
- 也可以强制指定函数名：`{"type":"function","function":{"name":"get_weather"}}`。

Parallel Function Calling（并行工具调用）：模型在一次响应里返回多个 tool_calls（比如同时查两个城市的天气），客户端可以并发执行、一次性回传多个 role=tool 消息，减少往返轮次。

各厂商实现差异：OpenAI/DeepSeek 走 OpenAI tools 协议；Claude 的 tool use 内容块结构不同但思想一致；部分兼容网关不支持强制 tool_choice，需要降级为 auto。
