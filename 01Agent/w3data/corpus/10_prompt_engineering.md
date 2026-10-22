# Prompt Engineering：从 LangGPT 到结构化输出

## LangGPT：结构化提示词框架

LangGPT 用类似编程语言的结构（Markdown 标题分层：Role/Profile/Skills/Constraints/Workflow/OutputFormat/Initialization）写系统提示词，把"一段模糊的自然语言"升级成"一份角色说明书"。实测中结构化提示词在"输出结构命中率"上显著优于随意写的提示词，代价是 prompt token 数量成倍增加。

## Zero-shot / Few-shot / CoT

- **Zero-shot**：直接下指令，不给例子。最省 token，是默认起手。
- **Few-shot**：在提示词里给几个"输入→输出"示例，让模型模仿格式和口径。锁格式、锁语气最有效，成本是示例的 token。
- **CoT（Chain-of-Thought）**：要求模型"一步一步思考再给答案"，把中间推理显式写出来。多步推理任务上提升明显，代价是输出 token 变多、延迟变长，且思维链不等于真实因果（可能是事后合理化）。

选择原则：简单任务 Zero-shot；要稳定格式 Few-shot；多步推理、需要审计过程时 CoT。

## System Prompt vs User Prompt

System Prompt 是整段对话的"宪法"：设定角色、行为边界、输出规范，优先级高于用户消息，应放在 messages 数组最前面。User Prompt 是每轮的具体输入。实测同一个任务，把"角色定义 + 输出规范"放进 System、只把数据放 User，输出稳定性显著高于全部塞进一条 User 消息。

## 结构化输出的三种机制

1. **纯 Prompt 约定**：只在提示词里说"输出 JSON"。最不可靠：模型会加 Markdown 围栏、改字段名。
2. **JSON Mode**：API 参数 `response_format={"type":"json_object"}`。保证输出是合法 JSON 语法，但不保证字段名和 Schema 一致。
3. **Function Calling**：把 JSON Schema 挂在 `tools` 参数上。Schema 进入协议层，输出以"工具调用"形式返回，字段结构最稳。

分工记忆：JSON Mode 保证"是 JSON"，Schema 才保证"是哪个 JSON"。要数据用 JSON Mode，要动作（Agent 调工具）用 Function Calling。
