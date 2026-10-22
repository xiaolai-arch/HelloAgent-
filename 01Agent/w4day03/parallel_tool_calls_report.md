# 并行工具调用与 tool_choice 实验报告（W4 Day3）

- 生成时间：2026-10-21 16:53:27
- 模型：`mimo-v2.6-flash`（temperature=0）
- 工具：get_weather（模拟 0.3s IO）/ calculate

## 一、并行 Function Calling

- 问题：北京、上海、广州现在分别多少度？哪个最热？
- 模型在一条 assistant 消息里返回 **3 个 tool_calls**：['{"city": "北京"}', '{"city": "上海"}', '{"city": "广州"}']
- 线程池并发执行：墙钟 **0.31s**，若串行累计要 **0.93s**——每个工具 0.3s 模拟 IO，并行的收益就是把 sum 变成 max。
- 最终回答（finish_reason=stop）：三个城市当前气温：

| 城市 | 气温 | 天气 |
|------|------|------|
| 北京 | 9℃ | 晴 |
| 上海 | 17℃ | 多云 |
| 广州 | 26℃ | 阵雨 |

**广州最热**（26℃），比上

并行的三个要点：

1. **模型侧**：一次响应给出多个 tool_calls（各自带独立 id 和 arguments），
   调用之间没有隐藏依赖时才会并行——有依赖的（先查 A 再用 A 查 B）模型
   通常会拆成多轮。
2. **客户端侧**：线程池/asyncio 并发执行，然后把 N 条结果按
   `role=tool + tool_call_id` 一次性全部回传，不要查一条回一条。
3. **什么时候能并行**：工具是只读查询（天气/搜索/查库）时放心并行；
   写操作（下单/转账）要慎重——并发执行绕过了顺序保证。

## 二、tool_choice 四档对照（实测）

| 请求的 tool_choice | 问题 | finish_reason | 实际调用 |
|---|---|---|---|
| `auto` | 北京多少度？ | `tool_calls` | get_weather({"city": "北京"}) |
| `forced` | 北京多少度？ | `tool_calls` | get_weather({"city": "北京"}) |
| `required` | 北京多少度？ | `tool_calls` | get_weather({"city": "北京"}) |
| `forced` | 用一句话解释注意力机制（与天气无关） | `stop` | 无（直接 stop） |

### 逐档解读（对照本机实测结果）

- **auto**：按预期调用了工具——auto 是默认档：模型按"回答问题需不需要"自主决定，FC 的本意正是"可以调"而非"必须调"。
- **none（实测异常）**：请求了 tool_choice=none，模型仍然返回了 ['get_weather']——本网关没有落实 none 档。协议里的可选字段不一定被每个端点实现，**每个字段都要实测**，不能只信文档。业务上需要"绝不允许调工具"时，最稳的做法是请求里干脆不带 tools 字段。
- **required**：按预期强制调用（['get_weather']）。适合"必须经过工具"的业务（如查库后再答）。
- **强制指定函数名（实测异常）**：finish_reason=`stop`、无 tool_calls——本网关忽略了强制档，模型按普通 auto 语义直接回答了。若业务依赖"必须调某个函数"（如强制保存表单），在这类端点上要加客户端兜底校验。

## 三、GPT-4o / DeepSeek / Claude 的 FC 实现差异（静态梳理）

| 维度 | OpenAI / DeepSeek（兼容系） | Claude（Anthropic 系） |
|---|---|---|
| 声明方式 | 请求体 `tools` 数组，function 内嵌 parameters | 请求体 `tools`，`input_schema` 字段 |
| 模型输出 | `message.tool_calls[]`（function.name/arguments） | `content[]` 里的 `tool_use` 块（id/name/input） |
| 结果回传 | `role: "tool"` 消息 + tool_call_id | `role: "user"` 消息内 `tool_result` 块 |
| 强制调用 | `tool_choice: required` 或指定函数名 | `tool_choice: {"type": "tool", "name": ...}` |
| 并行控制 | 模型自行决定（parallel_tool_calls 参数可关） | 默认并行，disable_parallel_tool_use 可关 |
| 思想 | 完全一致：模型产意图、代码执行、结果回传、循环到 stop ||

记忆锚点：**OpenAI 系的工具结果是一条独立消息，Claude 系的工具结果
塞在 user 消息的块里**——做网关适配时最容易踩的差异。

## 四、复现方式

```bash
python 01Agent/w4day03/parallel_tool_calls.py
```
