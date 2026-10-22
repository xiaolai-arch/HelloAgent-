# MCP：Model Context Protocol

## MCP 是什么

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 在 2024 年 11 月开源的协议，用统一的方式把外部工具和数据源接给 LLM 应用。一句话：**MCP 之 于 工具生态，就像 USB 之于外设——写一次 MCP Server，任何支持 MCP 的客户端（Claude Desktop、IDE、Agent 框架）都能用。**

## 三层角色架构

- **Host（宿主）**：用户直接使用的 LLM 应用，如 Claude Desktop、Cursor、各种 Agent 客户端。Host 决定接哪些 Server、管理鉴权和用户确认。
- **Client（客户端）**：Host 内部与某个 Server 保持 1:1 连接的中间层，负责协议握手和消息转发。
- **Server（服务器）**：暴露具体能力的轻量进程，每个 Server 专注一类资源（文件系统、GitHub、数据库、浏览器……）。

## 协议细节

传输层：本地进程用 stdio（标准输入输出上的 JSON-RPC 2.0），远程服务用 Streamable HTTP。生命周期：initialize 握手（交换协议版本和能力声明）→ 发现（tools/list 列出工具、resources/list 列出资源、prompts/list 列出提示模板）→ 调用（tools/call）。Server 可以提供三类能力：Tools（模型可调用的函数）、Resources（可读取的数据，类似 GET）、Prompts（预置提示模板）。

## MCP vs Function Calling

- Function Calling 是**模型 API 内的机制**：Schema 挂在请求体 tools 上，工具定义写在每个应用自己的代码里，换个应用要重写一遍。
- MCP 是**应用之间的开放协议**：工具实现沉淀在独立的 Server 进程里，与具体 Host 解耦，任何人写一次、大家复用。

适用场景：自己一个应用内部调工具，Function Calling 直接了当；要把工具开放给整个生态、或接入现成的第三方 Server（文件系统、GitHub、Slack），用 MCP。两者不是竞争关系——MCP Server 列出的工具，最终还是通过 Function Calling 让模型决策调用的。

## 实现方式

官方 Python SDK（`pip install mcp`，FastMCP 风格）：用装饰器把普通 Python 函数声明成工具，几行代码就是一个 stdio MCP Server。调试可以用 MCP Inspector，或在客户端里用 stdio 直接拉起 Server 进程测试。
