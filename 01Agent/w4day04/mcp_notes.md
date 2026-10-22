# MCP 协议学习笔记：Model Context Protocol（W4 Day4）

> 对应任务：读 MCP 官方文档理解架构；理解 Host → Client → Server 三层角色；
> 能口述 MCP vs Function Calling 的区别和适用场景。
> 配套代码：Day5 的 `mcp_server.py` + `test_client.py`（本笔记的理论落地）。

---

## 0. 一句话理解

> **MCP 之于工具生态，就像 USB 之于外设：写一次 MCP Server，任何支持 MCP 的客户端都能用。**

Function Calling 解决"模型怎么表达要调工具"；MCP 解决"工具怎么跨应用复用"。
两者不是竞争关系，是上下两层：MCP Server 暴露的工具，最终还是通过
Function Calling 让模型决策调用（Day3 已实测验证这个衔接）。

---

## 1. 为什么需要 MCP

没有 MCP 的世界：每个 LLM 应用（Claude Desktop、Cursor、自研 Agent）
都要为"读文件/查 GitHub/查数据库"各写一遍胶水代码——M 个应用 × N 个
数据源 = M×N 份集成。

MCP 把 M×N 变成 M+N：应用方实现一次 MCP Client，工具方实现一次 MCP
Server，协议负责中间的握手、发现和调用。

```
没有 MCP：  App1 ──自定义代码──> GitHub        有 MCP：  App1 ─┐        ┌─> Server(GitHub)
            App1 ──自定义代码──> 文件系统                        App2 ─┼─ MCP ──┼─> Server(文件系统)
            App2 ──另一套代码──> GitHub                          App3 ─┘        └─> Server(数据库)
            ...（M×N 份胶水）                                        （M+N 份实现）
```

---

## 2. 三层角色架构

| 角色 | 是什么 | 职责 | 例子 |
|---|---|---|---|
| **Host（宿主）** | 用户直接使用的 LLM 应用 | 决定接哪些 Server、管理鉴权、收集用户确认、执行工具调用 | Claude Desktop、Cursor、自研 Agent |
| **Client（客户端）** | Host 内部的协议中间层 | 与单个 Server 保持 1:1 连接，负责握手和消息转发 | Host 进程里的 MCP client 模块 |
| **Server（服务器）** | 暴露能力的轻量进程 | 专注一类资源：文件系统/GitHub/数据库/浏览器…… | 官方 `mcp-server-git`、自己写的 `mcp_server.py` |

要点：

- **1 : N : N**：一个 Host 管多个 Client，每个 Client 只连一个 Server。
- Host 是**安全边界**：用户在 Host 层看到 Server 要调什么工具、批准或拒绝；
  Server 之间互相不可见，权限由 Host 按会话分配。
- Server 是**普通进程**：stdio 模式下就是 Host 拉起的子进程，崩了随 Host
  重启，没有魔法。

---

## 3. 协议细节（stdio 传输为例）

传输层：本地进程走 **stdio 上的 JSON-RPC 2.0**（Day5 的 test_client 就是在
裸实现这个）；远程服务走 Streamable HTTP。

生命周期与核心方法：

```text
1. initialize        双方交换 protocolVersion 和 capabilities（能力声明）
2. tools/list        客户端发现工具：名字、描述、输入 Schema
   resources/list    发现资源（可读数据，类 GET 语义）
   prompts/list      发现提示模板
3. tools/call        调用：{name, arguments} → {content, isError}
4. notifications     进度/日志/资源变更通知（可选）
```

Server 可以暴露三类能力：

| 能力 | 语义 | 类比 |
|---|---|---|
| **Tools** | 模型可调用的函数，可能有副作用 | POST |
| **Resources** | 客户端可读取的数据，只读 | GET |
| **Prompts** | 预置的提示词模板 | 静态配置 |

---

## 4. MCP vs Function Calling（口述版）

| 维度 | Function Calling | MCP |
|---|---|---|
| 层次 | 模型 API 内的机制（单次请求体） | 应用之间的开放协议（跨进程） |
| 工具定义在哪 | 每个应用的请求体里，换应用重写 | 独立 Server 进程里，写一次处处用 |
| Schema 谁约束 | API 的 tools 字段 | MCP 的 tools/list 返回，最终仍以 FC 形式给模型 |
| 适用场景 | 单应用内部调工具 | 跨应用复用、接入现成生态、给客户分发集成 |
| 谁执行 | 应用自己的代码 | Server 进程（Host 代理调用） |

**30 秒口述稿**：

> FC 是模型 API 层的机制，解决"模型怎么表达要调工具"；MCP 是应用层的开放
> 协议，解决"工具怎么写一次、处处复用"。MCP Server 把工具、资源、提示模板
> 以标准方式暴露出来，Host 里的 Client 完成发现和调用，而模型侧最终看到的
> 仍然是一份份 tools Schema——也就是说 MCP 不替代 FC，它站在 FC 上游把
> 工具的分发和复用标准化了。单应用内部直接 FC 最简单；要接生态、要多应用
> 共享工具、要把集成能力分发给客户，用 MCP。

---

## 5. 什么时候用 MCP（判断清单）

- ✅ 要接入现成的第三方 Server（文件系统、GitHub、Slack、Postgres……）
- ✅ 同一组工具要服务多个宿主（IDE + CLI + 自研 Agent）
- ✅ 要把集成能力作为产品分发给用户（用户自己的数据源）
- ❌ 单个应用内部、几个内部工具 → 直接 Function Calling，别加进程间通信
- ❌ 对延迟极度敏感 → stdio 子进程和多一跳协议都有成本

---

## 6. 安全与工程注意

1. **Host 是权限边界**：工具执行要用户确认（尤其写操作）；Server 拿到的
   凭证应最小化。
2. **提示注入经工具放大**：检索到的网页/文件内容可能含恶意指令，
   模型读到后可能调用有副作用的工具——高危工具要加人工确认和参数白名单。
3. **Server 崩溃与超时**：stdio Server 是子进程，要有重启和健康检查；
   tools/call 要有超时。
4. **版本兼容**：initialize 时的 protocolVersion 协商；升级 Server 前看
   changelog。

---

## 7. 自测十问（能答 7 个算过关）

1. MCP 三层角色各自负责什么？Host 为什么是安全边界？
2. tools / resources / prompts 三类能力分别什么语义？
3. stdio 传输下，一条 tools/call 请求的完整旅程经过哪些层？
4. MCP 和 Function Calling 是什么关系？谁替代谁？
5. initialize 握手交换了什么？为什么需要能力协商？
6. 什么场景不该用 MCP？
7. Server 进程崩了会发生什么？Host 该怎么处理？
8. 提示注入在 MCP 场景下为什么更危险？
9. MCP 的 M+N 是相对什么说的？
10. 一个 MCP Server 的工具列表，最终以什么形式被模型"看见"？

---

## 8. 参考资料

- 官方文档：https://modelcontextprotocol.io （Introduction / Architecture / Tools）
- 规范原文：https://spec.modelcontextprotocol.io
- TypeScript/Python SDK：`@modelcontextprotocol/sdk`、`pip install mcp`
- 官方 Server 合集：https://github.com/modelcontextprotocol/servers
