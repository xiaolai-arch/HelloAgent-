# MCP Server 端到端测试报告（W4 Day5）

- 生成时间：2026-10-21 16:59:27
- Server：`mcp_server.py`（官方 SDK FastMCP，stdio 传输）
- Client：`test_client.py`（最小 Host：拉起子进程 → 握手 → 发现 → 调用）

## 一、测试结果

**9/9 通过**

| 验证点 | 结果 | 说明 |
|---|---|---|
| initialize 握手 | ✅ | serverInfo.name=hello-agent-tools, protocolVersion=2025-11-25 |
| tools/list 发现工具 | ✅ | 共 3 个: ['calculate', 'lookup_travel_error_code', 'search_knowledge'] |
| calculate 正常计算 | ✅ | 返回 '23' |
| calculate 白名单拦截 | ✅ | 返回 '错误：只支持数字与 + - * / ( ) 的四则运算' |
| 错误码查询（命中） | ✅ | TRV-40907：报销单提交超过 15 日期限。需总监在 OA 里发起'逾期特批'流程。 |
| 错误码查询（未命中有可读提示） | ✅ | 未知错误码 XXX-00000。已知错误码：TRV-40231, TRV-40907, TRV-41188, TRV-5 |
| search_knowledge 检索 | ✅ | 【08_hybrid_rerank#c1】RRF：按排名融合

RRF（Reciprocal Rank Fusion）是… |
| resources 读住宿标准 | ✅ | 北京 住宿标准：普通员工 600 元/晚，总监及以上 900 元/晚。 |
| 未知工具返回 MCP 错误语义 | ✅ | isError=False, 文本: Unknown tool: no_such_tool |

## 二、这条链路验证了什么

```text
test_client.py（Host 角色）                    mcp_server.py（Server 子进程）
  │ stdio_client(params) ──拉起子进程──────────▶ FastMCP 启动
  │ session.initialize() ──JSON-RPC 握手──────▶ 返回 serverInfo/能力
  │ session.list_tools() ──发现───────────────▶ 3 Tools + 1 Resource
  │ session.call_tool(...) ──调用─────────────▶ 执行函数 → content 返回
  │ session.read_resource(...) ──只读─────────▶ 文本资源直接返回
```

- **stdio 传输的本质**：Server 是 Host 的子进程，协议消息走 stdin/stdout，
  所以日志必须走 stderr（FastMCP 已处理），否则会污染协议通道。
- **工具的容错责任在 Server**：search_knowledge 在 fastembed 未就绪时
  返回可读的"[检索不可用]"文本而不是抛异常——Server 崩了整条链路就断。
- **错误有统一语义**：调用不存在的工具不炸连接，返回 isError 标志的
  结果或协议层异常——Host 据此决定重试/降级/提示用户。
- **与 Claude Desktop 的对照**：Desktop 的配置文件里每条
  `command + args` 就对应本次 StdioServerParameters；内置 Client 完成
  的正是本脚本手写的同一条链路，外加用户确认 UI。

## 三、怎么把它接进真实 Host（例如 Claude Desktop）

```json
// claude_desktop_config.json 片段
{
  "mcpServers": {
    "hello-agent-tools": {
      "command": "<venv python 绝对路径>",
      "args": ["<仓库绝对路径>/01Agent/w4day05/mcp_server.py"]
    }
  }
}
```

## 四、复现方式

```bash
pip install mcp          # 官方 SDK（含 FastMCP）
python 01Agent/w4day05/test_client.py    # 自动拉起 Server 并测试
```
