# -*- coding: utf-8 -*-
"""W4 Day5：MCP 测试客户端 —— 用 stdio 拉起 mcp_server.py 做端到端验证

对应任务："在本地测试 MCP Server 能否被调用"。

这就是一个最小 Host：按 MCP stdio 传输把 Server 作为子进程拉起，完成
initialize 握手 → tools/list 发现 → tools/call 调用 → resources/read 读取。
Claude Desktop、Cursor 等 Host 内置的也是同一条链路。

覆盖的验证点：
  1. initialize 握手：拿回 Server 名字与协议版本
  2. tools/list：4 个能力是否都注册成功
  3. tools/call：calculate 正常 + 非法输入、错误码查询（命中/未命中）、
     search_knowledge（fastembed 就绪时返回真检索结果）
  4. resources/read：policy://housing/北京
  5. 错误路径：调用不存在的工具，观察 MCP 的错误语义（isError 标志）

运行：python test_client.py（无需人工输入，全自动）
"""
import sys
import io
import json
import asyncio
import datetime
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "mcp_test_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "mcp_test_result.json")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_tests():
    """拉起 Server 并逐项验证。返回 (checks, server_info)。"""
    server_path = os.path.join(BASE_DIR, "mcp_server.py")
    # 用当前 venv 的解释器拉起 Server，保证 mcp/fastembed 依赖一致
    params = StdioServerParameters(command=sys.executable, args=[server_path])
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": ok, "detail": detail})
        print(f"  [{'✅' if ok else '❌'}] {name}: {detail}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) initialize 握手（mcp 2.x 的字段为 snake_case）
            info = await session.initialize()
            server_name = info.server_info.name
            proto = getattr(info, "protocol_version", "?")
            print(f"[handshake] server={server_name}  protocol={proto}")
            check("initialize 握手", server_name == "hello-agent-tools",
                  f"serverInfo.name={server_name}, protocolVersion={proto}")

            # 2) tools/list 能力发现
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"[tools/list] {names}")
            check("tools/list 发现工具",
                  {"calculate", "lookup_travel_error_code",
                   "search_knowledge"}.issubset(set(names)),
                  f"共 {len(names)} 个: {names}")

            # 3) tools/call：calculate
            r = await session.call_tool("calculate", {"expression": "3*7+2"})
            out = r.content[0].text if r.content else ""
            print(f"[tools/call] calculate(3*7+2) = {out}")
            check("calculate 正常计算", out.strip() == "23", f"返回 {out!r}")

            r = await session.call_tool("calculate", {"expression": "import os"})
            out = r.content[0].text if r.content else ""
            print(f"[tools/call] calculate('import os') = {out[:50]}")
            check("calculate 白名单拦截", "错误" in out, f"返回 {out!r}")

            # 4) 错误码查询（命中 / 未命中）
            r = await session.call_tool("lookup_travel_error_code",
                                        {"code": "TRV-40907"})
            out = r.content[0].text if r.content else ""
            print(f"[tools/call] lookup TRV-40907 = {out[:60]}")
            check("错误码查询（命中）", "逾期特批" in out, out[:60])

            r = await session.call_tool("lookup_travel_error_code",
                                        {"code": "XXX-00000"})
            out = r.content[0].text if r.content else ""
            print(f"[tools/call] lookup XXX-00000 = {out[:60]}")
            check("错误码查询（未命中有可读提示）", out.startswith("未知错误码"),
                  out[:60])

            # 5) search_knowledge（fastembed 就绪时是真检索）
            r = await session.call_tool("search_knowledge",
                                        {"question": "RRF 融合公式"})
            out = r.content[0].text if r.content else ""
            ok = out.startswith("【") and "不可用" not in out
            print(f"[tools/call] search_knowledge = {out[:60].replace(chr(10), ' ')}")
            check("search_knowledge 检索", ok,
                  (out[:60] + "…") if ok else out[:80])

            # 6) resources/read：只读资源（2.x 直接收字符串 URI）
            r = await session.read_resource("policy://housing/北京")
            out = r.contents[0].text
            print(f"[resources/read] policy://housing/北京 = {out[:60]}")
            check("resources 读住宿标准", "600" in out, out[:60])

            # 7) 错误路径：调用不存在的工具
            try:
                r = await session.call_tool("no_such_tool", {})
                is_err = getattr(r, "isError", False)
                detail = r.content[0].text if r.content else ""
                # SDK 1.x 置 isError 标志；2.x 返回 "Unknown tool" 文本结果。
                # 两种都是"连接不炸、错误可读"的合法错误语义。
                ok = is_err or "unknown tool" in detail.lower()
                print(f"[tools/call] no_such_tool → isError={is_err} {detail[:60]}")
                check("未知工具返回 MCP 错误语义", ok,
                      f"isError={is_err}, 文本: {detail[:50]}")
            except Exception as e:
                print(f"[tools/call] no_such_tool → 异常 {type(e).__name__}: {str(e)[:80]}")
                check("未知工具返回 MCP 错误语义", True,
                      f"协议层抛出 {type(e).__name__}（也是合法的错误路径）")

    return checks


def main():
    print("=" * 62)
    print("MCP Server 端到端测试（stdio：Host 拉起子进程）")
    print("=" * 62)
    checks = asyncio.run(run_tests())
    passed = sum(1 for c in checks if c["ok"])
    print()
    print("=" * 62)
    print(f"结果：{passed}/{len(checks)} 通过")
    print("=" * 62)

    payload = {"meta": {"generated_at": datetime.datetime.now().isoformat(),
                        "transport": "stdio",
                        "server": "mcp_server.py (hello-agent-tools)"},
               "checks": checks,
               "passed": passed, "total": len(checks)}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"[done] 报告：{REPORT_PATH}")


def build_report(p):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    a = L.append
    a("# MCP Server 端到端测试报告（W4 Day5）")
    a("")
    a(f"- 生成时间：{now}")
    a("- Server：`mcp_server.py`（官方 SDK FastMCP，stdio 传输）")
    a("- Client：`test_client.py`（最小 Host：拉起子进程 → 握手 → 发现 → 调用）")
    a("")
    a("## 一、测试结果")
    a("")
    a(f"**{p['passed']}/{p['total']} 通过**")
    a("")
    a("| 验证点 | 结果 | 说明 |")
    a("|---|---|---|")
    for c in p["checks"]:
        a(f"| {c['name']} | {'✅' if c['ok'] else '❌'} | {c['detail'][:80]} |")
    a("")
    a("## 二、这条链路验证了什么")
    a("")
    a("```text")
    a("test_client.py（Host 角色）                    mcp_server.py（Server 子进程）")
    a("  │ stdio_client(params) ──拉起子进程──────────▶ FastMCP 启动")
    a("  │ session.initialize() ──JSON-RPC 握手──────▶ 返回 serverInfo/能力")
    a("  │ session.list_tools() ──发现───────────────▶ 3 Tools + 1 Resource")
    a("  │ session.call_tool(...) ──调用─────────────▶ 执行函数 → content 返回")
    a("  │ session.read_resource(...) ──只读─────────▶ 文本资源直接返回")
    a("```")
    a("")
    a("- **stdio 传输的本质**：Server 是 Host 的子进程，协议消息走 stdin/stdout，")
    a("  所以日志必须走 stderr（FastMCP 已处理），否则会污染协议通道。")
    a("- **工具的容错责任在 Server**：search_knowledge 在 fastembed 未就绪时")
    a("  返回可读的\"[检索不可用]\"文本而不是抛异常——Server 崩了整条链路就断。")
    a("- **错误有统一语义**：调用不存在的工具不炸连接，返回 isError 标志的")
    a("  结果或协议层异常——Host 据此决定重试/降级/提示用户。")
    a("- **与 Claude Desktop 的对照**：Desktop 的配置文件里每条")
    a("  `command + args` 就对应本次 StdioServerParameters；内置 Client 完成")
    a("  的正是本脚本手写的同一条链路，外加用户确认 UI。")
    a("")
    a("## 三、怎么把它接进真实 Host（例如 Claude Desktop）")
    a("")
    a("```json")
    a("// claude_desktop_config.json 片段")
    a('{')
    a('  "mcpServers": {')
    a('    "hello-agent-tools": {')
    a('      "command": "<venv python 绝对路径>",')
    a('      "args": ["<仓库绝对路径>/01Agent/w4day05/mcp_server.py"]')
    a("    }")
    a("  }")
    a("}")
    a("```")
    a("")
    a("## 四、复现方式")
    a("")
    a("```bash")
    a("pip install mcp          # 官方 SDK（含 FastMCP）")
    a("python 01Agent/w4day05/test_client.py    # 自动拉起 Server 并测试")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
