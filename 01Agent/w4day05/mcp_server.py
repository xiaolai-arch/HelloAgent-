# -*- coding: utf-8 -*-
"""W4 Day5：第一个 MCP Server（官方 Python SDK，FastMCP 风格）

对应任务：
  1. 用 MCP Python SDK 写一个简单的 MCP Server
  2. 实现自定义工具（文件读取 / 计算器 / API 查询 类）
  3. 在本地测试 MCP Server 能否被调用（配套 test_client.py）

Server 名为 hello-agent-tools，暴露 3 个 Tools + 1 个 Resource：
  calculate(expression)         四则运算（正则白名单，不裸 eval）
  lookup_travel_error_code(c)   HelloAI 差旅制度错误码查询（Day3 语料的 TRV-xxxxx）
  search_knowledge(question)    W3 语料库向量检索 top2（懒加载 fastembed，
                                未安装时返回可读错误而不是崩掉 Server）
  resource policy://{city}      只读资源：差旅住宿标准（演示 Resources 能力）

运行方式（stdio 传输）：
  echo-request 响应请求由 Host 拉起本进程；手动调试用：
    python mcp_server.py                     # 前台跑（日志走 stderr）
    python test_client.py                    # 用测试客户端连它（推荐验证方式）
"""
import os
import re
import sys

# mcp SDK 2.x：FastMCP 已更名为 MCPServer（API 同形）；1.x 的老教程用
# from mcp.server.fastmcp import FastMCP，效果一致。
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("hello-agent-tools")

# ---- 模拟数据：与 W3 语料 15_helloai_policy.md 保持一致 --------------------

ERROR_CODES = {
    "TRV-40231": "发票抬头与公司主体不一致。修改发票抬头后重新提交。",
    "TRV-40907": "报销单提交超过 15 日期限。需总监在 OA 里发起'逾期特批'流程。",
    "TRV-41188": "住宿金额超出城市标准且未附豁免材料。补传展会凭证或改按标准填写。",
    "TRV-50002": "缺少行程单。高铁票/登机牌不能替代行程单，需在 12306 或航空公司官网补开。",
}

HOUSING_STANDARD = {
    "北京": {"普通员工": 600, "总监及以上": 900},
    "上海": {"普通员工": 600, "总监及以上": 900},
    "深圳": {"普通员工": 600, "总监及以上": 900},
    "广州": {"普通员工": 500, "总监及以上": 750},
    "杭州": {"普通员工": 500, "总监及以上": 750},
    "成都": {"普通员工": 500, "总监及以上": 750},
}


# ---- Tools：模型可调用的函数（可能有副作用）--------------------------------

@mcp.tool()
def calculate(expression: str) -> str:
    """计算四则运算表达式，支持数字与 + - * / ( )"""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "错误：只支持数字与 + - * / ( ) 的四则运算"
    try:
        return str(eval(expression))  # noqa: S307  已白名单校验
    except Exception as e:
        return f"计算错误：{e}"


@mcp.tool()
def lookup_travel_error_code(code: str) -> str:
    """查询 HelloAI 差旅报销系统的错误码含义，如 TRV-40907"""
    code = code.strip().upper()
    if code in ERROR_CODES:
        return f"{code}：{ERROR_CODES[code]}"
    return f"未知错误码 {code}。已知错误码：{', '.join(sorted(ERROR_CODES))}"


@mcp.tool()
def search_knowledge(question: str) -> str:
    """在学习知识库（W3 语料：Transformer/RAG/Prompt/MCP 等主题）中检索相关段落"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "../w3data"))
        import ragkit
    except Exception as e:  # fastembed 未安装等环境问题：返回可读错误，别崩 Server
        return f"[检索不可用] 缺少依赖：{e}"
    emb = ragkit.get_embedder()
    if emb.kind != "semantic":
        return "[检索不可用] 语义向量模型未就绪（fastembed 未安装或未下载）"
    chunks = []
    for d in ragkit.load_corpus():
        for i, (text, _) in enumerate(ragkit.recursive_chunk(d["text"], 300)):
            chunks.append({"id": f"{d['id']}#c{i}", "text": text})
    qvec = emb.query_embed([question])[0]
    scored = sorted(((ragkit.cosine(qvec, v), c) for c, v in
                     zip(chunks, emb.passage_embed([c["text"] for c in chunks]))),
                    key=lambda x: -x[0])
    return "\n\n".join(f"【{c['id']}】{c['text']}" for _, c in scored[:2])


# ---- Resource：只读数据（类 GET 语义，客户端直接读取，不经模型决策）--------

@mcp.resource("policy://housing/{city}")
def get_housing_standard(city: str) -> str:
    """HelloAI 差旅住宿标准（元/晚）"""
    if city not in HOUSING_STANDARD:
        return f"暂无 {city} 的住宿标准。支持城市：{', '.join(HOUSING_STANDARD)}"
    s = HOUSING_STANDARD[city]
    return (f"{city} 住宿标准：普通员工 {s['普通员工']} 元/晚，"
            f"总监及以上 {s['总监及以上']} 元/晚。")


if __name__ == "__main__":
    # stdio 传输：协议消息走 stdin/stdout（JSON-RPC 2.0），日志只能走 stderr
    mcp.run(transport="stdio")
