# -*- coding: utf-8 -*-
"""第一次 LLM API 调用：从零跑通 chat/completions

这一天把前几天的"纸面知识"落到真正的 API 上：
- 你敲的每一个 token（day03）最终都通过 messages 数组发给模型
- 模型每生成一个 token，都是在词表概率分布上采样一次（day03 温度/Top-p）
- 多模态（day05）只是 content 里多了一种 image_url part，接口形状完全一样

三个实验：
  实验 1  一问一答，认识请求/响应的完整结构
  实验 2  多轮对话，理解"上下文"的真实实现方式
  实验 3  system prompt，感受人设对输出的控制力

网关与 Key 的配置方式（二选一，配置文件优先）：
1. 本地配置文件 ../../vision_config.json（已被 .gitignore 忽略，不会提交）：
   {"base_url": "https://your-gateway.example.com/v1", "api_key": "sk-xxx", "model": "xxx"}
2. 环境变量 COPILOT_PROVIDER_BASE_URL + COPILOT_PROVIDER_API_KEY
"""
import sys
import io
import os
import json
import urllib.request
import urllib.error

# Windows 终端默认编码是 GBK，打印中文/emoji 会抛 UnicodeEncodeError。
# 统一把 stdout 换成 UTF-8 且不炸的包装（day03/day05 脚本同样处理）。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---- 配置加载 ---------------------------------------------------------------
# 策略：配置文件优先，环境变量兜底（与 day05 保持一致，两个脚本可互换使用）。
# 安全原则：API Key 绝不硬编码进 .py —— 脚本要提交 git，而密钥只能待在
# 被 .gitignore 忽略的 vision_config.json 里（这也是 day05 那条 gitignore 的意义）。
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json")

_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

# .rstrip("/") 防止配置里写成 ".../v1/" 导致拼出 ".../v1//chat/completions"
BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
MODEL = _cfg.get("model", "gpt-4o")  # 配置文件没写 model 时的兜底默认值

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，内容格式：\n"
             '  {"base_url": "https://your-gateway.example.com/v1",\n'
             '   "api_key": "sk-xxx",\n'
             '   "model": "xxx"}\n'
             "  或设置环境变量 COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")
_src = "vision_config.json" if (_cfg.get("base_url") or _cfg.get("api_key")) else "环境变量"
print(f"[config] 网关: {BASE_URL}  模型: {MODEL}  (来源: {_src})")


def chat(messages, model=None, temperature=0.7, max_tokens=512):
    """OpenAI 兼容 POST {base_url}/chat/completions

    请求体只有四个核心字段：
      model       —— 用哪个模型
      messages    —— 对话历史（role + content 的数组，这就是"上下文"的全部实体）
      temperature —— 采样温度（day03：softmax(logits / T)，改分布形状）
      max_tokens  —— 生成侧 token 上限（思考型模型 = 思考+回答共享的总预算）

    返回值：解析后的响应 JSON（dict），结构见实验 1 的逐字段打印。
    """
    # json.dumps(dict) → JSON 字符串；.encode("utf-8") → 字节流。
    # HTTP 只能传字节，而 body 里有中文，必须显式 UTF-8（默认 ASCII 会报错）。
    body = json.dumps({
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",   # OpenAI 兼容网关的标准端点
        data=body,                         # 带 data 即自动变为 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 鉴权：key 放 HTTP 头，不进 URL
                 "Content-Type": "application/json"})   # 告诉网关 body 是 JSON
    try:
        # timeout=120：思考型模型可能要想十几秒才吐第一个可见 token，留给它余量
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)  # 响应 body 也是 JSON 字节流，load 直接变 dict
    except urllib.error.HTTPError as e:
        # 常见错误码：401=key 错误/过期，429=限流，400=参数不被该网关支持，
        # 404=端点路径错（常见于 base_url 多写/少写 /v1）
        detail = e.read().decode("utf-8", errors="replace")[:500]
        sys.exit(f"[error] HTTP {e.code}\n{detail}")


print("=" * 62)
print("实验 1：最小可用调用 —— 一问一答")
print("=" * 62)
# messages 是一个数组，每个元素 {"role": ..., "content": ...}，三种角色：
#   system    —— 设定行为/人设，整段对话的"宪法"，应放在数组最前面
#   user      —— 你说的话
#   assistant —— 模型说的话（多轮时要原样放回，模型才"记得"，见实验 2）
# 这是最小请求：只有 1 条 user 消息，没有 system，一轮即结束。
resp = chat([{"role": "user", "content": "用一句话解释什么是自注意力机制。"}])

# 响应 JSON 的主结构：
#   {id, model, choices: [{message: {role, content, ...}, finish_reason}],
#    usage: {prompt_tokens, completion_tokens, total_tokens}}
# choices 是数组（协议支持 n>1 一次要多条候选），默认 n=1，取 [0] 即可。
msg = resp["choices"][0]["message"]
print(f"[request ] role=user  content=用一句话解释什么是自注意力机制。")
print(f"[response] {msg['content']}")

# 响应里的三个值得认识的字段
# finish_reason：模型停止生成的原因。stop=自己说完了；length=撞上 max_tokens
# 被硬截断（详见 parameter_experiments.py 实验 2）；还有 tool_calls 等取值。
print(f"\n[fields  ] finish_reason={resp['choices'][0]['finish_reason']!r}"
      f"   # stop=正常说完；length=撞上 max_tokens 被截断")
# id：本次推理的唯一标识，排障时拿着它找网关/厂商查日志用
print(f"[fields  ] id={resp.get('id')}  model={resp.get('model')}")
u = resp["usage"]
# 计费与预算的核心：prompt_tokens（你发出去的，含 system/历史）+
# completion_tokens（模型生成的）= total_tokens。day03 数过的 token 在这里变现。
# 思考型模型的 completion 里还细分出 reasoning_tokens（内部思维链消耗）。
think = u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
print(f"[usage   ] prompt={u['prompt_tokens']}  completion={u['completion_tokens']}"
      f"  total={u['total_tokens']}  (其中思考 {think} tokens)")
print("[note    ] 计费按 token：prompt（你发的）+ completion（模型生成的）都算钱。")
# 本模型（mimo-v2.6-flash）是思考型模型：message 里除了 content 还有
# reasoning_content 字段（思维链），回答前先内部推理 —— 类似 o 系列 / R1。
print(f"[fields  ] message 里还有 reasoning_content 字段 —— 本模型是思考型模型，")
print("           回答前先产出思维链（类似 o 系列 / DeepSeek-R1 的 thinking）。")

print()
print("=" * 62)
print("实验 2：多轮对话 —— 「上下文」就是往数组里追加元素")
print("=" * 62)
# 关键认知：HTTP 接口本身是"无状态"的。所谓"模型记得上一句"，
# 其实是你每次把完整历史（含模型之前的回答）重发了一遍。
# 推论：轮次越多 → messages 越长 → prompt_tokens 越多 → 费用和延迟同步上涨，
# 这也是长对话需要截断/压缩（下下阶段的 RAG、记忆机制）的根本原因。
messages = [
    # system 放最前：它的内容会作为前缀进入每次请求，影响后续所有生成
    {"role": "system", "content": "你是一个只用一句中文回答问题的助手。"},
    {"role": "user", "content": "我正在学 LLM，请推荐一个入门概念。"},
]
resp1 = chat(messages)
# 从响应里取出纯文本。注意只取 content，不取 reasoning_content ——
# 后者是内部草稿，放进历史没有意义还费 token。
reply1 = resp1["choices"][0]["message"]["content"]
print(f"[turn 1] user: 我正在学 LLM，请推荐一个入门概念。")
print(f"[turn 1] assistant: {reply1}")

# "记忆"的全部实现：把模型的回答作为 assistant 消息 append 回数组。
# 少了这一步，下一次请求模型就完全不知道上一轮发生过什么。
messages.append({"role": "assistant", "content": reply1})
messages.append({"role": "user", "content": "那它和 Softmax 有什么关系？"})
resp2 = chat(messages)
reply2 = resp2["choices"][0]["message"]["content"]
print(f"[turn 2] user: 那它和 Softmax 有什么关系？   # 注意「它」依赖上一轮的推荐")
print(f"[turn 2] assistant: {reply2}")
print(f"\n[messages 数组最终形态] 共 {len(messages)} 条："
      f"system x1, user x2, assistant x1 —— 第二次请求把前 3 条原样重发了。")

print()
print("=" * 62)
print("实验 3：system prompt 的作用 —— 同样的问题，不同人设")
print("=" * 62)
# 控制变量：问题完全相同、temperature 相同，只改 system 内容。
# 同一模型对同一问题给出截然不同的风格 —— 差异全部来自 system 前缀。
question = "什么是幻觉（hallucination）？"
for persona in ["你是一个严谨的教科书，只陈述事实。",
                "你是一个夸张的科幻小说家，用宇宙史诗的口吻回答。"]:
    r = chat([{"role": "system", "content": persona},
              {"role": "user", "content": question}], temperature=0.7)
    print(f"\n[system] {persona}")
    # [:200] 只是本脚本打印时的显示截断，方便对照阅读；
    # API 侧的截断由 max_tokens 控制（本实验默认 512）。
    print(f"[answer] {r['choices'][0]['message']['content'][:200]}")

print()
print("=" * 62)
print("小结：第一次调用你真正需要记住的四件事")
print("=" * 62)
print("1. 端点是 POST {base_url}/chat/completions，鉴权头 Authorization: Bearer <key>")
print("2. 一切输入都在 messages 数组里；多轮=重发全部历史，接口本身无状态")
print("3. usage 里的 prompt/completion tokens 就是 day03 数过的那些，直接决定费用")
print("4. finish_reason 告诉你模型是说完了(stop)还是被 max_tokens 掐断(length)")
