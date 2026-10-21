"""
彻底跑通LLM API 调用
"""

import os
import sys
import io
import json
import urllib.request
import urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 加载配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json")

cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
url = cfg.get("base_url")
url = url.rstrip("/")
key = cfg.get("api_key")
mode = cfg.get("model", "mimo-v2.6-flash")

def chat(msg, model=None, temperature = 0.7, max_tokens = 512):
    model = model or mode # 未指定模型时，回退到配置文件里的模型
    body = json.dumps({
        "model": model,
        "messages": msg,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        url + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key,
        "Content-Type": "application/json"})
    try:
        # timeout=120:思考模型可能想要十几秒的响应时间
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # 常见错误码：401=key 错误/过期，429=限流，400=参数不被该网关支持，
        # 404=端点路径错（常见于 base_url 多写/少写 /v1）
        detail = e.read().decode("utf-8")
        sys.exit(f"[error] HTTP {e.code}\n{detail}")

resp = chat([{"role": "user", "content": "用一句话解释什么是自注意力机制。"}])

# json格式的数据
"""
id：本次访问的唯一id，排查故障可以去查网关日志
finish_reason：模型停止生成的原因（stop=自己说完了，length=撞上 max_tokens）
    其他原因见实验二
prompt_tokens（发出去的，含system/历史）+completion_token（模型生成的）= total_tokens
reasoning_token （内部思维链消耗）
"""
print(resp) # 返回的json格式数据
# choices是数组，这里面有输出的文本
message = resp["choices"][0]["message"]["content"]
print(message)

message = [
    # system 放最前面：它的内容会作为前缀进入每次请求，影响后续每次生成
    {"role": "system", "content": "你是一个只用一句中文回答问题的助手。"},
    {"role": "user", "content": "讲一下有关耄耋的故事"}
]

resp1 = chat(message)
# 从响应里取纯文本。注意只取content
reply1 = resp1["choices"][0]["message"]["content"]
print(reply1)

# "记忆"的全部实现：把模型的回答作为 assistant 消息 append 回数组。
# 少了这一步，下一次请求模型就完全不知道上一轮发生过什么。
message.append({"role": "assistant", "content": reply1})
message.append({"role": "assistant", "content": "我说的是在抖音上非常火的猫"})
resp2 = chat(message)
reply2 = resp2["choices"][0]["message"]["content"]
print(reply2)

# 控制变量：问题完全相同、temperature 相同，只改 system 内容。
# 同一模型对同一问题给出截然不同的风格 —— 差异全部来自 system 前缀。
question = "什么是幻觉（hallucination）？"
for persona in ["你是一个严谨的科学家，只陈述事实。",
               "你是一个夸张的科幻小说家，用宇宙史诗的口吻回答"]:
    r = chat([{"role": "assistant", "content": persona},
              {"role": "assistant", "content": question}], temperature = 0.7)
    print(f"\n[system] {persona}")
    # [:200] 只是本脚本打印时的显示截断，方便对照阅读；
    # API 侧的截断由 max_tokens 控制（本实验默认 512）。
    print(f"[answer] {r['choices'][0]['message']['content'][:200]}")