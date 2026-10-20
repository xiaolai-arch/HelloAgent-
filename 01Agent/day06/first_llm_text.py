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

message = resp["choices"][0]["message"]["content"]
print(message)