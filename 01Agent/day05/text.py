import sys          # 用于：重定向 stdout 编码、缺配置时退出程序
import io           # 用于：构造带指定编码的 TextIOWrapper 包装 stdout
import os           # 用于：拼配置文件路径、读取环境变量
import base64       # 用于：把图片二进制编码成 base64 文本（塞进 data URI）
import json         # 用于：读配置文件、序列化请求体、解析响应 JSON
import urllib.request  # 用于：发起 HTTP POST 请求调用 LLM API（无需第三方库）

# 三个步骤：1. 读取配置文件 2. 拼接 URL / headers / body 发起请求 3. 输出对应内容

# 1. 读取配置文件
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json") ## 配置文件路径

cfg = {} # 初始化变量字典
if os.path.exists(CONFIG_PATH): # 如果配置文件存在
    with open(CONFIG_PATH, "r", encoding="utf-8") as f: # 打开配置文件
        cfg = json.load(f) # 读取配置文件内容
URL = cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "") # 获取网关 URL 环境变量兜底
URL = URL.rstrip("/") # 去掉 URL 末尾的斜杠
API_KEY = cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "") # 获取 API Key
MODEL = cfg.get("model", "mimo-v2.6-flash") # 获取模型名称，默认值为 "mimo-v2.6-flash"

# 2. 拼接 URL / headers / body 发起请求
def chat(messages, model=None, max_tokens=512):
    model = model or MODEL
    # 请求体：三要素 = 用哪个模型 + 对话历史 + 输出 token 上限（只限回答，不限输入）
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }).encode("utf-8")
    req = urllib.request.Request(
        URL + "/chat/completions",                      # 端点：所有 OpenAI 兼容网关的通用补全路径
        data=body,                                      # 传了 data，urllib 自动使用 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 鉴权头：Bearer + 密钥（API 标准做法）
                 "Content-Type": "application/json"})   # 告诉服务端请求体是 JSON
    with urllib.request.urlopen(req, timeout=120) as r:  # 发起请求，最多等 120 秒防挂死
        return json.load(r)                       # 响应字节→Python 字典，含 choices / usage 等字段

def img_part(path):
    """把本地图片转成 base64 data URI 的 image_url content part"""
    with open(path, "rb") as f:                   # 读取 PNG 的原始二进制
        b64 = base64.b64encode(f.read()).decode("ascii")  # 二进制→base64 文本（base64 是纯 ASCII）
    # 组装成 OpenAI 多模态消息里的"图片部件"：data URI 内嵌图片，无需公网 URL
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}

img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_test.png") # 测试图片路径：锚定到脚本所在目录，不依赖运行时工作目录

# 3. 输出对应内容
messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "你好，你是谁，调用的什么大模型，你有什么作用，可以做什么？\n"
                "1. 图片主体是什么几何形状？什么颜色？\n"
                "2. 背景是什么颜色？\n"
                "3. 图里写了什么英文字母/单词？\n"
                "4. 用一句话描述整张图。"},
        img_part(img),
    ],
}] # 构造消息列

resp = chat(messages) 
msg = resp["choices"][0]["message"]
print(msg["content"])