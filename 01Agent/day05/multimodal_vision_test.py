# -*- coding: utf-8 -*-
"""多模态 LLM API 调用实验：识图 + 问答

前置知识（笔记见同目录 多模态视觉_实验指南.md）：
- "视觉编码器 + LLM" 经典架构：图片 → ViT 切 patch → 视觉编码器输出特征
  → 投影层对齐到 LLM 的 token 空间 → 与文本 token 拼接 → 一起送进 LLM。
- API 层面完全印证这个架构：messages 里把图片和文字放在同一条 user
  消息里，格式是 OpenAI 兼容的 image_url content part。

网关与 Key 的配置方式（二选一，配置文件优先）：
1. 本地配置文件 vision_config.json（已被 .gitignore 忽略，不会提交）：
   {"base_url": "https://your-gateway.example.com/v1", "api_key": "sk-xxx"}
2. 环境变量 COPILOT_PROVIDER_BASE_URL + COPILOT_PROVIDER_API_KEY
"""
import sys
import io
import os
import base64
import json
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json")

_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
MODEL = _cfg.get("model", "gpt-4o")

if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，内容格式：\n"
             '  {"base_url": "https://your-gateway.example.com/v1",\n'
             '   "api_key": "sk-xxx",\n'
             '   "model": "gpt-4o"}   # model 可省略，默认 gpt-4o\n'
             "  或设置环境变量 COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")
_src = "vision_config.json" if (_cfg.get("base_url") or _cfg.get("api_key")) else "环境变量"
print(f"[config] 网关: {BASE_URL}  模型: {MODEL}  (来源: {_src})")

# ---------------------------------------------------------------
# 实验用图片：不用外部素材，直接用代码生成一张 PNG（纯 Python + zlib，
# 无需 pillow），内容是"橙色背景 + 蓝色三角形 + 手写文字像素"，
# 这样模型答对了什么、答错了什么，我们心里有底。
# ---------------------------------------------------------------

def make_test_png(path="vision_test.png", w=480, h=320):
    """生成一张测试图：橙色天空、蓝色三角形（山）、白色像素文字 VLM"""
    # 简单 24 位 RGB 像素画布
    px = [[(255, 165, 0)] * w for _ in range(h)]  # 橙色背景

    # 蓝色三角形（居中偏下，"山"）
    apex = (w // 2, 60)
    base_y = h - 40
    half = 160
    for y in range(apex[1], base_y):
        t = (y - apex[1]) / (base_y - apex[1])
        span = int(half * t)
        cx = apex[0]
        for x in range(cx - span, cx + span + 1):
            if 0 <= x < w:
                px[y][x] = (30, 90, 200)  # 蓝色

    # 白色像素文字 "VLM"（逐像素画一个 5x7 点阵字体的放大版）
    font = {
        'V': ["1 0 0 0 1", "1 0 0 0 1", "1 0 0 0 1", "0 1 0 1 0", "0 0 1 0 0"],
        'L': ["1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 1 1 1 1"],
        'M': ["1 0 0 0 1", "1 1 0 1 1", "1 0 1 0 1", "1 0 0 0 1", "1 0 0 0 1"],
    }
    word = "VLM"
    scale = 8
    x0 = w // 2 - (3 * 5 * scale + 2 * scale) // 2
    y0 = 20
    for i, ch in enumerate(word):
        glyph = font[ch]
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == '1':
                    for dy in range(scale):
                        for dx in range(scale):
                            yy = y0 + r * scale + dy
                            xx = x0 + i * (5 * scale + scale) + c * scale + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                px[yy][xx] = (255, 255, 255)

    # 编码为 PNG（无压缩 filter type 0，纯 zlib）
    raw = b""
    for y in range(h):
        raw += b"\x00"  # filter type 0: None
        for x in range(w):
            raw += bytes(px[y][x])

    import zlib

    def chunk(typ, data):
        return len(data).to_bytes(4, "big") + typ + data + crc32(typ + data)

    def crc32(b):
        return (zlib.crc32(b) & 0xFFFFFFFF).to_bytes(4, "big")

    ihdr = (w).to_bytes(4, "big") + (h).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    print(f"[setup] 已生成测试图片 {path}  ({w}x{h}, {len(png)} bytes)")
    return path


def chat(messages, model=None, max_tokens=1024):
    model = model or MODEL
    """OpenAI 兼容 chat/completions 调用"""
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + API_KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def img_part(path):
    """把本地图片转成 base64 data URI 的 image_url content part"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


print("=" * 62)
print("实验：多模态 LLM 识图 + 问答（视觉编码器 + LLM 架构的 API 印证）")
print("=" * 62)

img = make_test_png()

# 一次请求，两条 content：一条文本，一条图片 —— 这正是
# "视觉 token 与文本 token 在同一条序列里" 在 API 上的样子。
messages = [{
    "role": "user",
    "content": [
        {"type": "text",
         "text": "请仔细观察这张图片，回答：\n"
                 "1. 图片主体是什么几何形状？什么颜色？\n"
                 "2. 背景是什么颜色？\n"
                 "3. 图里写了什么英文字母/单词？\n"
                 "4. 用一句话描述整张图。"},
        img_part(img),
    ],
}]

print("[call] 正在调用多模态模型 ...")
resp = chat(messages)
msg = resp["choices"][0]["message"]
print("-" * 62)
print(msg["content"])
print("-" * 62)
u = resp["usage"]
print(f"[usage] prompt={u['prompt_tokens']}  completion={u['completion_tokens']}  total={u['total_tokens']}")
print("[note] prompt_tokens 里包含图片折算成的视觉 token —— 这就是")
print("       '图片被编码器变成 token 序列' 最直接的证据。")
