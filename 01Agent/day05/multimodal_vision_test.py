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

# ============ 标准库导入 ============
import sys          # 用于：重定向 stdout 编码、缺配置时退出程序
import io           # 用于：构造带指定编码的 TextIOWrapper 包装 stdout
import os           # 用于：拼配置文件路径、读取环境变量
import base64       # 用于：把图片二进制编码成 base64 文本（塞进 data URI）
import json         # 用于：读配置文件、序列化请求体、解析响应 JSON
import urllib.request  # 用于：发起 HTTP POST 请求调用 LLM API（无需第三方库）

# Windows 终端默认 GBK 编码，打印 UTF-8 内容（中文/模型回复）会乱码或崩溃，
# 这里把 stdout 强制换成 UTF-8，遇到无法编码的字符用 replace 兜底不中断
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 读取配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../vision_config.json")

_cfg = {}                                          # 先给空字典，配置文件不存在时也不报错
if os.path.exists(CONFIG_PATH):                    # 只有文件真实存在才去读
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:  # 以 UTF-8 文本模式打开
        _cfg = json.load(f)                        # 把 JSON 文本解析成 Python 字典

# 三级取值：配置文件 > 环境变量 > 空字符串；rstrip("/") 去掉末尾斜杠，避免拼出 //chat/completions
BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")  # 同样：文件优先，环境变量兜底
MODEL = _cfg.get("model", "mimo-v2.6-flash")                # 模型名

# 网关地址或密钥缺失时直接终止程序——宁可提前报错，也不要等 HTTP 请求 401 才发现
if not BASE_URL or not API_KEY:
    sys.exit(f"[config] 缺少网关配置。请创建 {CONFIG_PATH}，内容格式：\n"
             '  {"base_url": "https://your-gateway.example.com/v1",\n'
             '   "api_key": "sk-xxx",\n'
             '   "model": "gpt-4o"}   # model 可省略，默认 gpt-4o\n'
             "  或设置环境变量 COPILOT_PROVIDER_BASE_URL / COPILOT_PROVIDER_API_KEY")

# 标记配置来源，方便排查"为什么用的不是我以为的配置"
_src = "vision_config.json" if (_cfg.get("base_url") or _cfg.get("api_key")) else "环境变量"
print(f"[config] 网关: {BASE_URL}  模型: {MODEL}  (来源: {_src})")

# ---------------------------------------------------------------
# 实验用图片：不用外部素材，直接用代码生成一张 PNG（纯 Python + zlib，
# 无需 pillow），内容是"橙色背景 + 蓝色三角形 + 手写文字像素"，
# 这样模型答对了什么、答错了什么，我们心里有底。
# ---------------------------------------------------------------

def make_test_png(path="vision_test.png", w=480, h=320):
    """生成一张测试图：橙色天空、蓝色三角形（山）、白色像素文字 VLM"""
    # 简单 24 位 RGB 像素画布：h 行 × w 列，每格是 (R, G, B) 三元组
    px = [[(255, 165, 0)] * w for _ in range(h)]  # 橙色背景 (255,165,0) 铺满整张画布

    # —— 画蓝色三角形（"山"）：从顶点逐行向下扫描，行号越低扫描范围越宽 ——
    apex = (w // 2, 60)      # 顶点坐标：水平居中，y=60 处
    base_y = h - 40          # 底边所在行：距底部留 40 像素
    half = 160               # 底边半宽：三角形最宽处向两侧各延伸 160 像素
    for y in range(apex[1], base_y):              # 从顶点行扫到底边行
        t = (y - apex[1]) / (base_y - apex[1])    # 当前行的归一化高度 0→1（相似三角形比例）
        span = int(half * t)                      # 当前行的半宽 = 底边半宽 × 比例
        cx = apex[0]                              # 中心列（= 顶点列）
        for x in range(cx - span, cx + span + 1):  # 把这一行 [中心-span, 中心+span] 填成蓝色
            if 0 <= x < w:                        # 边界保护：不出画布
                px[y][x] = (30, 90, 200)          # 蓝色 (30,90,200)

    # —— 画白色像素文字 "VLM"：5×7 点阵字体，'1' 表示该格涂白 ——
    font = {                                      # 手写的点阵字模，每行 5 位、共 5 行
        'V': ["1 0 0 0 1", "1 0 0 0 1", "1 0 0 0 1", "0 1 0 1 0", "0 0 1 0 0"],
        'L': ["1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 1 1 1 1"],
        'M': ["1 0 0 0 1", "1 1 0 1 1", "1 0 1 0 1", "1 0 0 0 1", "1 0 0 0 1"],
    }
    word = "VLM"                                  # 要画的单词（可改成别的字母做对照实验）
    scale = 8                                     # 每个点阵"像素"放大成 8×8 的真实像素块
    x0 = w // 2 - (3 * 5 * scale + 2 * scale) // 2  # 整串文字的起始列：水平居中（3 个字母 + 2 个间隙）
    y0 = 20                                       # 起始行：靠近图片顶部
    for i, ch in enumerate(word):                 # 遍历每个字母
        glyph = font[ch]                          # 取出该字母的 5×7 点阵
        for r, row in enumerate(glyph):           # 遍历字模的每一行
            for c, bit in enumerate(row):         # 遍历该行的每一位
                if bit == '1':                    # 只画点亮的点
                    for dy in range(scale):       # 把 1 个点放大成 scale×scale 的实心方块
                        for dx in range(scale):
                            yy = y0 + r * scale + dy  # 换算到画布坐标：字母起始行 + 字模行×缩放 + 偏移
                            xx = x0 + i * (5 * scale + scale) + c * scale + dx  # 列方向再加字母序号×字宽（含 1 个点的字距）
                            if 0 <= yy < h and 0 <= xx < w:  # 边界保护
                                px[yy][xx] = (255, 255, 255)  # 白色

    # —— 把像素画布编码成合法 PNG 文件（手写格式，不依赖 pillow）——
    raw = b""                                     # 原始扫描线数据（未压缩）
    for y in range(h):                            # PNG 逐行存储
        raw += b"\x00"                            # 每行开头 1 字节 filter type 0（None，不做过滤）
        for x in range(w):                        # 该行每个像素依次写入 R、G、B 各 1 字节
            raw += bytes(px[y][x])

    import zlib                                   # PNG 的 IDAT 数据用 zlib/deflate 压缩

    def chunk(typ, data):
        # PNG 的一个数据块 = 4 字节长度 + 4 字节类型 + 数据 + 4 字节 CRC
        return len(data).to_bytes(4, "big") + typ + data + crc32(typ + data)

    def crc32(b):
        # 对"类型+数据"算 CRC32 校验（PNG 规范要求），返回 4 字节大端
        return (zlib.crc32(b) & 0xFFFFFFFF).to_bytes(4, "big")

    # IHDR 块：图片宽、高、位深 8、颜色类型 2（真彩色 RGB）、压缩/过滤/交互均为 0
    ihdr = (w).to_bytes(4, "big") + (h).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    # 完整 PNG = 签名魔数 + IHDR + IDAT(压缩像素) + IEND(结束标记)
    png = (b"\x89PNG\r\n\x1a\n"                   # 8 字节文件签名，识别"这是 PNG"
           + chunk(b"IHDR", ihdr)                 # 头块：尺寸与格式
           + chunk(b"IDAT", zlib.compress(raw))   # 数据块：压缩后的像素
           + chunk(b"IEND", b""))                 # 结束块：空数据
    with open(path, "wb") as f:                   # 二进制模式写出
        f.write(png)
    print(f"[setup] 已生成测试图片 {path}  ({w}x{h}, {len(png)} bytes)")
    return path                                   # 返回文件路径，供后续读取编码


def chat(messages, model=None, max_tokens=1024):
    """OpenAI 兼容 chat/completions 调用"""
    model = model or MODEL                        # 没显式传模型名就用全局配置的默认模型
    # 请求体：三要素 = 用哪个模型 + 对话历史 + 输出 token 上限（只限回答，不限输入）
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens}).encode("utf-8")  # dict→JSON 文本→UTF-8 字节
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",           # 端点：所有 OpenAI 兼容网关的通用补全路径
        data=body,                                # 传了 data，urllib 自动使用 POST
        headers={"Authorization": "Bearer " + API_KEY,  # 鉴权头：Bearer + 密钥（API 标准做法）
                 "Content-Type": "application/json"})  # 告诉服务端请求体是 JSON
    with urllib.request.urlopen(req, timeout=120) as r:  # 发起请求，最多等 120 秒防挂死
        return json.load(r)                       # 响应字节→Python 字典，含 choices / usage 等字段


def img_part(path):
    """把本地图片转成 base64 data URI 的 image_url content part"""
    with open(path, "rb") as f:                   # 读取 PNG 的原始二进制
        b64 = base64.b64encode(f.read()).decode("ascii")  # 二进制→base64 文本（base64 是纯 ASCII）
    # 组装成 OpenAI 多模态消息里的"图片部件"：data URI 内嵌图片，无需公网 URL
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


# ============ 主流程 ============
print("=" * 62)
print("实验：多模态 LLM 识图 + 问答（视觉编码器 + LLM 架构的 API 印证）")
print("=" * 62)

img = make_test_png()                             # 第一步：生成已知内容的测试图（ground truth 在我们手里）

# 一次请求，两条 content：一条文本，一条图片 —— 这正是
# "视觉 token 与文本 token 在同一条序列里" 在 API 上的样子。
messages = [{
    "role": "user",                               # 角色：用户消息
    "content": [                                  # 多模态消息：content 是数组而不是字符串
        {"type": "text",                          # 部件 1：文本——提出四个观察问题
         "text": "请仔细观察这张图片，回答：\n"
                 "1. 图片主体是什么几何形状？什么颜色？\n"
                 "2. 背景是什么颜色？\n"
                 "3. 图里写了什么英文字母/单词？\n"
                 "4. 用一句话描述整张图。"},
        img_part(img),                            # 部件 2：图片——base64 编码的测试图
    ],
}]

print("[call] 正在调用多模态模型 ...")
resp = chat(messages)                             # 第二步：发出请求，拿到完整响应字典
msg = resp["choices"][0]["message"]               # 取第一条（也是唯一一条）候选回复的 message
print("-" * 62)
print(msg["content"])                             # 打印模型的自然语言回答
print("-" * 62)
u = resp["usage"]                                 # usage 字段：本次请求的 token 计费统计
print(f"[usage] prompt={u['prompt_tokens']}  completion={u['completion_tokens']}  total={u['total_tokens']}")
# prompt_tokens = 输入侧（问题文本 + 图片折算的视觉 token）之和
# completion_tokens = 模型输出的回答；total = 两者相加
print("[note] prompt_tokens 里包含图片折算成的视觉 token —— 这就是")
print("       '图片被编码器变成 token 序列' 最直接的证据。")
