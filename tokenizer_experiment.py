# -*- coding: utf-8 -*-
"""Token 计数实验：对比不同编码器、不同语言、不同模型上下文窗口"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import tiktoken

def count(text, enc_name):
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode(text)
    return len(ids), ids

def show_tokens(text, enc_name="cl100k_base", label=""):
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode(text)
    # tiktoken 0.14: decode_single_token_bytes
    pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
    print(f"  [{label}] tokens={len(ids)}  pieces={pieces}")
    return len(ids)

print("=" * 60)
print("实验 1：同一段英文，不同 tokenizer 编码")
print("=" * 60)
en = "Hello, world! Transformer models split text into tokens."
for name in ["gpt2", "cl100k_base", "o200k_base"]:
    n, _ = count(en, name)
    print(f"  {name:12s}: {n} tokens")
show_tokens(en, "cl100k_base", "cl100k 切分")

print()
print("=" * 60)
print("实验 2：中英文 Token 效率对比（cl100k_base）")
print("=" * 60)
cases = [
    ("英文短句", "The quick brown fox jumps over the lazy dog."),
    ("中文短句", "敏捷的棕色狐狸跳过了懒狗。",
     ),
    ("中文常见句", "今天天气很好，我们一起去公园散步吧。"),
    ("英文对应", "The weather is very nice today, let's go for a walk in the park together."),
    ("数字", "1234567890"),
    ("代码", "def softmax(x):\n    return np.exp(x) / np.exp(x).sum()"),
    ("emoji", "😀🎉🔥"),
]
for item in cases:
    label, text = item[0], item[1]
    n, ids = count(text, "cl100k_base")
    chars = len(text)
    print(f"  {label:8s}: {chars:3d} 字符 -> {n:3d} tokens  (1 token ≈ {chars/n:.2f} 字符)  {text[:30]}")

print()
print("=" * 60)
print("实验 3：中文为什么费 token —— 逐 token 查看")
print("=" * 60)
show_tokens("今天天气很好", label="今天天气很好")
show_tokens("Hello world", label="Hello world")

print()
print("=" * 60)
print("实验 4：o200k_base (GPT-4o) vs cl100k_base (GPT-4/3.5) 中文对比")
print("=" * 60)
zh = "人工智能大语言模型的分词器对中文的支持越来越好了。"
n1, _ = count(zh, "cl100k_base")
n2, _ = count(zh, "o200k_base")
print(f"  文本: {zh}")
print(f"  cl100k_base (GPT-4/3.5): {n1} tokens")
print(f"  o200k_base  (GPT-4o)   : {n2} tokens  (新 tokenizer 中文更省)")

print()
print("=" * 60)
print("实验 5：常用模型 Context Window 对比")
print("=" * 60)
models = [
    ("GPT-3.5-turbo",        "16K",    "2022.11"),
    ("GPT-4 (8K/32K)",       "8K/32K", "2023.03"),
    ("GPT-4o",               "128K",   "2024.05"),
    ("GPT-5",                "400K",   "2025.08"),
    ("Claude 3 Haiku/Sonnet/Opus", "200K", "2024.03"),
    ("Claude 4.x 系列",       "200K (1M beta)", "2025"),
    ("Gemini 1.5 Pro",       "1M-2M",  "2024"),
    ("Gemini 2.x",           "1M",     "2025"),
    ("DeepSeek-V3",          "64K-128K", "2024.12"),
    ("Qwen2.5",              "32K-128K", "2024.09"),
    ("Llama 3.1 405B",       "128K",   "2024.07"),
]
print(f"  {'模型':<28} {'Context':<16} 发布")
for m, c, d in models:
    print(f"  {m:<28} {c:<16} {d}")

print()
print("=" * 60)
print("实验 6：token 与费用的直觉（以 1K tokens 为基准）")
print("=" * 60)
para = "Large language models do not read characters; they read tokens. " * 5
n, _ = count(para, "cl100k_base")
print(f"  重复句 x5: {len(para)} 字符 -> {n} tokens")
print(f"  经验法则: 英文 ~4 字符/token; 中文 ~1-1.5 字符/token")
