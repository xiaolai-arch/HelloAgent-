# Tokenizer、采样参数与 Context Window — 图文实验指南

> 本文四部分：① 官方文档要点 → ② 温度 / Top-p / Top-K → ③ tiktoken 实测 → ④ Context Window 对比。
> 实测代码见同目录 `tokenizer_experiment.py`（已实际运行，所有数字为真实输出）。

---

## 1. Tokenizer 官方文档要点

### 1.1 OpenAI 官方说明（Glossary / Cookbook）

核心事实：

- **Token 是模型读文本的最小单位**。模型不读字符，也不读词，而是读 token。
- **经验值：100 tokens ≈ 75 个英文单词 ≈ 400 字符**，即 **1 英文词 ≈ 1.33 tokens，1 token ≈ 4 个英文字符**。
- token 可以是：一个完整单词（`Hello`）、单词的一部分（`ing`）、一个空格+词（` world`）、标点、emoji，甚至一个字节。
- **不同模型用不同的 tokenizer**（编码器/BPE 词表不同），同一段文本在不同模型下的 token 数**不一样**。
- OpenAI 提供官方统计工具：**tiktoken**（Python 库，开源）。

BPE（Byte-Pair Encoding）训练过程简述：

1. 从字符/字节级开始
2. 统计语料中最常见的相邻 token 对，合并成新 token
3. 重复数万步，得到固定词表（cl100k_base 约 10 万条）

### 1.2 Anthropic 官方说明（Token Counting 文档）

- Claude 使用**自家 tokenizer**，token 数与 OpenAI 不同，不能跨厂商直接换算。
- 官方估算方法（无需调 API）：**约 3.5 个字符 ≈ 1 个 token**，或 **15 个中文字 ≈ 20 tokens** 量级。
- 官方提供 **count_tokens API**（Messages API 的 `/v1/messages/count_tokens`）精确统计一段对话的 token 数。
- Claude 的 context window 为 **200K tokens**（约 50 万英文单词，"可以读完《哈利·波特》全集"量级）。

### 1.3 关键结论

| 项目 | 事实 |
|---|---|
| token ≠ 单词 ≠ 字符 | 英文约 1 词 1.3 token；中文可能 1 字 1~2 token |
| 中文更"贵" | 同一语义，中文消耗的 token 通常多于英文 |
| tokenizer 是模型私有的 | 换模型 = 换词表，token 数会变 |
| 计费按 token | 输入、输出都按 token 计费 |

---

## 2. 温度 / Top-p / Top-K 采样区别

生成每个 token 时，模型输出的是**词表上一个概率分布**（logits → softmax）。三个参数都是在**修改这个分布**后再掷骰子。

### 2.1 温度（Temperature, T）

```
P(token) = softmax( logits / T )
```

| T 值 | 效果 | 适用 |
|---|---|---|
| T → 0 | 分布趋近 one-hot，几乎总选最高概率词，**确定性输出** | 代码、抽取、分类、翻译 |
| T = 1 | 原始分布，不动 | 一般对话 |
| T > 1 | 分布变平，低概率词机会增大，**更有创意/更不稳定** | 写诗、头脑风暴 |
| T < 0（API 不允许） | — | — |

**一句话：温度只改"分布的尖锐程度"，不截断候选。**

举例（假设某位置 logits）：

```
词A: p=0.60   词B: p=0.25   词C: p=0.10   词D: p=0.05

T=0.1 → 几乎 100% 选 A
T=1.0 → 按 0.60/0.25/0.10/0.05 抽
T=2.0 → 分布被拉平，比如变成 0.38/0.28/0.20/0.14，C、D 机会大增
```

### 2.2 Top-K 采样

**规则：只保留概率最高的 K 个候选，其余直接置零，然后在剩余里按（可能经温度调整的）概率抽样。**

```
Top-K = 3 → 只从 词A、词B、词C 里选，D 被淘汰
```

- 简单粗暴，但有个缺陷：**K 是固定数**。当分布很尖时（第一名 99%），强行留 3 个候选没意义；当分布很平时（四个都是 25%），K=3 又可能误杀。
- GPT-2 时代的主流做法。

### 2.3 Top-p（Nucleus，核采样）

**规则：按概率从高到低累加，累加值首次 ≥ p 的那个集合为候选，其余淘汰。**

```
Top-p = 0.75 →  A(0.60) + B(0.25) = 0.85 ≥ 0.75 → 候选 = {A, B}
Top-p = 0.95 →  A+B+C = 0.95 ≥ 0.95 → 候选 = {A, B, C}
```

- **候选数量自适应**：分布尖 → 候选可能只有 1~2 个；分布平 → 候选自动变多。
- 比 Top-K 更优雅，现代 API（OpenAI、Anthropic）的主力参数。

### 2.4 三者对比与组合使用

| | 改分布形状？ | 截断候选？ | 截断方式 |
|---|---|---|---|
| Temperature | ✅ 全局拉尖/拉平 | ❌ | — |
| Top-K | ❌ | ✅ | 固定个数 K |
| Top-p | ❌ | ✅ | 累计概率阈值 p |

常见组合：

- **严谨任务**：`temperature=0`（或 0.1）+ `top_p=1`
- **通用对话**：`temperature=0.7~1.0` + `top_p=0.9`（两个通常只调一个，另一个保持默认）
- **创意写作**：`temperature=1.0~1.3` + `top_p=0.95`

执行顺序：**先温度调整 logits → 再 Top-K 截断 → 再 Top-p 截断 → 最后 softmax 抽样**（各家实现细节略有差异）。

---

## 3. tiktoken 实测（真实运行结果）

安装：

```bash
pip install tiktoken
```

### 3.1 同一段英文，不同 tokenizer 的 token 数不同

文本：`Hello, world! Transformer models split text into tokens.`

| 编码器 | 所属模型 | token 数 |
|---|---|---|
| gpt2 | GPT-2 | 12 |
| cl100k_base | GPT-4 / GPT-3.5 | 11 |
| o200k_base | GPT-4o / GPT-5 | 11 |

cl100k_base 的具体切分：

```
['Hello', ',', ' world', '!', ' Transformer', ' models', ' split', ' text', ' into', ' tokens', '.']
```

注意 ` world` 带着**前导空格**是一个 token——英文里空格附着在后面的词上。

### 3.2 中英文 Token 效率对比（cl100k_base）

| 文本 | 字符数 | token 数 | 1 token ≈ 字符 |
|---|---|---|---|
| The quick brown fox jumps over the lazy dog. | 44 | 10 | 4.40 |
| 敏捷的棕色狐狸跳过了懒狗。 | 13 | 25 | **0.52** |
| 今天天气很好，我们一起去公园散步吧。 | 18 | 22 | **0.82** |
| The weather is very nice today, let's go for a walk... | 73 | 18 | 4.06 |
| 1234567890 | 10 | 4 | 2.50 |
| def softmax(x): ... | 54 | 17 | 3.18 |
| 😀🎉🔥 | 3 | **8** | **0.38** |

**英文 ~4 字符/token，中文 ~0.5-1 字符/token，emoji 最夸张（1 个 emoji ≈ 2-3 token）。**

### 3.3 中文为什么费 token —— 逐 token 查看

```
"今天天气很好"  → 8 个 token: ['今','天','天','气','很','好', ...]  几乎一字一 token
"Hello world"   → 2 个 token: ['Hello', ' world']
```

BPE 词表以英文语料为主训练，**常用英文词整词收录，而汉字大多只能退化到单字甚至字节级**，所以中文天然更耗 token、同等 context window 能装的中文内容更少。

### 3.4 新 tokenizer 对中文更友好（实测）

文本：`人工智能大语言模型的分词器对中文的支持越来越好了。`

| 编码器 | token 数 |
|---|---|
| cl100k_base（GPT-4/3.5） | 29 |
| o200k_base（GPT-4o/GPT-5） | **16** |

o200k_base 词表扩到 20 万条并纳入更多中文常见词，**同一句中文省了约 45% 的 token**。这是厂商升级 tokenizer 的直接收益（同样的 context window 能装更多中文）。

### 3.5 自己动手统计任意文本

```python
import tiktoken

def num_tokens(text: str, model: str = "gpt-4o") -> int:
    """统计任意文本在指定模型下的 token 数"""
    enc = tiktoken.encoding_for_model(model)  # 也可用 tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

print(num_tokens("今天天气很好", model="gpt-4o"))       # o200k_base
print(num_tokens("Hello world", model="gpt-4-turbo"))  # cl100k_base

# 查看具体切分
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("今天天气很好")
print([enc.decode_single_token_bytes(i) for i in ids])
```

> 注意：Claude、Gemini、国产模型不使用 OpenAI 的 tokenizer，tiktoken 统计的数字只能作近似参考；精确值需用各家的 count_tokens API。

---

## 4. Context Window 对比

Context Window = 模型单次请求能"看见"的 token 总量（**输入 + 输出共享**）。

### 4.1 主流模型一览（2025-2026 时点，以官方为准）

| 模型 | Context Window | 发布时间 |
|---|---|---|
| GPT-3.5-turbo | 16K | 2022.11 |
| GPT-4 | 8K / 32K | 2023.03 |
| GPT-4o | 128K | 2024.05 |
| o3 / o4-mini | 200K | 2025 |
| GPT-5 系列 | 400K | 2025.08 |
| Claude 3 Haiku / Sonnet / Opus | 200K | 2024.03 |
| Claude 4.x（Sonnet / Opus） | 200K（1M 长上下文 beta） | 2025 |
| Claude Opus 4.6 / Sonnet 4.6 | 200K（Sonnet 支持 1M beta） | 2025.11 |
| Gemini 1.5 Pro | 1M–2M | 2024 |
| Gemini 2.x | 1M | 2025 |
| Llama 3.1 405B | 128K | 2024.07 |
| DeepSeek-V3 / R1 | 64K–128K | 2024.12 / 2025.01 |
| Qwen2.5 / Qwen3 | 32K–128K（部分 1M） | 2024–2025 |

### 4.2 直觉换算

以"约 4 字符/token（英文）、1 字符/token（中文）"估算：

| Context | ≈ 英文词数 | ≈ 中文字数 | 直觉 |
|---|---|---|---|
| 8K | ~6K 词 | ~8K 字 | 一篇长论文 |
| 32K | ~24K 词 | ~32K 字 | 一本中篇小说 |
| 128K | ~96K 词 | ~128K 字 | 《哈利·波特》1/3 |
| 200K | ~150K 词 | ~200K 字 | 《哈利·波特》全集 |
| 400K | ~300K 词 | ~400K 字 | 数本小说 |
| 1M–2M | ~750K–1.5M 词 | ~1M–2M 字 | 整个代码仓库 / 全套教材 |

### 4.3 使用时的注意事项

1. **上下文 ≠ 全部有效**。长上下文存在"lost in the middle"现象：中间部分的信息利用率低于开头和结尾。
2. **价格与长度非线性**：很多模型对超长上下文（如 >200K）加价计费（Gemini 的 1M/2M 档、Claude 1M beta 档）。
3. **输出通常有独立上限**：如 context 200K 但 max output 可能只有 32K–64K，两者是包含关系。
4. **长上下文 ≠ 无限记忆**：多轮对话超窗后需要截断或压缩（这也是 RAG / 记忆机制存在的原因）。

---

## 5. 一页速查

```
Token：模型的阅读单位。英文 ~4 字符/token，中文 ~0.5-1 字符/token。
Tokenizer：模型私有词表（BPE）。换模型 token 数就变。统计用 tiktoken（仅限 OpenAI 系）。

采样三参数：
  Temperature  —— 调分布形状（尖 ↔ 平），不截断
  Top-K        —— 按个数截断（固定 K 个）
  Top-p        —— 按累计概率截断（自适应候选数，主流）

Context Window：输入+输出共享的 token 上限。
  GPT-4o 128K | GPT-5 400K | Claude 200K(1M beta) | Gemini 1M-2M
```

---

## 参考来源

- [OpenAI Glossary — Token](https://platform.openai.com/docs/glossary#token)
- [OpenAI — Tokenizer tool](https://platform.openai.com/tokenizer)
- [OpenAI Cookbook — How to count tokens with tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
- [tiktoken GitHub](https://github.com/openai/tiktoken)
- [Anthropic Docs — Token Counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [Anthropic Docs — Context Windows](https://docs.anthropic.com/en/docs/build-with-claude/context-windows)
- [Google Gemini — Long context](https://ai.google.dev/gemini-api/docs/long-context)
