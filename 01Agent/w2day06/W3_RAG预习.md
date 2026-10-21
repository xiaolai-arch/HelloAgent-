# W3 预习：RAG（检索增强生成）

> 用途：W3 开课前的地图。课程材料下发后，用这份文档对照“哪些概念已经见过、哪些要动手”。
> 说明：仓库里没有 W3 官方讲义，本文基于公开资料整理，不冒充课程原文。
> 目标：开课前能说清 RAG 的两条流水线、五个关键决策和一套分层评估指标。

---

## 0. 一句话理解

> **RAG = 开卷考试：先从外部知识库找到相关资料，再把资料放进上下文，让模型带着证据回答。**

模型参数里的知识是“闭卷记忆”：有截止日期、会过时、无法引用、不能按权限隔离。RAG 是“外部记忆”：可更新、可引用、可过滤、可审计。

从 Context Engineering 的角度看，RAG 不是“外挂一个向量库”，而是：

> **在推理前，按需选择最相关的外部上下文，并在有限的 token 预算内把它组织成模型能用的形式。**

---

## 1. 为什么需要 RAG

| 问题 | 模型裸跑 | RAG |
|---|---|---|
| 知识截止 | 只能答训练时见过的东西 | 可以接最新文档 |
| 私有数据 | 不知道公司内部制度/代码 | 可以检索内部知识库 |
| 引用来源 | 经常给不出可核查出处 | 可以返回文档、页码、链接 |
| 更新成本 | 重新训练/微调，慢且贵 | 更新索引即可 |
| 权限控制 | 无法按用户过滤知识 | 可在检索层做 ACL |
| 幻觉 | 可能编造事实 | 有证据约束，但仍可能错 |

### 1.1 什么时候不需要 RAG

- **知识库很小且静态**：小于约 200k tokens（约 500 页）时，直接把全文放进上下文 + prompt caching 可能更简单、更准。Anthropic 在 Contextual Retrieval 一文里明确提过这个边界。
- **任务主要是推理/计算**：数学、逻辑、代码执行，缺的不是知识，而是工具或推理。
- **数据结构化且查询明确**：SQL / API 比向量检索更合适。
- **延迟极度敏感**：检索 + 重排会增加延迟；如果单次全文上下文已经够快，未必需要 RAG。
- **模型已经知道且不常变**：通用常识、稳定概念，直接问可能更好。

> 先问“模型缺的是知识，还是推理，还是权限，还是实时性？”再决定要不要上 RAG。

---

## 2. RAG 的两条流水线

```text
离线索引（Ingestion）
文档 → 解析 → 清洗 → 切分 → 元数据 → 向量化 → 向量索引
                                      ↘ BM25 / 关键词索引

在线查询（Query）
问题 → 改写 → 召回（Dense + Sparse）→ 融合 → 重排 → 上下文打包 → 生成 → 引用 → 评估
```

两条流水线的失败要分开排查：

- **离线没做好**：资料没进来、解析乱了、切分断了、元数据缺失。
- **在线没做好**：查询跑偏、召回没中、排序不对、上下文太吵、生成不忠实。

很多人一上来就换 embedding 模型，但真正的问题可能在 PDF 解析或 chunk 边界。

---

## 3. 离线索引：把文档变成可检索的单元

### 3.1 各阶段

| 阶段 | 目标 | 关键决策 | 常见失败 |
|---|---|---|---|
| 解析 Parsing | 把 PDF/HTML/Word/代码变成干净文本 | 是否保留版面、表格、标题层级、页码 | 阅读顺序错、表格散架、页眉页脚混入 |
| 清洗 Cleaning | 去噪、去重、脱敏 | 去多少、是否保留代码/公式 | 把有用信息洗掉，或留下大量模板文本 |
| 切分 Chunking | 切成适合检索的单元 | 大小、重叠、边界、父子关系 | 语义被切断、chunk 缺上下文 |
| 元数据 Metadata | 让检索可过滤、可引用 | 来源、标题、章节、页码、时间、ACL、版本 | 无法按权限过滤，或引用不可追溯 |
| 向量化 Embedding | 把文本变成语义向量 | 模型、维度、语言、领域、成本 | 查询和文档不同模型/版本，语义空间不一致 |
| 索引 Index | 支持快速相似度搜索 | 向量库、ANN 算法、混合索引 | recall 不够、延迟高、更新困难 |

### 3.2 Chunking 策略

| 策略 | 做法 | 适合 | 风险 |
|---|---|---|---|
| 固定长度 | 每 N tokens 切一刀，带 overlap | 快速基线 | 容易切断句子/表格 |
| 递归切分 | 先按段落，再按句子，最后按 token | 通用文档 | 仍可能切断跨段逻辑 |
| 结构感知 | 按 Markdown 标题、HTML section、代码 AST 切 | 技术文档、代码、手册 | 依赖解析质量 |
| 语义切分 | 用 embedding 相似度找边界 | 长文、主题变化明显 | 成本高、边界不稳定 |
| 父子块 / small-to-big | 小 chunk 检索，返回父 chunk | 长文档问答 | 需要维护层级 |
| 句子窗口 | 命中句子后带前后邻居 | 事实型问答 | 上下文可能仍然不足 |
| Contextual Retrieval | 给每个 chunk 预置 50~100 tokens 的文档级背景 | 文档碎片化严重 | 预处理成本；需要缓存优化 |

经验区间（不是定律）：

- chunk 大小：**200~800 tokens** 起步；事实问答偏小，摘要/长文推理偏大。
- overlap：**10%~20%**；太大浪费预算，太小容易断。
- 表格、代码、公式：尽量保持完整，不要按字符硬切。
- 每个 chunk 最好带上：`title`、`section`、`source`、`page`、`timestamp`、`acl`。

### 3.3 Contextual Retrieval（值得知道）

传统 RAG 切完块后，chunk 可能失去“它来自哪份文档、哪一节、什么时间”的背景。Anthropic 的做法是让模型为每个 chunk 生成一段简短背景，再拼到 chunk 前面，然后做 embedding 和 BM25。

Anthropic 公布的实验结果（其自有 benchmark）：

- Contextual Embeddings：top-20 检索失败率降低 **35%**（5.7% → 3.7%）。
- 加上 Contextual BM25：降低 **49%**（5.7% → 2.9%）。
- 再加 reranking：降低 **67%**（5.7% → 1.9%）。

注意：这是特定数据集和配置下的结果，不是“任何场景都能提升 67%”。但它说明一个原则：**chunk 的上下文背景，和 chunk 本身一样重要。**

### 3.4 Embedding 模型怎么选

| 维度 | 要问的问题 |
|---|---|
| 语言 | 中文/多语言支持够不够？ |
| 领域 | 代码、法律、医学、金融是否有领域适配？ |
| 维度 | 维度越高越准，但存储/检索成本越高 |
| 最大输入 | 你的 chunk 会不会超过模型上限？ |
| 成本/延迟 | 在线 query embedding 的延迟能否接受？ |
| 版本 | 查询和文档是否用同一模型、同一版本？ |
| 归一化 | 是否做了 L2 normalization？相似度度量是否匹配？ |

常见选择：

- 通用多语言：`text-embedding-3-small/large`、`bge-m3`、`jina-embeddings-v3`、`voyage-3`。
- 中文/领域：优先看 MTEB/C-MTEB 榜单和实际业务评测，不要只看榜单。
- 稀疏检索：BM25 仍然是强基线，尤其是错误码、产品型号、专有名词。

### 3.5 向量库怎么选

| 选项 | 特点 | 适合 |
|---|---|---|
| FAISS | 库，不是服务；本地、快、灵活 | 原型、离线实验 |
| Chroma | 轻量、开发体验好 | 本地开发、小规模 |
| pgvector | 跟 PostgreSQL 一起用，事务/权限成熟 | 已有 Postgres 的业务 |
| Qdrant | 过滤、混合检索、生产化较好 | 中大规模服务 |
| Milvus | 大规模、分布式 | 大规模向量检索 |
| Weaviate | 内置混合检索/模块 | 快速搭服务 |

选型时不要只看“谁最快”，还要看：元数据过滤、ACL、增量更新、备份、可观测性、运维成本。

---

## 4. 在线检索：把对的资料找出来

### 4.1 检索方法

| 方法 | 强项 | 弱项 | 适用 |
|---|---|---|---|
| Dense（向量） | 语义匹配、同义改写 | 精确词、错误码、罕见词容易漏 | 自然语言问答 |
| Sparse（BM25） | 精确匹配、可解释、便宜 | 不懂同义词 | 型号、错误码、专有名词 |
| Hybrid | 两者互补 | 需要融合策略 | 大多数生产 RAG |
| Metadata filter | 权限、时间、版本、类型 | 依赖元数据质量 | 企业知识库 |
| MMR | 增加结果多样性 | 可能牺牲最相关结果 | 摘要、多角度回答 |

Hybrid 常用 **RRF（Reciprocal Rank Fusion）**：

```text
score(d) = Σ_i 1 / (k + rank_i(d))
```

`k` 常取 60。它只看排名，不看原始分数，简单稳健。

### 4.2 查询改写

| 方法 | 做法 | 适合 | 风险 |
|---|---|---|---|
| 同义词扩展 | 加同义词/缩写 | 术语不统一 | 引入噪声 |
| Multi-query | 生成多个改写，分别检索后融合 | 口语化问题 | token/延迟增加 |
| HyDE | 先让模型写一个假答案，再用它检索 | 问题很短、语义稀疏 | 假答案跑偏 |
| 问题分解 | 把多跳问题拆成子问题 | 多跳、比较类 | 分解错误 |
| 指代消解 | 把“它/这个”补全 | 多轮对话 | 补错 |

查询改写的原则：**先保证不改变用户意图，再追求召回。** 改写后要保留原始 query 作为兜底。

### 4.3 Top-k、重排与上下文打包

- **召回**：先取较多候选，常见 20~150 条。
- **重排**：用 cross-encoder 对 `(query, chunk)` 逐对打分，取 top 5~20。
- **打包**：去重、按相关性排序、控制 token、加引用编号。
- **顺序**：关键信息不要全放在中间；“lost in the middle” 研究表明模型对中间位置的信息召回更差。
- **边界**：用 `<context>`、`<document>` 标签把资料和指令分开，防止资料里的指令被当成系统指令。

一个实用的上下文打包顺序：

```text
[系统指令]
[输出契约]
[检索到的资料，按相关性排序，带编号]
[用户问题]
[最后提醒：只根据资料回答，不足则说不知道]
```

### 4.4 长上下文 vs RAG

| 场景 | 更适合 |
|---|---|
| 小、静态、可整体放入 | 全文上下文 + prompt caching |
| 大、动态、频繁更新 | RAG |
| 需要权限隔离 | RAG + metadata/ACL |
| 需要精确引用 | RAG + citation |
| 多跳、探索式 | Agentic RAG / 工具调用 |
| 超长代码库 | 混合：索引 + grep/glob + 文件系统 |

长上下文不是 RAG 的替代品。窗口再大，注意力预算和成本仍然有限；RAG 解决的是“选什么进来”，不是“窗口够不够大”。

---

## 5. 生成：让答案有据可查

### 5.1 回答 Prompt 的推荐结构

```text
<instructions>
你是一个基于资料回答问题的助手。
1. 只使用 <context> 中的信息回答。
2. 每个关键结论后面用 [编号] 标注来源。
3. 如果资料不足，明确说“根据现有资料无法确定”，不要编造。
4. 如果资料之间冲突，列出冲突点，并说明你采用的版本。
</instructions>

<context>
[1] 文档标题 / 章节 / 页码
正文……

[2] 文档标题 / 章节 / 页码
正文……
</context>

<question>
用户问题
</question>

<output_format>
## 结论
...
## 依据
- ... [1]
## 不确定项
...
</output_format>
```

### 5.2 Grounding 的几个动作

- **引用编号**：每个关键结论对应至少一个 chunk。
- **引用可追溯**：编号能映射回文档、页码、URL。
- **允许弃权**：资料不足时明确说“不知道”，比编造好。
- **冲突处理**：不同文档说法不一致时，列出冲突，不要强行合并。
- **温度**：事实型问答用低温度（0~0.3），创意型可以高一些。
- **后验检查**：生成后检查每个引用是否真的支持该结论。

### 5.3 常见生成失败

| 失败 | 表现 | 修法 |
|---|---|---|
| 不忠实 | 答案超出资料 | 强化“只根据 context”，加引用校验 |
| 漏引用 | 有结论没来源 | 输出 schema 强制 `[n]` |
| 引用错 | 引用了不支持的 chunk | 生成后做 NLI/LLM 校验 |
| 过度合并 | 把冲突资料说成一致 | 显式要求列出冲突 |
| 上下文稀释 | 资料太多，重点被淹没 | 减少 top-k、重排、压缩 |
| 先验污染 | 模型用训练知识覆盖资料 | 明确“资料优先”，必要时对比 |

---

## 6. 评估：不要只看“答得像不像”

RAG 评估要分层，不然出了问题不知道是检索还是生成。

### 6.1 先建 Golden Set

建议 50~200 条；最小验证集至少 20~30 条起步。要覆盖：

- **可回答**：答案在资料里，且能定位到 chunk。
- **不可回答**：资料里没有，模型应该弃权。
- **多跳**：需要组合 2~3 个 chunk。
- **歧义**：问题有多种解释，模型应该澄清。
- **时效**：旧版本 vs 新版本，模型应该用新的。
- **对抗**：资料里有误导性内容/注入指令。
- **权限**：不同用户能看到不同资料。

每条最好包含：`question`、`gold_answer`、`gold_chunk_ids`、`source`、`type`、`difficulty`。

### 6.2 指标

| 层级 | 指标 | 说明 |
|---|---|---|
| 检索 | Recall@k | 相关 chunk 是否出现在前 k 条 |
| 检索 | Precision@k | 前 k 条里有多少是相关的 |
| 检索 | MRR | 第一个相关结果排名的倒数 |
| 检索 | nDCG@k | 考虑排序位置的相关性 |
| 检索 | Hit rate | 至少命中一条相关 chunk 的比例 |
| 生成 | Faithfulness | 答案中的 claim 是否被 context 支持 |
| 生成 | Answer Relevance | 答案是否回答了问题 |
| 生成 | Context Precision | 上下文中相关内容的排序质量 |
| 生成 | Context Recall | 上下文是否覆盖了标准答案所需信息 |
| 生成 | Noise Sensitivity | 无关内容对答案的影响 |
| 端到端 | Correctness | 最终答案对不对 |
| 端到端 | Citation Accuracy | 引用是否支持结论 |
| 端到端 | Abstention Rate | 该弃权时是否弃权 |
| 工程 | Latency / Cost | P50/P95 延迟、每问答成本 |

RAGAS 提供了 Context Precision、Context Recall、Faithfulness、Response Relevancy、Noise Sensitivity 等指标，可以作为起点，但不要把它当成唯一真相。

### 6.3 LLM-as-judge 的坑

- **位置偏置**：更偏好前面的答案。
- **长度偏置**：更偏好长答案。
- **自我偏好**：更偏好同模型生成的答案。
- **Prompt 敏感**：评分 prompt 改一个字，分数就变。
- **无法判断事实**：judge 自己也可能错。

做法：

- 用人工标注一小批（50~100 条）校准 judge。
- 固定评分 rubric 和模型版本。
- 关键指标（faithfulness、citation）尽量用规则/人工抽检补充。
- 记录 judge 与被人工标注的一致率。

### 6.4 错误分析 taxonomy

| 层 | 问题 | 典型修法 |
|---|---|---|
| 数据 | 文档缺失、过期、权限错 | 补数据、加时间/ACL、版本管理 |
| 解析 | PDF 乱序、表格散架 | 换解析器、版面分析、人工校验 |
| 切分 | 语义断裂、chunk 缺背景 | 调大小/重叠、结构切分、contextual retrieval |
| 向量 | 模型不匹配、语言不支持 | 换模型、统一版本、归一化 |
| 查询 | 改写跑偏、指代未消解 | 保留原 query、加约束、对话状态 |
| 召回 | 相关 chunk 没进 top-k | 混合检索、提高 k、元数据过滤 |
| 排序 | 相关 chunk 排太后 | rerank、RRF、MMR |
| 打包 | 上下文太吵/太长 | 降 top-k、去重、压缩、排序 |
| 生成 | 不忠实、漏引用 | 强化 grounding、引用校验、弃权 |
| 评估 | 指标与业务不一致 | 重建 golden set、加人工评审 |

---

## 7. 进阶模式（知道什么时候用，不要一上来全上）

| 模式 | 解决什么 | 代价/风险 |
|---|---|---|
| Contextual Retrieval | chunk 缺文档背景 | 预处理成本 |
| Parent-child / small-to-big | 检索粒度 vs 生成粒度冲突 | 层级维护 |
| HyDE | 查询太短、语义稀疏 | 假答案跑偏 |
| Multi-query | 口语化、同义表达 | token/延迟 |
| Hybrid + RRF | 语义 + 精确匹配 | 融合调参 |
| Rerank | 召回多但排序差 | 延迟/成本 |
| GraphRAG | 多跳、全局关系、实体网络 | 构图成本、更新复杂 |
| Agentic RAG | 需要探索、工具调用、动态决策 | 延迟、复杂度、失控风险 |
| Self-RAG / Corrective RAG | 生成后自检、失败再检索 | 多次调用、成本 |
| Multimodal RAG | 表格、图片、图表、扫描件 | 解析/对齐困难 |
| Memory + RAG | 用户偏好/历史 + 外部知识 | 隐私、冲突、过期 |

一个判断原则：

> **先解决“检索不到”和“上下文太吵”，再考虑 GraphRAG/Agentic RAG。** 大多数 RAG 问题不是缺高级架构，而是解析、切分、评估没做好。

---

## 8. W3 五天动手计划（建议）

| 天 | 目标 | 产出 |
|---|---|---|
| Day1 | 建语料：选 20~50 篇文档（课程笔记 + 3 篇 PDF/网页），解析、清洗、切分 | `chunks.jsonl`，随机抽 20 个 chunk 人工检查 |
| Day2 | 最小 dense RAG：embedding + 向量索引 + top-k 检索 | `retrieval.py`，20 条问题的 Recall@5 |
| Day3 | Hybrid：BM25 + dense + RRF；加 rerank（可用 API 或本地模型） | 对比表：dense vs hybrid vs hybrid+rerank |
| Day4 | 生成：引用、弃权、冲突处理；建 30 条 golden set | `rag_answer.py`，30 条问答结果 |
| Day5 | 评估与错误分析：检索/生成/端到端指标；写报告 | `rag_report.md`，至少 3 个改进点 |

建议的交付物：

```text
01Agent/w3day0x/
├── data/                 # 原始文档（注意版权/隐私）
├── chunks.jsonl          # 切分结果
├── eval_set.json         # golden set
├── rag_pipeline.py       # 索引 + 检索 + 生成
├── rag_report.md         # 指标、错误分析、改进
└── README.md
```

---

## 9. 最小代码骨架（示意）

> 下面是结构示意，不是完整实现；W3 会把它跑通并加评估。

```python
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """按字符的简化切分；真实项目应按 token/结构切。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query_vec: list[float], chunks: list[Chunk], vectors: list[list[float]], k: int = 5):
    scored = [(cosine(query_vec, vec), chunk) for chunk, vec in zip(chunks, vectors)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:k]


def build_prompt(question: str, hits: list[tuple[float, Chunk]]) -> str:
    context = "\n\n".join(
        f"[{i}] {chunk.source}\n{chunk.text}"
        for i, (_, chunk) in enumerate(hits, start=1)
    )
    return f"""<instructions>
只使用 <context> 中的信息回答。每个关键结论用 [编号] 标注来源。
资料不足时明确说“根据现有资料无法确定”，不要编造。
</instructions>

<context>
{context}
</context>

<question>
{question}
</question>
"""


def answer(question: str) -> str:
    query_vec = embed([question])[0]  # 需要实现 embed()
    hits = retrieve(query_vec, chunks, vectors, k=5)
    prompt = build_prompt(question, hits)
    return call_llm(prompt)  # 需要实现 call_llm()
```

真实实现要补：

- token-based chunking（不是字符）
- embedding 批处理与缓存
- 向量库/ANN 索引
- BM25 + RRF
- rerank
- metadata/ACL 过滤
- 引用映射
- 评估脚本

---

## 10. 术语表

| 术语 | 含义 |
|---|---|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| Chunk | 文档切分后的检索单元 |
| Embedding | 把文本映射成向量 |
| Vector DB | 存储和检索向量的数据库 |
| ANN | Approximate Nearest Neighbor，近似最近邻 |
| HNSW | 一种常用的 ANN 图索引 |
| BM25 | 基于词频的稀疏检索算法 |
| Hybrid Search | 向量 + 关键词混合检索 |
| RRF | Reciprocal Rank Fusion，按排名融合 |
| Rerank | 对召回结果重新打分排序 |
| Cross-encoder | 同时看 query 和文档的 rerank 模型 |
| Contextual Retrieval | 给 chunk 预置文档背景再索引 |
| HyDE | 用假答案做检索 |
| MMR | 最大边际相关性，增加结果多样性 |
| Recall@k | 前 k 条中是否包含相关结果 |
| MRR | 第一个相关结果的排名倒数 |
| nDCG | 考虑排序位置的相关性指标 |
| Faithfulness | 答案是否被上下文支持 |
| Context Precision | 上下文中相关内容的排序质量 |
| Context Recall | 上下文是否覆盖标准答案所需信息 |
| Grounding | 让答案有据可查 |
| ACL | 访问控制列表 |
| Agentic RAG | 由 Agent 决定何时、如何检索 |

---

## 11. 开课前 10 问

能答上 7 个，W3 会轻松很多：

1. RAG 的离线索引和在线查询各包含哪些步骤？
2. chunk 大小和 overlap 如何影响召回与生成？
3. dense 和 sparse 检索各自强在哪？为什么要 hybrid？
4. RRF 为什么只看排名，不看分数？
5. rerank 和 embedding 检索的区别是什么？
6. 什么是 contextual retrieval？它解决什么问题？
7. Recall@k 和 Faithfulness 分别在评估什么？
8. 为什么“答对了”不代表 RAG 没问题？如何定位是检索错还是生成错？
9. 长上下文模型还需要 RAG 吗？什么情况下不需要？
10. 如果资料不足，系统应该怎么回答？如何在 Prompt 和评估里体现？

---

## 12. 参考资料

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)  
  https://arxiv.org/abs/2005.11401
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023)  
  https://arxiv.org/abs/2312.10997
- Anthropic, *Introducing Contextual Retrieval* (2024-09-19)  
  https://www.anthropic.com/news/contextual-retrieval
- Anthropic, *Effective context engineering for AI agents* (2025-09-29)  
  https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Ragas, *List of available metrics*  
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Chroma, *Context Rot* (2025-07-14)  
  https://research.trychroma.com/context-rot
- Liu et al., *Lost in the Middle* (2023)  
  https://arxiv.org/abs/2307.03172
- LangChain, *RAG from Scratch*  
  https://github.com/langchain-ai/rag-from-scratch
