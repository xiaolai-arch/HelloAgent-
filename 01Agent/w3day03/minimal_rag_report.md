# 最小 RAG 链路实验报告（W3 Day3）

- 生成时间：2026-10-21 16:18:38
- 语料：15 篇 → 递归切分 72 chunks（目标 300 字符）
- 向量：`BAAI/bge-small-zh-v1.5`（semantic，查询/文档同模型）
- 生成模型：`mimo-v2.6-flash`，temperature=0.2
- 黄金集：15 题（可回答/多跳/精确词/不可回答）

## 一、链路全景

```text
离线索引：corpus/*.md → recursive_chunk(300字) → bge passage_embed → Chroma
在线查询：question → bge query_embed → Chroma.query(top5)
          → <context> 打包(带编号) → chat(只准根据资料回答+弃权) → 带引用回答
```

与 LangChain 组件的对应关系（本仓库零框架手写，概念一一对应）：

| 本实验 | LangChain 对应组件 |
|---|---|
| `ragkit.load_corpus()` | DocumentLoader（TextLoader/DirectoryLoader） |
| `ragkit.recursive_chunk()` | RecursiveCharacterTextSplitter |
| `ragkit.get_embedder()` | Embeddings（OpenAIEmbeddings/HuggingFaceEmbeddings） |
| Chroma collection | VectorStore（Chroma） |
| `build_prompt()` | ChatPromptTemplate + 自定义 context 组装 |
| `ragkit.chat()` | ChatOpenAI / LLM |

## 二、生成侧 Prompt（全文）

要点：`<context>` 标签隔离资料与指令（防注入）、编号引用、明确弃权条款。

```text
<instructions>
你是一个基于资料回答问题的助手。
1. 只使用 <context> 中的信息回答，不要用你自己的知识补充。
2. 每个关键结论后面用 [编号] 标注来源。
3. 如果资料不足，明确说"根据现有资料无法确定"，不要编造。
</instructions>

<context>
{context}
</context>

<question>
{question}
</question>
```

## 三、评估结果

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | 0.867 | 相关 chunk 进了 top5 的问题比例 |
| MRR | 0.606 | 第一个相关 chunk 排名倒数的均值 |
| HitRate | 0.733 | 至少命中一条的比例（top5 内） |
| 可回答题 Recall@5 | 0.846 | |
| 弃权准确率 | 0.933 | 不可回答的题该弃权、可回答的不该 |
| 不可回答题弃权率 | 1.0 | 资料没有时应明确说不知道 |
| tokens | prompt=11699 / completion=2739 | 15 次问答合计 |

### 逐题结果

| 题 | 类型 | Recall@5 | MRR | 弃权 | 回答摘录 |
|---|---|---|---|---|---|
| Q1 | fact | 1 | 1.00 | ✅ | Transformer 相比 RNN 的优势主要有三点 [1]：  1. **并行性更好**：RNN 必须按时间步串行计… |
| Q2 | fact | 1 | 0.50 | ✅ | ## 为什么除以 sqrt(d_k)  **核心原因：维度越大，点积的方差越大；不缩放会把 softmax 推入饱和区，… |
| Q3 | fact | 1 | 1.00 | ✅ | 温度调高会让输出更随机，是因为它改变了概率分布的形状，具体机制如下：  温度的工作方式是**先给模型输出的 logits… |
| Q4 | fact | 1 | 1.00 | ✅ | 根据资料，DeepSeek-V3 的**总参数量为 671B**，**每 token 只激活 37B 参数**[1]。 … |
| Q5 | fact | 1 | 1.00 | ✅ | LLaVA 是开源的经典多模态方案，其架构由三部分组成 [1]：  1. **视觉编码器**：采用 CLIP ViT [… |
| Q6 | fact | 1 | 1.00 | ✅ | ## RRF 融合公式  **公式**：每个文档的融合分等于它在各路检索结果中排名倒数之和 [1]：  ```text … |
| Q7 | fact | 1 | 1.00 | ✅ | MCP 的三层角色架构如下 [1]：  1. **Host（宿主）**：用户直接使用的 LLM 应用，例如 Claude… |
| Q8 | fact | 1 | 0.33 | ✅ | RAGAS 的四大生成层指标如下 [4]：  1. **Faithfulness（忠实度）**：答案中的每个论断是否都被… |
| Q9 | exact-term | 1 | 1.00 | ✅ | 根据资料，`TRV-40907` 表示**报销单提交超过 15 日期限**，需要**由总监在 OA 系统里发起"逾期特批… |
| Q10 | fact | 1 | 1.00 | ✅ | 深圳出差普通员工的住宿标准是 **600 元/晚** [1]。… |
| Q11 | multi-hop | 0 | 0.00 | ✅ | ## 出差申请时限  - 员工须提前 **3 个工作日**在 OA 系统提交《出差申请单》，注明目的地、事由、预计费用；… |
| Q12 | multi-hop | 0 | 0.00 | ❌ | **最终机制：Function Calling。** MCP Server 列出的工具，最终还是通过 Function … |
| Q13 | unanswerable | 0 | 0.00 | ✅ | 根据现有资料无法确定。提供的资料中没有关于 HelloAI 公司年会一等奖的任何信息。… |
| Q14 | unanswerable | 0 | 0.00 | ✅ | 根据现有资料无法确定。提供的资料中没有关于 HelloAgent 学习仓库在 GitHub 上 star 数量的任何信息… |
| Q15 | fact | 1 | 0.25 | ✅ | 根据现有资料，以下几种场景不该上 RAG，直接把资料放进上下文更好 [4]：  1. **知识库小且静态**——资料能整… |

### 检索命中明细（不可回答题 + 没命中的题）

**Q11 员工要坐高铁去北京出差，最晚要提前几天提交申请？住宿标准每天多少钱、餐补每天多少？**（type=multi-hop）

- · 15_helloai_policy#c1：交通标准  - 高铁：单程 4 小时以内一律二等座；超过 4 小时可订一等座。 - 飞机：单程 15…
- · 15_helloai_policy#c3：餐补与津贴  - 出差期间餐补 **100 元/天**，不需要发票，按自然日计算（含往返当天）。 -…
- · 15_helloai_policy#c0：# HelloAI 公司差旅与报销制度（2026 年版）  > 本文档为 W3 RAG 实验专用的虚…
- · 15_helloai_policy#c2：住宿标准  / 城市 / 普通员工 / 总监及以上 / /---/---/---/ / 北京/上海/…
- · 15_helloai_policy#c4：常见错误码  OA 报销系统常见错误码：  - `TRV-40231`：发票抬头与公司主体不一致。修…

**Q12 MCP Server 暴露的工具最终靠什么机制让模型决定调用？这个机制的完整流程是哪几步？**（type=multi-hop）

- · 13_mcp#c4：适用场景：自己一个应用内部调工具，Function Calling 直接了当；要把工具开放给整个生态…
- · 12_react_function_calling#c1：为什么 ReAct 比单纯的 Reasoning（直接让模型想到底）更强：模型的知识是静态的、会幻觉…
- · 12_react_function_calling#c3：这个"模型决策 → 代码执行 → 结果回传"的循环就是 Agent 的执行闭环：Function C…
- · 10_prompt_engineering#c2：System Prompt vs User Prompt  System Prompt 是整段对话的…
- · 13_mcp#c2：协议细节  传输层：本地进程用 stdio（标准输入输出上的 JSON-RPC 2.0），远程服务用…

**Q13 HelloAI 公司年会一等奖是什么？**（type=unanswerable）

- · 09_rag_eval#c4：Golden Set（黄金测试集）  最小 20~30 条起步，必须覆盖：可回答（答案能定位到具体 …
- · 04_moe_multimodal#c1：DeepSeek-V3 总参数 671B、每 token 只激活 37B——容量接近千亿级大模型，推…
- · 11_context_engineering#c2：与 RAG 的关系  从上下文工程的视角看，RAG 不是"外挂一个向量库"，而是"在推理前按需选择最…
- · 02_attention#c1：计算流程分四步：第一步，Q 和 K 做点积，得到每对位置之间的原始相关性分数；第二步，除以 sqrt…
- · 03_tokenizer_sampling#c1：OpenAI 的 tiktoken 库可以在本地精确统计任意文本在 GPT 系列编码（如 o200k…

**Q14 HelloAgent 学习仓库在 GitHub 上有多少个 star？**（type=unanswerable）

- · 07_rag_pipeline#c4：经验区间（不是定律）：chunk 大小 200~800 token 起步，事实问答偏小、长文推理偏大…
- · 04_moe_multimodal#c1：DeepSeek-V3 总参数 671B、每 token 只激活 37B——容量接近千亿级大模型，推…
- · 01_transformer#c4：主流模型的取舍  现代大模型基本都是 Decoder-only 结构，只保留掩码自注意力 + FFN…
- · 06_vector_db#c0：# 向量数据库：存向量、找近邻  ## 解决什么问题  RAG 需要在一个几万到几亿条的向量库里快速…
- · 09_rag_eval#c4：Golden Set（黄金测试集）  最小 20~30 条起步，必须覆盖：可回答（答案能定位到具体 …

## 四、结论

- **检索层**：Recall@5 = 0.867。语义向量对"换说法"的问题（如 MCP 三层角色、RAGAS 指标）召回稳定；
  没命中的题值得逐一看 chunk：是没切好（语义被切断）还是确实语义太远。
- **弃权是生成侧行为**：检索层对不可回答的问题照样返回 top5（向量空间里
  总有"最近"的）；能不能守住"资料不足就说不知道"，取决于 Prompt 的弃权条款
  和模型的指令遵循——这正是分层评估的意义：检索没漏 ≠ 回答可靠。
- **成本结构**：15 次问答 prompt tokens 合计 11699，其中每次都重复发送的 context 占大头；
  上下文工程的"预算"视角（阶段一 W2 延伸）在 RAG 里直接变现为钱。

## 五、复现方式

```bash
pip install chromadb fastembed
python 01Agent/w3day03/minimal_rag.py
```
