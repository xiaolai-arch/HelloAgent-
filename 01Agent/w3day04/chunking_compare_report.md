# Chunking 策略对比实验报告（W3 Day4）

- 生成时间：2026-10-21 16:22:40
- 语料：15 篇（与 Day3 完全同一份）；黄金集 15 题
- 向量：`BAAI/bge-small-zh-v1.5`（semantic）；检索统一 top5；全程未调 LLM

## 一、三种切分策略原理

| 策略 | 做法 | 优点 | 风险 |
|---|---|---|---|
| 固定长度 | 每 N 字符切一刀，带 overlap | 快、均匀、可复现 | 无视语义边界，句子被拦腰切断 |
| 递归字符 | 段落→句子→子句逐级回退找边界 | 通用默认，边界完整 | 跨段逻辑仍可能被切开 |
| 语义切分 | 相邻句向量相似度骤降处断开 | 贴着主题边界切 | 要先给每句编码，成本高；阈值敏感 |

## 二、总成绩单（同一把尺子量 6 种配置）

| 配置 | chunk 数 | 平均长度 | Recall@5 | MRR | HitRate |
|---|---|---|---|---|---|
| fixed-300-50 | 65 | 271.5 | 1.0 | 0.583 | 0.733 |
| fixed-300-0 | 59 | 256.7 | 1.0 | 0.678 | 0.8 |
| recursive-150 | 132 | 113.6 | 0.667 | 0.35 | 0.467 |
| recursive-300 | 72 | 209.2 | 0.867 | 0.606 | 0.733 |
| recursive-500 | 43 | 351.6 | 0.867 | 0.733 | 0.733 |
| semantic | 58 | 254.0 | 0.933 | 0.578 | 0.8 |

本组实验的冠军：**fixed-300-0**。

## 三、chunk_size 与 overlap 的影响

- **大小（recursive 150/300/500）**：Recall@5 分别为 0.667 / 0.867 / 0.867。太小→一句话被劈成两半、单 chunk 信息不足；太大→一个 chunk 里混进多个主题，向量被"平均"稀释，且进上下文后挤占预算。经验区间 200~800 token 的说法在本次小语料上方向一致。
- **overlap（fixed 300，重叠 50 vs 0）**：Recall@5 1.0 vs 1.0，MRR 0.583 vs 0.678。重叠让跨在边界上的信息在两个 chunk 里各有一份完整拷贝，是纯字面切分的重要止血手段；代价是存储和索引各多一份。
- **语义切分**：Recall@5 0.933，平均块长 254.0 字。它贴着主题边界切，块与块之间差异大（有的长有的短）；代价是每句都要编码一次，语料大时成本可观。

## 四、定性分析

### 4.1 "拦腰切"实锤（断点两侧拼不成完整句读的）

fixed-300-50 拦腰切断句子 **10 处**（示例：…。
- **Decode ▌ 面套残差连接（Resid…）；recursive-300 只有 **7 处**。固定切分把一句完整的话切成两半时，两个 chunk 各剩半句，哪个都检索不到完整语义——这就是它需要 overlap 止血的原因。

### 4.2 多跳问题为什么难（gold 关键词分散度）

- **Q1**（fact）：关键词分散在 7 个 chunk——并行 → 01_transformer#c3,02_attention#c3,12_react_function_calling#c3,12_react_function_calling#c4,14_agent_patterns#c1,14_agent_patterns#c3,14_agent_patterns#c4；长程依赖 → 01_transformer#c3
- **Q2**（fact）：关键词分散在 2 个 chunk——方差 → 02_attention#c1；饱和 → 02_attention#c1,08_hybrid_rerank#c0
- **Q3**（fact）：关键词分散在 7 个 chunk——分布 → 01_transformer#c2,03_tokenizer_sampling#c1,03_tokenizer_sampling#c2,06_vector_db#c3；softmax → 01_transformer#c2,02_attention#c0,02_attention#c1,02_attention#c3,03_tokenizer_sampling#c2
- **Q5**（fact）：关键词分散在 4 个 chunk——CLIP ViT → 04_moe_multimodal#c3；投影 → 01_transformer#c2,04_moe_multimodal#c1,04_moe_multimodal#c2,04_moe_multimodal#c3
- **Q6**（fact）：关键词分散在 2 个 chunk——60 → 08_hybrid_rerank#c1,15_helloai_policy#c2；量纲 → 08_hybrid_rerank#c1
- **Q7**（fact）：关键词分散在 6 个 chunk——Host → 13_mcp#c1,13_mcp#c3；Client → 13_mcp#c1；Server → 13_mcp#c0,13_mcp#c1,13_mcp#c2,13_mcp#c3,13_mcp#c4,13_mcp#c5
- **Q8**（fact）：关键词分散在 2 个 chunk——Faithfulness → 09_rag_eval#c2,09_rag_eval#c5；Answer Relevancy → 09_rag_eval#c2；Context Precision → 09_rag_eval#c2,09_rag_eval#c5；Context Recall → 09_rag_eval#c2,09_rag_eval#c5
- **Q11**（multi-hop）：关键词分散在 6 个 chunk——3 个工作日 → 15_helloai_policy#c0；600 → 15_helloai_policy#c2；100 → 03_tokenizer_sampling#c1,07_rag_pipeline#c2,08_hybrid_rerank#c3,15_helloai_policy#c3
- **Q12**（multi-hop）：关键词分散在 5 个 chunk——tools → 10_prompt_engineering#c3,12_react_function_calling#c2,12_react_function_calling#c4,13_mcp#c2,13_mcp#c3；tool_calls → 12_react_function_calling#c2,12_react_function_calling#c4；role=tool → 12_react_function_calling#c2,12_react_function_calling#c4
- **Q15**（fact）：关键词分散在 9 个 chunk——静态 → 07_rag_pipeline#c4,12_react_function_calling#c1；SQL → 06_vector_db#c3,07_rag_pipeline#c4；延迟 → 05_embedding#c3,07_rag_pipeline#c4,08_hybrid_rerank#c2,10_prompt_engineering#c1,11_context_engineering#c1,11_context_engineering#c2,14_agent_patterns#c2

答案需要的信息物理上住在不同 chunk 里，任何单一 chunk 都不完整：
这时该修的不是 embedding 模型，而是切分粒度（父子块/small-to-big：
小块检索、大块喂给模型）或 Contextual Retrieval（给 chunk 预置文档背景）。

## 五、结论（按本组数据说话）

1. **chunk 太小明确有害**：recursive-150（132 个碎块）Recall@5 只有 0.667，是全场最差——单块信息太薄，哪个块都凑不齐 min_keyword_hits 的相关性门槛。
2. **大窗口讨好 Recall，排序质量看 MRR**：fixed-300 系列用"大窗+含混主题"拿下最高 Recall@5（1.0），但那是块大、覆盖广带来的；真正衡量"最好的结果排多靠前"的 MRR，冠军是 recursive-500（0.733）。只看 Recall 会得出"切得越粗越好"的错误结论。
3. **overlap 不是免费午餐**：fixed-300 的 MRR 反而 overlap=50（0.583）低于 overlap=0（0.678）——重复内容进多个 chunk，排名被同一内容的拷贝稀释。overlap 是硬切的止血带，不是普遍增益。
4. **语义切分是"边界质量"和"块大小"的折中**：Recall@5 0.933、平均块长 254.0 字，在不需要人工调参的情况下拿到了接近最好的综合表现；代价是每句都要过一遍 embedding（语料大时成本可观）。
5. **评估驱动调参**：配置没有万能冠军，把"同向量、同检索、同黄金集"的
   对照流程跑一遍、Recall 和 MRR 一起看，是唯一可靠的方法。

## 六、复现方式

```bash
pip install chromadb fastembed
python 01Agent/w3day04/chunking_compare.py
```
