# RAG 评估：分层指标与 RAGAS

## 为什么要分层

"答得像不像"没法定位问题。RAG 出错只有三个来源：没检索到（召回）、检索到但排序差（排序）、检索对了但答案写错（生成）。分层评估才能把锅甩对位置。

## 检索层指标

- **Recall@k**：相关的 chunk 是否出现在前 k 条里（不看排名，只看在不在）。
- **Precision@k**：前 k 条里有几条是相关的。
- **MRR**（Mean Reciprocal Rank）：第一个相关结果排名的倒数，排名第一得 1 分、第二得 0.5 分，衡量"最好的结果排得有多靠前"。
- **Hit Rate**：至少命中一条相关 chunk 的问题比例。

## 生成层指标（RAGAS 四大件）

- **Faithfulness（忠实度）**：答案中的每个论断是否都被上下文支持。低 = 模型在编造或超出资料发挥。
- **Answer Relevancy（答案相关性）**：答案是否真的回答了问题。低 = 答非所问或跑题。
- **Context Precision（上下文精确率）**：检索到的资料里，相关的是否排在前面——考的是排序质量。
- **Context Recall（上下文召回率）**：标准答案需要的信息，上下文里是否都覆盖了——低 = 检索层漏了东西，生成再好也白搭。

RAGAS 实现这些指标主要靠 LLM-as-judge：让一个强模型按固定 rubric 对 (question, context, answer, ground_truth) 打分。它的坑：位置偏置（偏好前面的答案）、长度偏置（偏好长答案）、自我偏好、对 prompt 措辞敏感、judge 自己也会判错。工程上要用人工标注的小集合校准 judge，固定评分 prompt 和模型版本。

## Golden Set（黄金测试集）

最小 20~30 条起步，必须覆盖：可回答（答案能定位到具体 chunk）、不可回答（资料里没有，考察弃权）、多跳（要组合 2~3 个 chunk）、精确词（错误码/型号，考察 BM25）、时效冲突。每条包含 question、gold_answer、gold 来源标注、类型。没有 golden set，一切优化都是凭感觉。

## 错误分析方法论

指标低了按固定顺序排查：先看 Context Recall（漏没漏）→ 再看检索排序（Context Precision）→ 最后看生成忠实度（Faithfulness）。先解决"检索不到"和"上下文太吵"，再考虑 GraphRAG、Agentic RAG 这些高级架构——大多数 RAG 问题不是缺高级架构，而是解析、切分、评估没做好。
