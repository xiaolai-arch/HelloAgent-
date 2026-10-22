# 向量数据库 CRUD 实验报告（Chroma，W3 Day2）

- 生成时间：2026-10-22 15:59:22
- Chroma 版本：`1.5.9`（PersistentClient 本地持久化）
- 向量：`BAAI/bge-small-zh-v1.5`（semantic，与 w3day01 同一套，查询/文档同模型）

## 一、向量数据库解决什么问题

RAG 要在几万到几亿条向量里找 top-k 近邻，暴力遍历是 O(N·d)，库大了扛不住。
向量数据库用 ANN（近似最近邻）索引换速度：Chroma 内置 HNSW 图索引，
允许极小的召回损失，换几个数量级的提速。但 Day2 的对账实验说明：
**它算的相似度和你手算的余弦是同一个东西**，增量价值在索引速度，不在语义理解。

## 二、CRUD 全流程记录

| 步骤 | 操作 | 关键点 |
|---|---|---|
| Create | `get_or_create_collection(hnsw:space=cosine)` | Collection ≈ 表；距离度量建表时定 |
| Insert | `add(ids, embeddings, documents, metadatas)`，10 条 | id/向量/原文/元数据 四件套 |
| Search | `query(query_embeddings, n_results, where)` | 返回 distance=1−cosine；where 做 元数据过滤 |
| Update | `upsert(...)` 覆盖 card-06 | 知识过时→更新向量即可，无需重训 |
| Delete | `delete(ids=[...])` | count 与查询结果即时联动 |

### 2.1 相似度搜索结果（含元数据过滤）

**查询：怎么把文档切分比较合适**

| id | cosine | 原文 |
|---|---|---|
| card-05 | 0.587 | Rerank 用交叉编码器对查询和文档逐对打分，精度高但慢。 |
| card-09 | 0.442 | Plan-and-Execute 先拆解任务清单再逐步执行，可全局重规划。 |
| card-03 | 0.435 | Chunk 大小建议 200 到 800 个 token，重叠取 10% 到 20%。 |

暴力遍历对账：✅ top-k 完全一致

**查询：多智能体是怎么协作的**

| id | cosine | 原文 |
|---|---|---|
| card-08 | 0.548 | MCP 协议分 Host、Client、Server 三层，工具实现与宿主解耦。 |
| card-07 | 0.422 | Function Calling 由模型决定调用哪个工具，代码负责真正执行。 |
| card-09 | 0.419 | Plan-and-Execute 先拆解任务清单再逐步执行，可全局重规划。 |

暴力遍历对账：✅ top-k 完全一致

**查询：检索结果怎么融合排序**（where={'topic': 'rag'}）

| id | cosine | 原文 |
|---|---|---|
| card-04 | 0.668 | RRF 按排名融合多路检索结果，k 常取 60。 |
| card-01 | 0.471 | RAG 先检索外部资料，再让模型带着证据回答问题。 |
| card-05 | 0.430 | Rerank 用交叉编码器对查询和文档逐对打分，精度高但慢。 |

暴力遍历对账：✅ top-k 完全一致

三组查询对账结果：**3/3 与暴力遍历完全一致**——向量库的检索语义就是余弦 top-k，ANN 只影响速度和极端情况下的排序抖动。

### 2.2 更新与删除

- upsert card-06："ReAct 循环是思考、行动、观察交替进行，直到给出最终答案。" → "Self-RAG 让模型自己判断什么时候需要检索，检索完再自我校验。"
- 换向量后再问"ReAct 循环是什么"，top2 = [('card-07', 0.462), ('card-08', 0.46)]
- 删除 card-09 后 count = 9，相关查询 top2 = ['card-07', 'card-10']

## 三、Chroma vs Qdrant 怎么选

| 维度 | Chroma | Qdrant |
|---|---|---|
| 定位 | 嵌入式库（像 SQLite） | 独立服务（像 PostgreSQL） |
| 安装 | `pip install chromadb` 即用 | Docker 部署 / 云服务 |
| 适合 | 本地开发、原型、小项目 | 生产环境、中大规模 |
| 元数据过滤 | where 条件（够用） | 更强：范围/嵌套/全文过滤 |
| 混合检索 | 需自己拼 BM25 | 内置 sparse/dense 混合 |
| 分布式/副本 | 无 | 支持 |
| 运维成本 | 零 | 需要管服务、备份、监控 |

一句话：**原型期用 Chroma 快速起步，生产化了再按运维能力和规模迁 Qdrant/pgvector**；
两者 API 概念同构（collection/point、upsert、filter、top-k），迁移成本主要是工程适配。

## 四、复现方式

```bash
pip install chromadb fastembed
python 01Agent/w3day02/chroma_crud.py
```
