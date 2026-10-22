# -*- coding: utf-8 -*-
"""W3 Day2：向量数据库 Chroma 的 CRUD 操作

对应任务：
  1. 安装 Chroma（最简单、适合入门的向量库）
  2. 跑通 CRUD：创建 Collection → 插入向量 → 相似度搜索 → 更新 → 删除
  3. 对比 Chroma 和 Qdrant 的差异（报告第三节）

实验设计（5 幕，按真实使用顺序）：
  幕1 Create   建持久化 Collection（cosine 距离），说明"表"的概念
  幕2 Insert   用 Day1 的 bge-small-zh 语义向量插入 10 条知识卡片（带元数据）
  幕3 Search   相似度搜索 + 元数据过滤（where），并与"暴力遍历"对账：
               向量库返回的 top-k 和自己手算余弦的 top-k 必须一致——
               这一步是为了证明"向量库的本质就是存向量+算相似度"，
               它的增量价值只是 ANN 索引带来的速度，不是更"懂"语义
  幕4 Update   upsert 更新一条向量，验证查询结果随之改变
  幕5 Delete   按 id 删除，验证 count 与查询结果
  彩蛋：统计库内向量与查询的两两余弦分布，直观感受"检索=在几何空间找近邻"

产出：
  - 终端打印每一步的操作与结果
  - chroma_crud_report.md   实验报告（含 Chroma vs Qdrant 对比）
  - chroma_crud_result.json 原始数据

依赖：chromadb（pip install chromadb）、fastembed（语义向量，见 w3day01）。
"""
import sys
import io
import os
import json
import gc
import shutil
import time
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "chroma_crud_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "chroma_crud_result.json")
DB_PATH = os.path.join(BASE_DIR, "chroma_demo_db")  # 持久化目录（每次运行重建）


def wipe_db(tag):
    """删除演示库目录。Windows 专有坑：即使 client.close() 释放了大部分句柄，
    chroma.sqlite3 仍可能被进程内后台线程（异步落盘/遥测）或外部程序
    （杀毒、IDE 索引）短暂占用（WinError 32）。标准对策是"等一下重试"，
    最终仍失败就警告而不是让整个实验以崩溃收尾——报告和数据早已落盘。"""
    for i in range(5):
        try:
            shutil.rmtree(DB_PATH)
            print(f"[clean] 已清理演示库目录（{tag}）")
            return True
        except PermissionError:
            time.sleep(0.4 * (i + 1))
    print(f"[warn] {DB_PATH} 仍被其他进程占用，暂不删除，可稍后手动删（不影响结果）")
    return False

import ragkit  # noqa: E402

# ---- 10 条"知识卡片"：两条主题（rag / agent），检索和过滤演示都用它们 ----
CARDS = [
    {"id": "card-01", "topic": "rag", "text": "RAG 先检索外部资料，再让模型带着证据回答问题。"},
    {"id": "card-02", "topic": "rag", "text": "向量数据库用 ANN 索引加速近邻搜索，牺牲少量精度换速度。"},
    {"id": "card-03", "topic": "rag", "text": "Chunk 大小建议 200 到 800 个 token，重叠取 10% 到 20%。"},
    {"id": "card-04", "topic": "rag", "text": "RRF 按排名融合多路检索结果，k 常取 60。"},
    {"id": "card-05", "topic": "rag", "text": "Rerank 用交叉编码器对查询和文档逐对打分，精度高但慢。"},
    {"id": "card-06", "topic": "agent", "text": "ReAct 循环是思考、行动、观察交替进行，直到给出最终答案。"},
    {"id": "card-07", "topic": "agent", "text": "Function Calling 由模型决定调用哪个工具，代码负责真正执行。"},
    {"id": "card-08", "topic": "agent", "text": "MCP 协议分 Host、Client、Server 三层，工具实现与宿主解耦。"},
    {"id": "card-09", "topic": "agent", "text": "Plan-and-Execute 先拆解任务清单再逐步执行，可全局重规划。"},
    {"id": "card-10", "topic": "agent", "text": "Reflection 让模型自我批判并修改草稿，适合可迭代的质量任务。"},
]

QUERIES = [
    {"q": "怎么把文档切分比较合适", "tag": "chunking", "filter": None},
    {"q": "多智能体是怎么协作的", "tag": "multi-agent", "filter": None},
    {"q": "检索结果怎么融合排序", "tag": "rrf+filter", "filter": {"topic": "rag"}},
]


def brute_force_topk(query_vec, card_vecs, k):
    """暴力遍历版 top-k：和 Chroma 的 ANN 结果对账用。"""
    scored = [(ragkit.cosine(query_vec, v), cid) for cid, v in card_vecs.items()]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def main():
    import chromadb

    # 每次运行重建库，保证实验可复现（真实项目里恰恰相反：持久化就是卖点）
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    emb = ragkit.get_embedder()
    print(f"[embedder] kind={emb.kind}  model={emb.name}")
    trace = {"embedder": {"kind": emb.kind, "name": emb.name},
             "degraded": getattr(emb, "warn", None)}

    print()
    print("=" * 62)
    print("幕 1  Create：创建 Collection（≈ 关系库里的'表'）")
    print("=" * 62)
    # 开库前先清掉上次运行残留的目录（上次清理失败/中断时会有残留，
    # 否则下面 add 固定 id 会撞 DuplicateIDError）；此刻本进程还没开句柄
    if os.path.exists(DB_PATH):
        wipe_db("清残留")
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(
        name="knowledge_cards",
        # hnsw:space=cosine：距离度量用余弦（Chroma 返回 distance = 1 - cosine）
        metadata={"hnsw:space": "cosine"})
    print(f"[ok] collection={col.name}  count={col.count()}  距离度量=cosine")

    print()
    print("=" * 62)
    print("幕 2  Insert：向量化 + 插入（id / 向量 / 原文 / 元数据 四件套）")
    print("=" * 62)
    texts = [c["text"] for c in CARDS]
    vecs = emb.passage_embed(texts)
    col.add(ids=[c["id"] for c in CARDS],
            embeddings=vecs,
            documents=texts,
            metadatas=[{"topic": c["topic"]} for c in CARDS])
    print(f"[ok] 插入 {col.count()} 条，向量维度={len(vecs[0])}")
    print("[note] 每条记录四件套：id（主键）、embedding（向量）、")
    print("       document（原文，可省略）、metadata（过滤/引用用）")

    print()
    print("=" * 62)
    print("幕 3  Search：相似度搜索 + 元数据过滤，并与暴力遍历对账")
    print("=" * 62)
    search_results = []
    for spec in QUERIES:
        qvec = emb.query_embed([spec["q"]])[0]
        extra = {"where": spec["filter"]} if spec["filter"] else {}
        res = col.query(query_embeddings=[qvec], n_results=3, **extra)
        hits = list(zip(res["ids"][0], res["distances"][0], res["documents"][0]))
        print(f"\n[query] {spec['q']}" +
              (f"   (where: {spec['filter']})" if spec["filter"] else ""))
        for cid, dist, doc in hits:
            print(f"   {cid}  dist={dist:.3f} (cos={1 - dist:.3f})  {doc}")

        # 对账：同样的向量，自己暴力算余弦取 top3，和 Chroma 的结果比
        scope = {c["id"]: v for c, v in zip(CARDS, vecs)
                 if not spec["filter"] or c["topic"] == spec["filter"]["topic"]}
        bf = brute_force_topk(qvec, scope, 3)
        bf_ids = [cid for _, cid in bf]
        agree = bf_ids == [cid for cid, _, _ in hits]
        print(f"   [对账] 暴力遍历 top3 = {bf_ids}  与 Chroma 一致: {agree}")
        search_results.append({
            "query": spec["q"], "filter": spec["filter"],
            "chroma": [{"id": i, "distance": d, "cosine": 1 - d, "doc": doc}
                       for i, d, doc in hits],
            "brute_force": [{"id": i, "cosine": round(s, 3)} for s, i in bf],
            "agree": agree,
        })

    print()
    print("=" * 62)
    print("幕 4  Update：upsert 按 id 覆盖，向量换掉后检索结果跟着变")
    print("=" * 62)
    new_text = "Self-RAG 让模型自己判断什么时候需要检索，检索完再自我校验。"
    new_vec = emb.passage_embed([new_text])[0]
    col.upsert(ids=["card-06"], embeddings=[new_vec],
               documents=[new_text], metadatas=[{"topic": "agent"}])
    after = col.get(ids=["card-06"])
    print(f"[upsert] card-06 原文: {after['documents'][0]}")
    qvec = emb.query_embed(["ReAct 循环是什么"])[0]
    res = col.query(query_embeddings=[qvec], n_results=2)
    print(f"[query ] 'ReAct 循环是什么' → top2: "
          f"{list(zip(res['ids'][0], [round(1 - d, 3) for d in res['distances'][0]]))}")
    print("[note] card-06 已换成 Self-RAG 内容，对'ReAct'的相似度应明显下降——")
    print("       向量库里的知识过时时，更新向量即可，不用重训模型")
    update_info = {"old": CARDS[5]["text"], "new": new_text,
                   "top2_after": list(zip(res["ids"][0],
                                          [round(1 - d, 3) for d in res["distances"][0]]))}

    print()
    print("=" * 62)
    print("幕 5  Delete：按 id 删除，count 与查询联动")
    print("=" * 62)
    col.delete(ids=["card-09"])
    print(f"[delete] card-09 后 count={col.count()}")
    res = col.query(query_embeddings=emb.query_embed(["任务拆解执行"]), n_results=2)
    print(f"[query ] '任务拆解执行' → top2: {res['ids'][0]}   # card-09 不再出现")
    delete_info = {"count_after": col.count(),
                   "top2_after_delete": res["ids"][0]}

    payload = {
        "meta": {"generated_at": datetime.datetime.now().isoformat(),
                 "chroma_version": __import__("chromadb").__version__,
                 **trace},
        "cards": CARDS,
        "search": search_results,
        "update": update_info,
        "delete": delete_info,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")
    # 默认清掉演示库，避免把二进制目录带进 git；想留着重看就加 --keep-db
    if "--keep-db" not in sys.argv:
        client.close()   # 释放 SQLite / HNSW 文件句柄（Windows 删目录的前提）
        gc.collect()     # 兜底：回收尚未释放的连接对象
        wipe_db("--keep-db 可保留")


def build_report(p):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agree_n = sum(1 for s in p["search"] if s["agree"])
    L = []
    a = L.append
    a("# 向量数据库 CRUD 实验报告（Chroma，W3 Day2）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- Chroma 版本：`{p['meta']['chroma_version']}`（PersistentClient 本地持久化）")
    a(f"- 向量：`{p['meta']['embedder']['name']}`（{p['meta']['embedder']['kind']}，"
      f"与 w3day01 同一套，查询/文档同模型）")
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    a("")
    a("## 一、向量数据库解决什么问题")
    a("")
    a("RAG 要在几万到几亿条向量里找 top-k 近邻，暴力遍历是 O(N·d)，库大了扛不住。")
    a("向量数据库用 ANN（近似最近邻）索引换速度：Chroma 内置 HNSW 图索引，")
    a("允许极小的召回损失，换几个数量级的提速。但 Day2 的对账实验说明：")
    a("**它算的相似度和你手算的余弦是同一个东西**，增量价值在索引速度，不在语义理解。")
    a("")
    a("## 二、CRUD 全流程记录")
    a("")
    a("| 步骤 | 操作 | 关键点 |")
    a("|---|---|---|")
    a("| Create | `get_or_create_collection(hnsw:space=cosine)` | Collection ≈ 表；距离度量建表时定 |")
    a(f"| Insert | `add(ids, embeddings, documents, metadatas)`，{len(p['cards'])} 条 | id/向量/原文/元数据 四件套 |")
    a("| Search | `query(query_embeddings, n_results, where)` | 返回 distance=1−cosine；where 做 元数据过滤 |")
    a("| Update | `upsert(...)` 覆盖 card-06 | 知识过时→更新向量即可，无需重训 |")
    a("| Delete | `delete(ids=[...])` | count 与查询结果即时联动 |")
    a("")
    a("### 2.1 相似度搜索结果（含元数据过滤）")
    a("")
    for s in p["search"]:
        flt = f"（where={s['filter']}）" if s["filter"] else ""
        a(f"**查询：{s['query']}**{flt}")
        a("")
        a("| id | cosine | 原文 |")
        a("|---|---|---|")
        for h in s["chroma"]:
            a(f"| {h['id']} | {h['cosine']:.3f} | {h['doc']} |")
        a("")
        a("暴力遍历对账：" + ("✅ top-k 完全一致" if s["agree"] else
                             f"❌ 不一致：{[x['id'] for x in s['brute_force']]}"))
        a("")
    a(f"三组查询对账结果：**{agree_n}/3 与暴力遍历完全一致**——"
      "向量库的检索语义就是余弦 top-k，ANN 只影响速度和极端情况下的排序抖动。")
    a("")
    a("### 2.2 更新与删除")
    a("")
    a(f"- upsert card-06：\"{p['update']['old']}\" → \"{p['update']['new']}\"")
    a(f"- 换向量后再问\"ReAct 循环是什么\"，top2 = {p['update']['top2_after']}")
    a(f"- 删除 card-09 后 count = {p['delete']['count_after']}，"
      f"相关查询 top2 = {p['delete']['top2_after_delete']}")
    a("")
    a("## 三、Chroma vs Qdrant 怎么选")
    a("")
    a("| 维度 | Chroma | Qdrant |")
    a("|---|---|---|")
    a("| 定位 | 嵌入式库（像 SQLite） | 独立服务（像 PostgreSQL） |")
    a("| 安装 | `pip install chromadb` 即用 | Docker 部署 / 云服务 |")
    a("| 适合 | 本地开发、原型、小项目 | 生产环境、中大规模 |")
    a("| 元数据过滤 | where 条件（够用） | 更强：范围/嵌套/全文过滤 |")
    a("| 混合检索 | 需自己拼 BM25 | 内置 sparse/dense 混合 |")
    a("| 分布式/副本 | 无 | 支持 |")
    a("| 运维成本 | 零 | 需要管服务、备份、监控 |")
    a("")
    a("一句话：**原型期用 Chroma 快速起步，生产化了再按运维能力和规模迁 Qdrant/pgvector**；")
    a("两者 API 概念同构（collection/point、upsert、filter、top-k），迁移成本主要是工程适配。")
    a("")
    a("## 四、复现方式")
    a("")
    a("```bash")
    a("pip install chromadb fastembed")
    a("python 01Agent/w3day02/chroma_crud.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
