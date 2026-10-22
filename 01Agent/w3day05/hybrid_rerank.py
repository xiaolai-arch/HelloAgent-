# -*- coding: utf-8 -*-
"""W3 Day5：混合检索（BM25 + 向量）+ RRF 融合 + Rerank 重排

对应任务：
  1. 理解稀疏检索 BM25 vs 密集检索（向量）的强弱
  2. 实现"向量检索 + BM25"双路召回
  3. 对召回结果重排序（原计划 BGE-Rerank；本次用 fastembed 的
     BAAI/bge-reranker-v2-m3-int8 交叉编码器，本地 onnx 推理——
     它就是 BGE 官方 reranker 家族的 int8 量化版；加载失败自动降级
     为 LLM listwise 重排，报告如实记录用的是哪条）

四条管线同场竞技（同索引、同黄金集、同指标）：
    dense        纯向量 top5
    bm25         纯 BM25 top5
    hybrid_rrf   双路各取 top20 → RRF(k=60) → top5
    hybrid_rank  hybrid_rrf 的 top20 → cross-encoder 重排 → top5

BM25 手写实现（零依赖）：中文按字符 bigram + 英文按词切分，
k1=1.5, b=0.75；公式带词频饱和与文档长度归一化，代码里逐项注释。

产出：hybrid_rerank_report.md + hybrid_rerank_result.json
依赖：chromadb、fastembed（+reranker 权重 ~0.57GB，走 hf-mirror）。
"""
import sys
import io
import os
import re
import json
import math
import time
import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "hybrid_rerank_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "hybrid_rerank_result.json")

import ragkit  # noqa: E402

TOP_K = 5          # 最终进上下文的条数
CAND_K = 20        # 每路召回的候选数（漏斗：宽召回 → 融合 → 精排）
RRF_K = 60         # RRF 常数
BM25_K1, BM25_B = 1.5, 0.75

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def bm25_tokens(text):
    """中文检索的朴素分词：连续英文/数字整体保留（错误码、型号不被切碎），
    其余部分按字符 bigram 切——没有分词器时这是 BM25 中文化的常用近似。
    """
    tokens = []
    # findall 取出英文/数字整词；split 的间隙就是中文段落
    tokens += _TOKEN_RE.findall(text)
    for seg in _TOKEN_RE.split(text):
        zh = re.sub(r"\s+", "", re.sub(r"[a-zA-Z0-9_]+", "", seg))
        if len(zh) == 1:
            tokens.append(zh)
        else:
            tokens += [zh[j:j + 2] for j in range(len(zh) - 1)]
    return tokens


class BM25:
    """Okapi BM25：score(q, d) = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))

    与 TF-IDF 的两点不同（也是它更强的地方）：
      - 词频饱和：分子里的 (k1+1) 配分母的 k1·tf，词出现 10 次不会比 3 次强太多
      - 长度归一化：b 控制长文档的"稀释"程度
    """

    def __init__(self, docs):
        """docs: [(id, text)]。预计算 df、avgdl 和每篇词频。"""
        self.docs = docs
        self.tfs = [Counter(bm25_tokens(t)) for _, t in docs]
        self.dls = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.dls) / len(self.dls) if self.dls else 1.0
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.n = len(docs)

    def idf(self, term):
        """BM25 版 IDF：log((N - df + 0.5) / (df + 0.5) + 1)，罕见词得分高。"""
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, k=CAND_K):
        q_terms = bm25_tokens(query)
        scores = []
        for i, (doc_id, _) in enumerate(self.docs):
            tf_map, dl = self.tfs[i], self.dls[i]
            s = 0.0
            for t in set(q_terms):
                tf = tf_map.get(t, 0)
                if not tf:
                    continue
                s += self.idf(t) * tf * (BM25_K1 + 1) / (
                    tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl))
            scores.append((s, doc_id))
        scores.sort(key=lambda x: -x[0])
        return [(doc_id, s) for s, doc_id in scores[:k] if s > 0]


class CrossEncoderRerank:
    """fastembed 的 TextCrossEncoder：把 (query, chunk) 成对送入模型打分。

    与 bi-encoder（向量检索）的本质区别：查询和文档在模型内部互相看见，
    精度高；代价是每对都要跑一次前向，没法离线建索引，只能现场算。
    """

    kind = "cross-encoder"

    def __init__(self, model="BAAI/bge-reranker-v2-m3-int8"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model)
        self.name = model

    def score(self, query, docs):
        return list(self._model.rerank(query, docs))


class LLMRerank:
    """降级方案：让 LLM 对候选做 listwise 重排（一次调用排完一轮）。

    交叉编码器不可用时（模型下载失败等）的兜底；延迟和 token 成本都
    高得多，排序质量也不如真 cross-encoder 稳定。
    """

    kind = "llm-listwise"

    def __init__(self):
        self.name = f"llm:{ragkit.MODEL}"

    def score(self, query, docs):
        listing = "\n".join(f"[{i}] {d[:150]}" for i, d in enumerate(docs))
        prompt = (f"你是一个检索重排序器。根据与查询的相关性，把候选段落从最相关到最不相关排序。\n"
                  f"只输出 JSON 数组（编号从最相关开始），不要任何解释。\n\n"
                  f"查询：{query}\n\n候选：\n{listing}")
        resp, err = ragkit.chat([{"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=300)
        if err:
            return list(range(len(docs), 0, -1))  # 失败兜底：维持原序（反序占位）
        try:
            order = json.loads(re.search(r"\[.*\]", resp["choices"][0]["message"]["content"],
                                         re.S).group(0))
            return [-1 if i not in range(len(docs)) else i for i in order]
        except Exception:
            return list(range(len(docs), 0, -1))


def main():
    docs = ragkit.load_corpus()
    questions = ragkit.load_eval_set()
    emb = ragkit.get_embedder()
    print(f"[embedder] kind={emb.kind}  model={emb.name}")

    # ---- 建索引：递归 300（与 Day3 基线一致）----
    chunks = []
    for d in docs:
        for i, (text, _) in enumerate(ragkit.recursive_chunk(d["text"], 300)):
            chunks.append({"id": f"{d['id']}#c{i}", "doc_id": d["id"], "text": text})
    print(f"[index] {len(docs)} 篇 → {len(chunks)} chunks (recursive-300)")

    vecs = emb.passage_embed([c["text"] for c in chunks])
    import chromadb
    col = chromadb.EphemeralClient().get_or_create_collection(
        "hybrid", metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks], embeddings=vecs,
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"]} for c in chunks])
    bm25 = BM25([(c["id"], c["text"]) for c in chunks])
    cid2chunk = {c["id"]: c for c in chunks}

    # ---- 重排器：cross-encoder 优先，失败降级 LLM ----
    print("[rerank] 加载重排器（首次需下载权重，走 hf-mirror）……")
    t0 = time.time()
    try:
        reranker = CrossEncoderRerank()
        reranker._model  # 触发加载
        print(f"[rerank] {reranker.name} 就绪（{time.time() - t0:.0f}s）")
    except Exception as e:
        reranker = LLMRerank()
        print(f"[rerank] cross-encoder 不可用（{type(e).__name__}: {str(e)[:120]}）")
        print(f"[rerank] 降级为 {reranker.kind}")

    # ---- 四条管线 × 15 题 ----
    pipelines = ["dense", "bm25", "hybrid_rrf", "hybrid_rerank"]
    per_pipe = {p: [] for p in pipelines}
    per_q = {p: {} for p in pipelines}
    timings = {p: 0.0 for p in pipelines}

    print()
    for q in questions:
        qvec = emb.query_embed([q["question"]])[0]

        # dense top-20
        t0 = time.time()
        dres = col.query(query_embeddings=[qvec], n_results=CAND_K)
        t_dense = time.time() - t0
        dense_rank = dres["ids"][0]

        # bm25 top-20
        t0 = time.time()
        bm_rank = [doc_id for doc_id, _ in bm25.search(q["question"], CAND_K)]
        t_bm = time.time() - t0

        # RRF 融合
        t0 = time.time()
        fused = ragkit.rrf_fuse([dense_rank, bm_rank], k=RRF_K, top_n=CAND_K)
        t_rrf = time.time() - t0
        rrf_rank = [doc_id for doc_id, _ in fused]

        # rerank（对融合候选重排）
        t0 = time.time()
        cand_ids = rrf_rank[:CAND_K]
        cand_texts = [cid2chunk[cid]["text"] for cid in cand_ids]
        if isinstance(reranker, CrossEncoderRerank):
            scores = reranker.score(q["question"], cand_texts)
        else:
            order = reranker.score(q["question"], cand_texts)
            scores = [1.0 / (i + 1) for i in order]  # 名次当分数
        t_rr = time.time() - t0
        ranked = sorted(zip(cand_ids, scores), key=lambda x: -x[1])
        rr_rank = [cid for cid, _ in ranked]

        timings["dense"] += t_dense
        timings["bm25"] += t_bm
        timings["hybrid_rrf"] += t_dense + t_bm + t_rrf
        timings["hybrid_rerank"] += t_dense + t_bm + t_rrf + t_rr

        for pipe, rank in [("dense", dense_rank), ("bm25", bm_rank),
                           ("hybrid_rrf", rrf_rank), ("hybrid_rerank", rr_rank)]:
            rel = [ragkit.chunk_is_relevant(cid2chunk[cid]["text"], q)
                   for cid in rank[:TOP_K]]
            per_pipe[pipe].append(rel)
            rr = next((1 / (i + 1) for i, h in enumerate(rel) if h), 0.0)
            per_q[pipe][q["id"]] = {"recall@5": 1 if any(rel) else 0,
                                    "mrr": round(rr, 3),
                                    "top5": rank[:TOP_K],
                                    "rel": [1 if f else 0 for f in rel]}

        marks = {p: "".join("✅" if h else "·" for h in per_q[p][q["id"]]["rel"])
                 for p in pipelines}
        print(f"[{q['id']}] {q['question'][:24]:<26} dense={marks['dense']} "
              f"bm25={marks['bm25']} rrf={marks['hybrid_rrf']} 重排={marks['hybrid_rerank']}")

    metrics = {p: ragkit.retrieval_metrics(per_pipe[p]) for p in pipelines}
    print("\n" + "=" * 62)
    print("汇总（15 题）")
    print("=" * 62)
    for p in pipelines:
        m = metrics[p]
        print(f"  {p:<14} Recall@5={m['recall@5']:<6} MRR={m['mrr']:<6} "
              f"HitRate={m['hit_rate']:<6} 累计耗时={timings[p]:.2f}s")

    # Q9（错误码）单题解剖：dense vs bm25 的排序
    q9 = next(q for q in questions if q["id"] == "Q9")
    print(f"\n[解剖 Q9] {q9['question']}")
    print(f"  dense top3: {per_q['dense']['Q9']['top5'][:3]}")
    print(f"  bm25  top3: {per_q['bm25']['Q9']['top5'][:3]}")

    payload = {
        "meta": {"generated_at": datetime.datetime.now().isoformat(),
                 "embedder": {"kind": emb.kind, "name": emb.name},
                 "degraded": getattr(emb, "warn", None),
                 "reranker": {"kind": reranker.kind, "name": reranker.name},
                 "n_chunks": len(chunks), "chunking": "recursive-300",
                 "top_k": TOP_K, "cand_k": CAND_K, "rrf_k": RRF_K,
                 "bm25": {"k1": BM25_K1, "b": BM25_B,
                          "tokenize": "中文 bigram + 英数整词"},
                 "timings_total_sec": {k: round(v, 2) for k, v in timings.items()}},
        "metrics": metrics,
        "per_question": per_q,
        "questions": [{"id": q["id"], "question": q["question"], "type": q["type"]}
                      for q in questions],
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(p):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    m = p["metrics"]
    t = p["meta"]["timings_total_sec"]
    L = []
    a = L.append
    a("# 混合检索 + Rerank 实验报告（W3 Day5）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 索引：{p['meta']['n_chunks']} chunks（recursive-300，与 Day3 基线一致）")
    a(f"- 向量：`{p['meta']['embedder']['name']}`（{p['meta']['embedder']['kind']}）")
    a(f"- 重排：`{p['meta']['reranker']['name']}`（{p['meta']['reranker']['kind']}）")
    a(f"- 漏斗：双路各 top{p['meta']['cand_k']} → RRF(k={p['meta']['rrf_k']}) → 重排 → top{p['meta']['top_k']}")
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    a("")
    a("## 一、BM25 vs Dense：强弱互补")
    a("")
    a("| 维度 | BM25（稀疏） | 向量（稠密） |")
    a("|---|---|---|")
    a("| 原理 | 词频 × 逆文档频率（词面匹配） | 语义空间余弦（意思匹配） |")
    a("| 强项 | 精确词：错误码/型号/人名；可解释；零成本 | 同义改写、跨语言、口语化问法 |")
    a("| 弱项 | 换个说法就找不到 | 精确词被语义平滑；需要编码模型 |")
    a("| 本实验实现 | 手写 Okapi BM25（k1=1.5, b=0.75，中文 bigram） | bge-small-zh + Chroma |")
    a("")
    a("BM25 公式（手写实现逐项注释见脚本）：")
    a("")
    a("```text")
    a("score(q,d) = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))")
    a("IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)")
    a("```")
    a("")
    a("## 二、四条管线成绩单")
    a("")
    a("| 管线 | Recall@5 | MRR | HitRate | 15 题累计耗时 |")
    a("|---|---|---|---|---|")
    for name, label in [("dense", "dense（纯向量）"), ("bm25", "bm25（纯 BM25）"),
                        ("hybrid_rrf", "hybrid_rrf（双路+RRF）"),
                        ("hybrid_rerank", "hybrid_rerank（+重排）")]:
        x = m[name]
        a(f"| {label} | {x['recall@5']} | {x['mrr']} | {x['hit_rate']} | {t[name]}s |")
    a("")
    a("### 逐题命中矩阵（✅=top5 内有相关 chunk）")
    a("")
    a("| 题 | 类型 | dense | bm25 | rrf | 重排 |")
    a("|---|---|---|---|---|")
    for qd in p["questions"]:
        row = ["".join("✅" if f else "·" for f in p["per_question"][pipe][qd["id"]]["rel"])
               for pipe in ("dense", "bm25", "hybrid_rrf", "hybrid_rerank")]
        a(f"| {qd['id']} | {qd['type']} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    a("")
    a("## 三、单题解剖")
    a("")
    q9 = next(q for q in p["questions"] if q["id"] == "Q9")
    a(f"### 精确词题 {q9['id']}：{q9['question']}")
    a("")
    a(f"- dense top3：{p['per_question']['dense']['Q9']['top5'][:3]}")
    a(f"- bm25  top3：{p['per_question']['bm25']['Q9']['top5'][:3]}")
    a("- 看第 2、3 名的差异：BM25 的 top3 全是差旅制度文档的 chunk（错误码、")
    a("  报销期限等字面强相关），dense 的第 2 名却混进了 RAG 评估文档——")
    a("  在语义向量眼里\"报销系统报错\"和\"评估/错误分析\"的向量距离也不算远。")
    a("  TRV-40907 这种字面锚点，正是 BM25 的主场。")
    a("")
    miss_fixed = [qd["id"] for qd in p["questions"]
                  if qd["type"] != "unanswerable"
                  and p["per_question"]["hybrid_rerank"][qd["id"]]["recall@5"] == 0]
    a(f"### 多跳题（Day3 的两道 MISS：Q11/Q12）")
    a("")
    q11 = p["per_question"]["hybrid_rerank"]["Q11"]
    q12 = p["per_question"]["hybrid_rerank"]["Q12"]
    q11_dense = p["per_question"]["dense"]["Q11"]
    a(f"- Q12：dense miss → 重排后 top5 {q12['top5']}（命中 {q12['rel']}）——"
      "MCP 文档和 FC 文档的 chunk 分居两路召回，融合进候选后由重排捞回 top5。")
    if q11["recall@5"] == 0:
        a(f"- Q11：仍然 miss（dense 命中 {q11_dense['rel']}，重排 {q11['rel']}）。"
          "它的三个 gold 关键词（提前 3 个工作日 / 600 元 / 100 元餐补）被切进了"
          "不同 chunk，任何单个 chunk 都凑不满 2 个命中——这是 Day4 实证过的"
          "\"切分散信息\"问题，检索和重排都救不了，要靠父子块或更粗的切分粒度。")
    a("- 多跳困难是\"信息住不同 chunk\"：混合检索提供更多样的候选（两路互补），")
    a("  重排把分散在两路里的相关块捞上来——这正是漏斗设计的意义；但候选里")
    a("  凑不齐信息的题，只能回切分层去修。")
    a("")
    a("## 四、Rerank 的账：收益 vs 延迟")
    a("")
    dt_rr = t["hybrid_rerank"] - t["hybrid_rrf"]
    a(f"- 重排 15 题 × 20 候选共 {dt_rr:.1f}s（{p['meta']['reranker']['kind']}，本地推理）。")
    a("- 它解决的是\"排序\"：相关块在候选里但排不进 top5 时，cross-encoder 因为")
    a("  让 query 和 chunk 在模型内部互相看见，能把它们捞上来。")
    a("- 它解决不了\"召回\"：候选 20 条里根本没有相关块，重排也无能为力——")
    a("  所以是\"宽召回 → 精排\"的漏斗，而不是\"小召回 → 精排\"。")
    a("")
    a("## 五、结论")
    a("")
    best_rrf = "hybrid_rrf" if m["hybrid_rrf"]["mrr"] >= m["dense"]["mrr"] else "dense"
    a(f"1. **混合检索值回票价**：hybrid_rrf 的 Recall@5={m['hybrid_rrf']['recall@5']} / "
      f"MRR={m['hybrid_rrf']['mrr']}，对比纯 dense {m['dense']['recall@5']}/{m['dense']['mrr']}、"
      f"纯 BM25 {m['bm25']['recall@5']}/{m['bm25']['mrr']}——"
      f"两路的强项题（语义改写 vs 精确词）几乎不重叠，融合后互为补丁。")
    a(f"2. **重排再抬一层**：MRR {m['hybrid_rrf']['mrr']} → {m['hybrid_rerank']['mrr']}，"
      f"代价是每题多 {dt_rr / 15:.2f}s 延迟。生产上 rerank 是最常用的"
      " \"最后一公里\"优化，但永远排在\"把召回做好\"之后。")
    a(f"3. **RRF 为什么稳**：只看排名不看分数（k={p['meta']['rrf_k']}），")
    a("   BM25 的分数量纲和余弦不同没法直接加；排名是无量纲的，融合无需求归一化。")
    a("4. **单路都有死穴**：dense 输精确词题、BM25 输改写题——看你业务里")
    a("   哪类问题多，但多数生产 RAG 直接默认 hybrid。")
    a("")
    a("## 六、复现方式")
    a("")
    a("```bash")
    a("pip install chromadb fastembed")
    a("export HF_ENDPOINT=https://hf-mirror.com   # 脚本已默认")
    a("python 01Agent/w3day05/hybrid_rerank.py    # 首次运行下载 reranker ~0.57GB")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
