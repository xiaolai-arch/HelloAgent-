# -*- coding: utf-8 -*-
"""W3 Day3：最小 RAG 链路 —— 文档→切分→向量化→检索→带引用生成

对应任务：
  1. 理解 RAG 标准流程：文档→切分→Embedding→存向量库→检索→生成
  2. 用 LangChain 的 Document Loader 加载本地文档
     —— 本仓库的约定是零框架（与 W1/W2 一致），因此加载/切分/检索全部
        手写实现（ragkit + 本脚本），报告里给出与 LangChain 组件的对应关系
  3. 实现"提问→检索→生成回答"的最小 RAG 链路

链路（对应 RAG 两条流水线，全部可在这一个文件里读完）：

  离线：load_corpus() → recursive_chunk() → passage_embed() → chroma 集合
  在线：query → query_embed() → chroma.query(top5) → 组装带引用的 prompt
        → chat 生成（只准根据资料回答，资料不足要弃权）

评估（用 ../w3data/eval_set.json 黄金集，15 题）：
  - 检索层：Recall@5 / MRR / HitRate（按 gold_keywords 判定 chunk 是否相关）
  - 生成层：弃权检查（不可回答的题，答案里应出现"无法确定"类表述）
  - 逐题打印检索命中与回答摘录，原始数据落盘

产出：minimal_rag_report.md + minimal_rag_result.json
依赖：chromadb、fastembed（见 w3day01/02）；LLM 网关配置同 W1/W2 脚本。
"""
import sys
import io
import os
import json
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "minimal_rag_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "minimal_rag_result.json")

import ragkit  # noqa: E402

TOP_K = 5          # 进上下文的 chunk 数（召回窗口）
CHUNK_CHARS = 300  # chunk 目标大小（字符）；生产应按 token 计

# ---- 生成侧 Prompt（与 W3 预习文档 5.1 节的推荐结构一致）----
PROMPT_TPL = """<instructions>
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
</question>"""


def build_index():
    """离线流水线：读语料 → 递归切分 → 向量化 → 存入 Chroma（内存态）。"""
    docs = ragkit.load_corpus()
    chunks = []  # [{id, doc_id, title, text}]
    for d in docs:
        pieces = ragkit.recursive_chunk(d["text"], max_chars=CHUNK_CHARS)
        for i, (text, pos) in enumerate(pieces):
            chunks.append({"id": f"{d['id']}#c{i}", "doc_id": d["id"],
                           "title": d["title"], "text": text})

    emb = ragkit.get_embedder()
    vecs = emb.passage_embed([c["text"] for c in chunks])

    import chromadb
    client = chromadb.EphemeralClient()  # 内存态：Day2 演示过持久化，这里不重复
    col = client.get_or_create_collection("rag_index",
                                          metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks],
            embeddings=vecs,
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks])
    return emb, col, chunks


def retrieve(emb, col, question, k=TOP_K):
    """在线流水线前半段：问题 → 向量 → top-k。"""
    qvec = emb.query_embed([question])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    return [{"id": cid, "doc_id": m["doc_id"], "title": m["title"], "text": doc}
            for cid, m, doc in zip(res["ids"][0], res["metadatas"][0],
                                   res["documents"][0])]


def build_prompt(question, hits):
    """上下文打包：带编号的资料 + 防注入标签 + 弃权指令。"""
    ctx = "\n\n".join(f"[{i}] {h['title']}\n{h['text']}"
                      for i, h in enumerate(hits, start=1))
    return PROMPT_TPL.format(context=ctx, question=question)


def answer_question(question, hits):
    """在线流水线后半段：prompt → LLM → 回答。"""
    resp, err = ragkit.chat([{"role": "user", "content": build_prompt(question, hits)}],
                            temperature=0.2, max_tokens=1500)
    if err:
        return f"[API 错误] {err}", {"error": err}
    msg = resp["choices"][0]["message"]
    return msg["content"], {"usage": resp.get("usage", {}),
                            "finish_reason": resp["choices"][0]["finish_reason"]}


def main():
    print("=" * 62)
    print("离线：建索引（切分 → 向量化 → Chroma）")
    print("=" * 62)
    emb, col, chunks = build_index()
    n_docs = len(set(c["doc_id"] for c in chunks))
    print(f"[index] 语料 {n_docs} 篇 → {len(chunks)} 个 chunk "
          f"(目标 {CHUNK_CHARS} 字符) → 向量维度 {emb.dim}")
    print(f"[embedder] kind={emb.kind}  model={emb.name}")

    questions = ragkit.load_eval_set()
    print(f"[eval ] 黄金集 {len(questions)} 题")

    results = []
    print()
    for q in questions:
        hits = retrieve(emb, col, q["question"])
        # 检索层评估：每个位置是否命中 gold_keywords
        rel_flags = [ragkit.chunk_is_relevant(h["text"], q) for h in hits]
        ans, meta = answer_question(q["question"], hits)

        # 弃权检查：不可回答的题，回答里应出现弃权表述
        abstain_words = ["无法确定", "资料中未", "没有提到", "未提及", "资料不足",
                         "没有相关", "无法回答", "没有找到"]
        abstained = any(w in ans for w in abstain_words)
        expect_abstain = q["type"] == "unanswerable"

        rr = next((1.0 / (i + 1) for i, h in enumerate(rel_flags) if h), 0.0)
        usage = meta.get("usage", {})
        hit_str = ", ".join(f"{h['id']}({1 if f else 0})" for h, f in zip(hits, rel_flags))
        print(f"[{q['id']}] {q['question']}")
        print(f"  hits: [{hit_str}]")
        print(f"  answer({usage.get('completion_tokens', '?')}t): {ans[:120].replace(chr(10), ' ')}...")
        flag = "OK" if (expect_abstain and abstained) or \
                       (not expect_abstain and any(rel_flags)) else "MISS"
        print(f"  [{flag}] type={q['type']} recall@5={1 if any(rel_flags) else 0} "
              f"mrr={rr:.2f} abstained={abstained}")
        print()

        results.append({
            "id": q["id"], "question": q["question"], "type": q["type"],
            "hits": [{**h, "relevant": f} for h, f in zip(hits, rel_flags)],
            "recall@5": 1 if any(rel_flags) else 0,
            "mrr": round(rr, 3),
            "answer": ans, "abstained": abstained,
            "expect_abstain": expect_abstain,
            "abstain_correct": abstained == expect_abstain,
            "usage": usage, "finish_reason": meta.get("finish_reason"),
        })

    # ---- 汇总 ----
    answerable = [r for r in results if not r["expect_abstain"]]
    unans = [r for r in results if r["expect_abstain"]]
    metrics = {
        **ragkit.retrieval_metrics([[h["relevant"] for h in r["hits"]] for r in results]),
        "abstain_accuracy": round(sum(1 for r in results if r["abstain_correct"])
                                  / len(results), 3),
        "answerable_recall@5": round(sum(r["recall@5"] for r in answerable)
                                     / len(answerable), 3),
        "unanswerable_abstain_rate": round(sum(r["abstained"] for r in unans)
                                           / len(unans), 3) if unans else None,
        "total_prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in results),
        "total_completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in results),
    }
    print("=" * 62)
    print("汇总")
    print("=" * 62)
    print(f"  Recall@5={metrics['recall@5']}  MRR={metrics['mrr']}  "
          f"HitRate={metrics['hit_rate']}")
    print(f"  可回答题 Recall@5={metrics['answerable_recall@5']}  "
          f"不可回答题弃权率={metrics['unanswerable_abstain_rate']}")
    print(f"  tokens: prompt={metrics['total_prompt_tokens']} "
          f"completion={metrics['total_completion_tokens']}")

    payload = {
        "meta": {"generated_at": datetime.datetime.now().isoformat(),
                 "embedder": {"kind": emb.kind, "name": emb.name},
                 "degraded": getattr(emb, "warn", None),
                 "n_docs": n_docs, "n_chunks": len(chunks),
                 "chunk_chars": CHUNK_CHARS, "top_k": TOP_K,
                 "model": ragkit.MODEL, "prompt_template": PROMPT_TPL},
        "chunks_preview": [{"id": c["id"], "len": len(c["text"])} for c in chunks[:10]],
        "metrics": metrics,
        "results": results,
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
    L = []
    a = L.append
    a("# 最小 RAG 链路实验报告（W3 Day3）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 语料：{p['meta']['n_docs']} 篇 → 递归切分 {p['meta']['n_chunks']} chunks"
      f"（目标 {p['meta']['chunk_chars']} 字符）")
    a(f"- 向量：`{p['meta']['embedder']['name']}`（{p['meta']['embedder']['kind']}，查询/文档同模型）")
    a(f"- 生成模型：`{p['meta']['model']}`，temperature=0.2")
    a(f"- 黄金集：{len(p['results'])} 题（可回答/多跳/精确词/不可回答）")
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    a("")
    a("## 一、链路全景")
    a("")
    a("```text")
    a("离线索引：corpus/*.md → recursive_chunk(300字) → bge passage_embed → Chroma")
    a("在线查询：question → bge query_embed → Chroma.query(top5)")
    a("          → <context> 打包(带编号) → chat(只准根据资料回答+弃权) → 带引用回答")
    a("```")
    a("")
    a("与 LangChain 组件的对应关系（本仓库零框架手写，概念一一对应）：")
    a("")
    a("| 本实验 | LangChain 对应组件 |")
    a("|---|---|")
    a("| `ragkit.load_corpus()` | DocumentLoader（TextLoader/DirectoryLoader） |")
    a("| `ragkit.recursive_chunk()` | RecursiveCharacterTextSplitter |")
    a("| `ragkit.get_embedder()` | Embeddings（OpenAIEmbeddings/HuggingFaceEmbeddings） |")
    a("| Chroma collection | VectorStore（Chroma） |")
    a("| `build_prompt()` | ChatPromptTemplate + 自定义 context 组装 |")
    a("| `ragkit.chat()` | ChatOpenAI / LLM |")
    a("")
    a("## 二、生成侧 Prompt（全文）")
    a("")
    a("要点：`<context>` 标签隔离资料与指令（防注入）、编号引用、明确弃权条款。")
    a("")
    a("```text")
    a(p["meta"]["prompt_template"])
    a("```")
    a("")
    a("## 三、评估结果")
    a("")
    a(f"| 指标 | 数值 | 说明 |")
    a(f"|---|---|---|")
    a(f"| Recall@5 | {m['recall@5']} | 相关 chunk 进了 top5 的问题比例 |")
    a(f"| MRR | {m['mrr']} | 第一个相关 chunk 排名倒数的均值 |")
    a(f"| HitRate | {m['hit_rate']} | 至少命中一条的比例（top5 内） |")
    a(f"| 可回答题 Recall@5 | {m['answerable_recall@5']} | |")
    a(f"| 弃权准确率 | {m['abstain_accuracy']} | 不可回答的题该弃权、可回答的不该 |")
    a(f"| 不可回答题弃权率 | {m['unanswerable_abstain_rate']} | 资料没有时应明确说不知道 |")
    a(f"| tokens | prompt={m['total_prompt_tokens']} / completion={m['total_completion_tokens']} | 15 次问答合计 |")
    a("")
    a("### 逐题结果")
    a("")
    a("| 题 | 类型 | Recall@5 | MRR | 弃权 | 回答摘录 |")
    a("|---|---|---|---|---|---|")
    for r in p["results"]:
        excerpt = r["answer"][:60].replace("\n", " ").replace("|", "/")
        a(f"| {r['id']} | {r['type']} | {r['recall@5']} | {r['mrr']:.2f} "
          f"| {'✅' if r['abstain_correct'] else '❌'} | {excerpt}… |")
    a("")
    a("### 检索命中明细（不可回答题 + 没命中的题）")
    a("")
    for r in p["results"]:
        if r["type"] == "unanswerable" or r["recall@5"] == 0:
            a(f"**{r['id']} {r['question']}**（type={r['type']}）")
            a("")
            for h in r["hits"]:
                mark = "✅" if h["relevant"] else "·"
                text = h["text"][:50].replace("\n", " ").replace("|", "/")
                a(f"- {mark} {h['id']}：{text}…")
            a("")
    a("## 四、结论")
    a("")
    a(f"- **检索层**：Recall@5 = {m['recall@5']}。语义向量对\"换说法\"的问题"
      f"（如 MCP 三层角色、RAGAS 指标）召回稳定；")
    a("  没命中的题值得逐一看 chunk：是没切好（语义被切断）还是确实语义太远。")
    a("- **弃权是生成侧行为**：检索层对不可回答的问题照样返回 top5（向量空间里")
    a("  总有\"最近\"的）；能不能守住\"资料不足就说不知道\"，取决于 Prompt 的弃权条款")
    a("  和模型的指令遵循——这正是分层评估的意义：检索没漏 ≠ 回答可靠。")
    a("- **成本结构**：15 次问答 prompt tokens 合计 "
      f"{m['total_prompt_tokens']}，其中每次都重复发送的 context 占大头；")
    a("  上下文工程的\"预算\"视角（阶段一 W2 延伸）在 RAG 里直接变现为钱。")
    a("")
    a("## 五、复现方式")
    a("")
    a("```bash")
    a("pip install chromadb fastembed")
    a("python 01Agent/w3day03/minimal_rag.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
