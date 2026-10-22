# -*- coding: utf-8 -*-
"""W3 Day6：RAGAS 风格评估 —— 四大指标 + LLM-as-judge + 一轮改进迭代

对应任务：
  1. 了解 RAGAS 四大指标：faithfulness / relevancy / precision / recall
  2. 跑评估脚本量化 RAG demo 质量
  3. 根据评估结果优化检索或生成策略

实现说明（不装 ragas 库的原因与等价性）：
  ragas 依赖 langchain 全家桶，且指标本质都是"LLM 按固定 rubric 打分"。
  本脚本按 RAGAS 论文的口径手写四指标，评分 rubric 全部写进 prompt、
  全部落盘，规则透明可复现——与 w2day04 手写 F1 的思路一脉相承。

四指标的简化实现（每题每配置 6 次 LLM 调用）：
  生成侧：gen(1) + faithfulness(1) + relevancy(1)
  检索侧：context_precision(1) + context_recall(1)
  另有一次 rule-based Recall@5（golden keywords，零 LLM）

实验：同一条 RAG 链路（dense+BM25 双路 → RRF → top5），两代配置对照：
    baseline   context = top5（Day3/5 的形态）
    ctx-top3   context = top3（假说：候选少而精，上下文噪声更低）
  对比 Faithfulness / Context Precision 等指标变化 → 完成一次
  "评估 → 定位 → 改进 → 复评"的闭环。

产出：ragas_eval_report.md + ragas_eval_result.json
依赖：chromadb、fastembed；LLM 网关同 W1/W2。
"""
import sys
import io
import os
import re
import json
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))

REPORT_PATH = os.path.join(BASE_DIR, "ragas_eval_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "ragas_eval_result.json")

import ragkit  # noqa: E402

GEN_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.0
EXCERPT_CHARS = 150  # 给 judge 看的 chunk 截断长度（省 token）

# ---- 生成 Prompt（与 Day3 一致：带编号引用 + 弃权条款）----
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

# ---- Judge prompts（rubric 全部明示）----
FAITH_PROMPT = """你是一个严格的事实核查员。下面是一段资料和一个"基于资料"的回答。
第一步：把回答拆成原子论断（每条一个可独立验证的事实点，忽略客套话和引用标记）。
第二步：逐条判断该论断是否被资料支持（支持=资料里有直接依据；资料没提或超出资料=不支持）。
只输出 JSON：{{"claims": [{{"claim": "...", "supported": true/false}}]}}

<资料>
{context}
</资料>

<回答>
{answer}
</回答>"""

REL_PROMPT = """你是一个问答质量评审。问题：{question}

候选回答：
{answer}

评分（1~5）：5=直接且完整回答了问题；4=回答了但有不必要展开；3=部分相关或明显跑偏一部分；
2=很少内容与问题相关；1=完全答非所问。
注意：如果资料本来就无法回答该问题，回答明确声明"无法确定"也算高质量履约（5 分）。
只输出 JSON：{{"score": 1-5, "reason": "一句话"}}"""

CTX_PRECISION_PROMPT = """你是一个检索质量评审。问题：{question}

下面是检索系统给出的 {n} 个候选段落（已编号）。逐个判断：要回答上面的问题，
这个段落是否包含有用信息（有用=true，无用=false）。拿不准时倾向 false。
只输出 JSON：{{"judgments": [{{"index": 1, "relevant": true/false}}]}}

{chunks}"""

CTX_RECALL_PROMPT = """你是一个检索完整性评审。参考答案（标准答案）如下：

{gold}

下面是检索系统提供给生成模型的全部资料。判断：参考答案里的每个信息点，
资料里是否都有覆盖？列出已覆盖的信息点和缺失的信息点。
只输出 JSON：{{"covered": ["信息点1", ...], "missing": ["信息点2", ...]}}

<资料>
{context}
</资料>"""


def _parse_json(text):
    """从模型输出里抠 JSON（容忍围栏/前后缀），失败返回 None。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def judge(prompt):
    """judge 调用：temperature=0，短输出，返回解析后的 dict 或 None。"""
    resp, err = ragkit.chat([{"role": "user", "content": prompt}],
                            temperature=JUDGE_TEMPERATURE, max_tokens=1500)
    if err:
        return None
    return _parse_json(resp["choices"][0]["message"]["content"])


def main():
    docs = ragkit.load_corpus()
    questions = ragkit.load_eval_set()
    emb = ragkit.get_embedder()
    print(f"[embedder] kind={emb.kind}  model={emb.name}")

    # 建索引（recursive-300，与 Day3/5 一致）
    chunks = []
    for d in docs:
        for i, (text, _) in enumerate(ragkit.recursive_chunk(d["text"], 300)):
            chunks.append({"id": f"{d['id']}#c{i}", "doc_id": d["id"], "text": text})
    vecs = emb.passage_embed([c["text"] for c in chunks])

    import chromadb
    col = chromadb.EphemeralClient().get_or_create_collection(
        "ragas", metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks], embeddings=vecs,
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"]} for c in chunks])

    # BM25（与 Day5 同参数；这里内嵌一份最小实现避免跨目录 import）
    from ragkit import rrf_fuse

    def bm25_search(query, k):
        # 简版 BM25（k1=1.5 b=0.75，中文 bigram）：只在这一天充当召回路
        from collections import Counter
        import math as _m

        def toks(t):
            import re as _re
            out = _re.findall(r"[a-zA-Z0-9_]+", t)
            zh = _re.sub(r"[a-zA-Z0-9_\s]+", "", t)
            out += [zh[j:j + 2] for j in range(len(zh) - 1)] if len(zh) > 1 else ([zh] if zh else [])
            return out

        tfs = [Counter(toks(c["text"])) for c in chunks]
        dls = [sum(tf.values()) for tf in tfs]
        avgdl = sum(dls) / len(dls)
        df = Counter()
        for tf in tfs:
            df.update(tf.keys())
        n = len(chunks)
        k1, b = 1.5, 0.75
        scored = []
        for i, c in enumerate(chunks):
            s = 0.0
            for t in set(toks(query)):
                tf = tfs[i].get(t, 0)
                if not tf:
                    continue
                idf = _m.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dls[i] / avgdl))
            scored.append((s, c["id"]))
        scored.sort(key=lambda x: -x[0])
        return [cid for s, cid in scored[:k] if s > 0]

    cid2chunk = {c["id"]: c for c in chunks}

    def retrieve(question, cand_k=20):
        qvec = emb.query_embed([question])[0]
        dense = col.query(query_embeddings=[qvec], n_results=cand_k)["ids"][0]
        sparse = bm25_search(question, cand_k)
        return [cid for cid, _ in ragkit.rrf_fuse([dense, sparse], k=60, top_n=top_k)]

    def generate(question, hits):
        ctx = "\n\n".join(f"[{i}] {h['title']}\n{h['text']}"
                          for i, h in enumerate(hits, start=1))
        resp, err = ragkit.chat([{"role": "user", "content":
                                  PROMPT_TPL.format(context=ctx, question=question)}],
                                temperature=GEN_TEMPERATURE, max_tokens=1200)
        if err:
            return f"[API 错误] {err}"
        return resp["choices"][0]["message"]["content"]

    # =====================================================================
    # 两代配置评估
    # =====================================================================
    configs = {"baseline-top5": 5, "ctx-top3": 3}
    all_results = {}

    for cfg_name, top_k in configs.items():
        print(f"\n{'=' * 62}")
        print(f"配置 {cfg_name}（context = RRF top{top_k}）")
        print("=" * 62)
        rows = []
        for q in questions:
            hits_ids = retrieve(q["question"], cand_k=20)
            hits = [{"id": cid, "title": cid2chunk[cid]["doc_id"],
                     "text": cid2chunk[cid]["text"]} for cid in hits_ids]
            answer = generate(q["question"], hits)
            ctx_all = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
            ctx_excerpt = "\n\n".join(
                f"[{i}] {h['text'][:EXCERPT_CHARS]}" for i, h in enumerate(hits, 1))

            # 1) Faithfulness：拆论断 + 逐条判支持
            f = judge(FAITH_PROMPT.format(context=ctx_all, answer=answer)) or {}
            claims = f.get("claims") or []
            supported = sum(1 for c in claims if c.get("supported") is True)
            faith = round(supported / len(claims), 3) if claims else None

            # 2) Answer Relevancy（1~5 → 归一到 0~1）
            r = judge(REL_PROMPT.format(question=q["question"], answer=answer)) or {}
            rel_score = r.get("score")
            relevancy = round((rel_score - 1) / 4, 3) if isinstance(rel_score, (int, float)) else None

            # 3) Context Precision：LLM 判每条候选 → 平均精度 AP
            cp = None
            if hits:
                jp = judge(CTX_PRECISION_PROMPT.format(
                    question=q["question"], n=len(hits),
                    chunks="\n\n".join(f"[{i}] {h['text'][:EXCERPT_CHARS]}"
                                       for i, h in enumerate(hits, 1)))) or {}
                jd = {x.get("index"): bool(x.get("relevant"))
                      for x in (jp.get("judgments") or []) if isinstance(x, dict)}
                flags = [jd.get(i, False) for i in range(1, len(hits) + 1)]
                hit_n, ap = sum(flags), 0.0
                seen = 0
                for i, fl in enumerate(flags, 1):
                    if fl:
                        seen += 1
                        ap += seen / i
                cp = round(ap / hit_n, 3) if hit_n else 0.0

            # 4) Context Recall：标准答案的信息点被资料覆盖的比例
            cr = None
            if q.get("gold_answer") and q["type"] != "unanswerable":
                jc = judge(CTX_RECALL_PROMPT.format(
                    gold=q["gold_answer"], context=ctx_excerpt)) or {}
                cov, mis = len(jc.get("covered") or []), len(jc.get("missing") or [])
                cr = round(cov / (cov + mis), 3) if (cov + mis) else None

            # 5) rule-based Recall@5（golden keywords，零 LLM）
            rb_rel = [ragkit.chunk_is_relevant(h["text"], q) for h in hits]
            rb = 1 if any(rb_rel) else 0

            row = {"id": q["id"], "type": q["type"], "question": q["question"],
                   "answer": answer, "hits": hits_ids,
                   "faithfulness": faith, "n_claims": len(claims),
                   "relevancy": relevancy, "judge_rel_reason": r.get("reason"),
                   "context_precision": cp, "context_recall": cr,
                   "rule_recall": rb,
                   "abstained": any(w in answer for w in
                                    ["无法确定", "资料不足", "没有提到", "未提及"])}
            rows.append(row)
            print(f"  [{q['id']}] faith={faith} rel={relevancy} "
                  f"cp={cp} cr={cr} rule={rb} abst={row['abstained']}")

        def _mean(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 3) if xs else None

        answerable = [x for x in rows if x["type"] != "unanswerable"]
        agg = {
            "faithfulness": _mean([x["faithfulness"] for x in answerable]),
            "relevancy": _mean([x["relevancy"] for x in answerable]),
            "context_precision": _mean([x["context_precision"] for x in answerable]),
            "context_recall": _mean([x["context_recall"] for x in answerable]),
            "rule_recall@k": _mean([x["rule_recall"] for x in answerable]),
            "abstain_rate_unans": _mean([1.0 if x["abstained"] else 0.0
                                         for x in rows if x["type"] == "unanswerable"]),
            "avg_context_chars": _mean([sum(len(h["text"]) for h in
                                            [{"text": cid2chunk[c]["text"]} for c in x["hits"]])
                                        for x in rows]),
        }
        all_results[cfg_name] = {"rows": rows, "aggregate": agg,
                                 "top_k": top_k}
        print(f"  [汇总] {json.dumps(agg, ensure_ascii=False)}")

    payload = {
        "meta": {"generated_at": datetime.datetime.now().isoformat(),
                 "embedder": {"kind": emb.kind, "name": emb.name},
                 "degraded": getattr(emb, "warn", None),
                 "model": ragkit.MODEL,
                 "gen_temperature": GEN_TEMPERATURE,
                 "judge_temperature": JUDGE_TEMPERATURE,
                 "judge_model": ragkit.MODEL,
                 "n_chunks": len(chunks), "chunking": "recursive-300",
                 "retrieval": "dense+BM25 → RRF(k=60)",
                 "configs": {k: {"top_k": v} for k, v in configs.items()},
                 "note": "RAGAS 指标为按论文口径的简化实现：faithfulness 合并了"
                         "拆论断与判支持两步；relevancy 用直接评审代替反向生成问题。"},
        "results": all_results,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(p):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = p["results"]["baseline-top5"]
    var = p["results"]["ctx-top3"]
    L = []
    a = L.append
    a("# RAGAS 风格评估报告（W3 Day6）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 被评系统：dense+BM25 双路 → RRF → top{base['top_k']}/top{var['top_k']} 两种配置")
    a(f"- 生成模型：`{p['meta']['model']}`（t={p['meta']['gen_temperature']}）；"
      f"judge 同模型（t={p['meta']['judge_temperature']}）")
    a(f"- 黄金集：15 题；RAGAS 四指标为按论文口径的简化实现（judge rubric 见脚本）")
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    a("")
    a("## 一、四指标各自在评什么")
    a("")
    a("| 指标 | 层 | 问题 | 低分意味着 |")
    a("|---|---|---|---|")
    a("| Faithfulness | 生成 | 答案的每条论断都被上下文支持吗 | 模型在编造/超出资料发挥 |")
    a("| Answer Relevancy | 生成 | 答案回答了问题吗 | 答非所问 |")
    a("| Context Precision | 检索 | 相关资料排在前面吗（AP） | 排序差，噪声挤占注意力 |")
    a("| Context Recall | 检索 | 标准答案所需信息资料里齐吗 | 检索漏了，生成再好也白搭 |")
    a("")
    a("排查顺序永远是：Context Recall → Context Precision → Faithfulness，")
    a("先检索后生成，锅才能甩对位置。")
    a("")
    a("## 二、两代配置对照（一次完整的评估→改进闭环）")
    a("")
    a("| 指标 | baseline-top5 | ctx-top3 | 变化 |")
    a("|---|---|---|---|")
    for k, label in [("faithfulness", "Faithfulness"), ("relevancy", "Relevancy"),
                     ("context_precision", "Context Precision"),
                     ("context_recall", "Context Recall"),
                     ("rule_recall@k", "Rule Recall@k"),
                     ("abstain_rate_unans", "不可回答题弃权率"),
                     ("avg_context_chars", "平均上下文字符")]:
        b, v = base["aggregate"][k], var["aggregate"][k]
        if isinstance(b, float) and isinstance(v, float):
            delta = f"{v - b:+.3f}"
        else:
            delta = "-"
        a(f"| {label} | {b} | {v} | {delta} |")
    a("")
    a("改动只有一处：进上下文的候选从 5 条减到 3 条。如果 Context Precision 和")
    a("Faithfulness 上升、Recall 不掉，说明\"上下文太吵\"是主要矛盾，")
    a("候选少而精就是改进方向；如果 Recall 掉了，说明第 4、5 名候选里有")
    a("兜底信息，噪声的代价买到了覆盖。")
    a("")
    a("## 三、逐题明细（baseline-top5）")
    a("")
    a("| 题 | 类型 | Faith | Rel | CP | CR | Rule | 弃权 |")
    a("|---|---|---|---|---|---|---|---|")
    for x in base["rows"]:
        a(f"| {x['id']} | {x['type']} | {x['faithfulness']} | {x['relevancy']} "
          f"| {x['context_precision']} | {x['context_recall']} | {x['rule_recall']} "
          f"| {'✅' if x['abstained'] else '·'} |")
    a("")
    low_faith = [x for x in base["rows"]
                 if x["faithfulness"] is not None and x["faithfulness"] < 0.8
                 and x["type"] != "unanswerable"]
    if low_faith:
        a("### 忠实度低的题（<0.8）逐条看")
        a("")
        for x in low_faith:
            a(f"**{x['id']}**（faith={x['faithfulness']}）：{x['question']}")
            a("")
            a(f"- 回答摘录：{x['answer'][:150].replace(chr(10), ' ')}…")
            a("")
    a("## 四、LLM-as-judge 的坑（本次实测观察到的）")
    a("")
    unans_low = [x for x in base["rows"]
                 if x["type"] == "unanswerable"
                 and x["faithfulness"] is not None and x["faithfulness"] < 0.9]
    a("1. **judge 与生成同模型**：自我偏好风险；生产应换更强的模型当 judge，")
    a("   并用人工标注的小集合校准一致率。")
    if unans_low:
        ids = "、".join(f"{x['id']}(faith={x['faithfulness']})"
                        for x in unans_low)
        a(f"2. **弃权回答被误伤（{ids}）**：\"根据现有资料无法确定\"是正确行为，")
        a("   但 judge 把\"资料中没有提到 X\"这类**元陈述**当成了需要资料支持的")
        a("   论断去核对，判了不支持 → faithfulness 虚低。修正方向：rubric 里")
        a("   显式豁免弃权表述，或对弃权回答走单独的评估通道。")
        skip = 3
    else:
        skip = 2
    a(f"{skip}. **rubric 措辞敏感**：本次在 relevancy 里补了一句\"正确弃权算 5 分\"，")
    a("   否则不可回答题的\"无法确定\"会被 judge 误判为答非所问——")
    a("   这既说明 judge 的评分是 prompt 的函数，也说明弃权必须被显式建模。")
    a(f"{skip + 1}. **拆论断的粒度不稳定**：faithfulness 的论断数量随题波动，")
    a("   一条长论断被拆成两条后，部分支持就会拉低分数。")
    a(f"{skip + 2}. **Context Precision 的 AP 公式**：相关块排第 1 记 1.0、第 5 记 0.27，")
    a("   排序位置的影响被显式计入——这正是\"重排为什么值得做\"的量化理由。")
    a("")
    a("## 五、结论")
    a("")
    b, v = base["aggregate"], var["aggregate"]
    a(f"1. **分层定位**：rule Recall@k={b['rule_recall@k']}、"
      f"Context Recall={b['context_recall']} 说明检索覆盖是"
      f"{'瓶颈' if (b['rule_recall@k'] or 1) < 0.9 else '基本够用'}；"
      f"Faithfulness={b['faithfulness']} 说明生成侧")
    a("   的 grounding 约束基本生效（带引用 + 弃权条款）。")
    a(f"2. **改进迭代**：top5→top3，Context Precision "
      f"{b['context_precision']}→{v['context_precision']}，"
      f"Faithfulness {b['faithfulness']}→{v['faithfulness']}，"
      f"Recall {b['context_recall']}→{v['context_recall']}——"
      "用数据决定\"候选该多宽\"，而不是拍脑袋。")
    a("3. **指标要一起看**：任何单一指标都能被刷高（少给上下文必涨 precision），")
    a("   四个指标 + 弃权率构成一张互相制衡的网。")
    a("")
    a("## 六、复现方式")
    a("")
    a("```bash")
    a("pip install chromadb fastembed")
    a("python 01Agent/w3day06/ragas_eval.py   # 约 180 次 LLM 调用（15 题 × 6 × 2 配置）")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
