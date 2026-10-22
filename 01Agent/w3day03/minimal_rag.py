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
import sys        # 解释器相关：这里只用于改 stdout 编码、读命令行参数（本脚本没用到参数）
import io         # 流包装：把 stdout 底层字节流重新包一层
import os         # 路径拼接、文件存在判断
import json       # 结果数据落盘 JSON
import datetime   # 报告/数据里记录生成时间

# Windows 控制台默认 GBK 编码，print 中文/特殊符号可能直接 UnicodeEncodeError；
# 把 stdout 重新包装成 UTF-8、不可编码的字符用 ? 替换——保证实验不会"打印"这一步崩掉
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 本脚本所在目录（…/01Agent/w3day03）；__file__ 是当前脚本路径，abspath 转绝对路径
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "../w3data")))
# 把 ../w3data 插到模块搜索路径最前面，下面才能 import ragkit（共享工具库在那里）

REPORT_PATH = os.path.join(BASE_DIR, "minimal_rag_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "minimal_rag_result.json")
# 两个产出物的落盘路径：报告（人读的 Markdown）+ 原始数据（机器读的 JSON，可复现）

import ragkit  # noqa: E402
# 共享工具库：语料加载、递归切分、embedder、chat 网关、评估指标全在里面；
# noqa: E402 告诉检查工具"import 写在代码中间是故意的"（必须先配好 sys.path）

TOP_K = 5          # 检索召回窗口：每次进 prompt 的 chunk 数；太大→噪声多+token 贵，太小→漏证据
CHUNK_CHARS = 300  # chunk 目标大小（字符）；生产应按 token 计（模型的上下文按 token 算）

# ---- 生成侧 Prompt（与 W3 预习文档 5.1 节的推荐结构一致）----
# 三个设计点：
#   1. <instructions>：三条指令分别对应 防幻觉（只准用资料）/ 可溯源（标[编号]）/ 敢弃权
#   2. XML 风格标签把"资料"和"指令"隔开——防止资料里的文字被当成指令执行（提示注入）
#   3. {context}/{question} 是 str.format 的槽位，由 build_prompt() 填入
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
    # 读 ../w3data/corpus/*.md → [{id, title, text}]，按文件名排序保证可复现

    chunks = []  # [{id, doc_id, title, text}] —— 切分后的块，检索的最小单位
    for d in docs:
        pieces = ragkit.recursive_chunk(d["text"], max_chars=CHUNK_CHARS)
        # 递归字符切分：优先在标题/空行等大边界切，太长的块再用句号/逗号细切
        for i, (text, pos) in enumerate(pieces):
            # pos 是块在原文中的偏移（本实验没用到，留作扩展：比如高亮出处）
            chunks.append({"id": f"{d['id']}#c{i}", "doc_id": d["id"],
                           "title": d["title"], "text": text})
            # chunk id 形如 "07_rag_pipeline#c3"：文档id+块序号，可反查出处；
            # doc_id/title 存进 metadata，检索命中后拼引用用

    emb = ragkit.get_embedder()
    # 拿语义 embedder（fastembed + bge-zh）；没装 fastembed 会自动降级为 HashEmbed

    vecs = emb.passage_embed([c["text"] for c in chunks])
    # 文档侧编码：bge 约定 passage_embed 编文档、query_embed 编问题，
    # 两侧必须配套（同模型+同侧），向量空间才对齐，余弦才有意义

    import chromadb
    client = chromadb.EphemeralClient()
    # 内存态向量库：进程结束即消失。Day2 已演示过 PersistentClient 落盘；
    # 这里语料小、重建秒级，还顺带避开残留目录的 Windows 文件锁问题

    col = client.get_or_create_collection("rag_index",
                                          metadata={"hnsw:space": "cosine"})
    # 建集合（≈关系库的表）：距离度量用余弦，Chroma 返回的 distance = 1 - cosine

    col.add(ids=[c["id"] for c in chunks],
            embeddings=vecs,
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks])
    # 批量写入"四件套"：主键 id、向量、原文、元数据（对应 SQL 的 INSERT）
    return emb, col, chunks


def retrieve(emb, col, question, k=TOP_K):
    """在线流水线前半段：问题 → 向量 → top-k。"""
    qvec = emb.query_embed([question])[0]
    # 问题侧编码（bge 会给 query 加指令前缀）；[0] 取出这一条问题的向量

    res = col.query(query_embeddings=[qvec], n_results=k)
    # ANN 近邻搜索 top-k：注意返回的 ids/distances/documents/metadatas
    # 都是"每个查询一个列表"的嵌套结构，所以后面处处有 [0]

    return [{"id": cid, "doc_id": m["doc_id"], "title": m["title"], "text": doc}
            for cid, m, doc in zip(res["ids"][0], res["metadatas"][0],
                                   res["documents"][0])]
    # Chroma 返回四条平行列表，zip 回"一条命中 = 一个 dict"的结构，上层代码更好读


def build_prompt(question, hits):
    """上下文打包：带编号的资料 + 防注入标签 + 弃权指令。"""
    ctx = "\n\n".join(f"[{i}] {h['title']}\n{h['text']}"
                      for i, h in enumerate(hits, start=1))
    # 每条命中编上号 [1]…[k]（标题+正文），模型引用 [编号] 就能溯源到具体 chunk
    return PROMPT_TPL.format(context=ctx, question=question)
    # 填进模板的 {context}/{question} 两个槽位，得到完整 prompt 字符串


def answer_question(question, hits):
    """在线流水线后半段：prompt → LLM → 回答。"""
    resp, err = ragkit.chat([{"role": "user", "content": build_prompt(question, hits)}],
                            temperature=0.2, max_tokens=1500)
    # 整个 prompt 作为一条 user 消息发给网关；temperature=0.2 压低随机性
    # （要事实性回答，不要发散）；chat() 出错不抛异常而是返回 (None, 错误串)

    if err:
        return f"[API 错误] {err}", {"error": err}
    # 网关挂了/超时：回答文本给错误占位符，meta 里带错误详情，实验继续跑

    msg = resp["choices"][0]["message"]
    return msg["content"], {"usage": resp.get("usage", {}),
                            "finish_reason": resp["choices"][0]["finish_reason"]}
    # 正常路径：content 是回答文本；usage（token 用量=钱）和 finish_reason 进 meta


def main():
    print("=" * 62)
    print("离线：建索引（切分 → 向量化 → Chroma）")
    print("=" * 62)
    emb, col, chunks = build_index()
    # 跑离线流水线：拿到 embedder、向量集合、chunk 清单

    n_docs = len(set(c["doc_id"] for c in chunks))
    # 去重统计文档数（chunks 是切分后的块，数量远多于文档）

    print(f"[index] 语料 {n_docs} 篇 → {len(chunks)} 个 chunk "
          f"(目标 {CHUNK_CHARS} 字符) → 向量维度 {emb.dim}")
    print(f"[embedder] kind={emb.kind}  model={emb.name}")
    # 打印索引规模和用的哪个 embedder（降级时 kind 会变，这里能看出来）

    questions = ragkit.load_eval_set()
    # 黄金集：每题带 question/type/gold_keywords，是评估的"标准答案"

    print(f"[eval ] 黄金集 {len(questions)} 题")

    results = []  # 每题的完整记录：命中的 chunks、指标、回答、弃权判定
    print()
    for q in questions:
        # ---- 每题完整走一遍"检索 → 生成 → 评估" ----
        hits = retrieve(emb, col, q["question"])
        # ① 检索：问题向量 → top5 chunk

        rel_flags = [ragkit.chunk_is_relevant(h["text"], q) for h in hits]
        # ② 检索层评估：每个 chunk 文本是否含黄金集的关键词 → [True/False]×5

        ans, meta = answer_question(q["question"], hits)
        # ③ 生成：把 5 个 chunk 打包进 prompt，发给 LLM 拿回答

        # ---- 弃权检查：不可回答的题，回答里应出现弃权表述 ----
        abstain_words = ["无法确定", "资料中未", "没有提到", "未提及", "资料不足",
                         "没有相关", "无法回答", "没有找到"]
        abstained = any(w in ans for w in abstain_words)
        # 关键词匹配是简化实现（够教学用）；生产可用 NLI 模型或再调一次 LLM 判定

        expect_abstain = q["type"] == "unanswerable"
        # 黄金集的期望：unanswerable 题（资料里没有的）应该弃权，其他题不该弃权

        rr = next((1.0 / (i + 1) for i, h in enumerate(rel_flags) if h), 0.0)
        # MRR 单题分：第一个相关 chunk 排在第 i+1 位就记 1/(i+1)；全没中记 0
        # 排得越前分越高——衡量"排序质量"，比 recall 更严格

        usage = meta.get("usage", {})
        hit_str = ", ".join(f"{h['id']}({1 if f else 0})" for h, f in zip(hits, rel_flags))
        print(f"[{q['id']}] {q['question']}")
        print(f"  hits: [{hit_str}]")
        # 命中清单：括号里的 1/0 = 该 chunk 是否含黄金关键词（人看结果用）
        print(f"  answer({usage.get('completion_tokens', '?')}t): {ans[:120].replace(chr(10), ' ')}...")
        # 回答前 120 字符（换行压成空格），括号里是本次生成用的 completion tokens

        flag = "OK" if (expect_abstain and abstained) or \
                       (not expect_abstain and any(rel_flags)) else "MISS"
        # 判题标准：不可回答题→弃权了才 OK；可回答题→top5 里至少有一个相关 chunk
        print(f"  [{flag}] type={q['type']} recall@5={1 if any(rel_flags) else 0} "
              f"mrr={rr:.2f} abstained={abstained}")
        print()

        results.append({
            "id": q["id"], "question": q["question"], "type": q["type"],
            "hits": [{**h, "relevant": f} for h, f in zip(hits, rel_flags)],
            # {**h, ...}：复制命中 dict 并追加一个 relevant 字段（原文落档）
            "recall@5": 1 if any(rel_flags) else 0,
            "mrr": round(rr, 3),
            "answer": ans, "abstained": abstained,
            "expect_abstain": expect_abstain,
            "abstain_correct": abstained == expect_abstain,
            # 弃权判对了没有：该弃权的弃了 / 不该弃权的没弃，都算对
            "usage": usage, "finish_reason": meta.get("finish_reason"),
        })

    # ---- 汇总：把逐题结果聚合成整体指标 ----
    answerable = [r for r in results if not r["expect_abstain"]]
    unans = [r for r in results if r["expect_abstain"]]
    # 按黄金集把题分成两堆：可回答（看召回）/ 不可回答（看弃权）
    metrics = {
        **ragkit.retrieval_metrics([[h["relevant"] for h in r["hits"]] for r in results]),
        # 检索层整体指标：Recall@5 / MRR / HitRate（实现在 ragkit 里）

        "abstain_accuracy": round(sum(1 for r in results if r["abstain_correct"])
                                  / len(results), 3),
        # 弃权准确率：该弃权的弃了 + 不该弃权的没弃，占全部题的比例

        "answerable_recall@5": round(sum(r["recall@5"] for r in answerable)
                                     / len(answerable), 3),
        # 只看可回答题的召回——排除"不可回答题天然 recall=0"对整体指标的稀释

        "unanswerable_abstain_rate": round(sum(r["abstained"] for r in unans)
                                           / len(unans), 3) if unans else None,
        # 不可回答题里真的弃权了的比例——防幻觉能力的直接度量

        "total_prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in results),
        "total_completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in results),
        # 成本核算：15 次问答的 token 合计——每次重复发送的 context 是 RAG 主要开销
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
                 # 降级记录：fastembed 不可用时这里是警告文本，否则 None
                 "n_docs": n_docs, "n_chunks": len(chunks),
                 "chunk_chars": CHUNK_CHARS, "top_k": TOP_K,
                 "model": ragkit.MODEL, "prompt_template": PROMPT_TPL},
        # meta 记录"这次实验的全部环境与参数"——换参数重跑后可对照
        "chunks_preview": [{"id": c["id"], "len": len(c["text"])} for c in chunks[:10]],
        # 前 10 个 chunk 的 id 和长度：检查切分质量用（太短的块说明切碎了）
        "metrics": metrics,
        "results": results,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # ensure_ascii=False：中文原样写入而不是 \uXXXX 转义；indent=2 人可读

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(p):
    """把结果 dict 排成 Markdown 报告：纯排版，无业务逻辑。

    模式：L 是行缓冲，a = L.append 起短别名；一路 a() 攒行，最后 join 成整篇。
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 生成时间（展示用）
    m = p["metrics"]   # 汇总指标 dict
    L = []             # 行缓冲
    a = L.append       # 短别名：下面排版代码每行都少打几个字
    a("# 最小 RAG 链路实验报告（W3 Day3）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 语料：{p['meta']['n_docs']} 篇 → 递归切分 {p['meta']['n_chunks']} chunks"
      f"（目标 {p['meta']['chunk_chars']} 字符）")
    a(f"- 向量：`{p['meta']['embedder']['name']}`（{p['meta']['embedder']['kind']}，查询/文档同模型）")
    a(f"- 生成模型：`{p['meta']['model']}`，temperature=0.2")
    a(f"- 黄金集：{len(p['results'])} 题（可回答/多跳/精确词/不可回答）")
    # ↑ 头部信息块：语料规模、向量模型、生成模型、题量
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    # 降级时（没有 fastembed）在报告里如实标注，不静默
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
    # ↑ 手写实现 ↔ LangChain 组件对照表：学了原理，也要认识框架的名词
    a("")
    a("## 二、生成侧 Prompt（全文）")
    a("")
    a("要点：`<context>` 标签隔离资料与指令（防注入）、编号引用、明确弃权条款。")
    a("")
    a("```text")
    a(p["meta"]["prompt_template"])
    # 把 prompt 模板原文放进报告——prompt 是 RAG 生成侧最重要的资产，必须留档
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
    # ↑ 指标总表：检索层（Recall/MRR/Hit）+ 生成层（弃权）+ 成本（tokens）
    a("")
    a("### 逐题结果")
    a("")
    a("| 题 | 类型 | Recall@5 | MRR | 弃权 | 回答摘录 |")
    a("|---|---|---|---|---|---|")
    for r in p["results"]:
        excerpt = r["answer"][:60].replace("\n", " ").replace("|", "/")
        # 回答截 60 字符塞进表格；换行压空格、竖线换成斜杠（防止破坏 Markdown 表格）
        a(f"| {r['id']} | {r['type']} | {r['recall@5']} | {r['mrr']:.2f} "
          f"| {'✅' if r['abstain_correct'] else '❌'} | {excerpt}… |")
    a("")
    a("### 检索命中明细（不可回答题 + 没命中的题）")
    a("")
    for r in p["results"]:
        if r["type"] == "unanswerable" or r["recall@5"] == 0:
            # 只展开两类"值得人看"的题：弃权题（看它召回了什么噪声）+ MISS 题（找原因）
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
    # ↑ 结论三条：检索层表现、弃权机制、成本结构——每条都从本实验数据出发
    a("")
    a("## 五、复现方式")
    a("")
    a("```bash")
    a("pip install chromadb fastembed")
    a("python 01Agent/w3day03/minimal_rag.py")
    a("```")
    a("")
    return "\n".join(L)  # 行缓冲拼成整篇 Markdown 文本


if __name__ == "__main__":
    main()  # 作为脚本运行时才执行；被 import 时不执行（方便别的实验复用这里的函数）
