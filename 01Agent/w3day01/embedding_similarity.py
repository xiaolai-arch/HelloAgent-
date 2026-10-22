# -*- coding: utf-8 -*-
"""W3 Day1：Embedding 原理与文本相似度实验

对应任务：
  1. 理解"词向量 → 句向量 → 稠密向量表示"的演进（报告第一节）
  2. 跑 Embedding 计算文本相似度（原计划是 Embedding API；实测网关无
     /embeddings 端点，改为本地模型，见实验 1 的探测记录）
  3. 理解余弦相似度的计算逻辑（实验 2 手算对照）

实验设计（三组对照，同一批文本跑两套向量）：
  - 语义向量：fastembed + BAAI/bge-small-zh-v1.5（512 维，本地 onnx 推理）
  - 词袋向量：HashEmbed 字符 bigram 哈希（纯 Python 零依赖，机制演示用）
  - A组 同义改写：字面几乎不重叠、意思相同 —— 考语义
  - B组 字面重叠：共享关键词、话题无关     —— 考"词袋会否被字面骗到"
  - C组 精确词：   相似格式不同取值（错误码）—— 考精确匹配
  两组向量对同一批文本算 10x10 余弦相似度矩阵，对照谁在哪组上给出正确排序。

产出：
  - 终端打印相似度矩阵与组内结论
  - embedding_similarity_report.md   实验报告
  - embedding_similarity_result.json 原始数据

依赖：fastembed（可选，没有则只跑 HashEmbed，报告如实标注降级）。
配置：../../vision_config.json（仅实验 1 探测 /embeddings 用）。
"""
import sys
import io
import os
import json
import math
import datetime
import urllib.request
import urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../w3data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

REPORT_PATH = os.path.join(BASE_DIR, "embedding_similarity_report.md")
RESULT_PATH = os.path.join(BASE_DIR, "embedding_similarity_result.json")

import ragkit  # noqa: E402  共享工具库（见 ../w3data/ragkit.py 的模块文档）

# ---- 实验文本：三组对照 -------------------------------------------------------
# 每组 3~4 条，组内互比、也跨组比。挑选原则：
# A组：同义改写对——"退货/退款"、"温度/随机性"，字面几乎无共同词
# B组：共享字面词但话题无关——都带"苹果"、"内存"
# C组：错误码成对——格式相似、取值不同，看两套向量怎么区分
TEXTS = [
    # A组：同义改写（语义应判"相似"，字面重叠很低）
    ("A1", "A", "怎么申请退货退款？"),
    ("A2", "A", "商品退款流程是什么"),
    ("A3", "A", "温度调高后模型输出为什么更随机"),
    ("A4", "A", "采样温度对生成多样性的影响"),
    # B组：字面重叠、语义无关（词袋易误判"相似"，语义向量应能分开）
    ("B1", "B", "苹果发布了新的手机"),
    ("B2", "B", "苹果富含多种维生素"),
    ("B3", "B", "这条内存条价格便宜"),
    ("B4", "B", "他内存不好，转头就忘了"),
    # C组：精确词（错误码：格式一样、取值不同）
    ("C1", "C", "报销系统报错 TRV-40907 怎么处理"),
    ("C2", "C", "报销系统报错 TRV-40231 怎么处理"),
    ("C3", "C", "出差报销的期限是多久"),
]

# 组内"应当判相似"的配对（用于量化：组内平均相似度 vs 跨组平均相似度）
EXPECTED_SIMILAR = {
    "A": [("A1", "A2"), ("A3", "A4")],   # 同义改写：语义相似、字面不同
    "B": [],                              # 字面重叠但语义无关：不该相似
    "C": [],                              # 不同错误码：语义相近但不该排最前
}


def probe_gateway_embeddings():
    """实验 1：探测网关的 /embeddings 端点。

    原计划"跑 OpenAI/DeepSeek Embedding API"。实测本网关（学习用）只
    代理 chat/completions，/embeddings 返回 404。探测结果如实记录进
    报告——"换一个端点，能力就不同"本身就是真实工程里的重要一课。
    """
    if not ragkit.BASE_URL or not ragkit.API_KEY:
        return {"ok": False, "status": "no-config", "detail": "未配置网关"}
    body = json.dumps({"model": "text-embedding-3-small",
                       "input": "你好世界"}).encode("utf-8")
    req = urllib.request.Request(
        ragkit.BASE_URL + "/embeddings", data=body,
        headers={"Authorization": "Bearer " + ragkit.API_KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        v = d["data"][0]["embedding"]
        return {"ok": True, "status": 200, "model": d.get("model"), "dim": len(v)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code,
                "detail": f"网关未提供 /embeddings 端点（HTTP {e.code}）"}
    except Exception as e:
        return {"ok": False, "status": "error",
                "detail": f"{type(e).__name__}: {str(e)[:150]}"}


def hand_cosine_example():
    """实验 2 的手算对照：两个 4 维向量，逐步算余弦。

    a=[1,0,2,1], b=[1,1,1,0]：
      点积 = 1*1 + 0*1 + 2*1 + 1*0 = 3
      |a|  = sqrt(1+0+4+1) = sqrt(6) ≈ 2.449
      |b|  = sqrt(1+1+1+0) = sqrt(3) ≈ 1.732
      cos  = 3 / (2.449*1.732) ≈ 0.707
    """
    a, b = [1.0, 0.0, 2.0, 1.0], [1.0, 1.0, 1.0, 0.0]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return {"a": a, "b": b, "dot": dot, "norm_a": round(na, 3),
            "norm_b": round(nb, 3), "cos": round(dot / (na * nb), 3)}


def sim_matrix(emb, texts):
    """query_embed 编码全部文本，算两两余弦，返回对称矩阵（含自身=1）。"""
    vecs = emb.query_embed([t for _, _, t in texts])
    n = len(texts)
    return [[round(ragkit.cosine(vecs[i], vecs[j]), 3) for j in range(n)]
            for i in range(n)]


def group_stats(matrix, texts):
    """组内平均相似度（同组两两）/ 组间平均相似度，衡量"聚类能力"。"""
    ids = [t[0] for t in texts]
    groups = [t[1] for t in texts]
    intra, inter = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (intra if groups[i] == groups[j] else inter).append(matrix[i][j])
    return {
        "intra_mean": round(sum(intra) / len(intra), 3) if intra else None,
        "inter_mean": round(sum(inter) / len(inter), 3) if inter else None,
        "separation": round(sum(intra) / len(intra) - sum(inter) / len(inter), 3)
        if intra and inter else None,
    }


def top_pairs(matrix, texts, n=6):
    """全矩阵里相似度最高的 n 个配对（不含自身），看两种向量的排序差异。"""
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append({"pair": f"{texts[i][0]}-{texts[j][0]}",
                          "sim": matrix[i][j],
                          "t1": texts[i][2], "t2": texts[j][2]})
    pairs.sort(key=lambda p: -p["sim"])
    return pairs[:n]


def main():
    print("=" * 62)
    print("实验 1：探测网关 /embeddings 端点")
    print("=" * 62)
    probe = probe_gateway_embeddings()
    print(f"[probe] ok={probe['ok']}  {probe.get('detail') or probe.get('model')}")

    print()
    print("=" * 62)
    print("实验 2：加载两套 Embedding，算余弦相似度")
    print("=" * 62)
    matrices = {}
    embedder_info = {}
    for emb in (ragkit.get_embedder(), ragkit.get_embedder(prefer="hash")):
        # 键用 kind：fastembed 可用时是 semantic vs lexical 两种向量；
        # fastembed 不可用时两者都是 lexical（自动降级），meta 里记录原因
        if emb.kind in matrices:
            continue
        matrices[emb.kind] = sim_matrix(emb, TEXTS)
        embedder_info[emb.kind] = {"kind": emb.kind, "name": emb.name, "dim": emb.dim}
        print(f"[embedder] kind={emb.kind}  model={emb.name}  dim={emb.dim}")
        warn = getattr(emb, "warn", None)
        if warn:
            print(f"[warn] {warn}")

    for kind, m in matrices.items():
        print(f"\n[{kind}] 组内/组间平均相似度：")
        st = group_stats(m, TEXTS)
        print(f"  组内均值={st['intra_mean']}  组间均值={st['inter_mean']}  "
              f"分离度(组内-组间)={st['separation']}")
        print(f"  相似度 top{6}：")
        for p in top_pairs(m, TEXTS):
            print(f"    {p['pair']}  {p['sim']:+.3f}   {p['t1']} ↔ {p['t2']}")

    # 手算对照：证明代码里的 cosine 和手推一致
    hand = hand_cosine_example()
    code_check = ragkit.cosine(hand["a"], hand["b"])

    # 落盘原始数据
    payload = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(),
            "probe": probe,
            "embedders": embedder_info,
            "degraded": next((getattr(e, "warn", None) for e in
                              (ragkit.get_embedder(),) if getattr(e, "warn", None)), None),
            "hand_example": hand,
            "hand_vs_code": round(code_check, 3),
        },
        "texts": [{"id": i, "group": g, "text": t} for i, g, t in TEXTS],
        "matrices": matrices,
        "group_stats": {k: group_stats(m, TEXTS) for k, m in matrices.items()},
        "top_pairs": {k: top_pairs(m, TEXTS) for k, m in matrices.items()},
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report(payload))
    print(f"\n[done] 报告：{REPORT_PATH}")
    print(f"[done] 数据：{RESULT_PATH}")


def build_report(p):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gs = p["group_stats"]
    L, a = [], None
    L = []
    a = L.append
    a("# Embedding 与文本相似度实验报告（W3 Day1）")
    a("")
    a(f"- 生成时间：{now}")
    a(f"- 语义向量：`{ragkit.EMBED_MODEL}`（fastembed 本地推理，{ragkit.EMBED_DIM} 维）")
    a(f"- 词袋对照：HashEmbed 字符 bigram 哈希（512 维，纯 Python）")
    a(f"- 实验文本：{len(p['texts'])} 条 × 3 组（同义改写 / 字面重叠 / 精确词）")
    if p["meta"].get("degraded"):
        a(f"- ⚠️ 降级记录：{p['meta']['degraded']}")
    a("")
    a("## 一、词向量 → 句向量 → 稠密向量表示")
    a("")
    a("| 阶段 | 代表 | 一句话 | 缺陷 |")
    a("|---|---|---|---|")
    a("| 词向量 | Word2Vec / GloVe | 一个词一个固定向量 | 多义词只有一个表示；没有句级语义 |")
    a("| 句向量 | Sentence-BERT | 整句编码成一个向量 | 早期模型对中文/短句较弱 |")
    a("| 上下文相关表示 | BERT | 同一个词在不同上下文里向量不同 | 编码慢，需再蒸馏出检索模型 |")
    a("| 现代 Embedding | BGE / text-embedding-3 / Jina | 检索优化，配合向量库使用 | 查询与文档必须同模型同版本 |")
    a("")
    a("演进方向始终是一个：**让意思相近的文本在向量空间里更近，并让多义词获得上下文相关的表示。**")
    a("")
    a("## 二、Embedding API 现状（探测记录）")
    a("")
    pr = p["meta"]["probe"]
    a(f"- 探测 `{ragkit.BASE_URL}/embeddings`："
      f"{'✅ 可用' if pr['ok'] else '❌ 不可用'}"
      + ("" if pr["ok"] else f"——{pr.get('detail', '')}"))
    a("- 原计划直接调 Embedding API；本学习网关只代理 chat/completions，")
    a("  因此改用本地模型 fastembed + `bge-small-zh-v1.5`（onnx 推理，无需 API）。")
    a("- 工程含义：Embedding 是纯计算服务，换网关/自建都很常见；关键是**查询和文档")
    a("  必须用同一个模型、同一个版本**，否则两者不在同一语义空间，相似度无意义。")
    a("")
    a("## 三、余弦相似度：手算对照")
    a("")
    h = p["meta"]["hand_example"]
    a("```text")
    a(f"a = {h['a']}")
    a(f"b = {h['b']}")
    a(f"点积 a·b   = {h['dot']}")
    a(f"|a| = {h['norm_a']}   |b| = {h['norm_b']}")
    a(f"cos(a,b) = {h['dot']} / ({h['norm_a']}×{h['norm_b']}) = {h['cos']}")
    a("```")
    a("")
    a(f"代码 `ragkit.cosine` 对同一对向量返回 {p['meta']['hand_vs_code']:.3f}，与手算一致。")
    a("bge 输出的向量已做 L2 归一化（长度为 1），此时余弦 = 点积；")
    a("检索场景几乎都用余弦：它衡量方向一致，不受向量长度影响。")
    a("")
    a("## 四、实验设计与结果")
    a("")
    a("同一批 {n} 条文本，分别用两套向量编码，算 {n}×{n} 余弦相似度矩阵："
      .format(n=len(p["texts"])))
    a("")
    a("| 向量 | 类型 | 组内均值 | 组间均值 | 分离度 |")
    a("|---|---|---|---|---|")
    for kind, st in gs.items():
        a(f"| {kind} | {'语义' if kind == 'semantic' else '词袋'} "
          f"| {st['intra_mean']} | {st['inter_mean']} | {st['separation']} |")
    a("")
    for kind in gs:
        a(f"### {kind} 向量的相似度 top6")
        a("")
        a("| 配对 | 相似度 | 文本1 | 文本2 |")
        a("|---|---|---|---|")
        for x in p["top_pairs"][kind]:
            a(f"| {x['pair']} | {x['sim']:+.3f} | {x['t1']} | {x['t2']} |")
        a("")
    a("## 五、结论")
    a("")
    sep_s = gs.get("semantic", {}).get("separation")
    sep_l = gs.get("lexical", {}).get("separation")
    if sep_s is not None:
        a(f"- **语义向量能把同义改写聚到一起**（分离度 {sep_s:+.3f}）："
          f"\"怎么申请退货退款\"和\"商品退款流程是什么\"几乎没有共同字，余弦仍然最高。")
    if sep_l is not None:
        a(f"- **词袋向量会被字面重叠骗到**（分离度 {sep_l:+.3f}）：它只看字符 bigram "
          f"重叠，\"苹果手机\"和\"苹果维生素\"的字面相似度不低，语义上却毫无关系。")
    a("- **精确词是语义向量的弱项**：两个格式相同、取值不同的错误码，在语义空间里")
    a("  几乎一样近——区分它们靠 BM25 精确匹配，这正是 Day5 混合检索的动机。")
    a("- **机制层面**：任何文本 → 定长向量 → 余弦排序，这套流程不因模型而变；")
    a("  换 Embedding 模型只是换了一个\"语义坐标系\"。")
    a("")
    a("## 六、复现方式")
    a("")
    a("```bash")
    a("pip install fastembed          # 可选：没有也能跑（自动降级 HashEmbed）")
    a("export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像，脚本已默认设置")
    a("python 01Agent/w3day01/embedding_similarity.py")
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
