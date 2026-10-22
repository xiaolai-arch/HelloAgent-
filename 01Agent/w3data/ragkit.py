# -*- coding: utf-8 -*-
"""ragkit —— W3 共享工具库：语料加载 / LLM 调用 / Embedding / 检索指标

W3 六天的实验脚本（embedding、Chroma CRUD、最小 RAG、Chunking 对比、
混合检索+Rerank、RAGAS 评估）都要用同一批基础设施，抽到这里统一维护。
每个脚本通过 sys.path 把本目录加进来后 `import ragkit`。

设计约定（与 w1/w2 脚本保持一致的风格）：
- LLM 调用走标准库 urllib 的 OpenAI 兼容 /chat/completions，
  配置文件 ../../vision_config.json 优先，环境变量兜底，密钥不进代码。
- Embedding 有两套实现，自动降级：
    1. fastembed + BAAI/bge-small-zh-v1.5（真·语义向量，onnx 本地推理）
       —— 中文小模型，512 维；首次使用需下载约 90MB 权重。
       huggingface.co 在国内不可达，因此默认走 hf-mirror.com 镜像
       （已设 HF_ENDPOINT，可用环境变量覆盖）。
    2. HashEmbed：纯 Python 字符 n-gram 哈希词袋（零依赖，任何环境能跑）
       —— 它不是语义向量，只是"文本→定长向量→余弦相似度"的机制演示。
  fastembed 没装或模型下载失败时自动落到 2，并把 .kind 标为 lexical，
  实验报告据此如实说明当前用的是哪套。
"""
import os
import sys
import json
import math
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# huggingface.co 主站在国内普遍不可达；hf-mirror.com 是常用镜像。
# HF_HUB_DISABLE_XET=1：新版 huggingface-hub 默认走 Xet 存储协议，镜像站
# 会对其返回 401，关掉后回落到经典 HTTP 下载，镜像可用。
# setdefault = 用户显式设了这两个变量就尊重用户。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---- LLM 网关配置（与 first_llm_call.py 完全同一套策略）----
CONFIG_PATH = os.path.join(BASE_DIR, "../../vision_config.json")

_cfg = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

BASE_URL = (_cfg.get("base_url") or os.environ.get("COPILOT_PROVIDER_BASE_URL", "")).rstrip("/")
API_KEY = _cfg.get("api_key") or os.environ.get("COPILOT_PROVIDER_API_KEY", "")
MODEL = _cfg.get("model", "mimo-v2.6-flash")


def chat(messages, temperature=0.2, max_tokens=2000, extra=None, timeout=180):
    """OpenAI 兼容 POST /chat/completions，返回 (resp_dict, error_str)。

    与 w2day04 的 chat() 同款：extra 是参数注入点（tools / response_format），
    错误不抛异常而是原样返回，让上层实验决定怎么记录。
    """
    if not BASE_URL or not API_KEY:
        return None, "缺少网关配置（vision_config.json / 环境变量）"
    payload = {"model": MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + API_KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def chat_text(messages, **kw):
    """chat() 的便捷版：直接返回文本回复（出错时返回 None）。"""
    resp, err = chat(messages, **kw)
    if err:
        return None
    return resp["choices"][0]["message"]["content"]


# =============================================================================
# Embedding：语义（fastembed/bge-zh）优先，词袋哈希兜底
# =============================================================================

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512


class HashEmbed:
    """纯 Python 字符 n-gram 哈希"词袋"向量（零依赖兜底）。

    原理：文本拆成字符 bigram，每个 bigram 经稳定哈希映射到 512 维中的
    某几位做计数，最后 L2 归一化。它捕捉的是"字面重叠"，不捕捉语义——
    恰好用来做对照实验：证明"词袋向量对同义改写无能为力"。
    哈希用 md5 而不是内置 hash()：后者每次进程重启 salt 不同，向量不可复现。
    """

    kind = "lexical"
    name = "hash-bigram-512"
    dim = EMBED_DIM

    def _bigrams(self, text):
        text = "".join(text.lower().split())
        if len(text) == 1:
            return [text]
        return [text[i:i + 2] for i in range(len(text) - 1)]

    def embed(self, text):
        import hashlib
        vec = [0.0] * self.dim
        for bg in self._bigrams(text):
            h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return _l2norm(vec)

    # 与 fastembed 对齐的双接口
    def passage_embed(self, texts):
        return [self.embed(t) for t in texts]

    def query_embed(self, texts):
        return [self.embed(t) for t in texts]


class FastEmbed:
    """fastembed + bge-small-zh-v1.5：本地 onnx 推理的真语义向量。

    passage_embed 编码文档、query_embed 编码问题——bge 系列的 zh 模型
    给查询侧加了一句检索指令前缀（"为这个句子生成表示……"），
    非对称检索（短问题查长文档）下能提升召回，官方推荐这么用。
    """

    kind = "semantic"
    name = EMBED_MODEL

    def __init__(self):
        from fastembed import TextEmbedding  # 延迟导入：没装就走 HashEmbed
        self._model = TextEmbedding(EMBED_MODEL)
        self.dim = EMBED_DIM

    def passage_embed(self, texts):
        # fastembed 返回 numpy.float32：json.dump 不认，转成原生 float
        return [[float(x) for x in v] for v in self._model.passage_embed(texts)]

    def query_embed(self, texts):
        return [[float(x) for x in v] for v in self._model.query_embed(texts)]


_embedders = {"semantic": None, "lexical": None}


def get_embedder(prefer="auto"):
    """取 Embedder（懒加载；语义/词袋各缓存一个实例，避免重复加载模型）。

    prefer: "auto"  fastembed 可用就用语义向量，否则哈希词袋
            "hash"  强制哈希词袋（做对照实验时用）
    加载失败不抛异常：返回 HashEmbed 并带上 warn 字段说明降级原因。
    """
    if prefer != "hash":
        if _embedders["semantic"] is None:
            try:
                _embedders["semantic"] = FastEmbed()
                _embedders["semantic"].warn = None
            except Exception as e:
                h = HashEmbed()
                h.warn = (f"fastembed/{EMBED_MODEL} 不可用"
                          f"（{type(e).__name__}: {str(e)[:120]}），已降级为 HashEmbed 词袋向量")
                _embedders["semantic"] = h
        return _embedders["semantic"]
    if _embedders["lexical"] is None:
        _embedders["lexical"] = HashEmbed()
        _embedders["lexical"].warn = None
    return _embedders["lexical"]


def _l2norm(vec):
    s = math.sqrt(sum(x * x for x in vec))
    return [x / s for x in vec] if s else vec


def cosine(a, b):
    """余弦相似度。输入已 L2 归一化时等价于点积。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# =============================================================================
# 语料与切分
# =============================================================================

def load_corpus():
    """读取 corpus/*.md，返回 [{id, title, text}]，按文件名排序保证可复现。"""
    corpus_dir = os.path.join(BASE_DIR, "corpus")
    docs = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()
        title = fname[:-3]
        # 首个一级标题当作文档标题（有则用，无则用文件名）
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append({"id": fname[:-3], "title": title, "text": text})
    return docs


def load_eval_set():
    """读取 golden set（eval_set.json）。"""
    with open(os.path.join(BASE_DIR, "eval_set.json"), "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def recursive_chunk(text, max_chars=300, separators=("\n## ", "\n### ", "\n\n", "\n", "。", "；", "，")):
    """递归字符切分：优先在大边界切开，太长的块继续用下一级边界切。

    与 LangChain RecursiveCharacterTextSplitter 同思想：尽量让 chunk 停在
    语义边界上。这里按字符计长（教学实现）；生产应按 token 计。
    返回 [(chunk_text, 在原文中的起始位置)]。
    """
    def split(text, seps):
        # 找第一个在文本中出现的分隔符
        sep = next((s for s in seps if s in text), None)
        pieces = text.split(sep) if sep else [text]
        # split 会丢掉分隔符本身，把它接回块尾保持原文可拼接
        if sep:
            pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]
        return [p for p in pieces if p.strip()]

    def recurse(text, seps):
        if len(text) <= max_chars:
            return [(text, 0)] if text.strip() else []
        sub = split(text, seps)
        if len(sub) == 1:  # 没有任何边界可切，硬切
            mid = max_chars
            return [(text[:mid], 0)] + (recurse_tail(text[mid:], max_chars) if text[mid:] else [])
        out, offset = [], 0
        rest_seps = seps[1:] if len(seps) > 1 else seps
        for piece in sub:
            if len(piece) > max_chars:
                for t, pos in recurse(piece, rest_seps):
                    out.append((t, offset + pos))
            else:
                out.append((piece, offset))
            offset += len(piece)
        return out

    def recurse_tail(text, max_chars):
        return [(text[i:i + max_chars], i) for i in range(0, len(text), max_chars)]

    # 合并过短的相邻块（< max_chars 的 20%），避免一堆一两句话的碎块
    raw = recurse(text, separators)
    merged, buf, buf_pos = [], "", 0
    min_len = max(1, int(max_chars * 0.2))
    for t, pos in raw:
        if not buf:
            buf, buf_pos = t, pos
        elif len(buf) + len(t) <= max_chars:
            buf += t
        else:
            if len(buf) < min_len and merged:
                merged[-1] = (merged[-1][0] + buf, merged[-1][1])
            else:
                merged.append((buf, buf_pos))
            buf, buf_pos = t, pos
    if buf:
        if len(buf) < min_len and merged:
            merged[-1] = (merged[-1][0] + buf, merged[-1][1])
        else:
            merged.append((buf, buf_pos))
    return [(t.strip(), pos) for t, pos in merged if t.strip()]


# =============================================================================
# 检索评估指标（golden set 驱动）
# =============================================================================

def chunk_is_relevant(chunk_text, gold):
    """判定一个 chunk 是否算"相关"：命中关键词数 >= min_keyword_hits。

    关键词匹配用"包含"而不是全等：gold 写"600"，chunk 里"600 元/晚"算命中。
    无关键词（unanswerable 题）返回 False——它没有可命中的 chunk。
    """
    kws = gold.get("gold_keywords") or []
    if not kws:
        return False
    hits = sum(1 for k in kws if k in chunk_text)
    return hits >= gold.get("min_keyword_hits", 1)


def retrieval_metrics(ranked_relevant, ks=(1, 3, 5)):
    """由"每个问题的命中标记列表（按排名排序，True=相关）"计算指标。

    ranked_relevant: [[bool, bool, ...], ...] 外层每个元素是一个问题的
                     top-k 排名结果；这里不设上限，指标按 ks 截断。
    返回 {recall@k, mrr, hit_rate}（k 遍历 ks）。
    """
    out = {}
    for k in ks:
        rel_k = [row[:k] for row in ranked_relevant]
        recall = sum(sum(row) / 1 for row in rel_k if row) / len(rel_k) if rel_k else 0.0
        out[f"recall@{k}"] = round(recall, 3)
    rr = []
    for row in ranked_relevant:
        rr.append(0.0)
        for i, hit in enumerate(row):
            if hit:
                rr[-1] = 1.0 / (i + 1)
                break
    out["mrr"] = round(sum(rr) / len(rr), 3) if rr else 0.0
    out["hit_rate"] = round(sum(1 for row in ranked_relevant if any(row)) / len(ranked_relevant), 3) if ranked_relevant else 0.0
    return out


def rrf_fuse(rank_lists, k=60, top_n=None):
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))。

    rank_lists: [[doc_id, ...], ...] 每一路检索的排名列表（第 1 名排最前）。
    只用排名不用原始分数——两路分数量纲不同没法加，排名无量纲。
    返回按融合分降序的 [doc_id, score]。
    """
    scores = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n] if top_n else fused
