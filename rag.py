"""
RAG 检索层（与框架无关）。

职责：
  1. chunk_text     —— 把文档切成小块（递归分隔符 + overlap），对应面试题里的 chunking。
  2. LocalEmbedding —— 用本地 fastembed 模型（BAAI/bge-small-zh-v1.5）把文本变成向量，离线、免费、不耗 key。
  3. VectorStore    —— 包 chromadb 持久化向量库：建库(add_documents) / 检索(retrieve) / 元数据。

用法见 build_kb.py（建库）与 backends/raw.py（每轮把检索上下文注入提示词）。
"""

import os
import math
import re
import shutil
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

# Rerank 重排序模型：fastembed 的 cross-encoder（本机 fastembed 0.8 支持列表中最贴合
# 中文的是 BAAI/bge-reranker-base，与嵌入模型 BGE 同源）。首次运行会下载 ONNX 权重。
try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except Exception:  # 老版本 fastembed 没带 reranker 时降级为不可用（由开关控制）
    TextCrossEncoder = None

from rank_bm25 import BM25Okapi

# ============================================================
# 配置
# ============================================================
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "rawagent_kb"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 中文优化，512 维，首次自动下载约 120MB
RERANK_MODEL = "BAAI/bge-reranker-base"  # 重排序 cross-encoder（中文友好，与嵌入同源）
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# 混合检索 / 重排序相关
CANDIDATE_N = 20     # 向量召回与 BM25 召回各自取 top-N 候选，再融合 / 重排
RRF_K = 60           # RRF（Reciprocal Rank Fusion）常数，业界常用 60
MAX_RRF = 2.0 / (RRF_K + 1)  # 单个文档在「向量+BM25 双路都排第 1」时的最大 RRF 分

# 递归分隔符：优先按段落/句末切，最后才按字符切（结构感知切块）
_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


# ============================================================
# 切块（chunking）—— 面试题里的「可挖点」
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """递归字符切块：先按大分隔符切，块仍超长则继续下一级分隔符，直到满足 chunk_size。"""
    text = text.strip()
    if not text:
        return []

    def _split(segment: str, seps: list[str]) -> list[str]:
        if len(segment) <= chunk_size or not seps:
            return [segment]
        sep = seps[0]
        # 按分隔符切开，并把分隔符保留回每个片段尾部（除非分隔符是空串）
        parts = segment.split(sep)
        if sep:
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
        merged: list[str] = []
        buf = ""
        for p in parts:
            if len(buf) + len(p) <= chunk_size:
                buf += p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        # 若某块仍超长，递归用更细分隔符再切
        result: list[str] = []
        for m in merged:
            if len(m) > chunk_size and len(seps) > 1:
                result.extend(_split(m, seps[1:]))
            else:
                result.append(m)
        return result

    pieces = _split(text, _SEPARATORS)
    # 加 overlap：相邻块首部重复前一块的尾部，缓解语义在边界处断裂
    if overlap <= 0 or len(pieces) <= 1:
        return [p for p in pieces if p.strip()]
    out: list[str] = []
    for i, p in enumerate(pieces):
        if i == 0:
            out.append(p)
        else:
            prev = pieces[i - 1]
            prefix = prev[-overlap:] if len(prev) >= overlap else prev
            out.append(prefix + p)
    return [o for o in out if o.strip()]


# ============================================================
# 本地嵌入（fastembed）—— 离线、免费、不耗 key
# ============================================================
class LocalEmbedding:
    """封装 fastembed，把一批文本变成向量列表（与输入同序）。"""

    def __init__(self, model_name: str = EMBED_MODEL):
        # 懒加载：第一次 embed 时才下载/载入模型，避免导入即拖慢启动
        self._model = None
        self._name = model_name

    def _ensure(self):
        if self._model is None:
            self._model = TextEmbedding(model_name=self._name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        vectors = list(self._model.embed(texts))
        return [v.tolist() for v in vectors]


# ============================================================
# 中文分词（混合检索用）—— 零额外依赖按「字」切，可选 jieba 增强
# ============================================================
def _tokenize(text: str) -> list[str]:
    """BM25 关键词召回的中文分词。

    默认按字切分（中文单字命中率高、零依赖）。若环境已装 jieba 则自动改用分词，
    召回更精准。空白与标点会被丢弃，避免噪声。
    """
    text = re.sub(r"\s+", "", text)
    if not text:
        return []
    try:
        import jieba  # 可选增强：装了 jieba 就用语义分词
        return [t for t in jieba.cut(text) if t.strip()]
    except Exception:
        return list(text)


def _sigmoid(x: float) -> float:
    """把 cross-encoder 的无界 logit 映射到 (0,1)，便于复用 0~1 阈值做门控。"""
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# 向量库（chromadb 持久化）+ 混合检索 + Rerank 重排序
# ============================================================
class VectorStore:
    """chromadb 持久化向量库封装：建库 / 检索，自带来源元数据。

    检索链路（默认开启全部优化）：
        向量召回 top-N  ┐
                       ├─ RRF 融合 ─► cross-encoder Rerank 精排 ─► top-k
        BM25 召回 top-N┘
    可通过开关关闭 Rerank / 混合检索，退化为纯向量召回。
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection: str = DEFAULT_COLLECTION,
        rerank: bool = True,
        hybrid: bool = True,
        candidate_n: int = CANDIDATE_N,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # cosine 距离，与多数 embedding 评测一致
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = LocalEmbedding()
        self.use_rerank = rerank and TextCrossEncoder is not None
        self.use_hybrid = hybrid
        self.candidate_n = candidate_n
        # 惰性构建，避免无 RAG 时白加载
        self._reranker = None
        self._bm25 = None
        self._bm25_ids: list[str] | None = None

    def count(self) -> int:
        return self.collection.count()

    def add_documents(self, chunks: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """批量嵌入并写入（显式传 embeddings，避免 chromadb 默认联网模型）。

        用 upsert 而非 add：重复建库时相同 id 直接覆盖，幂等不报错。
        """
        if not chunks:
            return
        embeddings = self.embedder(chunks)
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ---------------- 惰性组件 ----------------
    def _ensure_reranker(self):
        if self._reranker is None and self.use_rerank:
            self._reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    def _ensure_bm25(self):
        """从持久化库一次性读出全部文档构建 BM25 索引（中文按字切分）。"""
        if self._bm25 is not None or not self.use_hybrid:
            return
        data = self.collection.get(include=["documents"])
        docs = data.get("documents") or []
        ids = data.get("ids") or []
        self._bm25_ids = ids
        if docs:
            self._bm25 = BM25Okapi([_tokenize(d) for d in docs])

    # ---------------- 召回 ----------------
    def _vector_recall(self, query: str, n: int) -> tuple[list[str], dict[str, float]]:
        """向量召回 top-n，返回 (id 列表, {id: 余弦相似度})。"""
        if self.count() == 0:
            return [], {}
        q_emb = self.embedder([query])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(n, self.count()),
            include=["distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        sims = {i: 1.0 - float(d) for i, d in zip(ids, dists)}
        return ids, sims

    def _bm25_recall(self, query: str, n: int) -> tuple[list[str], dict[str, float]]:
        """BM25 关键词召回 top-n，返回 (id 列表, {id: BM25 原始分})。"""
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_ids:
            return [], {}
        toks = _tokenize(query)
        if not toks:
            return [], {}
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        ids = [self._bm25_ids[i] for i in order]
        sims = {self._bm25_ids[i]: float(scores[i]) for i in order}
        return ids, sims

    @staticmethod
    def _rrf_fuse(*rank_lists, k: int = RRF_K) -> dict[str, float]:
        """Reciprocal Rank Fusion：把多路召回按排名融合成统一分数。

        公式 score = Σ 1/(k + rank)，与具体分值量纲无关，专门解决
        「向量相似度」和「BM25 分」量纲不可比、无法直接相加的问题。
        """
        fused: dict[str, float] = {}
        for ranks in rank_lists:
            for rank, doc_id in enumerate(ranks):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def _gate_score(
        self, top1: str, vector_sims: dict[str, float], fused: dict[str, float], rerank_scores: dict[str, float]
    ) -> float:
        """计算门控分数（0~1），复用 rag_threshold 语义：

        - 重排序开启且 top-1 来自 BM25 独有命中 → 用 rerank 的 sigmoid 分（bm25-only 无余弦分）
        - 否则优先用 top-1 的余弦相似度（与改造前阈值含义一致）
        - 纯向量召回时 fused 即余弦相似度，行为完全等价于改造前
        """
        if self.use_rerank and top1 in rerank_scores and top1 not in vector_sims:
            return _sigmoid(rerank_scores[top1])
        if top1 in vector_sims:
            return vector_sims[top1]
        # 退化情况：BM25 独有命中且无 rerank，用归一化 RRF 分近似
        return min(1.0, fused.get(top1, 0.0) / MAX_RRF) if MAX_RRF else 0.0

    def retrieve(self, query: str, k: int = 3) -> tuple[str, float, str]:
        """检索 top-k，返回 (拼接好的上下文串, 门控分数[0,1], 管线信息串)。

        门控分数语义保持与改造前一致：优先为 top-1 的余弦相似度（0=不相似,1=相同），
        仅在重排序命中 BM25 独有片段时用 rerank 的 sigmoid 分。空库返回 ("", 0.0, "空库")。
        """
        if self.count() == 0:
            return "", 0.0, "空库"

        # 1) 候选召回：向量 +（可选）BM25
        vector_ids, vector_sims = self._vector_recall(query, self.candidate_n)
        if self.use_hybrid:
            bm25_ids, _ = self._bm25_recall(query, self.candidate_n)
            fused = self._rrf_fuse(vector_ids, bm25_ids)
            pipeline = "向量+BM25(RRF)"
        else:
            fused = vector_sims
            pipeline = "纯向量"
        cand_ids = list(fused.keys())[: max(self.candidate_n, k)]

        # 2) 取候选文档与元数据（按 cand_ids 顺序返回）
        docs_meta = self.collection.get(ids=cand_ids, include=["documents", "metadatas"])
        cand_docs = (docs_meta.get("documents") or [])
        cand_metas = (docs_meta.get("metadatas") or [])
        doc_map = {i: d for i, d in zip(cand_ids, cand_docs)}
        meta_map = {i: m for i, m in zip(cand_ids, cand_metas)}

        # 3) Rerank 精排（可选）：cross-encoder 对候选逐条打分，取 top-k
        rerank_scores: dict[str, float] = {}
        if self.use_rerank and cand_ids:
            self._ensure_reranker()
            # rerank() 返回与 cand_docs 顺序一一对应的分数列表
            scores = list(self._reranker.rerank(query, cand_docs))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            final_ids = [cand_ids[i] for i in order]
            rerank_scores = {cand_ids[i]: float(scores[i]) for i in order}
            pipeline += "→Rerank"
        else:
            final_ids = cand_ids[:k]

        # 4) 组装上下文 + 门控分数
        top_k = final_ids[:k]
        gate = self._gate_score(top_k[0], vector_sims, fused, rerank_scores) if top_k else 0.0
        lines = []
        for i, cid in enumerate(top_k, 1):
            source = (meta_map.get(cid) or {}).get("source", "?")
            lines.append(f"[片段{i} | 来源:{source}]\n{doc_map.get(cid, '')}")
        return "\n\n".join(lines), gate, pipeline


# ============================================================
# 从目录建库（供 build_kb.py 调用）
# ============================================================
def build_from_directory(
    doc_dir: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    reset: bool = False,
):
    """读取 doc_dir 下所有 .md/.txt，切块→嵌入→写入 chromadb，返回 (store, summary)。"""
    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    store = VectorStore(persist_dir, collection)
    exts = ("*.md", "*.txt")
    files: list[Path] = []
    for ext in exts:
        files.extend(Path(doc_dir).rglob(ext))
    files = sorted(set(files))

    chunks_all: list[str] = []
    metas_all: list[dict] = []
    ids_all: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(doc_dir).as_posix()
        chunks = chunk_text(text, chunk_size, overlap)
        for i, c in enumerate(chunks):
            chunks_all.append(c)
            metas_all.append({"source": rel, "chunk_index": i})
            # 稳定 id：来源+块号，保证重复建库可去重
            ids_all.append(f"{rel}::{i}")

    before = store.count()
    store.add_documents(chunks_all, metas_all, ids_all)
    after = store.count()
    summary = {"files": len(files), "chunks": len(chunks_all), "before": before, "after": after}
    return store, summary
