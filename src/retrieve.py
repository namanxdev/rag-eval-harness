"""Dense retrieval, optional sparse fusion, optional cross-encoder rerank.

The two-stage shape is the point: a bi-encoder is cheap enough to score every
chunk but embeds the query and the chunk independently, so it cannot weigh how
well they match each other. A cross-encoder reads the pair together and is far
more accurate, but far too slow to run over the corpus -- so it only ever sees
the shortlist the first stage hands it.

That shortlist can come from dense retrieval, from BM25, or from the two fused
(see `src/sparse.py`). Which one is a config, not a code change, because the
question the harness answers is which of them a given corpus needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from src.chunking import Chunk
from src.index import ChunkIndex
from src.sparse import BM25Index, reciprocal_rank_fusion

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # local, no API key

RETRIEVERS = ("dense", "bm25", "hybrid")


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[Chunk]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [candidates[i] for i in order[:top_k]]


@dataclass(frozen=True)
class Ranking:
    """A ranked chunk list plus where each chunk came from.

    `sources` maps chunk_id to the 1-based rank that chunk held in each
    first-stage retriever, None where that retriever did not return it. Empty
    for single-retriever configs, where the attribution would be trivial.
    """
    chunks: list[Chunk]
    sources: dict[str, dict] = field(default_factory=dict)


def retrieve_ranked(index: ChunkIndex, query: str, k: int, doc_id: str | None = None,
                    reranker: Reranker | None = None, candidate_k: int = 30,
                    sparse: BM25Index | None = None,
                    retriever: str = "dense") -> Ranking:
    """Top k chunks with retriever attribution. See `retrieve` for the plain list."""
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}; expected one of {RETRIEVERS}")
    if retriever != "dense" and sparse is None:
        raise ValueError(f"retriever {retriever!r} needs a BM25Index")

    # Rerankers only ever reorder; the pool has to be wide enough that the
    # chunk they should promote is in it.
    pool = max(candidate_k, k) if reranker is not None else k

    sources: dict[str, dict] = {}
    if retriever == "dense":
        candidates = index.search(query, pool, doc_id=doc_id)
    elif retriever == "bm25":
        candidates = sparse.search(query, pool, doc_id=doc_id)
    else:
        # Fuse over the wider pool, not over the top k of each: a chunk that is
        # 8th on both lists should beat one that is 1st on a single list, and
        # it can only do that if both lists are long enough to contain it.
        wide = max(candidate_k, k)
        candidates, sources = reciprocal_rank_fusion(
            {"dense": index.search(query, wide, doc_id=doc_id),
             "bm25": sparse.search(query, wide, doc_id=doc_id)},
            limit=pool,
        )

    if reranker is not None:
        candidates = reranker.rerank(query, candidates, k)
    return Ranking(candidates[:k], sources)


def retrieve(index: ChunkIndex, query: str, k: int, doc_id: str | None = None,
             reranker: Reranker | None = None, candidate_k: int = 30,
             sparse: BM25Index | None = None, retriever: str = "dense") -> list[Chunk]:
    """Return the top k chunks, reranking a wider candidate pool when given one."""
    return retrieve_ranked(index, query, k, doc_id, reranker, candidate_k,
                           sparse, retriever).chunks
