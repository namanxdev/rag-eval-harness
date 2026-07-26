"""Dense retrieval, with an optional cross-encoder rerank stage.

The two-stage shape is the point: a bi-encoder is cheap enough to score every
chunk but embeds the query and the chunk independently, so it cannot weigh how
well they match each other. A cross-encoder reads the pair together and is far
more accurate, but far too slow to run over the corpus -- so it only ever sees
the shortlist the dense stage hands it.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.chunking import Chunk
from src.index import ChunkIndex

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # local, no API key


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


def retrieve(index: ChunkIndex, query: str, k: int, doc_id: str | None = None,
             reranker: Reranker | None = None, candidate_k: int = 30) -> list[Chunk]:
    """Return the top k chunks, reranking a wider candidate pool when given one."""
    if reranker is None:
        return index.search(query, k, doc_id=doc_id)
    candidates = index.search(query, max(candidate_k, k), doc_id=doc_id)
    return reranker.rerank(query, candidates, k)
