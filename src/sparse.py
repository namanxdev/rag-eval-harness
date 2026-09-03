"""BM25 over the same chunks the dense index holds, and rank fusion.

Dense retrieval fails in a specific, repeatable way: it matches on topic, so a
query naming a defined term ("the Territory", "Section 7.3", a party name) can
lose to a chunk that is merely about the same subject. BM25 fails the opposite
way. Fusing them is worth measuring precisely because the two error modes are
not the same error mode.

Fusion is Reciprocal Rank Fusion rather than a weighted sum of scores. Cosine
similarity and BM25 scores are on unrelated scales, so combining them requires
normalising, and every normalisation introduces a weight with no principled
value -- one more knob to fit to this corpus and to mislead on the next one.
RRF reads only the rank positions, so there is nothing to tune.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.chunking import Chunk

# Fusion constant from the original RRF paper. It damps the contribution of the
# top rank enough that one retriever cannot dominate on its own. Left at the
# published value deliberately; tuning it per corpus is how a fused ranking
# stops generalising.
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Used for both the index and the query.

    Deliberately plain: no stemming, no stopword list. Contract language is
    already repetitive, and every extra normalisation step is another thing
    that has to be reproduced exactly on the client's side to get the same
    numbers.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Lexical index over a chunk list, scoped at query time like the dense one.

    IDF is computed over the whole corpus, not per document. A term that is
    rare across the corpus is informative even when the search is scoped to one
    contract, and per-document statistics over a few dozen chunks are noise.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, limit: int, doc_id: str | None = None) -> list[Chunk]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = range(len(self.chunks))
        if doc_id is not None:
            idx = [i for i in idx if self.chunks[i].doc_id == doc_id]
        # Ties broken by chunk order so a run is reproducible. BM25 returns 0.0
        # for every chunk sharing no term with the query, and on a scoped search
        # that can be most of them.
        ranked = sorted(idx, key=lambda i: (-float(scores[i]), i))
        return [self.chunks[i] for i in ranked[:limit]]


def reciprocal_rank_fusion(rankings: dict[str, list[Chunk]], limit: int,
                           k: int = RRF_K) -> tuple[list[Chunk], dict[str, dict]]:
    """Fuse named rankings by 1/(k + rank), and record where each chunk came from.

    Returns the fused list and, per chunk_id, the rank it held in each input
    ranking (1-based, None when that retriever did not return it). The
    attribution is the commercially interesting half: "these are the queries
    dense retrieval alone gets wrong" needs per-query evidence, not a delta.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    sources: dict[str, dict] = {}

    for name, ranking in rankings.items():
        for rank, c in enumerate(ranking, 1):
            seen.setdefault(c.chunk_id, c)
            sources.setdefault(c.chunk_id, {n: None for n in rankings})
            sources[c.chunk_id][name] = rank
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)

    # Tie-break on the best rank any retriever gave the chunk, then chunk_id,
    # so equal scores resolve the same way on every run.
    def order(cid: str):
        best = min((r for r in sources[cid].values() if r is not None), default=10**9)
        return (-scores[cid], best, cid)

    fused = [seen[cid] for cid in sorted(scores, key=order)][:limit]
    return fused, sources
