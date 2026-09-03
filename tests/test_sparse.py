"""BM25 and rank fusion. No model loads here -- both are pure Python."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.chunking import Chunk
from src.retrieve import Ranking, retrieve, retrieve_ranked
from src.sparse import RRF_K, BM25Index, reciprocal_rank_fusion, tokenize

DOCS = {
    "a": ["the Territory means North America and its possessions",
          "governing law of the State of New York",
          "an unrelated recital about the parties"],
    "b": ["the Territory means the United Kingdom only",
          "indemnification and limitation of liability"],
}


def corpus() -> list[Chunk]:
    out, pos = [], 0
    for doc_id, texts in DOCS.items():
        pos = 0
        for n, t in enumerate(texts):
            out.append(Chunk(doc_id, f"{doc_id}:c:{n}", t, pos, pos + len(t), "test"))
            pos += len(t)
    return out


def chunk(cid: str, doc_id: str = "a") -> Chunk:
    return Chunk(doc_id, cid, cid, 0, 1, "test")


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------

def test_tokenize_is_lowercase_alphanumeric():
    assert tokenize("Section 7.3 -- the TERRITORY's scope!") == \
        ["section", "7", "3", "the", "territory", "s", "scope"]


def test_index_and_query_share_one_tokenizer():
    """A mismatch here silently halves recall and shows up as a model problem."""
    idx = BM25Index(corpus())
    assert idx.bm25.corpus_size == len(corpus())
    assert idx.search("TERRITORY", 1, doc_id="b")[0].chunk_id == "b:c:0"


# ---------------------------------------------------------------------------
# BM25 scoping and limits
# ---------------------------------------------------------------------------

def test_search_respects_the_doc_filter():
    idx = BM25Index(corpus())
    hits = idx.search("Territory", 5, doc_id="b")
    assert hits and all(c.doc_id == "b" for c in hits)


def test_search_respects_the_limit():
    idx = BM25Index(corpus())
    assert len(idx.search("Territory", 2)) == 2


def test_unscoped_search_spans_documents():
    idx = BM25Index(corpus())
    assert {c.doc_id for c in idx.search("Territory", 10)} == {"a", "b"}


def test_zero_score_chunks_are_ordered_deterministically():
    """Most chunks share no term with a query, so ties are the common case."""
    idx = BM25Index(corpus())
    first = [c.chunk_id for c in idx.search("zzzz nonexistent", 5)]
    assert first == [c.chunk_id for c in idx.search("zzzz nonexistent", 5)]


# ---------------------------------------------------------------------------
# reciprocal rank fusion
# ---------------------------------------------------------------------------

def test_agreement_beats_a_single_strong_vote():
    """The reason to fuse at all: two mediocre votes outweigh one loud one."""
    both = chunk("both")
    solo = chunk("solo")
    fused, _ = reciprocal_rank_fusion(
        {"dense": [chunk(f"d{i}") for i in range(7)] + [both],
         "bm25": [solo] + [chunk(f"s{i}") for i in range(6)] + [both]},
        limit=3,
    )
    assert fused[0].chunk_id == "both"
    assert 2 / (RRF_K + 8) > 1 / (RRF_K + 1)


def test_sources_record_rank_in_each_ranking():
    a, b = chunk("a"), chunk("b")
    _, sources = reciprocal_rank_fusion({"dense": [a, b], "bm25": [b]}, limit=5)
    assert sources["a"] == {"dense": 1, "bm25": None}
    assert sources["b"] == {"dense": 2, "bm25": 1}


def test_fusion_deduplicates():
    a = chunk("a")
    fused, _ = reciprocal_rank_fusion({"dense": [a], "bm25": [a]}, limit=5)
    assert [c.chunk_id for c in fused] == ["a"]


def test_fusion_respects_the_limit():
    cs = [chunk(f"c{i}") for i in range(10)]
    fused, _ = reciprocal_rank_fusion({"dense": cs, "bm25": cs[::-1]}, limit=4)
    assert len(fused) == 4


def test_fusion_is_deterministic_under_ties():
    """Every chunk here scores identically; the order must still be stable."""
    cs = [chunk(f"c{i}") for i in range(5)]
    runs = {tuple(c.chunk_id for c in
                  reciprocal_rank_fusion({"dense": cs, "bm25": cs}, limit=5)[0])
            for _ in range(3)}
    assert len(runs) == 1


def test_empty_rankings_fuse_to_nothing():
    fused, sources = reciprocal_rank_fusion({"dense": [], "bm25": []}, limit=5)
    assert fused == [] and sources == {}


# ---------------------------------------------------------------------------
# retrieve() wiring
# ---------------------------------------------------------------------------

class StubIndex:
    """Stands in for ChunkIndex so these tests need no embedding model."""

    def __init__(self, hits: list[Chunk]):
        self.hits = hits
        self.calls: list[tuple] = []

    def search(self, query: str, limit: int, doc_id: str | None = None) -> list[Chunk]:
        self.calls.append((query, limit, doc_id))
        return [c for c in self.hits if doc_id is None or c.doc_id == doc_id][:limit]


def test_dense_default_is_unchanged_by_the_sparse_parameter():
    """Invariant: sparse=None must behave exactly as the pre-hybrid code did."""
    cs = [chunk(f"c{i}") for i in range(10)]
    idx = StubIndex(cs)
    assert [c.chunk_id for c in retrieve(idx, "q", 3, doc_id="a")] == ["c0", "c1", "c2"]
    assert idx.calls == [("q", 3, "a")]


def test_dense_widens_the_pool_only_for_a_reranker():
    cs = [chunk(f"c{i}") for i in range(10)]
    idx = StubIndex(cs)
    reranker = SimpleNamespace(rerank=lambda q, cands, k: list(reversed(cands))[:k])
    out = retrieve(idx, "q", 3, doc_id="a", reranker=reranker, candidate_k=8)
    assert idx.calls == [("q", 8, "a")]
    assert [c.chunk_id for c in out] == ["c7", "c6", "c5"]


def test_bm25_only_never_touches_the_dense_index():
    cs = [chunk(f"c{i}") for i in range(5)]
    idx, sparse = StubIndex(cs), BM25Index(corpus())
    out = retrieve(idx, "Territory", 2, doc_id="b", sparse=sparse, retriever="bm25")
    assert idx.calls == []
    assert all(c.doc_id == "b" for c in out)


def test_hybrid_returns_attribution():
    cs = [chunk(f"c{i}") for i in range(5)]
    ranking = retrieve_ranked(StubIndex(cs), "Territory", 3, doc_id=None,
                              sparse=BM25Index(corpus()), retriever="hybrid")
    assert isinstance(ranking, Ranking)
    assert ranking.sources
    assert all(set(v) == {"dense", "bm25"} for v in ranking.sources.values())


def test_single_retriever_configs_carry_no_attribution():
    """Attribution on a one-retriever config would be noise, not evidence."""
    assert retrieve_ranked(StubIndex([chunk("c0")]), "q", 1).sources == {}


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError, match="unknown retriever"):
        retrieve(StubIndex([]), "q", 1, retriever="magic")


def test_hybrid_without_a_sparse_index_is_rejected():
    with pytest.raises(ValueError, match="needs a BM25Index"):
        retrieve(StubIndex([]), "q", 1, retriever="hybrid")

