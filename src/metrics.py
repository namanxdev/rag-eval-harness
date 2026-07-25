"""Retrieval metrics computed against character-level gold spans.

A chunk counts as relevant when it comes from the right document AND overlaps
at least one gold span. The document check matters because retrieval here runs
across the whole corpus, not within a single contract: a chunk whose offsets
happen to overlap the gold range of a *different* contract is not a hit.
"""

from __future__ import annotations

from src.chunking import Chunk

Span = tuple[int, int]


def is_relevant(chunk: Chunk, doc_id: str, gold_spans: list[Span]) -> bool:
    return chunk.doc_id == doc_id and any(chunk.overlaps(s, e) for s, e in gold_spans)


def precision_at_k(retrieved: list[Chunk], doc_id: str, gold: list[Span], k: int) -> float:
    """Fraction of the top k that are relevant. Divided by k, not by len(top)."""
    return sum(is_relevant(c, doc_id, gold) for c in retrieved[:k]) / k


def recall_at_k(retrieved: list[Chunk], doc_id: str, gold: list[Span],
                all_chunks: list[Chunk], k: int) -> float | None:
    """Share of all relevant chunks in the corpus that made the top k.

    Returns None when no chunk in the corpus is relevant, which would otherwise
    score a misleading 0. That cannot happen with the shipped eval set -- every
    query has a gold span inside its document -- but it guards anyone who
    regenerates the set with different filters.
    """
    total = sum(is_relevant(c, doc_id, gold) for c in all_chunks)
    if total == 0:
        return None
    return sum(is_relevant(c, doc_id, gold) for c in retrieved[:k]) / total


def mrr(retrieved: list[Chunk], doc_id: str, gold: list[Span]) -> float:
    for i, c in enumerate(retrieved, 1):
        if is_relevant(c, doc_id, gold):
            return 1 / i
    return 0.0


def span_coverage(retrieved: list[Chunk], doc_id: str, gold: list[Span]) -> float | None:
    """Fraction of gold characters actually present in the retrieved text.

    This is the chunking-invariant metric. precision@k and recall@k both count
    chunks, so their values shift when the chunk size shifts even if the same
    text is returned; counting characters does not.
    """
    covered: set[int] = set()
    total = 0
    for gs, ge in gold:
        total += ge - gs
        for c in retrieved:
            if c.doc_id != doc_id:
                continue
            lo, hi = max(gs, c.start), min(ge, c.end)
            if lo < hi:
                covered.update(range(lo, hi))
    return len(covered) / total if total else None
