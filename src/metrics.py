"""Retrieval metrics computed against character-level gold spans.

A chunk counts as relevant when it comes from the right document AND overlaps
at least one gold span.

Retrieval is scoped to the contract each question is asked of (a Qdrant payload
filter on `doc_id` -- see `src/index.py`), so in a normal run every candidate
already comes from the right document and the `doc_id` check never fires. It is
kept as a guard: if that filter is ever dropped, or an eval set is regenerated
with different filters, a chunk whose offsets happen to overlap the gold range
of a *different* contract must not count as a hit. Character offsets are only
meaningful relative to the document they came from.
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
    """Share of all relevant chunks that made the top k.

    Only chunks from the query's own document can be relevant, so the
    denominator is every relevant chunk in that contract.

    Returns None when nothing is relevant at all, which would otherwise score a
    misleading 0. That cannot happen with the shipped eval set -- every query
    has a gold span inside its document -- but it guards anyone who regenerates
    the set with different filters.
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


def span_coverage_fractions(retrieved: list[Chunk], doc_id: str,
                            gold: list[Span]) -> list[float]:
    """Per-span coverage: what fraction of *each* gold span's characters is present.

    `span_coverage` pools characters across spans and returns one number, which
    is why it cannot tell one fully-covered span from two half-covered ones.
    This keeps the spans separate; the grounding metrics below are built on it.
    """
    out = []
    for gs, ge in gold:
        if ge <= gs:
            out.append(0.0)
            continue
        covered: set[int] = set()
        for c in retrieved:
            if c.doc_id != doc_id:
                continue
            lo, hi = max(gs, c.start), min(ge, c.end)
            if lo < hi:
                covered.update(range(lo, hi))
        out.append(len(covered) / (ge - gs))
    return out


def span_recall_at_k(retrieved: list[Chunk], doc_id: str, gold: list[Span], k: int,
                     threshold: float = 0.5) -> float | None:
    """Fraction of DISTINCT gold spans covered at >= threshold of their characters.

    Differs from span_coverage, which pools characters across spans and so
    cannot distinguish one fully-covered span from two half-covered ones.
    Returns None when gold is empty.
    """
    if not gold:
        return None
    fracs = span_coverage_fractions(retrieved[:k], doc_id, gold)
    return sum(f >= threshold for f in fracs) / len(fracs)


def complete_grounding_at_k(retrieved: list[Chunk], doc_id: str, gold: list[Span], k: int,
                            threshold: float = 0.5) -> bool | None:
    """True only when EVERY gold span clears the threshold.

    The binary the buyer cares about: could a correct, fully-cited answer be
    produced from this retrieval, or not. A query answered from one of the two
    clauses it depends on is a wrong answer with a confident citation, and
    pooled coverage scores that 0.5 and calls it half a success.
    """
    if not gold:
        return None
    fracs = span_coverage_fractions(retrieved[:k], doc_id, gold)
    return all(f >= threshold for f in fracs)


def uncovered_spans(retrieved: list[Chunk], doc_id: str, gold: list[Span], k: int,
                    threshold: float = 0.5) -> list[tuple[Span, float]]:
    """The gold spans that failed the threshold, each with what it did get.

    This is what turns a failing grounding score into something showable: the
    exact clause the answer would have been missing.
    """
    fracs = span_coverage_fractions(retrieved[:k], doc_id, gold)
    return [(s, f) for s, f in zip(gold, fracs, strict=True) if f < threshold]
