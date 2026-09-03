"""Grounding completeness: does retrieval support a fully-cited answer?

The point of these metrics is the gap between them and `span_coverage`. Pooled
character coverage cannot tell one fully-covered clause from two half-covered
ones, and only the second of those is a wrong answer with a confident citation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunking import Chunk
from src.metrics import (
    complete_grounding_at_k,
    span_coverage,
    span_recall_at_k,
    uncovered_spans,
)

DOC = "d1"


def chunk(start: int, end: int, doc_id: str = DOC, n: int = 0) -> Chunk:
    """A chunk covering [start, end). Text is filler; only offsets are scored."""
    return Chunk(doc_id, f"{doc_id}:test:{n}", "x" * (end - start), start, end, "test")


# ---------------------------------------------------------------------------
# the cases the spec names
# ---------------------------------------------------------------------------

def test_single_span_fully_covered():
    gold = [(100, 200)]
    got = [chunk(0, 300)]
    assert span_recall_at_k(got, DOC, gold, 5) == 1.0
    assert complete_grounding_at_k(got, DOC, gold, 5) is True
    assert uncovered_spans(got, DOC, gold, 5) == []


def test_two_spans_one_covered():
    gold = [(100, 200), (500, 600)]
    got = [chunk(0, 300)]
    assert span_recall_at_k(got, DOC, gold, 5) == 0.5
    assert complete_grounding_at_k(got, DOC, gold, 5) is False
    missed = uncovered_spans(got, DOC, gold, 5)
    assert [s for s, _ in missed] == [(500, 600)]
    assert missed[0][1] == 0.0


def test_two_spans_both_covered():
    gold = [(100, 200), (500, 600)]
    got = [chunk(0, 300, n=0), chunk(400, 700, n=1)]
    assert span_recall_at_k(got, DOC, gold, 5) == 1.0
    assert complete_grounding_at_k(got, DOC, gold, 5) is True


def test_empty_gold_returns_none():
    got = [chunk(0, 300)]
    assert span_recall_at_k(got, DOC, [], 5) is None
    assert complete_grounding_at_k(got, DOC, [], 5) is None


def test_threshold_boundary_is_inclusive():
    """Exactly at the threshold counts as covered; a character less does not."""
    gold = [(0, 100)]
    assert complete_grounding_at_k([chunk(0, 50)], DOC, gold, 5, threshold=0.5) is True
    assert complete_grounding_at_k([chunk(0, 49)], DOC, gold, 5, threshold=0.5) is False
    assert span_recall_at_k([chunk(0, 50)], DOC, gold, 5, threshold=0.5) == 1.0
    assert span_recall_at_k([chunk(0, 49)], DOC, gold, 5, threshold=0.5) == 0.0


# ---------------------------------------------------------------------------
# behaviour that must not regress
# ---------------------------------------------------------------------------

def test_only_top_k_counts():
    """A span covered by the 6th chunk is not grounded at k=5."""
    gold = [(0, 10), (900, 910)]
    got = [chunk(0, 100, n=i) for i in range(5)] + [chunk(900, 1000, n=5)]
    assert complete_grounding_at_k(got, DOC, gold, 5) is False
    assert complete_grounding_at_k(got, DOC, gold, 6) is True


def test_wrong_document_never_grounds():
    """Offsets are only meaningful against the document they came from."""
    gold = [(100, 200)]
    got = [chunk(0, 300, doc_id="other")]
    assert span_recall_at_k(got, DOC, gold, 5) == 0.0
    assert complete_grounding_at_k(got, DOC, gold, 5) is False


def test_arguments_are_not_mutated():
    gold = [(100, 200), (500, 600)]
    got = [chunk(0, 300)]
    gold_before, got_before = list(gold), list(got)
    span_recall_at_k(got, DOC, gold, 1)
    complete_grounding_at_k(got, DOC, gold, 1)
    uncovered_spans(got, DOC, gold, 1)
    assert gold == gold_before and got == got_before


def test_partial_coverage_from_two_chunks_unions():
    """Halves of a span in two chunks add up; they are not counted twice."""
    gold = [(0, 100)]
    got = [chunk(0, 60, n=0), chunk(40, 100, n=1)]
    assert complete_grounding_at_k(got, DOC, gold, 5) is True
    assert span_recall_at_k(got, DOC, gold, 5) == 1.0


# ---------------------------------------------------------------------------
# the divergence from span_coverage, on the real eval set
# ---------------------------------------------------------------------------

EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


@pytest.fixture(scope="module")
def q001():
    if not EVAL_SET.exists():
        pytest.skip(f"{EVAL_SET} not built; run python data/prepare.py --download")
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    q = next(q for q in data["queries"] if q["query_id"] == "q001")
    assert len(q["gold_spans"]) == 2, "q001 is the multi-span fixture; it must stay one"
    return q


def test_q001_pooled_coverage_disagrees_with_grounding(q001):
    """The failure the pooled metric hides.

    q001 (Exclusivity) needs two clauses: a 196-char span and a 530-char one.
    Retrieve only the long one and pooled span_coverage reads 0.73 -- healthy --
    while the answer is missing an entire clause it depends on.
    """
    doc_id = q001["doc_id"]
    gold = [tuple(s) for s in q001["gold_spans"]]
    short, long = gold
    assert (short[1] - short[0], long[1] - long[0]) == (196, 530)

    only_long = [chunk(long[0], long[1], doc_id=doc_id)]
    assert span_recall_at_k(only_long, doc_id, gold, 5) == 0.5
    assert complete_grounding_at_k(only_long, doc_id, gold, 5) is False
    pooled = span_coverage(only_long, doc_id, gold)
    assert pooled > 0.7, "pooled coverage looks like partial success"

    # And the reverse: the same 0.5 span_recall with pooled coverage far below
    # it. One number cannot stand in for the other in either direction.
    only_short = [chunk(short[0], short[1], doc_id=doc_id)]
    assert span_recall_at_k(only_short, doc_id, gold, 5) == 0.5
    assert complete_grounding_at_k(only_short, doc_id, gold, 5) is False
    assert span_coverage(only_short, doc_id, gold) < 0.3


def test_q001_missed_span_is_reportable(q001):
    """A failing query must yield the actual clause text that was missing."""
    doc_id = q001["doc_id"]
    gold = [tuple(s) for s in q001["gold_spans"]]
    long = gold[1]
    missed = uncovered_spans([chunk(long[0], long[1], doc_id=doc_id)], doc_id, gold, 5)
    assert [s for s, _ in missed] == [gold[0]]
