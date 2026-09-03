"""Failure taxonomy: one auditable label per failing query.

The synthetic cases below drive each branch directly. The integration test at
the bottom re-derives labels from the committed `results/per_query.json`
without loading a model -- chunking is pure Python, so a client can reproduce
every label from two committed files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.chunking import Chunk, chunk_documents
from src.evaluate import GROUNDING_AT, Config, Result
from src.taxonomy import FIX_FOR, LABEL_ORDER, classify, taxonomy_summary

ROOT = Path(__file__).resolve().parent.parent
DOC = "d1"
TEXT = "".join(f"{i % 10}" for i in range(1000))


def chunks(*bounds: tuple[int, int]) -> list[Chunk]:
    return [Chunk(DOC, f"{DOC}:t:{n}", TEXT[s:e], s, e, "t")
            for n, (s, e) in enumerate(bounds)]


def tiled(width: int = 100, n: int = 10) -> list[Chunk]:
    """A chunker that covers the document with no gaps, like the real ones."""
    return chunks(*[(i * width, (i + 1) * width) for i in range(n)])


def scenario(all_chunks, ranked, gold, *, span_recall=0.0, grounded=False):
    """A one-query Result plus the eval set and index it was scored against."""
    eval_set = {
        "documents": [{"doc_id": DOC, "title": "t", "n_chars": len(TEXT), "text": TEXT}],
        "queries": [{"query_id": "q001", "doc_id": DOC, "clause_type": "Test",
                     "query": "q", "gold_spans": [list(s) for s in gold],
                     "gold_texts": [TEXT[s:e] for s, e in gold]}],
    }
    row = {
        "query_id": "q001",
        "clause_type": "Test",
        "doc_id": DOC,
        f"span_recall@{GROUNDING_AT}": span_recall,
        f"complete_grounding@{GROUNDING_AT}": grounded,
        "ranked_chunk_ids": [c.chunk_id for c in ranked],
    }
    result = Result(Config("test", "t", False), {}, {}, [row])
    return result, eval_set, SimpleNamespace(chunks=all_chunks)


def label_of(*args, **kwargs) -> str:
    labels = classify(*scenario(*args, **kwargs))
    assert len(labels) == 1
    return labels[0].label


# ---------------------------------------------------------------------------
# one test per branch
# ---------------------------------------------------------------------------

def test_passing_queries_get_no_label():
    all_chunks = tiled()
    result, eval_set, index = scenario(
        all_chunks, all_chunks[:5], [(10, 90)], span_recall=1.0, grounded=True)
    assert classify(result, eval_set, index) == []


def test_chunk_severance():
    """A clause split across two chunks, only one of which was retrieved."""
    all_chunks = tiled()
    ranked = [all_chunks[5]] + all_chunks[:4]      # c5 in top 5, c6 nowhere
    assert label_of(all_chunks, ranked, [(590, 700)]) == "chunk-severance"


def test_retrieval_miss():
    """The chunk holding the clause never entered the candidate pool."""
    all_chunks = tiled()
    assert label_of(all_chunks, all_chunks[:6], [(610, 700)]) == "retrieval-miss"


def test_rank_miss():
    """The chunk was retrieved, just below k."""
    all_chunks = tiled()
    ranked = all_chunks[:5] + [all_chunks[6]]      # c6 at rank 5, k is 5
    assert label_of(all_chunks, ranked, [(610, 700)]) == "rank-miss"


def test_ceiling_bound_needs_a_chunker_that_drops_text():
    """Fires only when text is missing from the index entirely.

    c0 stops at 100 and c1 starts at 200, so the characters between are in no
    chunk at all. The one relevant chunk is already at rank 0: no ranking
    change can ground this query, which is exactly what the label means.
    """
    gapped = chunks((0, 100), (200, 300))
    assert label_of(gapped, gapped, [(50, 200)]) == "ceiling-bound"


def test_partial_grounding():
    """Everything that could help was retrieved and it still was not enough."""
    gapped = chunks((0, 100), (200, 300), (300, 400), (400, 500), (500, 600), (600, 700))
    ranked = gapped[:5]
    # [250,650) is well covered by c1..c4; [50,200) gets only its first 50
    # characters from c0, which sits at rank 0.
    assert label_of(gapped, ranked, [(50, 200), (250, 650)], span_recall=0.5) \
        == "partial-grounding"


# ---------------------------------------------------------------------------
# ordering: a query matching two conditions takes the earlier one
# ---------------------------------------------------------------------------

def test_severance_beats_retrieval_miss():
    """Both are true; chunking is the earlier fault, so chunking wins."""
    all_chunks = tiled()
    ranked = [all_chunks[5]] + all_chunks[:4]
    # [590,700) is severed across c5/c6; [810,890) sits in c8, absent from the pool.
    assert label_of(all_chunks, ranked, [(590, 700), (810, 890)]) == "chunk-severance"


def test_severance_beats_rank_miss():
    all_chunks = tiled()
    ranked = [all_chunks[5]] + all_chunks[:4] + [all_chunks[8]]
    assert label_of(all_chunks, ranked, [(590, 700), (810, 890)]) == "chunk-severance"


def test_retrieval_miss_beats_rank_miss():
    """One clause below k, another never retrieved: the harder fault wins."""
    all_chunks = tiled()
    ranked = all_chunks[:5] + [all_chunks[6]]     # c6 below k, c8 absent
    assert label_of(all_chunks, ranked, [(610, 700), (810, 890)]) == "retrieval-miss"


def test_a_covered_severed_span_does_not_trigger_severance():
    """The scoping rule: a split clause that came back whole is not the fault.

    [590,700) is severed across c5 and c6 but both are in the top 5, so it is
    grounded. The failure is [810,890), which is absent from the pool.
    """
    all_chunks = tiled()
    ranked = [all_chunks[5], all_chunks[6]] + all_chunks[:3]
    assert label_of(all_chunks, ranked, [(590, 700), (810, 890)]) == "retrieval-miss"


# ---------------------------------------------------------------------------
# evidence and summary contracts
# ---------------------------------------------------------------------------

def test_evidence_is_populated_and_json_serialisable():
    all_chunks = tiled()
    labels = classify(*scenario(all_chunks, all_chunks[:6], [(610, 700)]))
    ev = labels[0].evidence
    assert ev, "evidence must not be empty"
    assert ev["uncovered"] and ev["uncovered"][0]["text"]
    assert ev["missing_chunks"][0]["chunk_id"] == all_chunks[6].chunk_id
    json.dumps(ev)          # raises if anything is not serialisable


def test_every_label_has_a_fix():
    assert set(FIX_FOR) == set(LABEL_ORDER)


def test_taxonomy_summary_shape():
    from src.taxonomy import FailureLabel
    labels = [FailureLabel(f"q{i:03d}", lb, {"x": 1}) for i, lb in enumerate(
        ["rank-miss"] * 4 + ["chunk-severance"] * 3 + ["retrieval-miss"] * 2
        + ["partial-grounding"])]
    s = taxonomy_summary(labels, top_n=3)

    assert s["n_failing"] == 10
    assert list(s["counts"]) == ["rank-miss", "chunk-severance", "retrieval-miss",
                                 "partial-grounding"]
    assert len(s["patterns"]) == 3
    assert s["patterns"][0] == {
        "label": "rank-miss", "count": 4, "share": 0.4,
        "fix": FIX_FOR["rank-miss"], "examples": ["q000", "q001"],
    }
    assert sum(s["counts"].values()) == s["n_failing"]
    json.dumps(s)


def test_taxonomy_summary_of_nothing():
    s = taxonomy_summary([])
    assert s == {"n_failing": 0, "counts": {}, "patterns": []}


def test_ties_break_on_label_order_not_insertion_order():
    from src.taxonomy import FailureLabel
    labels = [FailureLabel("q1", "rank-miss", {}), FailureLabel("q2", "chunk-severance", {})]
    assert list(taxonomy_summary(labels)["counts"]) == ["chunk-severance", "rank-miss"]


# ---------------------------------------------------------------------------
# integration: re-derive the labels from committed artifacts, no model needed
# ---------------------------------------------------------------------------

PER_QUERY = ROOT / "results" / "per_query.json"
EVAL_SET = ROOT / "data" / "eval_set.json"


@pytest.fixture(scope="module")
def committed_run():
    if not (PER_QUERY.exists() and EVAL_SET.exists()):
        pytest.skip("no committed run; python run_eval.py first")
    rows = json.loads(PER_QUERY.read_text(encoding="utf-8"))
    if not any("ranked_chunk_ids" in r for rs in rows.values() for r in rs):
        pytest.skip("committed run predates ranked_chunk_ids; rerun run_eval.py")
    return rows, json.loads(EVAL_SET.read_text(encoding="utf-8"))


def test_labels_partition_the_failing_set(committed_run):
    """Every failing query gets exactly one label; no passing query gets one."""
    rows, eval_set = committed_run
    strategies = {"naive": "naive"}      # config name prefix -> chunking strategy

    for name, per_query in rows.items():
        strategy = strategies.get(name, "clause" if name.startswith("clause") else None)
        if strategy is None:
            continue
        cs = chunk_documents(eval_set["documents"], strategy)
        result = Result(Config(name, strategy, False), {}, {}, per_query)
        labels = classify(result, eval_set, SimpleNamespace(chunks=cs))

        failing = {r["query_id"] for r in per_query
                   if r[f"complete_grounding@{GROUNDING_AT}"] is False}
        labelled = [lb.query_id for lb in labels]

        assert len(labelled) == len(set(labelled)), f"{name}: duplicate label"
        assert set(labelled) == failing, f"{name}: labels do not cover the failing set"
        assert all(lb.label in FIX_FOR for lb in labels)
        assert all(lb.evidence for lb in labels)
        json.dumps([lb.evidence for lb in labels])
