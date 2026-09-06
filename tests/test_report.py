"""The client readout. The rule under test: every number traces to a run."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from src.chunking import Chunk
from src.evaluate import GROUNDING_AT, METRIC_KEYS, Config, Result
from src.report import build_backlog, lever_diff, write_client_report
from src.taxonomy import classify, taxonomy_summary

DOC = "d1"
TEXT = "".join(str(i % 10) for i in range(1000))
GROUNDING_KEY = f"complete_grounding@{GROUNDING_AT}"

BASE = Config("clause", "clause", False, "structural clause boundaries")


def tiled(width: int = 100, n: int = 10) -> list[Chunk]:
    return [Chunk(DOC, f"{DOC}:t:{i}", TEXT[i * width:(i + 1) * width],
                  i * width, (i + 1) * width, "t") for i in range(n)]


def eval_set(queries: list[dict]) -> dict:
    return {
        "meta": {
            "name": "test-corpus",
            "source": {"dataset": "a test corpus"},
            "sampling": {"notes": ["nothing was sampled"]},
            "counts": {"documents": 1, "queries": len(queries),
                       "clause_types": 1, "gold_spans": sum(
                           len(q["gold_spans"]) for q in queries)},
        },
        "documents": [{"doc_id": DOC, "title": "t", "n_chars": len(TEXT), "text": TEXT}],
        "queries": queries,
    }


def query(qid: str, gold: list[tuple[int, int]]) -> dict:
    return {"query_id": qid, "doc_id": DOC, "clause_type": "Termination",
            "query": "when does it end?", "gold_spans": [list(g) for g in gold],
            "gold_texts": [TEXT[s:e] for s, e in gold]}


def row(qid: str, ranked: list[Chunk], *, grounded: bool, span_recall: float = 0.5):
    r = {k: 0.5 for k in METRIC_KEYS}
    r |= {"query_id": qid, "clause_type": "Termination", "doc_id": DOC,
          f"span_recall@{GROUNDING_AT}": span_recall,
          GROUNDING_KEY: grounded,
          "ranked_chunk_ids": [c.chunk_id for c in ranked]}
    return r


def result(config: Config, rows: list[dict], chunks: list[Chunk], grounding: float):
    aggregates = {k: 0.5 for k in METRIC_KEYS} | {GROUNDING_KEY: grounding}
    return Result(config, {"n_chunks": len(chunks), "median_chars": 100},
                  aggregates, rows, SimpleNamespace(chunks=chunks))


# ---------------------------------------------------------------------------
# attributing a delta to a lever
# ---------------------------------------------------------------------------

def test_lever_diff_names_only_what_changed():
    assert lever_diff(BASE, BASE) == []
    assert lever_diff(BASE, Config("x", "naive", False)) == ["chunking"]
    assert lever_diff(BASE, Config("x", "clause", True)) == ["reranking"]
    assert lever_diff(BASE, Config("x", "clause", False, retriever="hybrid")) \
        == ["hybrid retrieval"]
    assert lever_diff(BASE, Config("x", "clause", False, doc_filter=False)) \
        == ["metadata filtering"]


def test_lever_diff_reports_every_lever_a_config_changed():
    both = Config("x", "clause", True, retriever="hybrid")
    assert lever_diff(BASE, both) == ["hybrid retrieval", "reranking"]


def test_backlog_uses_only_configs_that_isolate_one_lever():
    """A config that changes two things cannot attribute its delta to either."""
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=False)]
    baseline = result(BASE, rows, chunks, 0.40)
    combined = result(Config("clause + hybrid + rerank", "clause", True,
                             retriever="hybrid"), rows, chunks, 0.90)

    labels = classify(baseline, eval_set([query("q001", [(610, 700)])]), baseline.index)
    backlog = build_backlog(baseline, [baseline, combined],
                            taxonomy_summary(labels))
    assert [e["measured_by"] for e in backlog] == ["not measured"]


def test_backlog_reports_a_measured_delta_when_one_lever_was_isolated():
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=False)]
    baseline = result(BASE, rows, chunks, 0.40)
    hybrid = result(Config("clause + hybrid", "clause", False, retriever="hybrid"),
                    rows, chunks, 0.62)

    labels = classify(baseline, eval_set([query("q001", [(610, 700)])]), baseline.index)
    backlog = build_backlog(baseline, [baseline, hybrid], taxonomy_summary(labels))

    assert backlog[0]["lever"] == "hybrid retrieval"
    assert backlog[0]["measured_by"] == "clause + hybrid"
    assert backlog[0]["grounding_delta"] == "+22.0%"


def test_backlog_reports_a_negative_delta_honestly():
    """A lever that made things worse must say so, not be dropped."""
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=False)]
    baseline = result(BASE, rows, chunks, 0.60)
    hybrid = result(Config("clause + hybrid", "clause", False, retriever="hybrid"),
                    rows, chunks, 0.48)

    labels = classify(baseline, eval_set([query("q001", [(610, 700)])]), baseline.index)
    backlog = build_backlog(baseline, [baseline, hybrid], taxonomy_summary(labels))
    assert backlog[0]["grounding_delta"] == "-12.0%"


def test_backlog_is_ranked_by_failures_addressed():
    chunks = tiled()
    rows = [row("q001", [chunks[5]] + chunks[:4], grounded=False),   # severance
            row("q002", [chunks[5]] + chunks[:4], grounded=False),   # severance
            row("q003", chunks[:6], grounded=False)]                 # retrieval-miss
    es = eval_set([query("q001", [(590, 700)]), query("q002", [(590, 700)]),
                   query("q003", [(810, 890)])])
    baseline = result(BASE, rows, chunks, 0.0)

    backlog = build_backlog(baseline, [baseline],
                            taxonomy_summary(classify(baseline, es, baseline.index)))
    assert [e["queries_addressed"] for e in backlog] == [2, 1]
    assert backlog[0]["lever"] == "chunking"


# ---------------------------------------------------------------------------
# the document itself
# ---------------------------------------------------------------------------

@pytest.fixture
def written(tmp_path):
    chunks = tiled()
    rows = [row("q001", [chunks[5]] + chunks[:4], grounded=False),
            row("q002", chunks[:5], grounded=True, span_recall=1.0)]
    es = eval_set([query("q001", [(590, 700)]), query("q002", [(10, 90)])])
    baseline = result(BASE, rows, chunks, 0.5)
    naive = result(Config("naive", "naive", False), rows, chunks, 0.75)
    path = write_client_report([baseline, naive], es, tmp_path)
    return path, path.read_text(encoding="utf-8"), tmp_path


def test_all_four_sections_are_present(written):
    _, text, _ = written
    for heading in ("## 1. Where retrieval stands today", "## 2. Top failure patterns",
                    "## 3. Fix backlog", "## 4. Scope"):
        assert heading in text


def test_headline_matches_the_measured_aggregate(written):
    """The prose number and the table number are the same number."""
    _, text, _ = written
    assert "| `complete_grounding@5` | 0.500 |" in text
    assert "50% of questions (1 of 2)" in text


def test_failure_examples_quote_the_real_missing_clause(written):
    _, text, _ = written
    assert "**q001**" in text
    assert TEXT[590:610] in text, "the missed clause text must be shown, not summarised"


def test_scope_section_comes_from_meta(written):
    _, text, _ = written
    assert "1 documents, 2 queries" in text
    assert "a test corpus" in text
    assert "nothing was sampled" in text
    assert "Not covered" in text


def test_no_placeholder_numbers_leak_into_the_report(written):
    """Nothing in the report may be a nan, a None, or an empty format slot."""
    _, text, _ = written
    assert not re.search(r"\bnan\b|\bNone\b|\{\}", text)


def test_taxonomy_json_is_written_alongside(written):
    _, _, out_dir = written
    data = json.loads((out_dir / "taxonomy.json").read_text(encoding="utf-8"))
    assert data["config"] == "clause"
    assert [lb["query_id"] for lb in data["labels"]] == ["q001"]
    assert data["labels"][0]["evidence"]["uncovered"]


def test_baseline_is_selectable_by_name(tmp_path):
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=True, span_recall=1.0)]
    es = eval_set([query("q001", [(10, 90)])])
    a = result(BASE, rows, chunks, 0.5)
    b = result(Config("naive", "naive", False), rows, chunks, 0.75)

    text = write_client_report([a, b], es, tmp_path, "naive").read_text(encoding="utf-8")
    assert "Baseline configuration: **naive**" in text


def test_a_clean_run_says_so_rather_than_inventing_a_pattern(tmp_path):
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=True, span_recall=1.0)]
    es = eval_set([query("q001", [(10, 90)])])
    text = write_client_report([result(BASE, rows, chunks, 1.0)], es,
                               tmp_path).read_text(encoding="utf-8")
    assert "No query failed" in text
    assert "No failures to address" in text


def test_a_result_without_an_index_is_refused(tmp_path):
    """Better to fail than to publish a taxonomy built on rebuilt chunks."""
    rows = [row("q001", [], grounded=False)]
    bare = Result(BASE, {}, {k: 0.5 for k in METRIC_KEYS}, rows)
    with pytest.raises(ValueError, match="no index"):
        write_client_report([bare], eval_set([query("q001", [(10, 90)])]), tmp_path)


def test_empty_results_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        write_client_report([], eval_set([]), tmp_path)


def test_backlog_falls_back_to_a_pair_that_isolates_the_lever(tmp_path):
    """No `naive + rerank` config exists, but clause vs clause+rerank isolates it."""
    chunks = tiled()
    rows = [row("q001", chunks[:5] + [chunks[6]], grounded=False)]
    es = eval_set([query("q001", [(610, 700)])])

    baseline = result(Config("naive", "naive", False), rows, chunks, 0.58)
    clause = result(BASE, rows, chunks, 0.66)
    reranked = result(Config("clause + rerank", "clause", True), rows, chunks, 0.56)

    backlog = build_backlog(baseline, [baseline, clause, reranked],
                            taxonomy_summary(classify(baseline, es, baseline.index)))
    entry = next(e for e in backlog if e["lever"] == "reranking")
    assert entry["measured_by"] == "clause -> clause + rerank"
    assert entry["grounding_delta"] == "-10.0%"


def test_the_pair_runs_away_from_the_baseline_setting_not_toward_it():
    """Turning reranking on cost 10 points; it must not be reported as a gain."""
    chunks = tiled()
    rows = [row("q001", chunks[:5] + [chunks[6]], grounded=False)]
    es = eval_set([query("q001", [(610, 700)])])

    baseline = result(Config("naive", "naive", False), rows, chunks, 0.58)
    results = [baseline,
               result(BASE, rows, chunks, 0.66),
               result(Config("clause + rerank", "clause", True), rows, chunks, 0.56)]
    backlog = build_backlog(baseline, results,
                            taxonomy_summary(classify(baseline, es, baseline.index)))
    assert next(e for e in backlog
                if e["lever"] == "reranking")["grounding_delta"].startswith("-")


def test_report_names_the_best_measured_configuration(tmp_path):
    """A lever fixing no labelled failure still has to reach the client."""
    chunks = tiled()
    rows = [row("q001", [chunks[5]] + chunks[:4], grounded=False)]
    es = eval_set([query("q001", [(590, 700)])])

    baseline = result(BASE, rows, chunks, 0.66)
    hybrid = result(Config("clause + hybrid", "clause", False, retriever="hybrid"),
                    rows, chunks, 0.70)
    text = write_client_report([baseline, hybrid], es, tmp_path).read_text(encoding="utf-8")

    assert "highest measured grounding rate" in text
    assert "**clause + hybrid** at 0.700" in text
    assert "+4.0% against the baseline" in text


def test_no_best_config_line_when_the_baseline_already_wins(tmp_path):
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=True, span_recall=1.0)]
    es = eval_set([query("q001", [(10, 90)])])
    text = write_client_report([result(BASE, rows, chunks, 0.9),
                                result(Config("naive", "naive", False), rows, chunks, 0.4)],
                               es, tmp_path).read_text(encoding="utf-8")
    assert "highest measured grounding rate" not in text


def test_multiple_changed_levers_read_as_a_sentence(tmp_path):
    chunks = tiled()
    rows = [row("q001", chunks[:5], grounded=True, span_recall=1.0)]
    es = eval_set([query("q001", [(10, 90)])])
    best = result(Config("clause + hybrid", "clause", False, retriever="hybrid"),
                  rows, chunks, 0.70)
    text = write_client_report([result(Config("naive", "naive", False), rows, chunks, 0.58),
                                best], es, tmp_path).read_text(encoding="utf-8")
    assert "chunking and hybrid retrieval" in text
