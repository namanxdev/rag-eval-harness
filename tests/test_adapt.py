"""Corpus adapter: client documents in, eval set schema out.

The schema is the boundary between our benchmark and the client's data, so the
test that matters is the round trip -- the reference corpus, pushed back
through the adapter, has to come out equivalent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.chunking import chunk_documents
from src.evaluate import chunk_stats, document_metadata

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "data" / "eval_set.json"


def _load_adapt():
    """data/ is a directory of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("adapt", ROOT / "data" / "adapt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapt = _load_adapt()


def write_corpus(tmp_path: Path, docs: dict[str, str], queries: list[dict]):
    """Lay out a client corpus on disk the way the adapter expects to find it."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    for doc_id, text in docs.items():
        # Bytes, so no newline translation moves an offset out from under a span.
        (docs_dir / f"{doc_id}.txt").write_bytes(text.encode("utf-8"))
    qpath = tmp_path / "queries.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in queries), encoding="utf-8")
    return docs_dir, qpath


# ---------------------------------------------------------------------------
# round trip against the reference corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference():
    if not EVAL_SET.exists():
        pytest.skip(f"{EVAL_SET} not built; run python data/prepare.py --download")
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def test_round_trip_of_the_reference_corpus(reference, tmp_path):
    """CUAD out through the adapter and back must be the same eval set."""
    docs_dir, qpath = write_corpus(
        tmp_path,
        {d["doc_id"]: d["text"] for d in reference["documents"]},
        [{"query_id": q["query_id"], "doc_id": q["doc_id"], "query": q["query"],
          "clause_type": q["clause_type"], "gold_spans": q["gold_spans"]}
         for q in reference["queries"]],
    )
    out = adapt.adapt(docs_dir, qpath, name="round-trip")

    assert out["meta"]["counts"] == reference["meta"]["counts"]

    for got, want in zip(out["documents"], sorted(reference["documents"],
                                                  key=lambda d: d["doc_id"]), strict=True):
        # `title` is the filename here rather than the contract's own title;
        # everything the metrics touch has to match exactly.
        assert (got["doc_id"], got["n_chars"], got["text"]) == \
               (want["doc_id"], want["n_chars"], want["text"])

    by_id = {q["query_id"]: q for q in reference["queries"]}
    for q in out["queries"]:
        want = by_id[q["query_id"]]
        for field in ("doc_id", "clause_type", "query", "gold_spans", "gold_texts"):
            assert q[field] == want[field], f"{q['query_id']}: {field} differs"


def test_round_trip_by_gold_text_recovers_the_offsets(reference, tmp_path):
    """A client who quotes clauses gets the same spans as one who counts characters."""
    unique = [q for q in reference["queries"]
              if all({d["doc_id"]: d["text"] for d in reference["documents"]}
                     [q["doc_id"]].count(g) == 1 for g in q["gold_texts"])][:10]
    assert unique, "expected some gold texts to occur exactly once"

    docs_dir, qpath = write_corpus(
        tmp_path,
        {d["doc_id"]: d["text"] for d in reference["documents"]},
        [{"query_id": q["query_id"], "doc_id": q["doc_id"], "query": q["query"],
          "gold_texts": q["gold_texts"]} for q in unique],
    )
    out = adapt.adapt(docs_dir, qpath)
    by_id = {q["query_id"]: q for q in out["queries"]}
    for want in unique:
        assert by_id[want["query_id"]]["gold_spans"] == want["gold_spans"]


def test_output_feeds_the_harness_unchanged(reference, tmp_path):
    """`run_eval.py --eval-set <adapted>` must work with no other changes."""
    docs_dir, qpath = write_corpus(
        tmp_path,
        {d["doc_id"]: d["text"] for d in reference["documents"][:3]},
        [{"doc_id": q["doc_id"], "query": q["query"], "gold_spans": q["gold_spans"]}
         for q in reference["queries"]
         if q["doc_id"] in {d["doc_id"] for d in reference["documents"][:3]}],
    )
    out = adapt.adapt(docs_dir, qpath)

    # Everything run_eval.py reads before it loads a model.
    assert set(out["meta"]["counts"]) >= {"documents", "queries", "clause_types"}
    chunks = chunk_documents(out["documents"], "clause")
    assert chunk_stats(chunks, out["queries"])["n_chunks"] > 0


# ---------------------------------------------------------------------------
# refusing to guess
# ---------------------------------------------------------------------------

def test_ambiguous_gold_text_names_the_doc_the_text_and_the_count(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path,
        {"acme": "the term of this agreement. " * 3},
        [{"doc_id": "acme", "query": "term?", "gold_texts": ["the term"]}],
    )
    with pytest.raises(adapt.AdapterError) as exc:
        adapt.adapt(docs_dir, qpath)
    message = str(exc.value)
    assert "acme" in message and "3 times" in message and "the term" in message


def test_gold_text_that_is_absent_is_an_error_not_a_skip(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_texts": ["not in here"]}])
    with pytest.raises(adapt.AdapterError, match="not found"):
        adapt.adapt(docs_dir, qpath)


def test_unsupported_file_type_is_rejected_not_skipped(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]]}])
    (docs_dir / "notes.docx").write_bytes(b"PK\x03\x04")
    with pytest.raises(adapt.AdapterError) as exc:
        adapt.adapt(docs_dir, qpath)
    assert ".docx" in str(exc.value) and ".pdf" in str(exc.value)


def test_out_of_bounds_span_is_rejected(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 9999]]}])
    with pytest.raises(adapt.AdapterError, match="outside"):
        adapt.adapt(docs_dir, qpath)


def test_query_pointing_at_an_unknown_document_is_rejected(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "beta", "query": "q", "gold_spans": [[0, 5]]}])
    with pytest.raises(adapt.AdapterError, match="not among the documents"):
        adapt.adapt(docs_dir, qpath)


def test_both_or_neither_gold_form_is_rejected(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]],
          "gold_texts": ["a short"]}])
    with pytest.raises(adapt.AdapterError, match="both"):
        adapt.adapt(docs_dir, qpath)

    qpath.write_text(json.dumps({"doc_id": "acme", "query": "q"}), encoding="utf-8")
    with pytest.raises(adapt.AdapterError, match="neither"):
        adapt.adapt(docs_dir, qpath)


def test_duplicate_doc_id_is_rejected(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]]}])
    (docs_dir / "acme.md").write_bytes(b"a different contract")
    with pytest.raises(adapt.AdapterError, match="duplicate doc_id"):
        adapt.adapt(docs_dir, qpath)


def test_empty_extraction_is_rejected(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract", "scanned": "   \n  "},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]]}])
    with pytest.raises(adapt.AdapterError, match="no text"):
        adapt.adapt(docs_dir, qpath)


# ---------------------------------------------------------------------------
# scope caps
# ---------------------------------------------------------------------------

def test_document_cap_exits_non_zero_with_the_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapt, "MAX_DOCUMENTS", 2)
    docs_dir, qpath = write_corpus(
        tmp_path, {f"d{i}": "a short contract" for i in range(3)},
        [{"doc_id": "d0", "query": "q", "gold_spans": [[0, 5]]}])

    with pytest.raises(SystemExit) as exc:
        adapt.adapt(docs_dir, qpath)
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "3 documents" in err and "48 bytes" in err


def test_byte_cap_exits_non_zero_with_the_total(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapt, "MAX_TOTAL_BYTES", 10)
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a contract longer than ten bytes"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]]}])

    with pytest.raises(SystemExit) as exc:
        adapt.adapt(docs_dir, qpath)
    assert exc.value.code != 0
    assert "1 documents" in capsys.readouterr().err


def test_caps_match_the_contracted_scope():
    assert (adapt.MAX_DOCUMENTS, adapt.MAX_TOTAL_BYTES) == (250, 2 * 1024**3)


# ---------------------------------------------------------------------------
# provenance and metadata
# ---------------------------------------------------------------------------

def test_extraction_is_recorded_per_document(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]]}])
    source = adapt.adapt(docs_dir, qpath)["meta"]["source"]

    assert source["extractors"] == ["utf-8-passthrough"]
    record = source["documents"][0]
    assert record["doc_id"] == "acme" and record["n_chars"] == 16
    assert len(record["text_sha256"]) == 64
    assert "invalidates" in source["extraction_note"]


def test_metadata_reaches_the_index_as_filterable_payload(tmp_path):
    docs_dir, qpath = write_corpus(
        tmp_path, {"acme": "a short contract"},
        [{"doc_id": "acme", "query": "q", "gold_spans": [[0, 5]],
          "filters": {"entity": "Acme"}}])
    out = adapt.adapt(docs_dir, qpath,
                      metadata={"acme": {"entity": "Acme", "year": 2024}})

    assert document_metadata(out["documents"]) == {"acme": {"entity": "Acme", "year": 2024}}
    assert out["queries"][0]["filters"] == {"entity": "Acme"}


def test_pdf_extraction_records_the_pypdf_version(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with (docs_dir / "blank.pdf").open("wb") as fh:
        writer.write(fh)

    # A blank page extracts to nothing, which the adapter must refuse rather
    # than index as a document with no text.
    qpath = tmp_path / "queries.jsonl"
    qpath.write_text(json.dumps({"doc_id": "blank", "query": "q",
                                 "gold_spans": [[0, 1]]}), encoding="utf-8")
    with pytest.raises(adapt.AdapterError, match="no text"):
        adapt.adapt(docs_dir, qpath)

    text, extractor = adapt.extract(docs_dir / "blank.pdf")
    assert text.strip() == ""
    assert extractor == f"pypdf {pypdf.__version__}"
