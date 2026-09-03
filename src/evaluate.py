"""Evaluation loop: chunk -> index -> retrieve -> score -> report."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient

from src.chunking import Chunk, chunk_documents
from src.index import ChunkIndex, Encoder
from src.metrics import (
    complete_grounding_at_k,
    is_relevant,
    mrr,
    precision_at_k,
    recall_at_k,
    span_coverage,
    span_recall_at_k,
    uncovered_spans,
)
from src.retrieve import Reranker, retrieve_ranked
from src.sparse import BM25Index

# One ranked list per query, scored at several depths. Every config shares the
# same depth so the only difference between them is chunking and ordering.
RANK_DEPTH = 30
P_AT = 5
R_AT = 10
COVERAGE_AT = 5

# Grounding is scored at the same depth as coverage so the two are directly
# comparable: same chunks in, different question asked of them.
GROUNDING_AT = 5
# A gold span counts as retrieved when at least this share of its characters is
# present. Half a clause is not a citation. The value is fixed, not tuned per
# corpus -- moving it to make a config look better is how these numbers stop
# meaning anything.
GROUNDING_THRESHOLD = 0.5

# How much of a missed clause to quote in per_query.json. Enough to recognise
# the clause, short enough that the file stays readable.
MISSED_SPAN_CHARS = 120

METRIC_KEYS = [
    f"precision@{P_AT}",
    f"recall@{R_AT}",
    "mrr",
    f"span_coverage@{COVERAGE_AT}",
    f"span_recall@{GROUNDING_AT}",
    f"complete_grounding@{GROUNDING_AT}",
]


@dataclass
class Config:
    name: str
    strategy: str
    rerank: bool
    description: str = ""
    retriever: str = "dense"     # "dense" | "bm25" | "hybrid" -- see src/sparse.py
    # Both filters are levers under test, not fixed plumbing. `doc_filter`
    # scopes retrieval to the document a question is asked of; `metadata_filter`
    # applies whatever a query declared in its optional `filters` field. Turning
    # either off is how the harness measures what it was worth.
    doc_filter: bool = True
    metadata_filter: bool = True


CONFIGS = [
    Config("naive", "naive", False, "fixed 1000-char windows, 150 overlap"),
    Config("clause", "clause", False, "structural clause boundaries"),
    Config("clause + rerank", "clause", True, "clause boundaries, cross-encoder rerank of top 30"),
    Config("clause + bm25", "clause", False, "lexical BM25 only, no embeddings",
           retriever="bm25"),
    Config("clause + hybrid", "clause", False, "dense + BM25 fused with RRF",
           retriever="hybrid"),
    Config("clause + hybrid + rerank", "clause", True,
           "dense + BM25 fused, then cross-encoder rerank", retriever="hybrid"),
    Config("clause, unfiltered", "clause", False,
           "no document scoping: what the metadata filter is worth",
           doc_filter=False),
]

# Document fields that describe the eval set itself rather than the document.
# Everything else a document declares becomes filterable payload.
DOC_RESERVED = frozenset({"doc_id", "title", "n_chars", "text"})


def document_metadata(documents: list[dict]) -> dict[str, dict]:
    """Per-document metadata to index alongside each chunk.

    CUAD documents declare nothing beyond the reserved fields, so this is empty
    on the reference corpus and the payload is unchanged. Client corpora carry
    effective date, entity, document type, version; `data/adapt.py` passes them
    through, and a query filters on them via its `filters` field.
    """
    return {d["doc_id"]: {k: v for k, v in d.items() if k not in DOC_RESERVED}
            for d in documents}


@dataclass
class Result:
    config: Config
    stats: dict[str, float]
    aggregates: dict[str, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    # The index this config was scored against. The failure taxonomy needs the
    # full chunk list to tell "never retrieved" from "ranked too low", and it
    # must be the same chunks, not a rebuild that might have drifted.
    index: ChunkIndex | None = None


def chunk_stats(chunks: list[Chunk], queries: list[dict]) -> dict[str, float]:
    """Properties of the chunker alone, measured before any retrieval happens.

    Two of these bound what retrieval can possibly achieve:

    `spans_in_one_chunk` -- a gold span sliced across three chunks cannot be
    returned whole by any ranking, however good.

    `precision_ceiling` -- precision@k is capped by how many chunks are
    relevant at all. A query with one relevant chunk can never score above
    1/k. Because that count is a function of chunk size, the ceiling differs
    per strategy, and raw precision@k cannot be compared between them.
    """
    sizes = sorted(len(c.text) for c in chunks)

    whole = total = 0
    for q in queries:
        for s, e in q["gold_spans"]:
            hits = sum(1 for c in chunks
                       if c.doc_id == q["doc_id"] and c.start < e and s < c.end)
            whole += hits == 1
            total += 1

    relevant = []
    for q in queries:
        gold = [tuple(s) for s in q["gold_spans"]]
        relevant.append(sum(1 for c in chunks if c.doc_id == q["doc_id"]
                            and any(c.start < e and s < c.end for s, e in gold)))

    return {
        "n_chunks": len(chunks),
        "median_chars": sizes[len(sizes) // 2],
        "spans_in_one_chunk": whole / total if total else 0.0,
        "relevant_per_query": statistics.mean(relevant),
        "precision_ceiling": statistics.mean(min(r, P_AT) for r in relevant) / P_AT,
    }


def run_config(cfg: Config, eval_set: dict, reranker: Reranker | None,
               index: ChunkIndex, stats: dict[str, float],
               sparse: BM25Index | None = None) -> Result:
    queries = eval_set["queries"]
    chunks = index.chunks

    texts = {d["doc_id"]: d["text"] for d in eval_set["documents"]}

    rows = []
    for q in queries:
        gold = [tuple(s) for s in q["gold_spans"]]
        ranking = retrieve_ranked(index, q["query"], RANK_DEPTH,
                                  doc_id=q["doc_id"] if cfg.doc_filter else None,
                                  reranker=reranker if cfg.rerank else None,
                                  candidate_k=RANK_DEPTH,
                                  sparse=sparse, retriever=cfg.retriever,
                                  where=q.get("filters") if cfg.metadata_filter else None)
        ranked = ranking.chunks
        row = {
            "query_id": q["query_id"],
            "clause_type": q["clause_type"],
            "doc_id": q["doc_id"],
            f"precision@{P_AT}": precision_at_k(ranked, q["doc_id"], gold, P_AT),
            f"recall@{R_AT}": recall_at_k(ranked, q["doc_id"], gold, chunks, R_AT),
            "mrr": mrr(ranked, q["doc_id"], gold),
            f"span_coverage@{COVERAGE_AT}": span_coverage(ranked[:COVERAGE_AT], q["doc_id"], gold),
            f"span_recall@{GROUNDING_AT}": span_recall_at_k(
                ranked, q["doc_id"], gold, GROUNDING_AT, GROUNDING_THRESHOLD),
            f"complete_grounding@{GROUNDING_AT}": complete_grounding_at_k(
                ranked, q["doc_id"], gold, GROUNDING_AT, GROUNDING_THRESHOLD),
            "top_chunk_ids": [c.chunk_id for c in ranked[:P_AT]],
            "ranked_chunk_ids": [c.chunk_id for c in ranked],
        }

        # The clause the answer would have been missing. A grounding rate is a
        # number; this is the thing you put in front of a client.
        if row[f"complete_grounding@{GROUNDING_AT}"] is False \
                and (row[f"span_recall@{GROUNDING_AT}"] or 0) > 0:
            row["missed_spans"] = [
                {"start": s, "end": e, "covered": round(frac, 3),
                 "text": texts[q["doc_id"]][s:s + MISSED_SPAN_CHARS]}
                for (s, e), frac in uncovered_spans(
                    ranked, q["doc_id"], gold, GROUNDING_AT, GROUNDING_THRESHOLD)
            ]

        # Which retriever surfaced each relevant chunk that made the cut. This
        # is what backs "here are the queries dense retrieval alone gets
        # wrong" -- an aggregate delta cannot name a query.
        if ranking.sources:
            row["relevant_chunk_sources"] = {
                c.chunk_id: ranking.sources.get(c.chunk_id, {})
                for c in ranked[:GROUNDING_AT]
                if is_relevant(c, q["doc_id"], gold)
            }

        rows.append(row)

    aggregates = {}
    for key in METRIC_KEYS:
        # Booleans mean over to a rate; None means the metric was undefined for
        # that query (no gold spans) and must not be scored as a zero.
        vals = [r[key] for r in rows if r[key] is not None]
        aggregates[key] = statistics.mean(vals) if vals else float("nan")

    return Result(cfg, stats, aggregates, rows, index)


def run_all(eval_set: dict, encoder: Encoder, reranker: Reranker | None,
            configs: list[Config] = CONFIGS, verbose: bool = True) -> list[Result]:
    client = QdrantClient(":memory:")
    indexes: dict[str, tuple[ChunkIndex, BM25Index, dict[str, float]]] = {}
    results = []

    for cfg in configs:
        if cfg.rerank and reranker is None:
            if verbose:
                print(f"  skipping {cfg.name}: reranking disabled")
            continue

        # Configs that share a chunking strategy share its index; embedding the
        # same chunks twice would double the cost of every run.
        if cfg.strategy not in indexes:
            chunks = chunk_documents(eval_set["documents"], cfg.strategy)
            meta = document_metadata(eval_set["documents"])
            stats = chunk_stats(chunks, eval_set["queries"])
            stats["truncated"] = encoder.truncated_fraction([c.text for c in chunks])
            if verbose:
                print(f"  {cfg.strategy}: {stats['n_chunks']} chunks, "
                      f"median {stats['median_chars']} chars, "
                      f"{stats['spans_in_one_chunk']:.1%} of gold spans intact in one chunk, "
                      f"{stats['truncated']:.1%} truncated at {encoder.max_tokens} tokens")
            indexes[cfg.strategy] = (
                ChunkIndex(chunks, encoder, collection=f"chunks_{cfg.strategy}",
                           client=client, progress=verbose, doc_metadata=meta),
                BM25Index(chunks, doc_metadata=meta),
                stats,
            )

        index, sparse, stats = indexes[cfg.strategy]
        results.append(run_config(cfg, eval_set, reranker, index, stats, sparse))
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def format_table(results: list[Result]) -> str:
    name_w = max(6, *(len(r.config.name) for r in results))
    widths = [max(len(k), 7) for k in METRIC_KEYS]
    head = (f"| {'config':<{name_w}} | chunks | median chars | "
            + " | ".join(f"{k:>{w}}" for k, w in zip(METRIC_KEYS, widths, strict=True)) + " |")
    rule = (f"|{'-' * (name_w + 2)}|{'-' * 8}|{'-' * 14}|"
            + "|".join("-" * (w + 2) for w in widths) + "|")
    lines = [head, rule]
    for r in results:
        cells = " | ".join(f"{r.aggregates[k]:>{w}.3f}"
                           for k, w in zip(METRIC_KEYS, widths, strict=True))
        lines.append(f"| {r.config.name:<{name_w}} | {r.stats['n_chunks']:>6} "
                     f"| {r.stats['median_chars']:>12} | {cells} |")
    return "\n".join(lines)


def write_report(results: list[Result], eval_set: dict, out_dir: Path,
                 encoder: Encoder, reranker: Reranker | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, src = eval_set["meta"], eval_set["meta"]["source"]
    counts = meta["counts"]

    by_type: dict[str, dict[str, list[float]]] = {}
    for r in results:
        for row in r.per_query:
            by_type.setdefault(row["clause_type"], {}).setdefault(r.config.name, []).append(row["mrr"])

    lines = [
        "# Retrieval evaluation: chunking, retrieval, reranking and filtering",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by `run_eval.py`.",
        "",
        "## Setup",
        "",
        f"- **Corpus**: {counts['documents']} contracts from {src['dataset']}, "
        f"{counts['queries']} queries across {counts['clause_types']} clause types, "
        f"{counts['gold_spans']} lawyer-annotated gold spans.",
        f"- **Embedding**: `{encoder.model_name}` ({encoder.dim}-dim), Qdrant in-memory, cosine.",
        f"- **Reranker**: `{reranker.model_name}`." if reranker else "- **Reranker**: disabled.",
        "- **Retrievers**: dense (embeddings), BM25 (lexical), and the two fused with "
        "Reciprocal Rank Fusion at k=60. RRF rather than a weighted score sum, because "
        "cosine and BM25 scores are on unrelated scales and normalising them introduces "
        "a weight with no principled value.",
        f"- **Ranking depth**: {RANK_DEPTH} chunks per query, retrieved from within the "
        "contract the question is asked of (a Qdrant payload filter on `doc_id`). CUAD's "
        "questions are templated per clause type and worded identically for every "
        "contract, so a corpus-wide search is unanswerable by construction.",
        "",
        "## Results",
        "",
        format_table(results),
        "",
        "`span_coverage` is the fraction of gold characters present in the top "
        f"{COVERAGE_AT} chunks. It is the only metric here that is comparable across "
        "chunking strategies -- see Limitations in the README.",
        "",
        f"`span_recall@{GROUNDING_AT}` is the fraction of *distinct* gold spans that got "
        f"at least {GROUNDING_THRESHOLD:.0%} of their characters retrieved, and "
        f"`complete_grounding@{GROUNDING_AT}` is the rate at which *every* gold span "
        "cleared that bar. They exist because pooled character coverage cannot tell one "
        "fully-retrieved clause from two half-retrieved ones. A query whose answer "
        "depends on two clauses, one returned whole and one missed entirely, scores 0.5 "
        "on coverage and reads as partial success; it is a wrong answer carrying a "
        "confident citation. Complete grounding is the rate at which a fully-cited "
        f"answer was possible at all. The {GROUNDING_THRESHOLD:.0%} threshold is fixed, "
        "not fitted per corpus.",
        "",
        "Per-query detail, including the text of every clause a failing query missed, "
        "is in `per_query.json` under `missed_spans`.",
        "",
        "## What chunking alone decides",
        "",
        "Both columns below are fixed before a single query runs.",
        "",
        "| strategy | chunks | median chars | gold spans intact in one chunk | "
        f"truncated at {encoder.max_tokens} tokens | relevant chunks per query | "
        "ceiling on precision@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for r in results:
        if r.config.strategy in seen:
            continue
        seen.add(r.config.strategy)
        s = r.stats
        lines.append(f"| {r.config.strategy} | {s['n_chunks']} | {s['median_chars']} "
                     f"| {s['spans_in_one_chunk']:.1%} | {s['truncated']:.1%} "
                     f"| {s['relevant_per_query']:.2f} | {s['precision_ceiling']:.3f} |")

    ceilings = {r.config.strategy: r.stats["precision_ceiling"] for r in results}
    attained = {r.config.name: r.aggregates[f"precision@{P_AT}"] /
                ceilings[r.config.strategy] for r in results}
    lines += [
        "",
        "The last column is why precision@5 must not be read as a ranking of the "
        "strategies. Larger chunks overlap more gold spans, so more chunks count as "
        "relevant, so precision@5 can climb without a single extra gold character "
        "being retrieved. As a share of its own ceiling, each config attained:",
        "",
        "| config | precision@5 | ceiling | attained |",
        "|---|---|---|---|",
    ] + [
        f"| {r.config.name} | {r.aggregates[f'precision@{P_AT}']:.3f} "
        f"| {ceilings[r.config.strategy]:.3f} | {attained[r.config.name]:.1%} |"
        for r in results
    ]

    lines += ["", "## Mean MRR by clause type", "",
              "Only 1-4 queries per clause type, so read these as texture rather than "
              "evidence; the aggregate table above is what carries weight.",
              "",
              "| clause type | n | " + " | ".join(r.config.name for r in results) + " |",
              "|---" * (len(results) + 2) + "|"]
    for ctype in sorted(by_type):
        n = len(by_type[ctype][results[0].config.name])
        cells = " | ".join(f"{statistics.mean(by_type[ctype][r.config.name]):.3f}" for r in results)
        lines.append(f"| {ctype} | {n} | {cells} |")

    lines += [
        "",
        "## Provenance",
        "",
        f"- Source: {src['dataset']}, {src['license']}, (c) {src['attribution']}.",
        f"- `{src['file']}` sha256 `{src['sha256']}`.",
        f"- Eval set built by `data/prepare.py` with seed {meta['sampling']['seed']}; "
        "rerunning it selects the same documents and queries.",
        "",
    ]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    (out_dir / "per_query.json").write_text(
        json.dumps({r.config.name: r.per_query for r in results}, indent=2),
        encoding="utf-8",
    )
    return report
