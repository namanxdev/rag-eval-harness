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
from src.metrics import mrr, precision_at_k, recall_at_k, span_coverage
from src.retrieve import Reranker, retrieve

# One ranked list per query, scored at several depths. Every config shares the
# same depth so the only difference between them is chunking and ordering.
RANK_DEPTH = 30
P_AT = 5
R_AT = 10
COVERAGE_AT = 5


@dataclass
class Config:
    name: str
    strategy: str
    rerank: bool
    description: str = ""


CONFIGS = [
    Config("naive", "naive", False, "fixed 1000-char windows, 150 overlap"),
    Config("clause", "clause", False, "structural clause boundaries"),
    Config("clause + rerank", "clause", True, "clause boundaries, cross-encoder rerank of top 30"),
]


@dataclass
class Result:
    config: Config
    stats: dict[str, float]
    aggregates: dict[str, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


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
               index: ChunkIndex, stats: dict[str, float]) -> Result:
    queries = eval_set["queries"]
    chunks = index.chunks

    rows = []
    for q in queries:
        gold = [tuple(s) for s in q["gold_spans"]]
        ranked = retrieve(index, q["query"], RANK_DEPTH, doc_id=q["doc_id"],
                          reranker=reranker if cfg.rerank else None,
                          candidate_k=RANK_DEPTH)
        rows.append({
            "query_id": q["query_id"],
            "clause_type": q["clause_type"],
            "doc_id": q["doc_id"],
            f"precision@{P_AT}": precision_at_k(ranked, q["doc_id"], gold, P_AT),
            f"recall@{R_AT}": recall_at_k(ranked, q["doc_id"], gold, chunks, R_AT),
            "mrr": mrr(ranked, q["doc_id"], gold),
            f"span_coverage@{COVERAGE_AT}": span_coverage(ranked[:COVERAGE_AT], q["doc_id"], gold),
            "top_chunk_ids": [c.chunk_id for c in ranked[:P_AT]],
        })

    keys = [f"precision@{P_AT}", f"recall@{R_AT}", "mrr", f"span_coverage@{COVERAGE_AT}"]
    aggregates = {}
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        aggregates[key] = statistics.mean(vals) if vals else float("nan")

    return Result(cfg, stats, aggregates, rows)


def run_all(eval_set: dict, encoder: Encoder, reranker: Reranker | None,
            configs: list[Config] = CONFIGS, verbose: bool = True) -> list[Result]:
    client = QdrantClient(":memory:")
    indexes: dict[str, tuple[ChunkIndex, dict[str, float]]] = {}
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
            stats = chunk_stats(chunks, eval_set["queries"])
            stats["truncated"] = encoder.truncated_fraction([c.text for c in chunks])
            if verbose:
                print(f"  {cfg.strategy}: {stats['n_chunks']} chunks, "
                      f"median {stats['median_chars']} chars, "
                      f"{stats['spans_in_one_chunk']:.1%} of gold spans intact in one chunk, "
                      f"{stats['truncated']:.1%} truncated at {encoder.max_tokens} tokens")
            indexes[cfg.strategy] = (
                ChunkIndex(chunks, encoder, collection=f"chunks_{cfg.strategy}",
                           client=client, progress=verbose),
                stats,
            )

        index, stats = indexes[cfg.strategy]
        results.append(run_config(cfg, eval_set, reranker, index, stats))
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def format_table(results: list[Result]) -> str:
    keys = [f"precision@{P_AT}", f"recall@{R_AT}", "mrr", f"span_coverage@{COVERAGE_AT}"]
    head = f"| {'config':<16} | chunks | median chars | " + " | ".join(f"{k:>18}" for k in keys) + " |"
    rule = f"|{'-'*18}|{'-'*8}|{'-'*14}|" + "|".join("-" * 20 for _ in keys) + "|"
    lines = [head, rule]
    for r in results:
        cells = " | ".join(f"{r.aggregates[k]:>18.3f}" for k in keys)
        lines.append(f"| {r.config.name:<16} | {r.stats['n_chunks']:>6} "
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
        "# Retrieval evaluation: naive vs clause-aware chunking",
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
