"""The client-facing readout.

`write_report` in evaluate.py is the developer report: every config, every
diagnostic, aimed at whoever is changing the harness. This is the other
document -- what the client bought. Four sections, in the order a buyer reads
them: where you are, what is going wrong, what to fix first, and what was and
was not covered.

One rule governs the whole file: **no number appears here that was not produced
by a run in this repo.** Every rate is read off a `Result`, every failure count
off a `FailureLabel`, every scope figure off the eval set's own `meta`. Where a
lever was not run, the backlog says "not measured" rather than estimating. A
readout that mixes measured and plausible numbers is worth less than one that
measures fewer things.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluate import (
    COVERAGE_AT,
    GROUNDING_AT,
    GROUNDING_THRESHOLD,
    P_AT,
    R_AT,
    RANK_DEPTH,
    Config,
    Result,
)
from src.taxonomy import FIX_FOR, LEVER, classify, taxonomy_summary

GROUNDING_KEY = f"complete_grounding@{GROUNDING_AT}"

BASELINE_ROWS = [
    (f"precision@{P_AT}", "of the top 5 chunks, the share that hit a gold clause"),
    (f"recall@{R_AT}", "of every chunk holding a gold clause, the share in the top 10"),
    ("mrr", "1 / the rank of the first correct chunk, averaged"),
    (f"span_coverage@{COVERAGE_AT}", "of all gold clause characters, the share retrieved"),
    (f"span_recall@{GROUNDING_AT}",
     f"of gold clauses, the share at least {GROUNDING_THRESHOLD:.0%} retrieved"),
    (GROUNDING_KEY, "share of questions where EVERY gold clause was retrieved"),
]


def lever_diff(baseline: Config, other: Config) -> list[str]:
    """Which levers differ between two configs, in the backlog's vocabulary.

    Derived from the configs themselves rather than a lookup table, so a delta
    can never be attributed to a lever that was not actually changed.
    """
    levers = []
    if other.strategy != baseline.strategy:
        levers.append("chunking")
    if other.retriever != baseline.retriever:
        levers.append("hybrid retrieval")
    if other.rerank != baseline.rerank:
        levers.append("reranking")
    if (other.doc_filter, other.metadata_filter) != \
            (baseline.doc_filter, baseline.metadata_filter):
        levers.append("metadata filtering")
    return levers


LEVER_FIELD = {
    "chunking": lambda c: c.strategy,
    "hybrid retrieval": lambda c: c.retriever,
    "reranking": lambda c: c.rerank,
    "metadata filtering": lambda c: (c.doc_filter, c.metadata_filter),
}


def _measured(lever: str, baseline: Result, results: list[Result]) -> tuple[str, str]:
    """Best measured grounding change for one lever, as (comparison, delta).

    A comparison counts only when the two configs differ in this one lever and
    nothing else -- a config that changes two things at once cannot attribute
    its delta to either.

    Preferred comparison is against the baseline itself. Where no config
    isolates the lever there, any other pair in the run that does is used
    instead, named in full so the client can see what was compared. The
    direction is always *away from the baseline's current setting*, so a
    negative delta means changing the lever made things worse -- never that the
    comparison was run backwards.
    """
    base_value = LEVER_FIELD[lever](baseline.config)
    pairs = [(a, b) for a in results for b in results
             if lever_diff(a.config, b.config) == [lever]
             and LEVER_FIELD[lever](a.config) == base_value]
    if not pairs:
        return "not measured", "not measured"

    # Against the baseline where possible; it is the config the client runs.
    direct = [(a, b) for a, b in pairs if a is baseline]
    a, b = max(direct or pairs,
               key=lambda ab: ab[1].aggregates[GROUNDING_KEY] - ab[0].aggregates[GROUNDING_KEY])
    delta = b.aggregates[GROUNDING_KEY] - a.aggregates[GROUNDING_KEY]
    label = b.config.name if a is baseline else f"{a.config.name} -> {b.config.name}"
    return label, f"{delta:+.1%}"


def build_backlog(baseline: Result, results: list[Result], summary: dict) -> list[dict]:
    """Failure patterns turned into ranked, evidenced work.

    Ranked by how many failing queries each lever accounts for, because that is
    the only ordering the measurements support. It is not an ordering by effort
    or by cost -- those are the client's to supply.
    """
    backlog = []
    for label, count in summary["counts"].items():
        lever = LEVER[label]
        if lever is None:
            continue
        comparison, delta = _measured(lever, baseline, results)
        backlog.append({
            "lever": lever,
            "label": label,
            "queries_addressed": count,
            "measured_by": comparison,
            "grounding_delta": delta,
        })
    return backlog


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def _join(items: list[str]) -> str:
    if not items:
        return "nothing"
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def write_client_report(results: list[Result], eval_set: dict, out_dir: Path,
                        baseline_name: str | None = None) -> Path:
    """Write `client_report.md` and the taxonomy labels behind it."""
    if not results:
        raise ValueError("no results to report on")
    baseline = next((r for r in results if r.config.name == baseline_name), results[0])

    if baseline.index is None:
        raise ValueError(
            f"{baseline.config.name} carries no index; the failure taxonomy needs "
            "the chunk list it was scored against")
    labels = classify(baseline, eval_set, baseline.index)
    summary = taxonomy_summary(labels)
    backlog = build_backlog(baseline, results, summary)

    meta = eval_set["meta"]
    counts, src = meta["counts"], meta["source"]
    n_queries = len(baseline.per_query)
    n_failing = summary["n_failing"]
    by_id = {lb.query_id: lb for lb in labels}
    queries = {q["query_id"]: q for q in eval_set["queries"]}

    L = [
        "# Retrieval quality baseline",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC from a "
        f"run of this repository on `{meta['name']}`. Every figure below was produced "
        "by that run; nothing is carried over from another engagement and nothing is "
        "estimated.",
        "",
        f"Baseline configuration: **{baseline.config.name}** "
        f"({baseline.config.description}).",
        "",
        "## 1. Where retrieval stands today",
        "",
        "| metric | value | what it means |",
        "|---|---|---|",
    ]
    L += [f"| `{key}` | {baseline.aggregates[key]:.3f} | {gloss} |"
          for key, gloss in BASELINE_ROWS]

    grounded = baseline.aggregates[GROUNDING_KEY]
    L += [
        "",
        f"**{grounded:.0%} of questions ({n_queries - n_failing} of {n_queries}) could be "
        "answered with every clause they depend on actually retrieved.** The remaining "
        f"{n_failing} can produce an answer, and it will carry a citation, but the "
        "citation will be incomplete. That is the number the rest of this report is "
        "about; the metrics above it are diagnostics, not outcomes.",
        "",
        f"A clause counts as retrieved when at least {GROUNDING_THRESHOLD:.0%} of its "
        f"characters appear in the top {GROUNDING_AT} chunks. The threshold is fixed "
        "across every run and every client.",
        "",
        "## 2. Top failure patterns",
        "",
    ]

    if not summary["patterns"]:
        L += ["No query failed on this configuration; there is no pattern to report.", ""]
    else:
        L += [f"Each of the {n_failing} failing questions carries exactly one label, "
              "assigned by rule from chunk offsets and rank positions -- no model "
              "judged them, and every label can be re-derived from `per_query.json` "
              "and `taxonomy.json`.",
              ""]
        for n, pattern in enumerate(summary["patterns"], 1):
            L += [
                f"### {n}. `{pattern['label']}` -- {pattern['count']} of {n_failing} "
                f"failures ({pattern['share']:.0%})",
                "",
                f"Points at: **{FIX_FOR[pattern['label']]}**.",
                "",
            ]
            for qid in pattern["examples"]:
                q, ev = queries[qid], by_id[qid].evidence
                L += [f"- **{qid}** ({q['clause_type']}, `{q['doc_id']}`) -- "
                      f"{ev[f'span_recall@{GROUNDING_AT}']:.0%} of its clauses retrieved. "
                      "Missing:"]
                L += [f"  > {u['text'].strip()[:200]}..." for u in ev["uncovered"][:2]]
                L.append("")

    L += ["## 3. Fix backlog", ""]
    if backlog:
        L += [
            "Ranked by how many failing questions each lever accounts for. `grounding "
            "delta` is the measured change in section 1's headline number when that "
            "lever alone was changed, always in the direction of moving away from the "
            "baseline's current setting -- so a negative number means the change made "
            "things worse. Where the comparison names two configurations, the lever "
            "was isolated between those two rather than against the baseline. Levers "
            "no pair of configurations isolated are marked *not measured* rather than "
            "guessed at.",
            "",
            "| # | lever | failures it addresses | measured by | grounding delta |",
            "|---|---|---|---|---|",
        ]
        L += [f"| {n} | {e['lever']} | {e['queries_addressed']} | `{e['measured_by']}` "
              f"| {e['grounding_delta']} |"
              for n, e in enumerate(backlog, 1)]
    else:
        L.append("No failures to address on this configuration.")

    L += ["", "### What every configuration scored", "",
          "| config | " + " | ".join(k for k, _ in BASELINE_ROWS) + " |",
          "|---" * (len(BASELINE_ROWS) + 1) + "|"]
    L += ["| " + ("**{}**" if r is baseline else "{}").format(r.config.name) + " | "
          + " | ".join(_fmt(r.aggregates[k]) for k, _ in BASELINE_ROWS) + " |"
          for r in results]

    # The backlog is driven by the failure taxonomy, so a lever that fixes no
    # *labelled* failure never appears in it even when it scored best overall.
    # State the best measured configuration outright rather than leaving the
    # client to find it in the table.
    best = max(results, key=lambda r: r.aggregates[GROUNDING_KEY])
    if best is not baseline:
        gain = best.aggregates[GROUNDING_KEY] - grounded
        L += ["",
              f"The highest measured grounding rate in this run was "
              f"**{best.config.name}** at {best.aggregates[GROUNDING_KEY]:.3f}, "
              f"{gain:+.1%} against the baseline. It changes "
              f"{_join(lever_diff(baseline.config, best.config))} relative to the "
              "baseline, so the gain cannot be attributed to a single lever from "
              "this run alone."]

    L += [
        "",
        "## 4. Scope",
        "",
        f"- **Corpus**: {counts['documents']} documents, {counts['queries']} queries "
        f"across {counts['clause_types']} clause types, {counts['gold_spans']} gold "
        "spans. Small enough that per-clause-type figures are texture, not evidence.",
        f"- **Source**: {src['dataset']}.",
        f"- **Retrieval depth**: every configuration ranked {RANK_DEPTH} chunks per "
        f"query and was scored on the top {GROUNDING_AT}.",
        "- **Relevance**: character overlap with a gold span, never string matching. "
        "A chunk counts only if it comes from the right document and overlaps a span "
        "a human marked.",
        "- **Not covered**: answer generation, faithfulness of generated text, latency, "
        "and cost. This measures what retrieval puts in front of the model, not what "
        "the model does with it.",
    ]
    if src.get("extractors"):
        L.append(f"- **Text extraction**: {', '.join(src['extractors'])}. Gold offsets "
                 "are valid only against text produced this way; re-extracting with a "
                 "different tool invalidates the judgments.")
    for note in meta.get("sampling", {}).get("notes", []):
        L.append(f"- {note}")
    L.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "client_report.md"
    report.write_text("\n".join(L), encoding="utf-8")

    (out_dir / "taxonomy.json").write_text(
        json.dumps({"config": baseline.config.name, "summary": summary,
                    "labels": [{"query_id": lb.query_id, "label": lb.label,
                                "evidence": lb.evidence} for lb in labels]}, indent=2),
        encoding="utf-8")
    return report
