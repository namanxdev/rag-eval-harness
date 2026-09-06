"""Why each query failed, decided from artifacts the harness already produced.

Every label here is derived from chunk offsets and rank positions. There is no
model call and no judge: a client can re-derive any label by hand from
`per_query.json` and the eval set, which is the point. A taxonomy a buyer
cannot audit is an opinion with a table around it.

A query is *failing* when `complete_grounding@GROUNDING_AT` is False -- some
clause its answer depends on was not retrieved. Passing queries get no label.
Each failing query gets exactly one label, first match wins, in the order
below, because the earlier faults cause the later ones: a clause sliced across
two chunks will also look like a ranking problem, and fixing the ranking will
not fix it.

    ceiling-bound      metric artifact, not a retrieval fault -> no fix
    chunk-severance    a needed clause is split across chunks  -> chunking
    retrieval-miss     the chunk never entered the candidate pool
                                                              -> hybrid, embeddings
    rank-miss          the chunk was in the pool, ranked too low -> reranking
    partial-grounding  something got through, not enough of it -> metadata, k

Scoping: labels 2-4 are judged against the spans that actually failed, not
against every gold span in the query. A clause that was split but retrieved
whole did not cause this failure, and labelling it "chunking" would send the
client to fix something that is working. `uncovered` in the evidence names the
spans each label is answering for.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.chunking import Chunk
from src.evaluate import GROUNDING_AT, GROUNDING_THRESHOLD, P_AT, RANK_DEPTH, Result
from src.index import ChunkIndex
from src.metrics import uncovered_spans

# What each label tells the client to go and change. The report's fix backlog
# is built from these, so the wording is the deliverable, not a comment.
FIX_FOR = {
    "ceiling-bound": "none -- metric artifact",
    "chunk-severance": "chunking",
    "retrieval-miss": "hybrid retrieval, embedding model",
    "rank-miss": "reranking",
    "partial-grounding": "metadata filtering, k tuning",
}

# The same fixes as a controlled vocabulary, so the backlog can match a label
# against the config that actually measured that lever instead of guessing from
# prose. `None` where no config could change the outcome.
LEVER = {
    "ceiling-bound": None,
    "chunk-severance": "chunking",
    "retrieval-miss": "hybrid retrieval",
    "rank-miss": "reranking",
    "partial-grounding": "metadata filtering",
}

LABEL_ORDER = list(FIX_FOR)

# How much of a missed clause to carry into the evidence.
EVIDENCE_CHARS = 120


@dataclass(frozen=True)
class FailureLabel:
    query_id: str
    label: str
    evidence: dict = field(default_factory=dict)


def _overlapping(chunks: list[Chunk], doc_id: str, span: tuple[int, int]) -> list[Chunk]:
    s, e = span
    return [c for c in chunks if c.doc_id == doc_id and c.start < e and s < c.end]


def classify(result: Result, eval_set: dict, index: ChunkIndex) -> list[FailureLabel]:
    """One label per failing query in `result`, in query order."""
    queries = {q["query_id"]: q for q in eval_set["queries"]}
    texts = {d["doc_id"]: d["text"] for d in eval_set["documents"]}
    by_id = {c.chunk_id: c for c in index.chunks}

    labels: list[FailureLabel] = []
    for row in result.per_query:
        if row[f"complete_grounding@{GROUNDING_AT}"] is not False:
            continue

        q = queries[row["query_id"]]
        doc_id = q["doc_id"]
        gold = [tuple(s) for s in q["gold_spans"]]
        ranked = [by_id[cid] for cid in row["ranked_chunk_ids"]]
        rank_of = {c.chunk_id: i for i, c in enumerate(ranked)}

        missed = uncovered_spans(ranked, doc_id, gold, GROUNDING_AT, GROUNDING_THRESHOLD)
        base = {
            "config": result.config.name,
            "clause_type": q["clause_type"],
            "doc_id": doc_id,
            f"span_recall@{GROUNDING_AT}": row[f"span_recall@{GROUNDING_AT}"],
            "uncovered": [
                {"start": s, "end": e, "covered": round(frac, 3),
                 "text": texts[doc_id][s:s + EVIDENCE_CHARS]}
                for (s, e), frac in missed
            ],
        }
        labels.append(_label_one(row, doc_id, gold, missed, ranked, rank_of, index, base))

    return labels


def _label_one(row, doc_id, gold, missed, ranked, rank_of, index, base) -> FailureLabel:
    qid = row["query_id"]

    # 1. ceiling-bound. The query has fewer relevant chunks than k, so its
    #    precision@k could never reach 1.0, and every relevant chunk is already
    #    in the top k -- nothing about retrieval is broken.
    #
    #    On a chunker that tiles its document contiguously this cannot fire:
    #    if every relevant chunk is retrieved then every gold character is
    #    retrieved, and the query is not failing. It is kept as a guard for
    #    chunkers that drop text (sliding windows with gaps, filtered corpora),
    #    where a query can be unfixable by ranking alone. Expect a count of 0
    #    on naive and clause; a non-zero count means the chunker is leaving
    #    text out of the index, which is worth knowing before tuning anything.
    relevant = [c for c in index.chunks
                if c.doc_id == doc_id and any(c.start < e and s < c.end for s, e in gold)]
    ceiling = min(len(relevant), P_AT) / P_AT
    ranks = {c.chunk_id: rank_of.get(c.chunk_id) for c in relevant}
    if ceiling < 1.0 and relevant and all(
            r is not None and r < GROUNDING_AT for r in ranks.values()):
        return FailureLabel(qid, "ceiling-bound", base | {
            "n_relevant_chunks": len(relevant),
            "ceiling_contribution": ceiling,
            "relevant_chunk_ranks": ranks,
        })

    # 2. chunk-severance. A clause the answer needed is split across chunks, so
    #    no single retrieved chunk carries it and the ranking would have to
    #    surface all the pieces to succeed. Judged only against clauses that
    #    actually failed.
    severed = []
    for (s, e), frac in missed:
        pieces = _overlapping(index.chunks, doc_id, (s, e))
        if len(pieces) > 1:
            severed.append({
                "span": [s, e],
                "covered": round(frac, 3),
                "n_chunks": len(pieces),
                "chunks": [{"chunk_id": c.chunk_id, "start": c.start, "end": c.end,
                            "rank": rank_of.get(c.chunk_id)} for c in pieces],
            })
    if severed:
        return FailureLabel(qid, "chunk-severance", base | {"severed_spans": severed})

    # The chunks that would have fixed this query: the ones overlapping a
    # clause that failed. Labels 3 and 4 split on where these ended up.
    culprits = [c for span, _ in missed for c in _overlapping(index.chunks, doc_id, span)]
    culprits = list({c.chunk_id: c for c in culprits}.values())

    # 3. retrieval-miss. The chunk never entered the candidate pool at all, so
    #    no amount of reranking could have saved it.
    absent = [c for c in culprits if c.chunk_id not in rank_of]
    if absent:
        in_doc = sum(1 for c in index.chunks if c.doc_id == doc_id)
        return FailureLabel(qid, "retrieval-miss", base | {
            "rank_depth": RANK_DEPTH,
            "chunks_in_doc": in_doc,
            "missing_chunks": [{"chunk_id": c.chunk_id, "start": c.start, "end": c.end}
                               for c in absent],
        })

    # 4. rank-miss. The chunk was retrieved, just not high enough. This is the
    #    one a reranker fixes.
    below = [c for c in culprits if rank_of[c.chunk_id] >= GROUNDING_AT]
    if below:
        return FailureLabel(qid, "rank-miss", base | {
            "k": GROUNDING_AT,
            "ranked_below_k": [{"chunk_id": c.chunk_id, "rank": rank_of[c.chunk_id]}
                               for c in below],
        })

    # 5. partial-grounding. Some of the clause came back, not enough of it, and
    #    none of the structural explanations above apply.
    return FailureLabel(qid, "partial-grounding", base | {
        "threshold": GROUNDING_THRESHOLD,
        "n_retrieved_culprit_chunks": len(culprits),
    })


def taxonomy_summary(labels: list[FailureLabel], top_n: int = 3) -> dict:
    """Counts by label, descending, plus up to 2 example query_ids per label."""
    counts = Counter(lb.label for lb in labels)
    total = len(labels)
    examples: dict[str, list[str]] = {}
    for lb in labels:
        examples.setdefault(lb.label, []).append(lb.query_id)

    # Ties broken by the label order so the same input always summarises the
    # same way; a report that reorders itself between runs is not evidence.
    ordered = sorted(counts, key=lambda lb: (-counts[lb], LABEL_ORDER.index(lb)))
    return {
        "n_failing": total,
        "counts": {lb: counts[lb] for lb in ordered},
        "patterns": [
            {
                "label": lb,
                "count": counts[lb],
                "share": counts[lb] / total if total else 0.0,
                "fix": FIX_FOR[lb],
                "examples": examples[lb][:2],
            }
            for lb in ordered[:top_n]
        ],
    }
