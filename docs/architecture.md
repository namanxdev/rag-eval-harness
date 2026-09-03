# Architecture

How a number gets from a contract to a report, and the contracts between the
pieces that make it along the way.

## The pipeline

```
eval_set.json                     the integration boundary (see Schema below)
     |
     |  chunk_documents(documents, strategy)          src/chunking.py
     v
list[Chunk]                       every chunk carries (start, end) into its doc
     |
     +---> ChunkIndex(chunks, encoder, doc_metadata)  src/index.py    dense
     +---> BM25Index(chunks, doc_metadata)            src/sparse.py   lexical
     |
     |  retrieve_ranked(index, query, k, ..., sparse, retriever, where)
     v                                                src/retrieve.py
Ranking(chunks, sources)          ranked chunks + which retriever found each
     |
     |  precision / recall / mrr / span_coverage / span_recall / grounding
     v                                                src/metrics.py
Result(config, stats, aggregates, per_query, index)   src/evaluate.py
     |
     +---> write_report()          results/report.md          developer
     +---> classify()              src/taxonomy.py -> one label per failure
     +---> write_client_report()   results/client_report.md   client
```

`run_eval.py` wires it together. `run_all()` is the loop; `run_config()` is one
configuration over every query.

## Module responsibilities

| module | owns | must not |
|---|---|---|
| `src/chunking.py` | Splitting documents while preserving offsets | Know anything about retrieval or scoring |
| `src/index.py` | Embeddings, the Qdrant collection, the filterable payload | Decide *what* to retrieve for a config |
| `src/sparse.py` | BM25 scoring and rank fusion | Load a model or touch the network |
| `src/retrieve.py` | Choosing and combining retrievers, reranking | Score anything against gold |
| `src/metrics.py` | Turning (retrieved, gold) into numbers | Know about configs or reports |
| `src/taxonomy.py` | Turning a failure into an auditable label | Call a model, or invent evidence |
| `src/evaluate.py` | The run loop and the developer report | Contain client-facing prose |
| `src/report.py` | The client readout | Compute a metric of its own |

The dependency direction is strictly downward in that table. `metrics.py`
importing `evaluate.py` would be a cycle and a design error; the metrics must
stay usable without a run.

## Index sharing

Configurations that share a chunking strategy share its indexes. Embedding the
same 972 chunks once per config would multiply the cost of a run by the number
of configs for no change in the result:

```python
indexes: dict[str, tuple[ChunkIndex, BM25Index, dict]] = {}
```

Keyed by `cfg.strategy`. The seven shipped configs therefore build two dense
indexes (naive, clause) and two BM25 indexes, not seven of each. Qdrant runs
in `:memory:` with one collection per strategy, so the two never share a
search space.

## The three retrievers

`retriever` on a `Config` selects the first stage. All three feed the same
optional rerank stage.

| value | first stage | notes |
|---|---|---|
| `"dense"` | `ChunkIndex.search` | The default. `sparse=None` is enough. |
| `"bm25"` | `BM25Index.search` | Never touches the embedding index. |
| `"hybrid"` | both, fused with RRF | Requires a `BM25Index` or raises. |

Fusion happens over the wider candidate pool, not the top k of each list: a
chunk ranked 8th by both retrievers should beat one ranked 1st by a single
retriever, and it can only do that if both lists are long enough to contain
it.

Reranking only ever reorders, so when a reranker is present the pool widens to
`max(candidate_k, k)` before it runs.

## Two filters, both evaluable

Neither filter is fixed plumbing — each is a `Config` field so a run can turn
it off and measure the difference.

- `doc_filter` scopes retrieval to the document a question is asked of.
- `metadata_filter` applies whatever a query declared in its `filters` field.

Document metadata reaches the index through `document_metadata()`, which
treats every document key except `doc_id`, `title`, `n_chars` and `text` as
filterable payload. Metadata may not shadow a chunk's own payload keys
(`CHUNK_PAYLOAD_KEYS`); a client field called `start` silently breaking offset
arithmetic is the kind of failure that surfaces as a wrong metric weeks later,
so it raises instead.

Both filters apply to the dense and the sparse side. A `where` that only
filtered one of them would quietly change what hybrid retrieval means.

## Invariants

Break these and the numbers stop meaning anything. The first is asserted in
code; the rest are design constraints.

1. **`chunk.text == document[chunk.start:chunk.end]`.** `chunk_documents`
   raises if a chunker violates it. Every relevance decision is offset
   arithmetic, so a chunk whose text and offsets disagree corrupts everything
   downstream.
2. **Relevance is character overlap against gold spans**, never string
   matching, and never a model's opinion.
3. **The harness runs CPU-only with no API key.** No hosted-model dependency
   in the core path.
4. **`python run_eval.py` with no arguments works and writes
   `results/report.md`.**
5. **`precision@k` is not comparable across chunking strategies.** Its ceiling
   is a function of chunk size, which is why `chunk_stats` reports
   `precision_ceiling`. Any new comparison must either use a
   chunking-invariant metric or report the ceiling alongside it.
6. **Determinism.** No sampling at run time, and ties broken explicitly —
   BM25 by chunk order, RRF by best input rank then chunk id. Three
   consecutive runs produced identical numbers.

## The schema is the boundary

Everything downstream reads this shape and nothing else. `data/prepare.py`
emits it from CUAD; `data/adapt.py` emits it from a client corpus; both are
validated by the same `validate()` so they cannot drift apart.

```jsonc
{
  "meta":      { "name", "generated_utc", "source", "sampling", "counts" },
  "documents": [ { "doc_id", "title", "n_chars", "text", ...metadata } ],
  "queries":   [ {
      "query_id":    "q001",
      "doc_id":      "...",
      "clause_type": "...",
      "query":       "...",
      "gold_spans":  [[6873, 7069], [7070, 7600]],   // sorted, non-overlapping
      "gold_texts":  ["...", "..."],                 // must equal text[start:end]
      "filters":     { "entity": "Acme" }            // optional, drives metadata_filter
  } ]
}
```

Extra keys on a document become filterable metadata. Extra keys on a query are
ignored except `filters`.

## Extending it

**A new chunking strategy** — add it to `STRATEGIES` in `chunking.py` and give
it a `Config`. It must satisfy invariant 1. If it does not tile its document
contiguously, expect the `ceiling-bound` and `partial-grounding` labels to
start firing; that is them working, not a bug.

**A new retriever** — add it to `RETRIEVERS` and a branch in
`retrieve_ranked`. Fuse it in `reciprocal_rank_fusion` by passing another named
ranking; the attribution recorded in `Ranking.sources` picks up the new name
automatically.

**A new metric** — add it to `metrics.py`, then to `METRIC_KEYS`. The
aggregates, the developer table and the client table all iterate that list, so
one entry is enough. Return `None` when the metric is undefined for a query
rather than `0.0`; the aggregator skips `None` and would otherwise average in
a misleading zero.

**A new failure label** — add it to `FIX_FOR` and `LEVER`, then a branch in
`_label_one` in the position its ordering demands. The labels are
first-match-wins because earlier faults cause later ones, so where you insert
it is part of the design, not a detail. Its evidence must be enough for a
human to verify the label without rerunning.
