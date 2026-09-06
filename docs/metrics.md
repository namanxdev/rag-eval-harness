# Metrics and failure labels

Exact definitions, with worked examples. This is the document to hand someone
who disputes a number.

Every metric below is computed in `src/metrics.py` against character-level gold
spans. None of them involves string matching, and none involves a model
judging relevance.

## The relevance predicate

Everything else is built on this:

```python
def is_relevant(chunk, doc_id, gold_spans) -> bool:
    return chunk.doc_id == doc_id and any(chunk.overlaps(s, e) for s, e in gold_spans)
```

A chunk overlaps a span when `chunk.start < end and start < chunk.end`. Note
the `doc_id` test comes first: character offsets are meaningful only relative
to the document that produced them, so a chunk from another contract whose
offsets happen to fall in the gold range is not a hit.

In a normal run retrieval is already scoped to one document, so that test never
fires. It is a guard for when the scope is removed — which is exactly what the
`clause, unfiltered` configuration does.

## Retrieval metrics

### `precision@k` — default k=5

Fraction of the top k that are relevant. **Divided by k, not by
`len(retrieved)`**, so returning three chunks when five were asked for is
penalised rather than flattered.

> **Not comparable across chunking strategies.** See [the ceiling](#the-precision-ceiling) below.

### `recall@k` — default k=10

Share of *all* relevant chunks that made the top k. The denominator is every
relevant chunk in the query's document, not just the retrieved ones.

Returns `None` when nothing is relevant at all, which would otherwise score a
misleading `0.0`. That cannot happen with the shipped eval set, where every
query has a gold span inside its document, but it guards a regenerated set.

Inherits the same comparability flaw as precision, via its denominator.

### `mrr`

Reciprocal of the rank of the first relevant chunk, `0.0` if none. Averaged
over queries. Measures how far a reader scrolls before hitting something
useful; says nothing about what comes after.

### `span_coverage@k` — default k=5

Fraction of gold *characters* present in the top k, pooled across all of a
query's spans. Characters are counted once even when several chunks overlap
them.

**This is the chunking-invariant metric.** Precision and recall count chunks,
so their values shift when chunk size shifts even if identical text is
returned. Counting characters does not.

### `span_recall@k` — default k=5, threshold 0.5

Fraction of **distinct** gold spans that got at least `threshold` of their own
characters retrieved. Returns `None` when the query has no gold spans.

### `complete_grounding@k` — default k=5, threshold 0.5

`True` only when **every** gold span clears the threshold. Aggregated as a rate
over queries.

This is the metric that matters: the binary of whether a correct, fully-cited
answer was possible from this retrieval at all.

## Why grounding exists: a worked example

`q001` (Exclusivity) has two gold spans:

| span | offsets | length |
|---|---|---|
| A | `[6873, 7069]` | 196 chars |
| B | `[7070, 7600]` | 530 chars |

Retrieve a chunk set covering **only B**:

| metric | value | reads as |
|---|---|---|
| `span_coverage@5` | 530 / 726 = **0.730** | healthy |
| `span_recall@5` | 1 of 2 spans = **0.500** | half the clauses missing |
| `complete_grounding@5` | **False** | the answer cannot be fully cited |

Pooled coverage says 0.73 because span B is long. An answer built from this
retrieval is missing an entire clause it depends on, and it will carry a
confident citation to the clause it *did* get. In a contract review or a
regulatory submission that is the expensive kind of wrong, and it is invisible
to every metric above `span_recall`.

The divergence runs both ways. Retrieve **only A**:

| metric | value |
|---|---|
| `span_coverage@5` | 196 / 726 = **0.270** |
| `span_recall@5` | **0.500** |

Same span recall, coverage less than half of what it was. Neither number can
stand in for the other. Both directions are asserted in
`tests/test_grounding.py`.

### On the threshold

50% is a convention, not a measurement. It is a module constant applied to
every configuration and every corpus, so it cannot be tuned to flatter a
result. A different threshold would move every grounding figure; it would not
change their ordering, because the same ranked lists are scored either way.

The threshold also means a span split across two chunks can still count as
grounded if the retrieved chunk holds enough of it — which is deliberate. The
question is whether an answer could be written, not whether the chunker was
tidy.

## Chunker statistics

Computed by `chunk_stats` before any retrieval runs. These bound what
retrieval can possibly achieve.

| stat | meaning |
|---|---|
| `spans_in_one_chunk` | Share of gold spans landing entirely inside a single chunk. A span sliced across three chunks cannot be returned whole by any ranking. |
| `relevant_per_query` | Mean number of chunks that count as relevant. |
| `precision_ceiling` | `mean(min(relevant, k)) / k` — the best `precision@k` this chunker permits. |
| `truncated` | Share of chunks the embedding model silently cuts off at its context limit. |

### The precision ceiling

A query with one relevant chunk can never score above `1/k` on `precision@k`.
Because that count is a function of chunk size, the ceiling differs per
strategy, and raw `precision@k` cannot rank strategies against each other.

Measured on the shipped corpus:

| strategy | relevant/query | ceiling on precision@5 | attained | share of ceiling |
|---|---|---|---|---|
| naive | 2.70 | 0.504 | 0.276 | 54.8% |
| clause | 1.58 | 0.312 | 0.208 | 66.7% |

Clause-aware chunking scores *lower* on raw precision and is the more precise
strategy. That inversion is why `span_coverage` and `complete_grounding` carry
the argument and precision is reported as a diagnostic.

`truncated` matters for the same reason: it is 1.8% for naive and 30.6% for
clause, so a short model context quietly penalises variable-length chunks —
the very strategy under test.

## Failure labels

Assigned by `src/taxonomy.py` to every query where `complete_grounding@5` is
`False`. Passing queries get no label. Each failing query gets **exactly one**,
first match wins.

The ordering is the design. Earlier faults cause later ones: a clause sliced
across two chunks will *also* look like a ranking problem, and fixing the
ranking will not fix it.

| # | label | fires when | points at |
|---|---|---|---|
| 1 | `ceiling-bound` | Fewer relevant chunks than k exist, and all of them are already in the top k | nothing — metric artifact |
| 2 | `chunk-severance` | A gold span that failed is split across more than one chunk | chunking |
| 3 | `retrieval-miss` | A chunk that would have covered a failed span never appeared in `ranked[:30]` | hybrid retrieval, embeddings |
| 4 | `rank-miss` | That chunk was in `ranked[:30]` but below k | reranking |
| 5 | `partial-grounding` | Something came back, not enough of it, and none of the above applies | metadata filtering, k tuning |

### Scoping: labels answer for the spans that failed

Labels 2–4 are judged against the gold spans that were actually under-covered,
not against every span in the query.

A clause that was split across chunks but retrieved whole did not cause this
failure. Blaming it would send the client to fix chunking when the real fault
was ranking — and on the naive strategy, where only 43% of spans survive in one
chunk, the unscoped rule would have swallowed nearly every failure into
`chunk-severance`.

### Two labels that never fire here

`ceiling-bound` and `partial-grounding` recorded zero on both shipped chunkers.
That is a property of the chunkers, not an accident.

Both `naive` and `clause` tile their document contiguously with no gaps. So if
every relevant chunk is retrieved, every gold character is retrieved, and the
query is not failing — which is exactly the precondition both labels require.
They are reachable only for a chunker that leaves text out of the index
(sliding windows with gaps, a filtered corpus).

They are kept, unit-tested against a deliberately gapped chunker, and their
zero counts reported rather than hidden. A non-zero count on a future chunker
means it is dropping text, which is worth knowing before tuning anything else.

### Evidence

Every label carries enough to verify it by hand without rerunning: the
uncovered spans with their coverage fractions and opening text, plus
label-specific detail — the chunk ids and ranks of a severed span's pieces, the
chunk ids missing from the candidate pool, the rank positions below k.

`results/taxonomy.json` holds all of it. Because chunking is pure Python, the
labels can be re-derived from that file and `data/eval_set.json` with no
embedding model, which is what `tests/test_taxonomy.py` does.

## Measured deltas in the fix backlog

A label says which lever the evidence *points at*. It does not establish that
changing that lever fixes that query — only a run with the lever changed does
that.

`src/report.py` computes which lever a delta belongs to by diffing the two
`Config` objects, never from a lookup table:

- different `strategy` → chunking
- different `retriever` → hybrid retrieval
- different `rerank` → reranking
- different `doc_filter` / `metadata_filter` → metadata filtering

A comparison counts only when the two configs differ in exactly one lever, so a
config that changes two things at once cannot lend its delta to either. The
direction is always *away from the baseline's current setting*, so a negative
delta means the change made things worse — never that the comparison was run
backwards. Where no pair of configurations isolated a lever, the entry reads
**not measured** rather than carrying an estimate.
