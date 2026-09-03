# Documentation

Reference material for the harness. The [top-level README](../README.md) is the
overview and the results; these are the details that would clutter it.

| document | read it when |
|---|---|
| [`v2.md`](v2.md) | You want to know what v2 added, why each piece exists, and what it changed for code that already used the harness. |
| [`architecture.md`](architecture.md) | You are changing the harness and need the data flow, the module contracts, and the invariants you must not break. |
| [`metrics.md`](metrics.md) | You need the exact definition of a metric or a failure label, with worked examples. This is the document to hand someone who disputes a number. |

## Conventions used throughout

**A span** is a `(start, end)` character offset pair into one document's text,
end-exclusive. Offsets are only meaningful relative to the document they came
from, which is why every relevance check tests `doc_id` first.

**Gold spans** are the human-annotated answer locations. They are never
inferred by this repo — CUAD's come from lawyers, a client's come from the
client.

**A chunk is relevant** when it comes from the query's document *and* overlaps
at least one gold span. There is no string matching anywhere in the scoring
path, and no model judges relevance.

**k** is a retrieval depth. Different metrics use different depths on purpose
(`precision@5`, `recall@10`), and every configuration ranks to the same
`RANK_DEPTH` first so the only difference between configs is chunking and
ordering.
