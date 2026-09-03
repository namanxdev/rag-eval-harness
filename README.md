# rag-eval-harness

Built from scratch on public documents (CUAD, CC BY 4.0) using published techniques. Contains no code or data from any employer.

A retrieval evaluation harness for RAG over legal contracts. It measures four levers — **chunking**, **hybrid retrieval**, **reranking**, and **metadata filtering** — against lawyer-annotated ground truth with character-level offsets, then labels every failing query with the lever that caused it.

The point of the repo is not the number that comes out. It is that the number is falsifiable: every chunk carries `(start, end)` offsets back into its source document, so "did we retrieve the answer?" is decided by span arithmetic rather than by string matching or vibes. Nothing here uses an LLM as a judge.

## Quickstart

```bash
pip install -r requirements.txt
python data/prepare.py --download     # fetches CUAD (18.3 MB), writes data/eval_set.json
python run_eval.py                    # writes results/report.md + results/client_report.md
pytest                                # or: make test
```

To run it against your own corpus instead of CUAD:

```bash
python data/adapt.py --docs ./my-docs --queries ./my-queries.jsonl --out data/mine.json
python run_eval.py --eval-set data/mine.json
```

No API keys. Both models (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) run locally on CPU, and Qdrant runs in `:memory:` mode — the real client API, no Docker. First run downloads ~200 MB of model weights; the seven-configuration eval then takes under ten minutes on a laptop CPU. The test suite loads no models and finishes in seconds.

`data/eval_set.json` is committed, so `python run_eval.py` works on a fresh clone without the download step.

## Results

**20 contracts, 50 queries, 21 clause types, 114 gold spans.** That is a small corpus, and the Limitations section below should be read next to this table rather than after it. Full output in [`results/report.md`](results/report.md); the client-facing readout is [`results/client_report.md`](results/client_report.md).

| config | chunks | precision@5 | recall@10 | MRR | span coverage@5 | span recall@5 | **complete grounding@5** |
|---|---|---|---|---|---|---|---|
| naive | 1103 | 0.276 | 0.751 | 0.607 | 0.705 | 0.701 | 0.580 |
| clause | 972 | 0.208 | 0.800 | 0.641 | 0.735 | 0.723 | 0.660 |
| clause + rerank | 972 | 0.204 | 0.827 | 0.617 | 0.709 | 0.695 | 0.560 |
| clause + bm25 | 972 | 0.196 | 0.820 | 0.616 | 0.659 | 0.662 | 0.580 |
| **clause + hybrid** | 972 | **0.240** | **0.857** | **0.700** | **0.823** | **0.818** | **0.700** |
| clause + hybrid + rerank | 972 | 0.208 | 0.827 | 0.605 | 0.717 | 0.705 | 0.580 |
| clause, unfiltered | 972 | 0.064 | 0.203 | 0.166 | 0.213 | 0.207 | 0.140 |

**Complete grounding is the column that matters.** It is the share of questions where *every* gold clause the answer depends on was retrieved — the binary of whether a correct, fully-cited answer was possible at all. The columns to its left are diagnostics for why it is what it is.

It is deliberately harsher than span coverage, and the gap between the two is the reason it exists. Pooled character coverage cannot distinguish one fully-retrieved clause from two half-retrieved ones. On `clause`, coverage reads 0.735 while complete grounding is 0.660: those points of apparent success are questions answerable only from part of what they needed. In contract review that is a wrong answer carrying a confident citation, and every metric to the left of it scores that as partial credit.

**Hybrid retrieval is the largest win available.** Fusing dense and BM25 with RRF beats dense alone on every column (grounding 0.660 → 0.700, coverage 0.735 → 0.823). Neither retriever wins on its own: BM25 alone scores 0.580 against dense's 0.660, and has the lowest span coverage of any clause-based config. Fusing two retrievers that each lose separately is the whole argument for fusion rather than choosing between them.

The mechanism is reordering, not candidate generation, and `per_query.json` records which retriever surfaced each relevant chunk so this is checkable rather than asserted. Fusion promoted **11 relevant chunks into the top 5 across 9 queries** that dense retrieval alone had ranked below 5; BM25 contributed no chunk that dense missed entirely. That is expected here and would not hold on a larger corpus: retrieval is scoped to one contract, and most contracts have few enough chunks that a 30-deep search already returns nearly all of them. Net effect on grounding, per query: 4 questions fixed, 2 broken.

**The reranker loses, and is reported anyway.** `clause + rerank` lifts recall@10 (0.800 → 0.827) by promoting relevant chunks out of ranks 11–30, but drops complete grounding from 0.660 to 0.560. `ms-marco-MiniLM` is trained on web passage ranking, and legal clause language is far outside that distribution. Stacked on hybrid retrieval it gives back most of hybrid's gain too (0.700 → 0.580).

**Document scoping is worth more than every other lever combined.** `clause, unfiltered` drops the `doc_id` payload filter and grounding collapses from 0.660 to 0.140. **26 of 50 queries are grounded only because the filter is there**, and without it 89.6% of retrieved top-5 chunks come from the wrong contract entirely.

That row is a diagnostic, not a candidate configuration. CUAD's questions are templated identically across all 510 contracts, so an unscoped search is unanswerable by construction and the collapse is partly an artifact of the corpus. It is included because it is the only way to put a measured number on what metadata filtering buys, and because the alternative — inventing effective dates and entity names for CUAD so the lever has something to filter on — would produce a more impressive table and no evidence. The filtering machinery is general (any document field, on both the dense and BM25 side); CUAD simply has nothing else to filter on.

### Why each query failed

Every failing query is given exactly one label, assigned by rule from chunk offsets and rank positions — no model judges them. On the `naive` baseline, 21 of 50 queries fail:

| label | count | points at |
|---|---|---|
| `chunk-severance` | 15 | chunking |
| `rank-miss` | 6 | reranking |
| `retrieval-miss` | 0 | hybrid retrieval, embeddings |
| `ceiling-bound` | 0 | nothing — metric artifact |
| `partial-grounding` | 0 | metadata filtering, `k` |

Each label carries the evidence for itself, so any of them can be re-derived by hand from [`results/taxonomy.json`](results/taxonomy.json) and `results/per_query.json`. The dominance of `chunk-severance` is the retrieval-side consequence of the 43% span-integrity figure below: fixed-width chunking cuts clauses in half, and no ranking function can put them back together.

### What chunking decides before retrieval runs

| strategy | gold spans intact in one chunk | relevant chunks per query | ceiling on precision@5 |
|---|---|---|---|
| naive | 43.0% | 2.70 | 0.504 |
| clause | **99.1%** | 1.58 | 0.312 |

Fixed-width chunking severs **57% of gold spans** across chunk boundaries. No ranking function can undo that: a clause split across three chunks cannot be returned whole. This is the mechanism behind the retrieval numbers, and it is measurable without running a retriever at all.

## How it works

**Ground truth.** CUAD ships 510 commercial contracts with 13,000+ clause spans annotated by lawyers, as SQuAD-style JSON where every answer carries an `answer_start` offset into the contract text. `data/prepare.py` converts a deterministic sample into `{query, doc_id, gold_spans}` rows. Ground truth is expert-labeled by someone else — not asserted by this repo.

**Relevance.** A retrieved chunk counts as relevant when it comes from the right document and overlaps any gold span:

```python
chunk.doc_id == doc_id and chunk.start < gold_end and gold_start < chunk.end
```

**Retrieval scope.** Queries are scoped to the contract they are asked about, via a Qdrant payload filter on `doc_id`. CUAD's questions are templated per clause type and worded identically for all 510 contracts — "Which state/country's law governs the interpretation of the contract?" describes the Governing Law clause of every contract equally well. A corpus-wide search on such a query is unanswerable by construction, and scoring it produces numbers that measure nothing. That claim is measured rather than asserted: the `clause, unfiltered` row is the same config with the filter removed, and MRR collapses from 0.641 to 0.166.

The payload is not limited to `doc_id`. Any field a document declares — effective date, entity, document type, version — is indexed alongside every chunk and filterable from a query's `filters` field, on both the dense and the BM25 side. CUAD declares none, so nothing is fabricated for it; the mechanism is there for corpora that do.

**Grounding.** `span_coverage` pools gold characters across all of a query's spans, so a question needing two clauses scores 0.5 when one is returned whole and the other missed entirely. `complete_grounding@k` is the binary that pooling hides: every gold span must have at least 50% of its characters retrieved. The threshold is a fixed constant, not fitted per corpus. Where a query fails it, `results/per_query.json` records the offsets and text of each clause that was missing.

**Failure taxonomy.** Every failing query is labelled by rule, first match wins: `ceiling-bound` (a metric artifact, not a retrieval fault), `chunk-severance` (a needed clause split across chunks), `retrieval-miss` (its chunk never entered the top 30), `rank-miss` (its chunk was in the pool but below k), `partial-grounding` (something came back, not enough of it). The conditions are evaluated against the spans that actually failed, so a clause that was split but retrieved whole is not blamed for a failure it did not cause. Each label carries the chunk ids, offsets and rank positions that produced it.

Two of the five labels never fire on this corpus, and that is a property of the chunkers rather than an accident: both `naive` and `clause` tile their document contiguously with no gaps, so if every relevant chunk is retrieved then every gold character is retrieved, and the query cannot be failing. `ceiling-bound` and `partial-grounding` are reachable only for a chunker that drops text. They are kept, and their zero counts reported, because a non-zero count on a future chunker is a signal worth having.

**The clause regex is the engineering.** The obvious pattern —

```python
re.compile(r"^(?:ARTICLE\s+[IVXLC]+|Section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*\s+[A-Z])", re.M)
```

— finds zero headers in 7 of these 20 contracts. Three failures, each found by reading the corpus rather than reasoning about it:

1. **Headers are indented** (`"            1. Recitals."`). `^` with no `[ \t]*` misses all of them.
2. **The number is followed by a period** (`"1. Recitals."`), so `\d+\s+[A-Z]` never fires. Some are spaced oddly: `"3 .SCHEDULED AND UNSCHEDULED DOWN TIME."`
3. **Some headers are mid-line.** Where the PDF-to-text extraction collapsed line breaks, sections run inline: `"...prior to its use with the VIP System.   2. VIP'S EQUIPMENT AND SERVICES."` A `^`-anchored pattern cannot see these at all.

And one false-positive trap: inner whitespace must be `[ \t]`, never `\s`. Contracts carry page-number footers (`"4\n\n\n\n\n\nIf to any of the Sellers:"`), and `\s+` bridges the blank lines — turning every page break into a spurious section boundary. The naive pattern's "hits" on one contract were *all* page footers.

The tuned pattern finds 1001 headers across the 20 contracts, leaving 2 with no recoverable structure at all (their extraction delaminated section numbers from their titles). Those two fall back to sentence-boundary size splitting, which is the honest outcome — not a silent one.

## Running it on your own corpus

`data/adapt.py` converts a directory of documents plus a query file into the same schema `data/prepare.py` produces, so nothing downstream needs to know which corpus it is looking at. `.txt` and `.md` are read directly; `.pdf` goes through `pypdf`. Anything else is rejected with a message rather than skipped, because a silently skipped document leaves queries pointing at something that is not in the index.

A document's `doc_id` is its filename without the extension, used verbatim. The query file is JSONL, one object per line:

```jsonl
{"query": "What law governs?", "doc_id": "acme-msa", "clause_type": "Governing Law",
 "gold_texts": ["This Agreement is governed by the laws of Delaware."]}
{"query": "When does it start?", "doc_id": "acme-msa", "gold_spans": [[27, 71]]}
```

Give `gold_texts` and the adapter locates each one; give `gold_spans` and it uses the offsets as supplied. A `gold_texts` entry that matches zero times, or more than once, is an error naming the document, the text and the match count — it will not pick one for you, because guessing there corrupts the ground truth every later number rests on.

Two things are enforced rather than trusted. The scope caps (250 documents, 2 GB) exit non-zero with the actual counts. And extraction is recorded: `meta.source` carries the extractor, its version, and a sha256 of every extracted document, because gold span offsets are only valid against the exact text that produced them and re-extracting with a different library invalidates every judgment silently otherwise.

## What a run produces

`python run_eval.py` writes four files. Two are for whoever is changing the harness, two are the engagement deliverable.

| file | for | contents |
|---|---|---|
| `results/report.md` | developer | every config, the chunking table, attained-vs-ceiling precision, mean MRR by clause type, provenance |
| `results/client_report.md` | client | baseline → failure patterns → fix backlog → scope |
| `results/per_query.json` | audit | per-query metrics, the full 30-deep ranking, `missed_spans`, and which retriever surfaced each relevant chunk |
| `results/taxonomy.json` | audit | one label per failing query with the evidence that produced it |

The client report is four sections and nothing else: where retrieval stands today, the top failure patterns with two real queries and the text of the clauses they missed, a fix backlog ranked by how many failing questions each lever accounts for, and a scope statement of what was and was not covered.

Its one governing rule is that **no number appears in it that was not produced by a run in this repo.** The backlog decides which lever a delta belongs to by diffing the two configs, so a change cannot be attributed to a lever that was not actually moved, and a config that changes two things at once is excluded from both. Where no pair of configs isolated a lever, the entry reads *not measured* rather than carrying an estimate. `--baseline <config>` picks which config stands in for the client's current setup; it defaults to the first one run.

## Tests

```bash
pytest          # 87 tests, ~25s, no model weights loaded
```

A repo whose entire pitch is measurement rigour had no tests. These are the ones that would catch a wrong number rather than a crash:

- **The metric divergence itself** — that pooled coverage and complete grounding disagree on `q001` in both directions, since a metric that never diverges from the one it replaces is not worth adding.
- **Adapter round trip** — CUAD's own 20 documents and 50 queries pushed back through `data/adapt.py` come out as an equivalent eval set, asserted field by field.
- **Taxonomy partition** — labels re-derived from the committed `results/per_query.json` must cover the failing set exactly, once each. Chunking is pure Python, so this runs without an embedding model and a client can reproduce it from two committed files.
- **Label ordering** — a query satisfying two conditions takes the earlier one, and a split-but-retrieved clause is not blamed for a failure it did not cause.
- **The dense path is unchanged** — `retrieve()` with no sparse index issues the identical single search it did before hybrid retrieval existed.
- **Backlog attribution** — a config that moves two levers cannot lend its delta to either, and a lever that made things worse reports a negative number rather than being dropped.

## Limitations

**Precision@k is not comparable across chunking strategies.** The number of gold-relevant chunks depends on how the document was split, so a strategy producing larger chunks can inflate precision without retrieving one extra character of the answer. That is not hypothetical here — it is exactly what the table shows. Naive chunking makes 2.70 chunks relevant per query against clause-aware's 1.58, which caps precision@5 at 0.504 versus 0.312. Measured against its own ceiling, clause-aware attains **66.7%** and naive **54.8%**: the strategy that looks worse on the raw metric is the more precise one. Span coverage is reported alongside as a chunking-invariant measure, since it counts gold characters rather than chunks. Recall@k inherits the same flaw in its denominator, and should be read the same way.

**This harness evaluates retrieval only.** Generation-side quality — faithfulness, answer relevance, hallucination — is a separate layer and is not measured here.

**The sample is small.** 20 contracts and 50 queries, with 1–4 queries per clause type. No confidence intervals are computed, and differences of a few points in the aggregate table are not statistically meaningful. The per-clause-type table in the report is texture, not evidence. The 43% vs 99.1% span-integrity gap is large enough to be real; the 0.607 vs 0.641 MRR gap is not, on its own.

**Relevance is binary and generous.** A chunk containing one character of a gold span counts the same as one containing the entire clause. Span coverage exists to compensate, and complete grounding is stricter still; when the three disagree, trust them in that order.

**The grounding threshold is a choice, not a measurement.** A gold span counts as retrieved at 50% of its characters. That number is fixed across every config and every corpus so it cannot be tuned to flatter a result, but it is still a convention. A different threshold would move every grounding figure in the table; it would not change their ordering, since the same ranked lists are scored either way.

**The failure taxonomy attributes, it does not prove.** A label says which lever the evidence points at, from chunk offsets and rank positions. It does not establish that changing that lever will fix that query — only a run with the lever changed does that, which is what the fix backlog's measured deltas are for. Where no configuration isolated a lever, the backlog says "not measured" rather than estimating.

**Retrieval is scoped to the correct document**, so this measures clause localization *within* a contract. It does not measure document routing across a corpus, which a production system would also need.

**Bigger models were tried and didn't win.** `BAAI/bge-base-en-v1.5` and `BAAI/bge-large-en-v1.5` (up to 15x the parameters of `all-MiniLM-L6-v2`) were swapped in as the embedder; on the clause config, bge-large edged out MiniLM (MRR 0.648 vs 0.641, recall@10 0.840 vs 0.800) but not by more than 50 queries can distinguish from noise, and it made naive chunking *worse*. `cross-encoder/ms-marco-MiniLM-L-12-v2` and `BAAI/bge-reranker-base` were swapped in as the reranker; `bge-reranker-base` was worse across every metric, and L12 traded a small precision/coverage gain for a worse MRR (0.580 vs 0.617) at 2.6x the size. The committed defaults are the smallest models tried, not merely the first ones tried.

**RAGAS is not wired in.** The intent was `IDBasedContextRecall` as a key-free cross-check, but `ragas==0.4.3` fails on import against current `langchain-community` (`ModuleNotFoundError: langchain_community.chat_models.vertexai`). Rather than pin a stale dependency tree or ship an unverified code path, it is left out. The metric it provides would duplicate `recall@k` in `src/metrics.py`, which is computed here directly from chunk IDs.

## Layout

```
data/prepare.py            CUAD -> eval_set.json (deterministic, validated, seeded)
data/adapt.py              a client's own documents -> the same schema
src/chunking.py            naive vs clause-aware, both carrying character offsets
src/index.py               embeddings + Qdrant in-memory, filterable metadata payload
src/sparse.py              BM25 + reciprocal rank fusion
src/retrieve.py            dense / BM25 / hybrid, plus optional cross-encoder rerank
src/metrics.py             precision@k, recall@k, MRR, span coverage, grounding
src/taxonomy.py            why each failing query failed, by rule
src/evaluate.py            run loop + developer report
src/report.py              client readout: baseline, patterns, fix backlog, scope
run_eval.py                entrypoint
tests/                     pytest; no model downloads, runs in seconds
results/report.md          generated: every config, every diagnostic
results/client_report.md   generated: the four-section readout
results/taxonomy.json      generated: one label + evidence per failing query
```

`data/prepare.py` verifies the CUAD checksum, asserts every gold span slices back to its own annotated text, and fails loudly rather than let a bad offset reach the metrics. Regenerating with the default seed reproduces the committed eval set exactly, apart from the `generated_utc` timestamp in `meta` — after a rerun, `git diff data/eval_set.json` shows that single line.

## Data

CUAD v1 (Contract Understanding Atticus Dataset), CC BY 4.0, © The Atticus Project, Inc. — <https://github.com/TheAtticusProject/cuad>

`CUADv1.json`, sha256 `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`. The 18.3 MB GitHub bundle is used rather than the 105.9 MB Zenodo release; the JSON is byte-identical in both (verified by CRC), and the extra 88 MB is PDF renderings of the same contracts.

Of CUAD's 41 clause types, 6 are excluded from the eval set: five are document metadata living in the header block (Document Name, Parties, Agreement Date, Effective Date, Expiration Date), where retrieval is header extraction and insensitive to body chunking; one (Competitive Restriction Exception) is phrased as a cross-reference rather than a standalone question. Pass `--include-metadata-clauses` to keep the metadata types. About 68% of CUAD's question/contract pairs are `is_impossible` with no gold span, and are filtered out.
