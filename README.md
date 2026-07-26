# rag-eval-harness

Built from scratch on public documents (CUAD, CC BY 4.0) using published techniques. Contains no code or data from any employer.

A retrieval evaluation harness that measures whether **clause-aware chunking** beats **fixed-width chunking** on legal contracts, scored against lawyer-annotated ground truth with character-level offsets.

The point of the repo is not the number that comes out. It is that the number is falsifiable: every chunk carries `(start, end)` offsets back into its source document, so "did we retrieve the answer?" is decided by span arithmetic rather than by string matching or vibes.

## Quickstart

```bash
pip install -r requirements.txt
python data/prepare.py --download     # fetches CUAD (18.3 MB), writes data/eval_set.json
python run_eval.py                    # writes results/report.md
```

No API keys. Both models (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) run locally on CPU, and Qdrant runs in `:memory:` mode — the real client API, no Docker. First run downloads ~200 MB of model weights; the eval itself takes a few minutes.

`data/eval_set.json` is committed, so `python run_eval.py` works on a fresh clone without the download step.

## Results

20 contracts, 50 queries, 21 clause types, 114 gold spans. Full output in [`results/report.md`](results/report.md).

| config | chunks | median chars | precision@5 | recall@10 | MRR | span coverage@5 |
|---|---|---|---|---|---|---|
| naive | 1103 | 1000 | 0.276 | 0.751 | 0.607 | 0.705 |
| clause | 972 | 797 | **0.208** | 0.800 | **0.641** | **0.735** |
| clause + rerank | 972 | 797 | 0.204 | **0.827** | 0.617 | 0.709 |

Clause-aware chunking wins on recall@10, MRR, and span coverage, and **loses on precision@5** — which is the most informative cell in the table. See Limitations.

The reranker is a split decision: it lifts recall@10 from 0.800 to 0.827 by promoting relevant chunks out of ranks 11–30, but it slightly *hurts* MRR (0.641 → 0.617) and span coverage. `ms-marco-MiniLM` is trained on web passage ranking, and legal clause language is far outside that distribution. Reported as measured rather than quietly dropped.

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

**Retrieval scope.** Queries are scoped to the contract they are asked about, via a Qdrant payload filter on `doc_id`. CUAD's questions are templated per clause type and worded identically for all 510 contracts — "Which state/country's law governs the interpretation of the contract?" describes the Governing Law clause of every contract equally well. A corpus-wide search on such a query is unanswerable by construction, and scoring it produces numbers that measure nothing. (Measured: corpus-wide, MRR collapses to 0.16.)

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

## Limitations

**Precision@k is not comparable across chunking strategies.** The number of gold-relevant chunks depends on how the document was split, so a strategy producing larger chunks can inflate precision without retrieving one extra character of the answer. That is not hypothetical here — it is exactly what the table shows. Naive chunking makes 2.70 chunks relevant per query against clause-aware's 1.58, which caps precision@5 at 0.504 versus 0.312. Measured against its own ceiling, clause-aware attains **66.7%** and naive **54.8%**: the strategy that looks worse on the raw metric is the more precise one. Span coverage is reported alongside as a chunking-invariant measure, since it counts gold characters rather than chunks. Recall@k inherits the same flaw in its denominator, and should be read the same way.

**This harness evaluates retrieval only.** Generation-side quality — faithfulness, answer relevance, hallucination — is a separate layer and is not measured here.

**The sample is small.** 20 contracts and 50 queries, with 1–4 queries per clause type. No confidence intervals are computed, and differences of a few points in the aggregate table are not statistically meaningful. The per-clause-type table in the report is texture, not evidence. The 43% vs 99.1% span-integrity gap is large enough to be real; the 0.607 vs 0.641 MRR gap is not, on its own.

**Relevance is binary and generous.** A chunk containing one character of a gold span counts the same as one containing the entire clause. Span coverage exists to compensate, and is the metric to trust when the two disagree.

**Retrieval is scoped to the correct document**, so this measures clause localization *within* a contract. It does not measure document routing across a corpus, which a production system would also need.

**Bigger models were tried and didn't win.** `BAAI/bge-base-en-v1.5` and `BAAI/bge-large-en-v1.5` (up to 15x the parameters of `all-MiniLM-L6-v2`) were swapped in as the embedder; on the clause config, bge-large edged out MiniLM (MRR 0.648 vs 0.641, recall@10 0.840 vs 0.800) but not by more than 50 queries can distinguish from noise, and it made naive chunking *worse*. `cross-encoder/ms-marco-MiniLM-L-12-v2` and `BAAI/bge-reranker-base` were swapped in as the reranker; `bge-reranker-base` was worse across every metric, and L12 traded a small precision/coverage gain for a worse MRR (0.580 vs 0.617) at 2.6x the size. The committed defaults are the smallest models tried, not merely the first ones tried.

**RAGAS is not wired in.** The intent was `IDBasedContextRecall` as a key-free cross-check, but `ragas==0.4.3` fails on import against current `langchain-community` (`ModuleNotFoundError: langchain_community.chat_models.vertexai`). Rather than pin a stale dependency tree or ship an unverified code path, it is left out. The metric it provides would duplicate `recall@k` in `src/metrics.py`, which is computed here directly from chunk IDs.

## Layout

```
data/prepare.py     CUAD -> eval_set.json (deterministic, validated, seeded)
src/chunking.py     naive vs clause-aware, both carrying character offsets
src/index.py        sentence-transformers embeddings + Qdrant in-memory
src/retrieve.py     dense retrieval + optional cross-encoder rerank
src/metrics.py      precision@k, recall@k, MRR, span coverage
src/evaluate.py     run loop + report generation
run_eval.py         entrypoint
results/report.md   generated
```

`data/prepare.py` verifies the CUAD checksum, asserts every gold span slices back to its own annotated text, and fails loudly rather than let a bad offset reach the metrics. Regenerating with the default seed reproduces the committed eval set exactly, apart from the `generated_utc` timestamp in `meta` — after a rerun, `git diff data/eval_set.json` shows that single line.

## Data

CUAD v1 (Contract Understanding Atticus Dataset), CC BY 4.0, © The Atticus Project, Inc. — <https://github.com/TheAtticusProject/cuad>

`CUADv1.json`, sha256 `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`. The 18.3 MB GitHub bundle is used rather than the 105.9 MB Zenodo release; the JSON is byte-identical in both (verified by CRC), and the extra 88 MB is PDF renderings of the same contracts.

Of CUAD's 41 clause types, 6 are excluded from the eval set: five are document metadata living in the header block (Document Name, Parties, Agreement Date, Effective Date, Expiration Date), where retrieval is header extraction and insensitive to body chunking; one (Competitive Restriction Exception) is phrased as a cross-reference rather than a standalone question. Pass `--include-metadata-clauses` to keep the metadata types. About 68% of CUAD's question/contract pairs are `is_impossible` with no gold span, and are filtered out.
