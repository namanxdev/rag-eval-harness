# Retrieval evaluation: chunking, retrieval, reranking and filtering

Generated 2026-09-03 13:14 UTC by `run_eval.py`.

## Setup

- **Corpus**: 20 contracts from CUAD v1 (Contract Understanding Atticus Dataset), 50 queries across 21 clause types, 114 lawyer-annotated gold spans.
- **Embedding**: `all-MiniLM-L6-v2` (384-dim), Qdrant in-memory, cosine.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Retrievers**: dense (embeddings), BM25 (lexical), and the two fused with Reciprocal Rank Fusion at k=60. RRF rather than a weighted score sum, because cosine and BM25 scores are on unrelated scales and normalising them introduces a weight with no principled value.
- **Ranking depth**: 30 chunks per query, retrieved from within the contract the question is asked of (a Qdrant payload filter on `doc_id`). CUAD's questions are templated per clause type and worded identically for every contract, so a corpus-wide search is unanswerable by construction.

## Results

| config                   | chunks | median chars | precision@5 | recall@10 |     mrr | span_coverage@5 | span_recall@5 | complete_grounding@5 |
|--------------------------|--------|--------------|-------------|-----------|---------|-----------------|---------------|----------------------|
| naive                    |   1103 |         1000 |       0.276 |     0.751 |   0.607 |           0.705 |         0.701 |                0.580 |
| clause                   |    972 |          797 |       0.208 |     0.800 |   0.641 |           0.735 |         0.723 |                0.660 |
| clause + rerank          |    972 |          797 |       0.204 |     0.827 |   0.617 |           0.709 |         0.695 |                0.560 |
| clause + bm25            |    972 |          797 |       0.196 |     0.820 |   0.616 |           0.659 |         0.662 |                0.580 |
| clause + hybrid          |    972 |          797 |       0.240 |     0.857 |   0.700 |           0.823 |         0.818 |                0.700 |
| clause + hybrid + rerank |    972 |          797 |       0.208 |     0.827 |   0.605 |           0.717 |         0.705 |                0.580 |
| clause, unfiltered       |    972 |          797 |       0.064 |     0.203 |   0.166 |           0.213 |         0.207 |                0.140 |

`span_coverage` is the fraction of gold characters present in the top 5 chunks. It is the only metric here that is comparable across chunking strategies -- see Limitations in the README.

`span_recall@5` is the fraction of *distinct* gold spans that got at least 50% of their characters retrieved, and `complete_grounding@5` is the rate at which *every* gold span cleared that bar. They exist because pooled character coverage cannot tell one fully-retrieved clause from two half-retrieved ones. A query whose answer depends on two clauses, one returned whole and one missed entirely, scores 0.5 on coverage and reads as partial success; it is a wrong answer carrying a confident citation. Complete grounding is the rate at which a fully-cited answer was possible at all. The 50% threshold is fixed, not fitted per corpus.

Per-query detail, including the text of every clause a failing query missed, is in `per_query.json` under `missed_spans`.

## What chunking alone decides

Both columns below are fixed before a single query runs.

| strategy | chunks | median chars | gold spans intact in one chunk | truncated at 256 tokens | relevant chunks per query | ceiling on precision@5 |
|---|---|---|---|---|---|---|
| naive | 1103 | 1000 | 43.0% | 1.8% | 2.70 | 0.504 |
| clause | 972 | 797 | 99.1% | 30.6% | 1.58 | 0.312 |

The last column is why precision@5 must not be read as a ranking of the strategies. Larger chunks overlap more gold spans, so more chunks count as relevant, so precision@5 can climb without a single extra gold character being retrieved. As a share of its own ceiling, each config attained:

| config | precision@5 | ceiling | attained |
|---|---|---|---|
| naive | 0.276 | 0.504 | 54.8% |
| clause | 0.208 | 0.312 | 66.7% |
| clause + rerank | 0.204 | 0.312 | 65.4% |
| clause + bm25 | 0.196 | 0.312 | 62.8% |
| clause + hybrid | 0.240 | 0.312 | 76.9% |
| clause + hybrid + rerank | 0.208 | 0.312 | 66.7% |
| clause, unfiltered | 0.064 | 0.312 | 20.5% |

## Mean MRR by clause type

Only 1-4 queries per clause type, so read these as texture rather than evidence; the aggregate table above is what carries weight.

| clause type | n | naive | clause | clause + rerank | clause + bm25 | clause + hybrid | clause + hybrid + rerank | clause, unfiltered |
|---|---|---|---|---|---|---|---|---|
| Audit Rights | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| Cap On Liability | 4 | 0.542 | 0.875 | 0.646 | 0.453 | 0.750 | 0.646 | 0.044 |
| Change Of Control | 1 | 0.111 | 1.000 | 0.111 | 0.167 | 0.500 | 0.111 | 0.000 |
| Exclusivity | 3 | 0.390 | 0.122 | 0.463 | 0.151 | 0.144 | 0.463 | 0.028 |
| Governing Law | 4 | 0.307 | 0.487 | 0.875 | 0.812 | 0.833 | 0.875 | 0.010 |
| Insurance | 3 | 1.000 | 0.833 | 0.778 | 0.381 | 0.833 | 0.750 | 0.147 |
| Ip Ownership Assignment | 3 | 0.833 | 0.516 | 0.460 | 0.500 | 0.667 | 0.444 | 0.035 |
| Irrevocable Or Perpetual License | 2 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.375 |
| Joint Ip Ownership | 2 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 | 0.667 | 0.500 |
| License Grant | 2 | 0.129 | 0.229 | 0.625 | 1.000 | 0.667 | 0.625 | 0.000 |
| Liquidated Damages | 1 | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.000 |
| Minimum Commitment | 3 | 0.400 | 0.500 | 0.208 | 0.417 | 0.667 | 0.208 | 0.197 |
| Most Favored Nation | 1 | 1.000 | 0.333 | 0.100 | 0.071 | 0.500 | 0.111 | 0.000 |
| No-Solicit Of Employees | 3 | 0.667 | 0.714 | 0.611 | 0.556 | 0.778 | 0.611 | 0.357 |
| Non-Compete | 4 | 0.417 | 0.315 | 0.403 | 0.550 | 0.436 | 0.278 | 0.000 |
| Notice Period To Terminate Renewal | 2 | 1.000 | 0.750 | 0.750 | 0.750 | 1.000 | 0.750 | 0.054 |
| Post-Termination Services | 2 | 0.417 | 0.208 | 0.375 | 1.000 | 0.417 | 0.375 | 0.000 |
| Renewal Term | 2 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| Rofr/Rofo/Rofn | 3 | 0.524 | 0.519 | 0.514 | 0.683 | 0.500 | 0.516 | 0.333 |
| Termination For Convenience | 2 | 0.667 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.126 |
| Uncapped Liability | 2 | 1.000 | 1.000 | 0.750 | 0.375 | 0.500 | 0.750 | 0.350 |

## Provenance

- Source: CUAD v1 (Contract Understanding Atticus Dataset), CC BY 4.0, (c) The Atticus Project, Inc..
- `CUADv1.json` sha256 `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`.
- Eval set built by `data/prepare.py` with seed 42; rerunning it selects the same documents and queries.
