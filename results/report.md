# Retrieval evaluation: naive vs clause-aware chunking

Generated 2026-07-25 11:53 UTC by `run_eval.py`.

## Setup

- **Corpus**: 20 contracts from CUAD v1 (Contract Understanding Atticus Dataset), 50 queries across 21 clause types, 114 lawyer-annotated gold spans.
- **Embedding**: `all-MiniLM-L6-v2` (384-dim), Qdrant in-memory, cosine.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Ranking depth**: 30 chunks per query, retrieved from within the contract the question is asked of (a Qdrant payload filter on `doc_id`). CUAD's questions are templated per clause type and worded identically for every contract, so a corpus-wide search is unanswerable by construction.

## Results

| config           | chunks | median chars |        precision@5 |          recall@10 |                mrr |    span_coverage@5 |
|------------------|--------|--------------|--------------------|--------------------|--------------------|--------------------|
| naive            |   1103 |         1000 |              0.276 |              0.751 |              0.607 |              0.705 |
| clause           |    972 |          797 |              0.208 |              0.800 |              0.641 |              0.735 |
| clause + rerank  |    972 |          797 |              0.204 |              0.827 |              0.617 |              0.709 |

`span_coverage` is the fraction of gold characters present in the top 5 chunks. It is the only metric here that is comparable across chunking strategies -- see Limitations in the README.

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

## Mean MRR by clause type

Only 1-4 queries per clause type, so read these as texture rather than evidence; the aggregate table above is what carries weight.

| clause type | n | naive | clause | clause + rerank |
|---|---|---|---|---|
| Audit Rights | 1 | 1.000 | 1.000 | 1.000 |
| Cap On Liability | 4 | 0.542 | 0.875 | 0.646 |
| Change Of Control | 1 | 0.111 | 1.000 | 0.111 |
| Exclusivity | 3 | 0.390 | 0.122 | 0.463 |
| Governing Law | 4 | 0.307 | 0.487 | 0.875 |
| Insurance | 3 | 1.000 | 0.833 | 0.778 |
| Ip Ownership Assignment | 3 | 0.833 | 0.516 | 0.460 |
| Irrevocable Or Perpetual License | 2 | 0.667 | 1.000 | 1.000 |
| Joint Ip Ownership | 2 | 1.000 | 1.000 | 0.667 |
| License Grant | 2 | 0.129 | 0.229 | 0.625 |
| Liquidated Damages | 1 | 0.500 | 1.000 | 0.500 |
| Minimum Commitment | 3 | 0.400 | 0.500 | 0.208 |
| Most Favored Nation | 1 | 1.000 | 0.333 | 0.100 |
| No-Solicit Of Employees | 3 | 0.667 | 0.714 | 0.611 |
| Non-Compete | 4 | 0.417 | 0.315 | 0.403 |
| Notice Period To Terminate Renewal | 2 | 1.000 | 0.750 | 0.750 |
| Post-Termination Services | 2 | 0.417 | 0.208 | 0.375 |
| Renewal Term | 2 | 0.750 | 1.000 | 1.000 |
| Rofr/Rofo/Rofn | 3 | 0.524 | 0.519 | 0.514 |
| Termination For Convenience | 2 | 0.667 | 1.000 | 1.000 |
| Uncapped Liability | 2 | 1.000 | 1.000 | 0.750 |

## Provenance

- Source: CUAD v1 (Contract Understanding Atticus Dataset), CC BY 4.0, (c) The Atticus Project, Inc..
- `CUADv1.json` sha256 `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`.
- Eval set built by `data/prepare.py` with seed 42; rerunning it selects the same documents and queries.
