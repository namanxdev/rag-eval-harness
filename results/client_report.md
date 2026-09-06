# Retrieval quality baseline

Generated 2026-09-03 13:14 UTC from a run of this repository on `cuad-retrieval-eval`. Every figure below was produced by that run; nothing is carried over from another engagement and nothing is estimated.

Baseline configuration: **naive** (fixed 1000-char windows, 150 overlap).

## 1. Where retrieval stands today

| metric | value | what it means |
|---|---|---|
| `precision@5` | 0.276 | of the top 5 chunks, the share that hit a gold clause |
| `recall@10` | 0.751 | of every chunk holding a gold clause, the share in the top 10 |
| `mrr` | 0.607 | 1 / the rank of the first correct chunk, averaged |
| `span_coverage@5` | 0.705 | of all gold clause characters, the share retrieved |
| `span_recall@5` | 0.701 | of gold clauses, the share at least 50% retrieved |
| `complete_grounding@5` | 0.580 | share of questions where EVERY gold clause was retrieved |

**58% of questions (29 of 50) could be answered with every clause they depend on actually retrieved.** The remaining 21 can produce an answer, and it will carry a citation, but the citation will be incomplete. That is the number the rest of this report is about; the metrics above it are diagnostics, not outcomes.

A clause counts as retrieved when at least 50% of its characters appear in the top 5 chunks. The threshold is fixed across every run and every client.

## 2. Top failure patterns

Each of the 21 failing questions carries exactly one label, assigned by rule from chunk offsets and rank positions -- no model judged them, and every label can be re-derived from `per_query.json` and `taxonomy.json`.

### 1. `chunk-severance` -- 15 of 21 failures (71%)

Points at: **chunking**.

- **q001** (Exclusivity, `ambassadoreyeweargroupinc-11-17-1997-ex-10-28-endorsement-ag`) -- 0% of its clauses retrieved. Missing:
  > The license hereby granted shall be exclusive as to the products described in subparagraphs 2.(a)(1) and (2) of this Agr...
  > Nothing in this Agreement shall be construed to prevent KI, Inc. from granting any other licenses for the use of KI's na...

- **q006** (Exclusivity, `americasshoppingmallinc-12-10-1999-ex-10-2-site-development`) -- 0% of its clauses retrieved. Missing:
  > HDI shall have the exclusive right to use of the "Deerskin" brand for a self-contained web site for the offering of Deer...

### 2. `rank-miss` -- 6 of 21 failures (29%)

Points at: **reranking**.

- **q014** (Rofr/Rofo/Rofn, `bicycletherapeuticsplc-03-10-2020-ex-10-11-service-agreement`) -- 50% of its clauses retrieved. Missing:
  > On or as soon as practicable following the Effective Date, it is intended that you will be granted an option under the O...

- **q025** (Cap On Liability, `deltathreeinc-19991102-s-1a-ex-10-19-6227850-ex-10-19-co-bra`) -- 75% of its clauses retrieved. Missing:
  > THE LIABILITY OF DELTATHREE FOR DAMAGES OR ALLEGED DAMAGES HEREUNDER, WHETHER IN CONTRACT, TORT OR ANY OTHER LEGAL THEOR...

## 3. Fix backlog

Ranked by how many failing questions each lever accounts for. `grounding delta` is the measured change in section 1's headline number when that lever alone was changed, always in the direction of moving away from the baseline's current setting -- so a negative number means the change made things worse. Where the comparison names two configurations, the lever was isolated between those two rather than against the baseline. Levers no pair of configurations isolated are marked *not measured* rather than guessed at.

| # | lever | failures it addresses | measured by | grounding delta |
|---|---|---|---|---|
| 1 | chunking | 15 | `clause` | +8.0% |
| 2 | reranking | 6 | `clause -> clause + rerank` | -10.0% |

### What every configuration scored

| config | precision@5 | recall@10 | mrr | span_coverage@5 | span_recall@5 | complete_grounding@5 |
|---|---|---|---|---|---|---|
| **naive** | 0.276 | 0.751 | 0.607 | 0.705 | 0.701 | 0.580 |
| clause | 0.208 | 0.800 | 0.641 | 0.735 | 0.723 | 0.660 |
| clause + rerank | 0.204 | 0.827 | 0.617 | 0.709 | 0.695 | 0.560 |
| clause + bm25 | 0.196 | 0.820 | 0.616 | 0.659 | 0.662 | 0.580 |
| clause + hybrid | 0.240 | 0.857 | 0.700 | 0.823 | 0.818 | 0.700 |
| clause + hybrid + rerank | 0.208 | 0.827 | 0.605 | 0.717 | 0.705 | 0.580 |
| clause, unfiltered | 0.064 | 0.203 | 0.166 | 0.213 | 0.207 | 0.140 |

The highest measured grounding rate in this run was **clause + hybrid** at 0.700, +12.0% against the baseline. It changes chunking and hybrid retrieval relative to the baseline, so the gain cannot be attributed to a single lever from this run alone.

## 4. Scope

- **Corpus**: 20 documents, 50 queries across 21 clause types, 114 gold spans. Small enough that per-clause-type figures are texture, not evidence.
- **Source**: CUAD v1 (Contract Understanding Atticus Dataset).
- **Retrieval depth**: every configuration ranked 30 chunks per query and was scored on the top 5.
- **Relevance**: character overlap with a gold span, never string matching. A chunk counts only if it comes from the right document and overlaps a span a human marked.
- **Not covered**: answer generation, faithfulness of generated text, latency, and cost. This measures what retrieval puts in front of the model, not what the model does with it.
- Only questions with is_impossible=false are eligible; ~68% of CUAD question/contract pairs are unanswerable and carry no gold span.
- Overlapping gold spans are merged so character-level coverage metrics cannot count the same character twice.
- Queries are the 'Details:' half of the CUAD question; the templated 'Highlight the parts...' prefix is identical across all types.
- Questions are drawn uniformly at random from the answerable ones, not ranked by clause rarity, so the mix stays representative of CUAD; the per-doc and per-clause caps supply the diversity.
