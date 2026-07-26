"""Convert CUAD v1 into a compact retrieval eval set.

CUAD ships as SQuAD-v2-style JSON: one paragraph per contract, where the
paragraph `context` is the entire contract text and every answer carries an
`answer_start` character offset into that text. Those offsets are what make
CUAD usable as retrieval ground truth without any labelling work of our own.

This script selects a small, deterministic, clause-type-diverse subset and
writes `data/eval_set.json`. Every other module in the harness depends on that
file's shape, so it is validated before being written.

Source : https://github.com/TheAtticusProject/cuad  (CUADv1.json, 40 MB)
License: CC BY 4.0, (c) The Atticus Project, Inc.

Usage:
    python data/prepare.py --download      # fetch CUADv1.json if missing, then convert
    python data/prepare.py                 # convert an already-downloaded CUADv1.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The 18.3 MB GitHub bundle. Its CUADv1.json is byte-identical to the one in
# the 105.9 MB Zenodo release (verified: crc32=1c88167f, 40,128,638 bytes);
# the extra 88 MB there is PDF/txt renderings of the same contracts.
CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_ZIP_MEMBER = "CUADv1.json"
CUAD_JSON_SHA256 = "ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae"

# CUAD encodes the clause type in the question id, after a double underscore:
#   "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Governing Law"
CLAUSE_TYPE_SEP = "__"

# Five of the 41 types are document metadata sitting in the header block, not
# clauses in the body: their gold spans are the title, the signature parties
# and the dates in the preamble. Retrieving them is header extraction, and it
# is insensitive to how the body is chunked -- which is precisely the variable
# under test. Excluded by default; re-enable with --include-metadata-clauses.
METADATA_CLAUSE_TYPES = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

# "Competitive Restriction Exception" phrases its question as a cross-reference
# ("the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of
# Customers above") rather than a self-contained question, so it does not work
# as a standalone retrieval query.
UNUSABLE_CLAUSE_TYPES = frozenset({"Competitive Restriction Exception"})


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_cuad(dest: Path) -> None:
    """Fetch data.zip and extract just CUADv1.json to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {CUAD_ZIP_URL} (18.3 MB) ...", file=sys.stderr)
    with urllib.request.urlopen(CUAD_ZIP_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(CUAD_ZIP_MEMBER) as src, dest.open("wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)", file=sys.stderr)


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def slugify(title: str, maxlen: int = 60) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:maxlen].rstrip("-") or "doc"


def clean_query(question: str) -> str:
    """Take the natural-language half of a CUAD question.

    Every CUAD question is templated as
        Highlight the parts (if any) of this contract related to "X" that
        should be reviewed by a lawyer. Details: <real question>
    The boilerplate half is identical across all 41 types and carries no
    retrieval signal, so only the Details half is used as the query.
    """
    _, _, details = question.partition("Details:")
    details = details or question
    details = details.replace(" ", " ")           # CUAD is full of nbsp
    return re.sub(r"\s+", " ", details).strip()


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent gold spans.

    264 of CUAD's 6,702 answered questions have annotations that overlap each
    other. Left unmerged they inflate the denominator of any character-level
    coverage metric (the same character counted twice), so they are merged
    here, once, rather than in each metric.
    """
    out: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------

def load_candidates(cuad: dict, allowed_types: frozenset[str]):
    """Flatten CUAD into per-document records of answerable questions."""
    docs = []
    for entry in cuad["data"]:
        paragraphs = entry["paragraphs"]
        if len(paragraphs) != 1:                       # true for all 510 in v1
            raise ValueError(f"{entry['title']}: expected 1 paragraph, got {len(paragraphs)}")
        text = paragraphs[0]["context"]
        questions = []
        for qa in paragraphs[0]["qas"]:
            if qa["is_impossible"] or not qa["answers"]:
                continue                                # ~68% of CUAD; no gold span to score
            clause_type = qa["id"].rsplit(CLAUSE_TYPE_SEP, 1)[-1]
            if clause_type not in allowed_types:
                continue
            spans = merge_spans([
                (a["answer_start"], a["answer_start"] + len(a["text"]))
                for a in qa["answers"]
            ])
            questions.append({
                "cuad_id": qa["id"],
                "clause_type": clause_type,
                "query": clean_query(qa["question"]),
                "cuad_question": qa["question"],
                "gold_spans": [[s, e] for s, e in spans],
            })
        docs.append({"title": entry["title"], "text": text, "questions": questions})
    return docs


def select(docs, *, n_docs, n_queries, min_chars, max_chars, min_q_per_doc,
           max_per_doc, max_per_clause, seed):
    """Pick documents and queries deterministically, capped for balance.

    Candidates are drawn uniformly at random rather than by clause rarity.
    Ordering by rarity does produce more distinct clause types, but it drops
    every common type -- Governing Law, Anti-Assignment, Termination For
    Convenience -- and those are exactly the clauses a contract-retrieval
    system is asked about. A uniform draw keeps the mix representative of
    CUAD and cannot be accused of selecting for the hypothesis; the per-doc
    and per-clause caps do the diversity work instead.
    """
    rng = random.Random(seed)

    pool = sorted(
        (d for d in docs
         if min_chars <= len(d["text"]) <= max_chars
         and len(d["questions"]) >= min_q_per_doc),
        key=lambda d: d["title"],
    )
    if len(pool) < n_docs:
        raise SystemExit(f"only {len(pool)} documents satisfy the filters; need {n_docs}")
    chosen = sorted(rng.sample(pool, n_docs), key=lambda d: d["title"])

    def shuffled(questions):
        qs = sorted(questions, key=lambda q: q["cuad_id"])   # stable base order
        random.Random(seed).shuffle(qs)
        return qs

    remaining = {d["title"]: shuffled(d["questions"]) for d in chosen}
    per_doc: dict[str, int] = {d["title"]: 0 for d in chosen}
    per_clause: dict[str, int] = {}

    # Round-robin across documents so no single contract dominates.
    picked = []
    order = [d["title"] for d in chosen]
    rng.shuffle(order)
    progressed = True
    while len(picked) < n_queries and progressed:
        progressed = False
        for title in order:
            if len(picked) >= n_queries:
                break
            if per_doc[title] >= max_per_doc:
                continue
            for i, q in enumerate(remaining[title]):
                if per_clause.get(q["clause_type"], 0) >= max_per_clause:
                    continue
                picked.append((title, remaining[title].pop(i)))
                per_doc[title] += 1
                per_clause[q["clause_type"]] = per_clause.get(q["clause_type"], 0) + 1
                progressed = True
                break

    if len(picked) < n_queries:
        print(f"warning: caps allowed only {len(picked)} of {n_queries} requested queries",
              file=sys.stderr)
    return chosen, picked


def build(chosen, picked, meta_extra):
    doc_ids, seen = {}, set()
    documents = []
    for d in chosen:
        doc_id = slugify(d["title"])
        suffix = 2
        while doc_id in seen:
            doc_id = f"{slugify(d['title'], 55)}-{suffix}"
            suffix += 1
        seen.add(doc_id)
        doc_ids[d["title"]] = doc_id
        documents.append({
            "doc_id": doc_id,
            "title": d["title"],
            "n_chars": len(d["text"]),
            "text": d["text"],
        })

    by_title = {d["title"]: d for d in chosen}
    queries = []
    for n, (title, q) in enumerate(sorted(picked, key=lambda p: (p[0], p[1]["cuad_id"])), 1):
        text = by_title[title]["text"]
        spans = [tuple(s) for s in q["gold_spans"]]
        queries.append({
            "query_id": f"q{n:03d}",
            "doc_id": doc_ids[title],
            "clause_type": q["clause_type"],
            "query": q["query"],
            "gold_spans": q["gold_spans"],
            "gold_texts": [text[s:e] for s, e in spans],
            "n_gold_chars": sum(e - s for s, e in spans),
            "cuad_id": q["cuad_id"],
            "cuad_question": q["cuad_question"],
        })

    return {
        "meta": meta_extra | {
            "counts": {
                "documents": len(documents),
                "queries": len(queries),
                "clause_types": len({q["clause_type"] for q in queries}),
                "gold_spans": sum(len(q["gold_spans"]) for q in queries),
            },
        },
        "documents": documents,
        "queries": queries,
    }


def validate(eval_set: dict) -> None:
    """Fail loudly rather than let a bad offset reach the metrics."""
    docs = {d["doc_id"]: d for d in eval_set["documents"]}
    if len(docs) != len(eval_set["documents"]):
        raise AssertionError("duplicate doc_id")

    seen_qids = set()
    for q in eval_set["queries"]:
        if q["query_id"] in seen_qids:
            raise AssertionError(f"duplicate query_id {q['query_id']}")
        seen_qids.add(q["query_id"])

        doc = docs.get(q["doc_id"])
        if doc is None:
            raise AssertionError(f"{q['query_id']}: unknown doc_id {q['doc_id']}")
        if not q["gold_spans"]:
            raise AssertionError(f"{q['query_id']}: no gold spans")

        prev_end = -1
        for (start, end), gold in zip(q["gold_spans"], q["gold_texts"], strict=True):
            if not 0 <= start < end <= len(doc["text"]):
                raise AssertionError(f"{q['query_id']}: span {start},{end} out of bounds")
            if start <= prev_end:
                raise AssertionError(f"{q['query_id']}: spans overlap or are unsorted")
            if doc["text"][start:end] != gold:
                raise AssertionError(f"{q['query_id']}: text at {start},{end} != gold_texts")
            prev_end = end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cuad", type=Path, default=ROOT / "data" / "CUADv1.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "eval_set.json")
    p.add_argument("--download", action="store_true", help="fetch CUADv1.json if it is missing")
    p.add_argument("--docs", type=int, default=20)
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=15_000)
    p.add_argument("--max-chars", type=int, default=80_000)
    p.add_argument("--min-q-per-doc", type=int, default=5)
    p.add_argument("--max-per-doc", type=int, default=3)
    p.add_argument("--max-per-clause", type=int, default=4)
    p.add_argument("--include-metadata-clauses", action="store_true",
                   help="also sample Document Name / Parties / dates (header extraction)")
    args = p.parse_args()

    if args.download and not args.cuad.exists():
        download_cuad(args.cuad)
    if not args.cuad.exists():
        raise SystemExit(f"{args.cuad} not found; re-run with --download")

    digest = sha256_file(args.cuad)
    if digest != CUAD_JSON_SHA256:
        print(f"warning: {args.cuad.name} sha256 {digest} != expected {CUAD_JSON_SHA256}",
              file=sys.stderr)

    cuad = json.loads(args.cuad.read_text(encoding="utf-8"))

    excluded = set(UNUSABLE_CLAUSE_TYPES)
    if not args.include_metadata_clauses:
        excluded |= set(METADATA_CLAUSE_TYPES)
    all_types = {qa["id"].rsplit(CLAUSE_TYPE_SEP, 1)[-1]
                 for e in cuad["data"] for qa in e["paragraphs"][0]["qas"]}
    allowed = frozenset(all_types - excluded)

    docs = load_candidates(cuad, allowed)
    chosen, picked = select(
        docs, n_docs=args.docs, n_queries=args.queries,
        min_chars=args.min_chars, max_chars=args.max_chars,
        min_q_per_doc=args.min_q_per_doc, max_per_doc=args.max_per_doc,
        max_per_clause=args.max_per_clause, seed=args.seed,
    )

    eval_set = build(chosen, picked, {
        "name": "cuad-retrieval-eval",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "data/prepare.py",
        "source": {
            "dataset": "CUAD v1 (Contract Understanding Atticus Dataset)",
            "internal_version": cuad.get("version"),
            "url": CUAD_ZIP_URL,
            "file": args.cuad.name,
            "sha256": digest,
            "size_bytes": args.cuad.stat().st_size,
            "license": "CC BY 4.0",
            "attribution": "The Atticus Project, Inc.",
            "corpus_totals": {"documents": len(cuad["data"])},
        },
        "sampling": {
            "seed": args.seed,
            "doc_char_range": [args.min_chars, args.max_chars],
            "min_answered_questions_per_doc": args.min_q_per_doc,
            "max_queries_per_doc": args.max_per_doc,
            "max_queries_per_clause_type": args.max_per_clause,
            "excluded_clause_types": sorted(excluded),
            "notes": [
                "Only questions with is_impossible=false are eligible; ~68% of CUAD "
                "question/contract pairs are unanswerable and carry no gold span.",
                "Overlapping gold spans are merged so character-level coverage "
                "metrics cannot count the same character twice.",
                "Queries are the 'Details:' half of the CUAD question; the templated "
                "'Highlight the parts...' prefix is identical across all types.",
                "Questions are drawn uniformly at random from the answerable ones, "
                "not ranked by clause rarity, so the mix stays representative of CUAD; "
                "the per-doc and per-clause caps supply the diversity.",
            ],
        },
    })

    validate(eval_set)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(eval_set, indent=2, ensure_ascii=False), encoding="utf-8")

    c = eval_set["meta"]["counts"]
    print(f"wrote {args.out} -- {c['documents']} documents, {c['queries']} queries, "
          f"{c['clause_types']} clause types, {c['gold_spans']} gold spans "
          f"({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
