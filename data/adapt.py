"""Turn a directory of client documents into the harness's eval set schema.

`data/prepare.py` builds the CUAD reference corpus. This builds the same shape
from whatever the client has, so a sprint runs on their contracts rather than
on ours. The schema is the boundary between "our benchmark" and "their data";
nothing downstream needs to know which one it is looking at.

    python data/adapt.py \\
      --docs ./client-docs \\
      --queries ./client-queries.jsonl \\
      --out ./data/client_eval_set.json

    python run_eval.py --eval-set ./data/client_eval_set.json

A document's `doc_id` is its filename without the extension, used verbatim --
`Acme MSA 2024.pdf` becomes `Acme MSA 2024`. That is the id the queries file
must refer to. It is not slugified, because a rule the operator has to
reverse-engineer to write a query file is a rule that will be got wrong.

Query file: one JSON object per line.

    {"query": "What is the governing law?", "doc_id": "Acme MSA 2024",
     "gold_texts": ["This Agreement shall be governed by the laws of Delaware."],
     "clause_type": "Governing Law"}

Give `gold_texts` (located in the document) or `gold_spans` (character offsets,
used as given). Optional: `query_id`, `clause_type`, `filters`.

Extraction is recorded, not assumed. Gold span offsets are only valid against
the exact text that produced them, so `meta.source` carries the extractor and
its version and the sha256 of every extracted document. Re-extracting with a
different library silently invalidates every judgment the client signed off,
and the hash is what makes that loud instead of silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# prepare.py is a sibling script, not a package. The schema contract lives in
# its `validate`, and there must be exactly one of those.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import merge_spans, validate            # noqa: E402

# Contractual scope of the Retrieval Quality Baseline Sprint. Enforced here
# rather than trusted to the operator: a run that quietly exceeds them produces
# numbers for work that was not sold.
MAX_DOCUMENTS = 250
MAX_TOTAL_BYTES = 2 * 1024**3          # 2 GB

TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = sorted(TEXT_SUFFIXES | PDF_SUFFIXES)

EXIT_CAP_EXCEEDED = 2


class AdapterError(Exception):
    """Something in the client's input is wrong and guessing would be worse."""


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def extract(path: Path) -> tuple[str, str]:
    """Return (text, extractor) for one document.

    Text is returned exactly as the extractor produced it. No whitespace
    normalisation, no unicode folding, no newline translation -- every one of
    those shifts character offsets, and offsets are the ground truth here.
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        # Bytes, not open(): text mode would translate CRLF to LF on some
        # platforms and not others, moving every offset in the file.
        return path.read_bytes().decode("utf-8"), "utf-8-passthrough"
    if suffix in PDF_SUFFIXES:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, f"pypdf {pypdf.__version__}"
    raise AdapterError(
        f"{path.name}: unsupported file type {suffix or '(none)'}. "
        f"Supported: {', '.join(SUPPORTED)}. Convert it or remove it from "
        "--docs; skipping it silently would leave queries pointing at a "
        "document that is not in the eval set."
    )


def collect(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        raise AdapterError(f"--docs {docs_dir} is not a directory")
    paths = sorted(p for p in docs_dir.rglob("*") if p.is_file())
    if not paths:
        raise AdapterError(f"--docs {docs_dir} contains no files")
    return paths


def check_caps(paths: list[Path]) -> int:
    """Enforce the sprint's scope caps. Returns the total source byte count."""
    total = sum(p.stat().st_size for p in paths)
    if len(paths) > MAX_DOCUMENTS or total > MAX_TOTAL_BYTES:
        print(
            f"scope cap exceeded: {len(paths)} documents, {total:,} bytes "
            f"({total / 1024**3:.2f} GB).\n"
            f"The sprint covers up to {MAX_DOCUMENTS} documents and "
            f"{MAX_TOTAL_BYTES / 1024**3:.0f} GB. Reduce --docs or agree a "
            "wider scope before running.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_CAP_EXCEEDED)
    return total


def load_documents(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Extract every document. Returns (documents, extraction records)."""
    documents, records, seen = [], [], {}
    for path in paths:
        doc_id = path.stem
        if doc_id in seen:
            raise AdapterError(
                f"duplicate doc_id {doc_id!r}: {seen[doc_id]} and {path}. "
                "doc_id is the filename without its extension, so two files "
                "with the same stem cannot both be indexed."
            )
        seen[doc_id] = path

        text, extractor = extract(path)
        if not text.strip():
            raise AdapterError(
                f"{path.name}: extracted no text with {extractor}. A scanned "
                "PDF needs OCR before it can be used as retrieval ground truth."
            )

        documents.append({"doc_id": doc_id, "title": path.name,
                          "n_chars": len(text), "text": text})
        records.append({
            "doc_id": doc_id,
            "file": path.name,
            "extractor": extractor,
            "source_bytes": path.stat().st_size,
            "n_chars": len(text),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return documents, records


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def locate(text: str, gold: str, doc_id: str) -> tuple[int, int]:
    """Find `gold` in `text`, refusing to guess.

    Zero matches means the judgment does not belong to this document, or the
    text was extracted differently from when it was written. Several matches
    means the harness would have to pick one, and picking wrong silently
    corrupts the ground truth every later number rests on.
    """
    count = text.count(gold)
    if count == 0:
        raise AdapterError(
            f"{doc_id}: gold text not found. Either it belongs to another "
            f"document or the text was extracted differently.\n  {gold[:200]!r}"
        )
    if count > 1:
        raise AdapterError(
            f"{doc_id}: gold text matches {count} times; the harness will not "
            f"choose between them. Extend the quote until it is unique, or "
            f"give gold_spans offsets instead.\n  {gold[:200]!r}"
        )
    start = text.index(gold)
    return start, start + len(gold)


def load_queries(path: Path, texts: dict[str, str]) -> list[dict]:
    if not path.is_file():
        raise AdapterError(f"--queries {path} is not a file")

    queries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path}:{lineno}: not valid JSON -- {exc}") from exc

        where = f"{path}:{lineno}"
        for required in ("query", "doc_id"):
            if not raw.get(required):
                raise AdapterError(f"{where}: missing {required!r}")
        doc_id = raw["doc_id"]
        if doc_id not in texts:
            raise AdapterError(
                f"{where}: doc_id {doc_id!r} is not among the documents in --docs. "
                f"Known: {', '.join(sorted(texts)[:5])}"
                f"{' ...' if len(texts) > 5 else ''}"
            )

        text = texts[doc_id]
        has_spans, has_texts = "gold_spans" in raw, "gold_texts" in raw
        if has_spans == has_texts:
            raise AdapterError(
                f"{where}: give exactly one of gold_spans or gold_texts, not "
                f"{'both' if has_spans else 'neither'}."
            )

        if has_spans:
            spans = [tuple(s) for s in raw["gold_spans"]]
            for start, end in spans:
                if not 0 <= start < end <= len(text):
                    raise AdapterError(
                        f"{where}: span ({start}, {end}) is outside {doc_id}, "
                        f"which is {len(text)} characters. Offsets must be "
                        "against the text this adapter extracts."
                    )
        else:
            spans = [locate(text, g, doc_id) for g in raw["gold_texts"]]
        spans = merge_spans(spans)

        query = {
            "query_id": raw.get("query_id") or f"q{len(queries) + 1:03d}",
            "doc_id": doc_id,
            "clause_type": raw.get("clause_type", "unspecified"),
            "query": raw["query"],
            "gold_spans": [list(s) for s in spans],
            "gold_texts": [text[s:e] for s, e in spans],
            "n_gold_chars": sum(e - s for s, e in spans),
        }
        if raw.get("filters"):
            query["filters"] = raw["filters"]
        queries.append(query)

    if not queries:
        raise AdapterError(f"{path} contains no queries")
    return queries


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build(documents: list[dict], queries: list[dict], records: list[dict],
          docs_dir: Path, queries_path: Path, total_bytes: int,
          name: str) -> dict:
    extractors = sorted({r["extractor"] for r in records})
    return {
        "meta": {
            "name": name,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "data/adapt.py",
            "source": {
                "dataset": f"client corpus at {docs_dir}",
                "queries_file": str(queries_path),
                "license": "client-owned; not redistributable",
                "attribution": "client",
                "extractors": extractors,
                "extraction_note":
                    "Gold span offsets are valid only against text produced by "
                    "these extractors. Re-extracting with a different library "
                    "or version invalidates every judgment; compare "
                    "text_sha256 before reusing an eval set.",
                "size_bytes": total_bytes,
                "documents": records,
            },
            "sampling": {
                "seed": None,
                "notes": ["Documents and queries supplied by the client; "
                          "nothing was sampled or excluded by the harness."],
            },
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


def adapt(docs_dir: Path, queries_path: Path, name: str = "client-eval",
          metadata: dict[str, dict] | None = None) -> dict:
    """The whole pipeline, without touching argv. Used by the tests."""
    paths = collect(docs_dir)
    total_bytes = check_caps(paths)
    documents, records = load_documents(paths)

    for doc in documents:
        for key, value in (metadata or {}).get(doc["doc_id"], {}).items():
            doc[key] = value

    queries = load_queries(queries_path, {d["doc_id"]: d["text"] for d in documents})
    eval_set = build(documents, queries, records, docs_dir, queries_path,
                     total_bytes, name)
    validate(eval_set)         # the same contract data/prepare.py is held to
    return eval_set


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", type=Path, required=True,
                   help="directory of client documents (%s)" % ", ".join(SUPPORTED))
    p.add_argument("--queries", type=Path, required=True,
                   help="JSONL file of queries and relevance judgments")
    p.add_argument("--out", type=Path, required=True, help="eval set to write")
    p.add_argument("--metadata", type=Path,
                   help="optional JSON mapping doc_id -> {field: value}. Indexed "
                        "alongside every chunk and filterable from a query's "
                        "`filters` field; this is what makes metadata filtering "
                        "measurable as a lever rather than assumed to help.")
    p.add_argument("--name", default="client-eval", help="name recorded in meta")
    args = p.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None
    try:
        eval_set = adapt(args.docs, args.queries, args.name, metadata)
    except AdapterError as exc:
        raise SystemExit(f"error: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(eval_set, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    c = eval_set["meta"]["counts"]
    print(f"wrote {args.out}")
    print(f"  {c['documents']} documents, {c['queries']} queries, "
          f"{c['gold_spans']} gold spans, {c['clause_types']} clause types")
    print(f"  extractors: {', '.join(eval_set['meta']['source']['extractors'])}")
    print(f"\nnext:  python run_eval.py --eval-set {args.out}")


if __name__ == "__main__":
    main()
