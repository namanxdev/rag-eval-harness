"""Entry point: python run_eval.py

Runs three configurations over the CUAD eval set and writes results/report.md.
Everything is local and CPU-only; no API key is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluate import format_table, run_all, write_report          # noqa: E402
from src.index import EMBED_MODEL, Encoder                            # noqa: E402
from src.retrieve import RERANK_MODEL, Reranker                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-set", type=Path, default=ROOT / "data" / "eval_set.json")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--rerank-model", default=RERANK_MODEL)
    p.add_argument("--query-prefix", default=None,
                   help="instruction prepended to queries only. Defaults to BGE's "
                        "recommended prefix for bge* models and none otherwise; "
                        "pass '' to disable.")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder stage")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.eval_set.exists():
        raise SystemExit(
            f"{args.eval_set} not found.\n"
            "Build it first:  python data/prepare.py --download"
        )

    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    verbose = not args.quiet
    counts = eval_set["meta"]["counts"]
    if verbose:
        print(f"eval set: {counts['documents']} documents, {counts['queries']} queries, "
              f"{counts['clause_types']} clause types")
        print(f"loading models ({args.embed_model}"
              f"{'' if args.no_rerank else ', ' + args.rerank_model})...")

    encoder = Encoder(args.embed_model, query_prefix=args.query_prefix)
    reranker = None if args.no_rerank else Reranker(args.rerank_model)

    results = run_all(eval_set, encoder, reranker, verbose=verbose)

    print()
    print(format_table(results))
    print()

    report = write_report(results, eval_set, args.results, encoder, reranker)
    print(f"wrote {report}")
    print(f"wrote {args.results / 'per_query.json'}")


if __name__ == "__main__":
    main()
