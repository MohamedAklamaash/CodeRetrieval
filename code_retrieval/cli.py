import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from code_retrieval.config import load_config
from code_retrieval.pipeline import METHODS, QueryOutcome, Retriever


def explain(outcome: QueryOutcome, symbol_id: str) -> str:
    candidate = outcome.candidates[symbol_id]
    reasons = []
    if candidate.in_seed_pool:
        reasons.append(f"seed (hybrid relevance {candidate.relevance:.3f})")
    for path in candidate.paths[:3]:
        reasons.append(f"{path.edge} of {path.parent} (hop {path.hop}, seed {path.seed})")
    if len(candidate.paths) > 3:
        reasons.append(f"+{len(candidate.paths) - 3} more paths")
    return "; ".join(reasons)


def command_index(args):
    config = load_config(args.config)
    retriever = Retriever(args.repo, config)
    index = retriever.index
    summary = {
        "repo": str(index.root), "symbols": len(index.symbols), "files": len(index.by_path),
        "kinds": Counter(s.kind for s in index.symbols), "skipped_files": index.skipped_files,
        "unparsable_files": index.unparsable_files, "newly_embedded": retriever.newly_embedded,
        "timings": {k: round(v, 3) for k, v in retriever.timings.items()},
    }
    print(json.dumps(summary, indent=2))


def command_query(args):
    config = load_config(args.config)
    config.expansion.max_depth = args.depth if args.depth is not None else config.expansion.max_depth
    retriever = Retriever(args.repo, config)
    try:
        outcome = retriever.query(args.query, args.method)
    finally:
        retriever.close()
    if args.json:
        print(json.dumps({
            "results": [{**retriever.index.by_id[sid].to_record(), "score": score, "why": explain(outcome, sid)}
                        for sid, score in outcome.ranked[:args.top_k]],
            "expansion": outcome.expansion.to_record() if outcome.expansion else None,
            "timings": {**retriever.timings, **outcome.timings},
        }, indent=2))
        return
    for position, (sid, score) in enumerate(outcome.ranked[:args.top_k], start=1):
        symbol = retriever.index.by_id[sid]
        print(f"{position:>2}. {symbol.path}:{symbol.start_line}-{symbol.end_line}  {symbol.qualified_name} "
              f"[{symbol.kind}]  score={score:.4f}")
        if args.explain:
            print(f"    why: {explain(outcome, sid)}")
        if args.snippet:
            for line in symbol.source.splitlines()[:args.snippet]:
                print(f"    | {line}")
    if outcome.expansion:
        print(f"\nexpansion: {json.dumps(outcome.expansion.to_record())}")
    timings = {**retriever.timings, **outcome.timings}
    print("timings: " + ", ".join(f"{k}={v:.3f}s" for k, v in timings.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-retrieval")
    parser.add_argument("--config", type=Path, default=None, help="TOML file overriding defaults")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="index a repository and warm the embedding cache")
    index.add_argument("--repo", type=Path, required=True)
    index.set_defaults(run=command_index)

    query = commands.add_parser("query", help="retrieve symbols for a natural-language query")
    query.add_argument("--repo", type=Path, required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--method", choices=METHODS, default="lsp-ranked")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--depth", type=int, default=None)
    query.add_argument("--explain", action="store_true")
    query.add_argument("--snippet", type=int, default=0, metavar="LINES")
    query.add_argument("--json", action="store_true")
    query.set_defaults(run=command_query)

    from code_retrieval.evaluation.cli import register
    register(commands)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
