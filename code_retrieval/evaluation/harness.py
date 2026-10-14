import json
import platform
import re
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from thefuzz import fuzz

from code_retrieval.config import Config
from code_retrieval.evaluation.gold import derive_gold
from code_retrieval.evaluation.swebench import Instance, checkout
from code_retrieval.indexing import RepositoryIndex
from code_retrieval.pipeline import Retriever, load_provider

CODE_IDENTIFIER = re.compile(r"`([A-Za-z_][\w.]*)`|\b([A-Za-z_]\w*_\w+|[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*)\b|\b(\w+)\(")
TRACKED_PACKAGES = ("bm25s", "sentence-transformers", "torch", "basedpyright", "tree-sitter", "tree-sitter-python", "numpy")


def name_match_ranking(query: str, index: RepositoryIndex, depth: int) -> list[str]:
    """Model-free stand-in for the legacy engine: fuzzy-match code-like identifiers in the query to symbol names."""
    targets = {t.split(".")[-1].lower() for groups in CODE_IDENTIFIER.findall(query) for t in groups if t}
    if not targets:
        return []
    scored = []
    for symbol in index.symbols:
        name = symbol.name.lower()
        score = max(fuzz.ratio(t, name) for t in targets)
        if score >= 70:
            scored.append((score + (20 if name in targets else 0), symbol.id))
    return [sid for _, sid in sorted(scored, key=lambda item: (-item[0], item[1]))[:depth]]


def run_manifest(config: Config, dataset_id: str, split_file: Path, revision: str) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha or None, "git_dirty": dirty,
        "dataset": {"id": dataset_id, "revision": revision, "file": str(split_file)},
        "config": config.to_dict(),
        "packages": {p: metadata.version(p) for p in TRACKED_PACKAGES},
        "python": platform.python_version(), "machine": platform.platform(),
    }


def evaluate_instance(instance: Instance, retriever: Retriever, depth: int) -> dict:
    index = retriever.index
    gold = derive_gold(instance.patch, index)
    query = instance.problem_statement
    bm25 = retriever.query(query, "bm25")
    dense = retriever.query(query, "dense")
    started = time.perf_counter()
    name_match = name_match_ranking(query, index, depth)
    name_match_seconds = time.perf_counter() - started
    pool = retriever.build_pool(query, expand=True)
    expansion = pool.expansion
    return {
        "instance_id": instance.instance_id, "repo": instance.repo, "base_commit": instance.base_commit,
        "status": "ok",
        "gold": gold.to_record(),
        "index": {"symbols": len(index.symbols), "files": len(index.by_path),
                  "skipped_files": len(index.skipped_files), "unparsable_files": len(index.unparsable_files),
                  "newly_embedded": retriever.newly_embedded},
        "timings": {**retriever.timings, **pool.timings,
                    "bm25_seconds": bm25.timings["total_seconds"], "dense_seconds": dense.timings["total_seconds"],
                    "name_match_seconds": name_match_seconds},
        "rankings": {"bm25": [sid for sid, _ in bm25.ranked], "dense": [sid for sid, _ in dense.ranked],
                     "name_match": name_match},
        "pool": {
            "seed_pool": pool.seed_pool, "relevance": pool.relevance,
            "lexical_rank": pool.lexical_rank, "dense_rank": pool.dense_rank,
            "expansion": {**expansion.to_record(), "request_log": expansion.request_log,
                          "paths": {sid: [asdict(p) for p in paths] for sid, paths in expansion.paths.items()}},
        },
    }


def completed_ids(results_file: Path) -> set[str]:
    if not results_file.exists():
        return set()
    return {json.loads(line)["instance_id"] for line in results_file.read_text().splitlines() if line.strip()}


def run(instances: list[Instance], config: Config, out_dir: Path, repos_dir: Path, worktrees_dir: Path,
        manifest: dict, retry_failures: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "per_query.jsonl"
    manifest_file = out_dir / "manifest.json"
    if not manifest_file.exists():
        manifest_file.write_text(json.dumps({**manifest, "instances": [i.instance_id for i in instances]}, indent=2))
    if retry_failures and results_file.exists():
        kept = [l for l in results_file.read_text().splitlines() if json.loads(l)["status"] == "ok"]
        results_file.write_text("".join(f"{l}\n" for l in kept))
    done = completed_ids(results_file)
    provider = load_provider(config) if config.dense.enabled else None
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    for position, instance in enumerate(instances, start=1):
        if instance.instance_id in done:
            continue
        started = time.perf_counter()
        retriever = None
        try:
            with checkout(repos_dir, worktrees_dir, instance) as worktree:
                retriever = Retriever(worktree, config, provider, lsp_log=out_dir / "lsp.log")
                record = evaluate_instance(instance, retriever, config.seeding.pool_size)
        except Exception as error:
            # One broken instance must not abort a multi-hour run; the failure is recorded and counted as a miss.
            record = {"instance_id": instance.instance_id, "repo": instance.repo, "status": "error",
                      "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        finally:
            if retriever is not None:
                retriever.close()
        record["wall_seconds"] = time.perf_counter() - started
        with results_file.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        expansion = record.get("pool", {}).get("expansion", {})
        print(f"[{position}/{len(instances)}] {instance.instance_id} {record['status']} "
              f"{record['wall_seconds']:.1f}s requests={expansion.get('requests')} "
              f"discovered={expansion.get('discovered')} gold={record.get('gold', {}).get('files')}", flush=True)
