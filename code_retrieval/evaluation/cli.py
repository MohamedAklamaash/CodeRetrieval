import random
from collections import defaultdict
from pathlib import Path

from code_retrieval.config import load_config
from code_retrieval.evaluation import harness
import json

from code_retrieval.evaluation.swebench import DATASET_IDS, Instance, download_split, load_instances, source_files_changed

DATA_DIR = Path("data")
REVISIONS_FILE = Path("configs/swebench_revisions.json")


def stratified_sample(instances: list[Instance], size: int, seed: int) -> list[Instance]:
    """Proportional per-repo allocation (at least one per repo), deterministic for a given seed."""
    if size >= len(instances):
        return instances
    by_repo = defaultdict(list)
    for instance in instances:
        by_repo[instance.repo].append(instance)
    rng = random.Random(seed)
    quotas = {repo: max(1, round(size * len(group) / len(instances))) for repo, group in by_repo.items()}
    while sum(quotas.values()) > size:
        largest = max(quotas, key=lambda repo: (quotas[repo], repo))
        quotas[largest] -= 1
    chosen = []
    for repo in sorted(by_repo):
        group = sorted(by_repo[repo], key=lambda i: i.instance_id)
        chosen.extend(rng.sample(group, quotas[repo]))
    return sorted(chosen, key=lambda i: (i.repo, i.instance_id))


def command_evaluate(args):
    config = load_config(args.config)
    split_file = DATA_DIR / "swebench" / f"{args.dataset}_{args.split}.jsonl"
    revision = json.loads(REVISIONS_FILE.read_text())[args.dataset]
    if not split_file.exists():
        download_split(args.dataset, args.split, revision, split_file)
    instances = load_instances(split_file)
    if args.min_files > 1:
        instances = [i for i in instances if len(source_files_changed(i.patch)) >= args.min_files]
    if args.sample:
        instances = stratified_sample(instances, args.sample, args.seed)
    # Seeded random processing order, so a time-boxed partial run is a random subset of the selection
    # rather than its alphabetically-first repositories.
    random.Random(args.seed).shuffle(instances)
    if args.limit:
        instances = instances[:args.limit]
    manifest = harness.run_manifest(config, DATASET_IDS[args.dataset], split_file, revision)
    manifest["selection"] = {"dataset": args.dataset, "split": args.split, "min_files": args.min_files,
                             "sample": args.sample, "seed": args.seed, "limit": args.limit}
    harness.run(instances, config, args.out, DATA_DIR / "repos", DATA_DIR / "worktrees", manifest,
                retry_failures=args.retry_failures)


def command_report(args):
    from code_retrieval.evaluation.report import write_report
    write_report(args.runs, args.out, load_config(args.config))


def register(commands):
    evaluate = commands.add_parser("evaluate", help="run the SWE-bench Lite retrieval benchmark (resumable)")
    evaluate.add_argument("--dataset", choices=sorted(DATASET_IDS), default="lite")
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--min-files", type=int, default=1, help="keep instances whose patch edits at least N source files")
    evaluate.add_argument("--sample", type=int, default=0, help="stratified-by-repo subset size")
    evaluate.add_argument("--seed", type=int, default=13)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--retry-failures", action="store_true")
    evaluate.set_defaults(run=command_evaluate)

    report = commands.add_parser("report", help="compute metrics, ablations, statistics and plots from runs")
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(run=command_report)
