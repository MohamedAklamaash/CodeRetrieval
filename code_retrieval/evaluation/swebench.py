import json
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from code_retrieval.evaluation.gold import changed_pre_image_lines
from code_retrieval.indexing import is_test_path

DATASET_IDS = {"lite": "princeton-nlp/SWE-bench_Lite", "verified": "princeton-nlp/SWE-bench_Verified"}
REQUIRED_FIELDS = ("instance_id", "repo", "base_commit", "patch", "problem_statement")


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    problem_statement: str


def load_instances(path: Path) -> list[Instance]:
    instances = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        record = json.loads(line)
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise ValueError(f"{path}:{line_number} is missing fields {missing}")
        instances.append(Instance(**{f: record[f] for f in REQUIRED_FIELDS}))
    return instances


def download_split(dataset: str, split: str, revision: str, destination: Path):
    from datasets import load_dataset

    destination.parent.mkdir(parents=True, exist_ok=True)
    load_dataset(DATASET_IDS[dataset], split=split, revision=revision).to_json(str(destination))


def source_files_changed(patch: str) -> list[str]:
    return [p for p in changed_pre_image_lines(patch) if p.endswith(".py") and not is_test_path(p)]


def bare_clone_path(repos_dir: Path, repo: str) -> Path:
    return repos_dir / f"{repo.replace('/', '__')}.git"


def ensure_clone(repos_dir: Path, repo: str) -> Path:
    path = bare_clone_path(repos_dir, repo)
    if not path.exists():
        subprocess.run(["git", "clone", "--bare", "--filter=blob:none", f"https://github.com/{repo}.git", str(path)],
                       check=True, capture_output=True)
    return path


@contextmanager
def checkout(repos_dir: Path, worktrees_dir: Path, instance: Instance):
    """Pre-fix repository state for one instance as a detached git worktree, removed afterwards."""
    bare = ensure_clone(repos_dir, instance.repo)
    worktree = (worktrees_dir / instance.instance_id).resolve()
    if worktree.exists():
        subprocess.run(["git", "-C", str(bare), "worktree", "remove", "--force", str(worktree)], capture_output=True)
    subprocess.run(["git", "-C", str(bare), "worktree", "prune"], check=True, capture_output=True)
    completed = subprocess.run(["git", "-C", str(bare), "worktree", "add", "--detach", str(worktree),
                                instance.base_commit], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"checkout of {instance.repo}@{instance.base_commit} failed: {completed.stderr.strip()}")
    try:
        yield worktree
    finally:
        subprocess.run(["git", "-C", str(bare), "worktree", "remove", "--force", str(worktree)], capture_output=True)
