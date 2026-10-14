import csv
import itertools
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from code_retrieval.config import Config, RankingConfig
from code_retrieval.evaluation.metrics import CUTOFFS, collapse_to_files, mean_metrics, query_metrics
from code_retrieval.expansion import DiscoveryPath, Expansion
from code_retrieval.pipeline import CandidatePool, pool_candidates
from code_retrieval.ranking import GATES, file_margin, file_of, hybrid_is_confident, rank

ALL_EDGES = ("called_by", "calls", "referenced_by", "inherits_from")
BASELINES = ("name_match", "bm25", "dense", "hybrid")
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 20261013


@dataclass(frozen=True)
class Variant:
    name: str
    strategy: str
    seeds: int = 10
    depth: int = 1
    edges: tuple[str, ...] = ALL_EDGES
    propagation_weight: float | None = None
    propagation_cap: float | None = None
    gate: str | None = None
    gate_threshold: float | None = None


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "per_query.jsonl").read_text().splitlines() if line.strip()]
    return manifest, records


def is_module(symbol_id: str) -> bool:
    return symbol_id.endswith("::module")


def seed_ranks(record: dict) -> dict[str, int]:
    expandable = [sid for sid in record["pool"]["seed_pool"] if not is_module(sid)]
    return {sid: rank for rank, sid in enumerate(expandable, start=1)}


def restricted_pool(record: dict, variant: Variant) -> CandidatePool:
    pool = record["pool"]
    ranks = seed_ranks(record)
    expansion = Expansion()
    for sid, paths in pool["expansion"]["paths"].items():
        kept = [DiscoveryPath(**p) for p in paths
                if p["hop"] <= variant.depth and ranks.get(p["seed"], 10**9) <= variant.seeds and p["edge"] in variant.edges]
        if kept:
            expansion.paths[sid] = kept
    return CandidatePool(relevance=pool["relevance"], lexical_rank=pool["lexical_rank"],
                         dense_rank=pool["dense_rank"], seed_pool=pool["seed_pool"], expansion=expansion, timings={})


def hybrid_ranking(record: dict) -> list[str]:
    relevance = record["pool"]["relevance"]
    return sorted(record["pool"]["seed_pool"], key=lambda sid: (-relevance[sid], sid))


def gate_open(record: dict, variant: Variant) -> bool:
    """Replays the online gate from stored signals: hybrid's file margin and BM25/dense top-file agreement."""
    margin = file_margin(hybrid_ranking(record), record["pool"]["relevance"])
    bm25, dense = record["rankings"]["bm25"], record["rankings"]["dense"]
    agree = bool(bm25) and bool(dense) and file_of(bm25[0]) == file_of(dense[0])
    return not hybrid_is_confident(margin, agree, variant.gate, variant.gate_threshold)


def variant_ranking(record: dict, variant: Variant, ranking_config: RankingConfig) -> list[str]:
    if record["status"] != "ok":
        return []
    if variant.strategy == "gated":
        if not gate_open(record, variant):
            return hybrid_ranking(record)
        variant = Variant(variant.name, "relationship", variant.seeds, variant.depth, variant.edges,
                          variant.propagation_weight, variant.propagation_cap)
    if variant.strategy in ("name_match", "bm25", "dense"):
        return record["rankings"][variant.strategy]
    if variant.strategy == "hybrid":
        return hybrid_ranking(record)
    config = deepcopy(ranking_config)
    if variant.propagation_weight is not None:
        config.propagation_weight = variant.propagation_weight
    if variant.propagation_cap is not None:
        config.propagation_cap = variant.propagation_cap
    pool = restricted_pool(record, variant)
    candidates = pool_candidates(pool)
    return [sid for sid, _ in rank(list(candidates.values()), pool.relevance, config, variant.strategy)]


def variant_expansion_seconds(record: dict, variant: Variant) -> float:
    """Latency the variant would have spent in LSP requests, summed from the logged per-request timings."""
    if record["status"] != "ok" or variant.strategy not in ("inherited", "relationship", "novelty", "gated"):
        return 0.0
    if variant.strategy == "gated" and not gate_open(record, variant):
        return 0.0
    return sum(r["seconds"] for r in record["pool"]["expansion"]["request_log"]
               if r["hop"] <= variant.depth and r["seed_rank"] <= variant.seeds and r["edge"] in variant.edges)


def per_query_metrics(records: list[dict], variant: Variant, ranking_config: RankingConfig) -> dict[str, list]:
    file_rows, function_rows = [], []
    for record in records:
        ranking = variant_ranking(record, variant, ranking_config)
        gold = record.get("gold") or {}
        gold_files = set(gold.get("files") or [])
        if not gold_files and record["status"] == "ok":
            continue
        file_rows.append((record["instance_id"], query_metrics(collapse_to_files(ranking), gold_files or {"<missing>"})))
        gold_symbols = set(gold.get("symbols") or [])
        if gold_symbols:
            function_rows.append((record["instance_id"],
                                  query_metrics([sid for sid in ranking if not is_module(sid)], gold_symbols)))
    return {"file": file_rows, "function": function_rows}


def summarize(rows: list[tuple[str, dict]]) -> dict:
    return {**mean_metrics([m for _, m in rows]), "n": len(rows)}


def tune(dev_records: list[dict], ranking_config: RankingConfig, strategy: str, name: str) -> tuple[Variant, list[dict]]:
    """Grid search on the dev split only, selecting by file-level MRR; ties prefer the cheaper configuration."""
    trials = []
    for seeds, depth, weight, cap in itertools.product((5, 10, 20), (1, 2), (0.1, 0.25, 0.5, 1.0), (0.5, 1.0)):
        variant = Variant(name, strategy, seeds, depth, ALL_EDGES, weight, cap)
        rows = per_query_metrics(dev_records, variant, ranking_config)
        file_summary = summarize(rows["file"])
        trials.append({"seeds": seeds, "depth": depth, "propagation_weight": weight, "propagation_cap": cap,
                       "file_mrr": file_summary["mrr"], "file_acc@5": file_summary["acc@5"]})
    best = max(trials, key=lambda t: (round(t["file_mrr"], 6), -t["depth"], -t["seeds"], -t["propagation_weight"]))
    variant = Variant(name, strategy, best["seeds"], best["depth"], ALL_EDGES,
                      best["propagation_weight"], best["propagation_cap"])
    return variant, trials


def tune_gate(dev_records: list[dict], ranking_config: RankingConfig, inner: Variant) -> tuple[Variant, list[dict]]:
    """Only the gate is tuned (on dev); the inner ranker keeps the dev-tuned relationship settings.
    Selection by file MRR; ties prefer the gate that opens least often (lower latency)."""
    trials = []
    options = [("agree", None)] + [(g, t) for g in ("margin", "agree_or_margin", "agree_and_margin")
                                   for t in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15)]
    for gate, threshold in options:
        variant = Variant("lsp-gated (tuned)", "gated", inner.seeds, inner.depth, inner.edges,
                          inner.propagation_weight, inner.propagation_cap, gate, threshold if threshold is not None else 0.0)
        file_summary = summarize(per_query_metrics(dev_records, variant, ranking_config)["file"])
        open_rate = sum(gate_open(r, variant) for r in dev_records if r["status"] == "ok") / len(dev_records)
        trials.append({"gate": gate, "threshold": threshold, "file_mrr": file_summary["mrr"],
                       "file_acc@5": file_summary["acc@5"], "open_rate": open_rate})
    best = max(trials, key=lambda t: (round(t["file_mrr"], 6), -t["open_rate"]))
    variant = Variant("lsp-gated (tuned)", "gated", inner.seeds, inner.depth, inner.edges, inner.propagation_weight,
                      inner.propagation_cap, best["gate"], best["threshold"] if best["threshold"] is not None else 0.0)
    return variant, trials


def gate_summary(records: list[dict], gated: Variant) -> dict:
    ok = [r for r in records if r["status"] == "ok" and r["gold"]["files"]]
    by_slice: dict[str, list[bool]] = defaultdict(list)
    for record in ok:
        by_slice[hybrid_slice(record)].append(gate_open(record, gated))
    return {"open_rate": sum(gate_open(r, gated) for r in ok) / len(ok) if ok else 0.0,
            "open_rate_by_slice": {k: sum(v) / len(v) for k, v in sorted(by_slice.items())}}


def main_variants(tuned: Variant, novelty: Variant, gated: Variant) -> list[Variant]:
    return [
        Variant("name_match", "name_match"), Variant("bm25", "bm25"), Variant("dense", "dense"),
        Variant("hybrid", "hybrid"),
        Variant("lsp (inherited)", "inherited", tuned.seeds, tuned.depth),
        Variant("lsp-ranked (default)", "relationship"),
        tuned,
        novelty,
        gated,
    ]


def ablation_variants(tuned: Variant) -> list[Variant]:
    base = dict(strategy="relationship", propagation_weight=tuned.propagation_weight,
                propagation_cap=tuned.propagation_cap)
    variants = [Variant(f"depth={d}", seeds=tuned.seeds, depth=d, **base) for d in (1, 2)]
    variants += [Variant(f"seeds={n}", seeds=n, depth=tuned.depth, **base) for n in (5, 10, 20)]
    variants += [Variant(f"only {e}", seeds=tuned.seeds, depth=tuned.depth, edges=(e,), **base) for e in ALL_EDGES]
    variants += [Variant(f"without {e}", seeds=tuned.seeds, depth=tuned.depth,
                         edges=tuple(x for x in ALL_EDGES if x != e), **base) for e in ALL_EDGES]
    return variants


def paired_comparison(rows_a: list[tuple[str, dict]], rows_b: list[tuple[str, dict]], metric: str) -> dict:
    a = dict(rows_a)
    b = dict(rows_b)
    shared = sorted(set(a) & set(b))
    diff = np.array([b[i][metric] - a[i][metric] for i in shared])
    rng = np.random.default_rng(RANDOM_SEED)
    resamples = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    means = diff[resamples].mean(axis=1)
    result = {"metric": metric, "n": len(shared), "mean_difference": float(diff.mean()),
              "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
              "improved": int((diff > 0).sum()), "worsened": int((diff < 0).sum())}
    if metric.startswith("acc@") or metric.startswith("hit@"):
        discordant = result["improved"] + result["worsened"]
        result["test"] = "McNemar exact"
        result["p_value"] = (float(stats.binomtest(result["improved"], discordant, 0.5).pvalue)
                             if discordant else 1.0)
    else:
        nonzero = diff[diff != 0]
        result["test"] = "Wilcoxon signed-rank"
        result["p_value"] = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else None
        result["effect_size_rank_biserial"] = (
            float((result["improved"] - result["worsened"]) / len(nonzero)) if len(nonzero) else 0.0)
    return result


def holm(p_values: list[float | None]) -> list[float | None]:
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for position, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, p * (len(indexed) - position)))
        adjusted[i] = running
    return adjusted


def headroom(records: list[dict], k: int) -> dict:
    """Of the queries where hybrid misses the gold file in its top-k, how many have gold reachable by expansion."""
    result = {}
    for seeds, depth in ((10, 1), (20, 1), (10, 2), (20, 2)):
        misses = 0
        reachable_symbol = reachable_file = 0
        for record in records:
            if record["status"] != "ok" or not record["gold"]["files"]:
                continue
            gold_files = set(record["gold"]["files"])
            hybrid_files = collapse_to_files(variant_ranking(record, Variant("hybrid", "hybrid"), RankingConfig()))
            if gold_files & set(hybrid_files[:k]):
                continue
            misses += 1
            pool = restricted_pool(record, Variant("h", "relationship", seeds, depth))
            discovered = set(pool.expansion.paths)
            if set(record["gold"]["symbols"]) & discovered:
                reachable_symbol += 1
            if gold_files & {sid.split("::", 1)[0] for sid in discovered}:
                reachable_file += 1
        result[f"seeds={seeds},depth={depth}"] = {
            "hybrid_misses": misses, "gold_symbol_discovered": reachable_symbol,
            "gold_file_discovered": reachable_file}
    return result


def gold_file_rank(ranking: list[str], gold_files: set[str]) -> int | None:
    return next((i for i, path in enumerate(collapse_to_files(ranking), start=1) if path in gold_files), None)


def hybrid_slice(record: dict) -> str:
    rank_of_gold = gold_file_rank(variant_ranking(record, Variant("hybrid", "hybrid"), RankingConfig()),
                                  set(record["gold"]["files"]))
    if rank_of_gold == 1:
        return "a: gold file ranked 1 by hybrid"
    if rank_of_gold is not None and rank_of_gold <= 10:
        return "b: gold file ranked 2-10 by hybrid"
    return "c: gold file outside hybrid top-10"


def slice_analysis(records: list[dict], challenger: Variant, ranking_config: RankingConfig) -> dict:
    """Hybrid vs challenger within slices defined by where hybrid places the gold file (pre-declared, H1 test)."""
    usable = [r for r in records if r["status"] == "ok" and r["gold"]["files"]]
    slices: dict[str, list[dict]] = defaultdict(list)
    for record in usable:
        slices[hybrid_slice(record)].append(record)
    result = {}
    for name, group in sorted(slices.items()):
        hybrid = summarize(per_query_metrics(group, Variant("hybrid", "hybrid"), ranking_config)["file"])
        other = summarize(per_query_metrics(group, challenger, ranking_config)["file"])
        result[name] = {"n": len(group), **{f"hybrid_{m}": hybrid[m] for m in ("mrr", "acc@1", "acc@5", "recall@5", "recall@10")},
                        **{f"challenger_{m}": other[m] for m in ("mrr", "acc@1", "acc@5", "recall@5", "recall@10")}}
    return result


def files_above_gold(ranking: list[str], gold_files: set[str]) -> list[str]:
    above = []
    for path in collapse_to_files(ranking):
        if path in gold_files:
            break
        above.append(path)
    return above


def overtakers(record: dict, ranking: list[str], baseline: list[str], gold_files: set[str]) -> list[dict]:
    """Files newly placed above the first gold file relative to the baseline, each with its top symbol's origin."""
    seeds = set(record["pool"]["seed_pool"])
    paths = record["pool"]["expansion"]["paths"]
    newly_above = set(files_above_gold(ranking, gold_files)) - set(files_above_gold(baseline, gold_files))
    rows, seen = [], set()
    for sid in ranking:
        path = sid.split("::", 1)[0]
        if path in gold_files:
            break
        if path in seen or path not in newly_above:
            continue
        seen.add(path)
        found_by = paths.get(sid, [])
        rows.append({"symbol": sid, "seed": sid in seeds, "relevance": record["pool"]["relevance"].get(sid),
                     "edges": sorted({p["edge"] for p in found_by}), "path_count": len(found_by)})
    return rows


def win_loss(records: list[dict], challenger: Variant, ranking_config: RankingConfig, limit: int = 12) -> dict:
    """Per-query change in gold-file rank versus hybrid, with the symbols that overtook gold on losses."""
    changes = []
    for record in records:
        if record["status"] != "ok" or not record["gold"]["files"]:
            continue
        gold_files = set(record["gold"]["files"])
        hybrid_ranking = variant_ranking(record, Variant("hybrid", "hybrid"), ranking_config)
        before = gold_file_rank(hybrid_ranking, gold_files)
        challenger_ranking = variant_ranking(record, challenger, ranking_config)
        after = gold_file_rank(challenger_ranking, gold_files)
        if before == after:
            continue
        gold_symbol_paths = [p for g in record["gold"]["symbols"] for p in record["pool"]["expansion"]["paths"].get(g, [])]
        changes.append({
            "instance_id": record["instance_id"], "hybrid_rank": before, "challenger_rank": after,
            "gold_reached_by_expansion_via": sorted({p["edge"] for p in gold_symbol_paths}),
            "overtakers": overtakers(record, challenger_ranking, hybrid_ranking, gold_files)[:5] if (after or 10**9) > (before or 10**9) else [],
        })
    worse = [c for c in changes if (c["challenger_rank"] or 10**9) > (c["hybrid_rank"] or 10**9)]
    better = [c for c in changes if c not in worse]
    overtaker_kinds = defaultdict(int)
    overtaker_edges = defaultdict(int)
    for change in worse:
        for row in change["overtakers"]:
            overtaker_kinds["boosted seed" if row["seed"] else "expansion-only"] += 1
            for edge in row["edges"]:
                overtaker_edges[edge] += 1
    gain_edges = defaultdict(int)
    for change in better:
        for edge in change["gold_reached_by_expansion_via"]:
            gain_edges[edge] += 1
    key = lambda c: abs((c["challenger_rank"] or 101) - (c["hybrid_rank"] or 101))
    return {"improved": len(better), "worsened": len(worse),
            "overtaker_kinds": dict(overtaker_kinds), "overtaker_edges": dict(overtaker_edges),
            "gold_edges_on_gains": dict(gain_edges),
            "largest_losses": sorted(worse, key=key, reverse=True)[:limit],
            "largest_gains": sorted(better, key=key, reverse=True)[:limit]}


def variant_request_count(record: dict, variant: Variant) -> int:
    if variant.strategy not in ("inherited", "relationship", "novelty", "gated"):
        return 0
    if variant.strategy == "gated" and not gate_open(record, variant):
        return 0
    return sum(1 for q in record["pool"]["expansion"]["request_log"]
               if q["hop"] <= variant.depth and q["seed_rank"] <= variant.seeds and q["edge"] in variant.edges)


def latency_summary(records: list[dict], variant: Variant) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return {}
    retrieval = [r["timings"].get("lexical_seconds", 0) + r["timings"].get("dense_seconds", 0) for r in ok]
    expansion = [variant_expansion_seconds(r, variant) for r in ok]
    total = [a + b for a, b in zip(retrieval, expansion)]
    requests = [variant_request_count(r, variant) for r in ok]
    return {
        "queries": len(ok),
        "retrieval_mean_s": statistics.mean(retrieval),
        "expansion_mean_s": statistics.mean(expansion), "expansion_median_s": statistics.median(expansion),
        "expansion_p95_s": float(np.percentile(expansion, 95)),
        "total_mean_s": statistics.mean(total), "total_median_s": statistics.median(total),
        "total_p95_s": float(np.percentile(total, 95)),
        "requests_mean": statistics.mean(requests),
        "lsp_startup_mean_s": statistics.mean(r["timings"].get("lsp_startup_seconds", 0) for r in ok),
        "index_mean_s": statistics.mean(r["timings"]["index_seconds"] for r in ok),
        "dense_build_mean_s": statistics.mean(r["timings"].get("dense_build_seconds", 0) for r in ok),
    }


def operational_summary(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    events = defaultdict(int)
    outcomes = defaultdict(int)
    for r in ok:
        for name, count in r["pool"]["expansion"]["events"].items():
            events[name] += count
        for request in r["pool"]["expansion"]["request_log"]:
            outcomes[request["outcome"]] += 1
    total_requests = sum(outcomes.values())
    return {
        "instances": len(records), "failed_instances": len(records) - len(ok),
        "failures": [{"instance_id": r["instance_id"], "error": r["error"]} for r in records if r["status"] != "ok"],
        "instances_without_gold": [r["instance_id"] for r in ok if not r["gold"]["files"]],
        "gold_files_outside_index": sorted({f for r in ok for f in r["gold"]["files_outside_index"]}),
        "lsp_requests": total_requests,
        "request_outcomes": dict(outcomes),
        "request_failure_rate": (total_requests - outcomes.get("ok", 0)) / total_requests if total_requests else 0.0,
        "events": dict(events),
        "mean_discovered_candidates": statistics.mean(r["pool"]["expansion"]["discovered"] for r in ok) if ok else 0,
        "mean_dropped_outside_index": statistics.mean(r["pool"]["expansion"]["dropped_outside_index"] for r in ok) if ok else 0,
        "function_gold_coverage": {
            "with_function_gold": sum(1 for r in ok if r["gold"]["symbols"]),
            "module_level_only": sum(1 for r in ok if not r["gold"]["symbols"] and r["gold"]["module_level"]),
            "unmappable_lines": sum(len(r["gold"]["unmappable"]) for r in ok),
            "gold_files_outside_index": sum(len(r["gold"]["files_outside_index"]) for r in ok),
        },
    }


def per_repo(records: list[dict], variants: list[Variant], ranking_config: RankingConfig) -> dict:
    by_repo = defaultdict(list)
    for record in records:
        by_repo[record["repo"]].append(record)
    return {repo: {v.name: summarize(per_query_metrics(group, v, ranking_config)["file"])["mrr"] for v in variants}
            | {"n": len(group)} for repo, group in sorted(by_repo.items())}


def write_table(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(name: str, records: list[dict], tuned: Variant, novelty: Variant, gated: Variant,
                ranking_config: RankingConfig) -> dict:
    variants = main_variants(tuned, novelty, gated)
    per_variant = {v.name: per_query_metrics(records, v, ranking_config) for v in variants}
    hybrid = per_variant["hybrid"]
    comparisons = []
    for challenger in ("lsp (inherited)", "lsp-ranked (default)", tuned.name, novelty.name, gated.name):
        for level in ("file", "function"):
            for metric in ("mrr", "acc@1", "acc@5", "acc@10", "ndcg@10"):
                comparisons.append({"challenger": challenger, "level": level,
                                    **paired_comparison(hybrid[level], per_variant[challenger][level], metric)})
    for row, adjusted in zip(comparisons, holm([c["p_value"] for c in comparisons])):
        row["p_holm"] = adjusted
    ablations = {v.name: {"file": summarize(per_query_metrics(records, v, ranking_config)["file"]),
                          "function": summarize(per_query_metrics(records, v, ranking_config)["function"]),
                          "latency": latency_summary(records, v)}
                 for v in ablation_variants(tuned)}
    return {
        "run": name,
        "methods": {v.name: {"file": summarize(per_variant[v.name]["file"]),
                             "function": summarize(per_variant[v.name]["function"]),
                             "latency": latency_summary(records, v)} for v in variants},
        "comparisons_vs_hybrid": comparisons,
        "ablations": ablations,
        "headroom@5": headroom(records, 5),
        "headroom@10": headroom(records, 10),
        "slices": {v.name: slice_analysis(records, v, ranking_config) for v in (tuned, novelty, gated)},
        "win_loss": {v.name: win_loss(records, v, ranking_config) for v in (tuned, novelty, gated)},
        "gate": gate_summary(records, gated),
        "operations": operational_summary(records),
        "per_repo_file_mrr": per_repo(records, variants, ranking_config),
        "per_query": {v.name: {level: dict(rows) for level, rows in per_variant[v.name].items()} for v in variants},
    }


def write_report(run_dirs: list[Path], out_dir: Path, config: Config):
    from code_retrieval.evaluation.figures import render_figures

    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {run_dir.name: load_run(run_dir) for run_dir in run_dirs}
    dev = [name for name, (manifest, _) in runs.items() if manifest["selection"]["split"] == "dev"]
    if len(dev) != 1:
        raise ValueError(f"expected exactly one dev run to tune on, found {dev}")
    tuned, trials = tune(runs[dev[0]][1], config.ranking, "relationship", "lsp-ranked (tuned)")
    novelty, novelty_trials = tune(runs[dev[0]][1], config.ranking, "novelty", "lsp-novelty (tuned)")
    gated, gate_trials = tune_gate(runs[dev[0]][1], config.ranking, tuned)
    report = {"tuned_on": dev[0], "tuned_variant": tuned.__dict__, "tuning_trials": trials,
              "novelty_variant": novelty.__dict__, "novelty_tuning_trials": novelty_trials,
              "gated_variant": gated.__dict__, "gate_tuning_trials": gate_trials,
              "ranking_config": config.ranking.__dict__, "runs": {}}
    for name, (manifest, records) in runs.items():
        report["runs"][name] = {"manifest": manifest, **analyze_run(name, records, tuned, novelty, gated, config.ranking)}

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    for name, run in report["runs"].items():
        rows = []
        for method, values in run["methods"].items():
            for level in ("file", "function"):
                rows.append({"run": name, "method": method, "level": level,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in values[level].items()}})
        write_table(out_dir / f"{name}_methods.csv", rows)
    render_figures(report, out_dir / "figures")
    (out_dir / "summary.md").write_text(markdown_summary(report))
    print((out_dir / "summary.md").read_text())


def fmt(value, digits=3):
    if value is None:
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown_summary(report: dict) -> str:
    tuned = report["tuned_variant"]
    novelty = report["novelty_variant"]
    lines = [f"# Results summary (generated by `code-retrieval report`)",
             "", f"Ranking weights tuned on `{report['tuned_on']}` only: seeds={tuned['seeds']}, depth={tuned['depth']}, "
             f"propagation_weight={tuned['propagation_weight']}, propagation_cap={tuned['propagation_cap']}. "
             f"Novelty ranker: seeds={novelty['seeds']}, depth={novelty['depth']}, "
             f"propagation_weight={novelty['propagation_weight']}, propagation_cap={novelty['propagation_cap']}.", ""]
    for name, run in report["runs"].items():
        sel = run["manifest"]["selection"]
        lines += [f"## {name}", "", f"Dataset `{run['manifest']['dataset']['id']}` split `{sel['split']}`, "
                  f"sample={sel['sample'] or 'all'}, min_files={sel.get('min_files', 1)}, "
                  f"instances={run['operations']['instances']}, failed={run['operations']['failed_instances']}.", ""]
        for level in ("file", "function"):
            n = next(iter(run["methods"].values()))[level].get("n", 0)
            lines += [f"### {level.capitalize()}-level (n={n})", "",
                      "| method | " + " | ".join(f"Acc@{k}" for k in CUTOFFS) + " | Recall@5 | MRR | nDCG@10 |",
                      "|---|" + "---|" * (len(CUTOFFS) + 3)]
            for method, values in run["methods"].items():
                m = values[level]
                if not m.get("n"):
                    continue
                lines.append(f"| {method} | " + " | ".join(fmt(m[f'acc@{k}']) for k in CUTOFFS)
                             + f" | {fmt(m['recall@5'])} | {fmt(m['mrr'])} | {fmt(m['ndcg@10'])} |")
            lines.append("")
        lines += ["### Paired comparison against hybrid", "",
                  "| challenger | level | metric | Δ mean | 95% CI | +/− | test | p | p (Holm) |", "|---|---|---|---|---|---|---|---|---|"]
        for c in run["comparisons_vs_hybrid"]:
            if c["metric"] in ("mrr", "acc@1", "acc@5"):
                lines.append(f"| {c['challenger']} | {c['level']} | {c['metric']} | {c['mean_difference']:+.3f} | "
                             f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['improved']}/{c['worsened']} | {c['test']} | "
                             f"{fmt(c['p_value'])} | {fmt(c['p_holm'])} |")
        lines += ["", "### Ablations (relationship ranker, tuned weights)", "",
                  "| variant | file Acc@1 | file Acc@5 | file MRR | function MRR | expansion mean s | requests/query |",
                  "|---|---|---|---|---|---|---|"]
        for variant, values in run["ablations"].items():
            lines.append(f"| {variant} | {fmt(values['file']['acc@1'])} | {fmt(values['file']['acc@5'])} | "
                         f"{fmt(values['file']['mrr'])} | {fmt(values['function'].get('mrr'))} | "
                         f"{fmt(values['latency'].get('expansion_mean_s'))} | {fmt(values['latency'].get('requests_mean'), 1)} |")
        lines += ["", "### Headroom: hybrid misses whose gold is discovered by expansion", "",
                  "| cutoff | config | hybrid misses | gold symbol discovered | gold file discovered |", "|---|---|---|---|---|"]
        for cutoff in ("headroom@5", "headroom@10"):
            for config, h in run[cutoff].items():
                lines.append(f"| {cutoff} | {config} | {h['hybrid_misses']} | {h['gold_symbol_discovered']} | {h['gold_file_discovered']} |")
        lines += ["", "### Slices by hybrid's placement of the gold file (file level)", "",
                  "| challenger | slice | n | hybrid MRR | challenger MRR | hybrid Acc@1 | challenger Acc@1 | hybrid R@10 | challenger R@10 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for challenger, slices in run["slices"].items():
            for name, sl in slices.items():
                lines.append(f"| {challenger} | {name} | {sl['n']} | {fmt(sl['hybrid_mrr'])} | {fmt(sl['challenger_mrr'])} | "
                             f"{fmt(sl['hybrid_acc@1'])} | {fmt(sl['challenger_acc@1'])} | {fmt(sl['hybrid_recall@10'])} | "
                             f"{fmt(sl['challenger_recall@10'])} |")
        for challenger, wl in run["win_loss"].items():
            lines += ["", f"### Win/loss versus hybrid: {challenger} (gold-file rank)", "",
                      f"- Gold file moved up in {wl['improved']} queries, down in {wl['worsened']}.",
                      f"- Files that overtook gold on losses, by origin: {wl['overtaker_kinds']}; by discovering edge: {wl['overtaker_edges']}.",
                      f"- Edges by which the gold symbol was reached on gains: {wl['gold_edges_on_gains']}.", "",
                      "| instance | hybrid rank | challenger rank | top overtaker | origin | edges |", "|---|---|---|---|---|---|"]
            for change in wl["largest_losses"][:8] + wl["largest_gains"][:8]:
                top = change["overtakers"][0] if change["overtakers"] else None
                lines.append(f"| {change['instance_id']} | {change['hybrid_rank']} | {change['challenger_rank']} | "
                             f"{top['symbol'] if top else '–'} | {('seed' if top['seed'] else 'expansion-only') if top else '–'} | "
                             f"{', '.join(top['edges']) if top else ', '.join(change['gold_reached_by_expansion_via']) or '–'} |")
        ops = run["operations"]
        lines += ["", "### Operations", "", f"- LSP requests: {ops['lsp_requests']}, outcomes {ops['request_outcomes']}, "
                  f"failure rate {ops['request_failure_rate']:.4f}", f"- Events: {ops['events']}",
                  f"- Mean discovered candidates/query: {ops['mean_discovered_candidates']:.1f}; "
                  f"mean locations dropped outside index: {ops['mean_dropped_outside_index']:.1f}",
                  f"- Function-level gold coverage: {ops['function_gold_coverage']}",
                  f"- Failed instances: {ops['failures']}",
                  f"- Instances with no derivable gold (excluded from n): {ops['instances_without_gold']}",
                  f"- Gold files not present in the index (automatic misses): {ops['gold_files_outside_index']}", "",
                  "### Latency (seconds; expansion derived from logged per-request timings)", "",
                  "| method | retrieval mean | expansion mean | expansion p95 | total median | total p95 | requests |",
                  "|---|---|---|---|---|---|---|"]
        for method, values in run["methods"].items():
            lat = values["latency"]
            if lat:
                lines.append(f"| {method} | {fmt(lat['retrieval_mean_s'])} | {fmt(lat['expansion_mean_s'])} | "
                             f"{fmt(lat['expansion_p95_s'])} | {fmt(lat['total_median_s'])} | {fmt(lat['total_p95_s'])} | "
                             f"{fmt(lat['requests_mean'], 1)} |")
        first = next(iter(run["methods"].values()))["latency"]
        if first:
            lines += ["", f"One-time per checkout (mean): index {first['index_mean_s']:.2f}s, "
                      f"embedding build {first['dense_build_mean_s']:.1f}s (cache-dependent), "
                      f"LSP startup {first['lsp_startup_mean_s']:.2f}s.", ""]
        lines += ["### File MRR per repository", "", "| repo | n | " + " | ".join(run["methods"]) + " |",
                  "|---|---|" + "---|" * len(run["methods"])]
        for repo, values in run["per_repo_file_mrr"].items():
            lines.append(f"| {repo} | {values['n']} | " + " | ".join(fmt(values[m]) for m in run["methods"]) + " |")
        lines.append("")
    return "\n".join(lines)
