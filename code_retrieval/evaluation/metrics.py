import math

CUTOFFS = (1, 3, 5, 10)


def collapse_to_files(ranked_symbol_ids: list[str]) -> list[str]:
    files = []
    seen = set()
    for symbol_id in ranked_symbol_ids:
        path = symbol_id.split("::", 1)[0]
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def query_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    """Binary-relevance metrics for one query; `ranked` is deduplicated at the evaluated granularity."""
    first_hit = next((i for i, item in enumerate(ranked, start=1) if item in gold), None)
    metrics = {"mrr": 1.0 / first_hit if first_hit else 0.0}
    for k in CUTOFFS:
        retrieved = set(ranked[:k])
        found = len(gold & retrieved)
        metrics[f"acc@{k}"] = float(found == len(gold))
        metrics[f"recall@{k}"] = found / len(gold)
        metrics[f"hit@{k}"] = float(found > 0)
        dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in gold)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
