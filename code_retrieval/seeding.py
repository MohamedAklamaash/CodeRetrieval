from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Seed:
    symbol_index: int
    rank: int
    fused_score: float
    lexical_rank: int | None
    dense_rank: int | None


def top_indices(scores: np.ndarray, depth: int) -> list[int]:
    candidates = np.flatnonzero(scores > 0) if scores.max(initial=0) > 0 else np.arange(len(scores))
    order = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))
    return order[:depth]


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings.values():
        for rank, symbol_index in enumerate(ranking, start=1):
            fused[symbol_index] = fused.get(symbol_index, 0.0) + 1.0 / (k + rank)
    return fused


def seed(lexical_scores: np.ndarray | None, dense_scores: np.ndarray | None,
         channel_depth: int, rrf_k: int, limit: int) -> list[Seed]:
    rankings = {}
    if lexical_scores is not None:
        rankings["lexical"] = top_indices(lexical_scores, channel_depth)
    if dense_scores is not None:
        rankings["dense"] = top_indices(dense_scores, channel_depth)
    fused = reciprocal_rank_fusion(rankings, rrf_k)
    positions = {name: {s: r for r, s in enumerate(ranking, start=1)} for name, ranking in rankings.items()}
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        Seed(symbol_index=i, rank=rank, fused_score=score,
             lexical_rank=positions.get("lexical", {}).get(i), dense_rank=positions.get("dense", {}).get(i))
        for rank, (i, score) in enumerate(ordered, start=1)
    ]
