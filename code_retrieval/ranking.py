from dataclasses import dataclass

import numpy as np

from code_retrieval.config import RankingConfig
from code_retrieval.expansion import DiscoveryPath


@dataclass(frozen=True)
class Candidate:
    symbol_id: str
    relevance: float
    in_seed_pool: bool
    paths: tuple[DiscoveryPath, ...]


def full_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def hybrid_relevance(lexical_scores: np.ndarray | None, dense_scores: np.ndarray | None, rrf_k: int) -> np.ndarray:
    """Full-corpus RRF normalized to [0, 1], so expansion-only candidates have a query relevance too."""
    channels = [s for s in (lexical_scores, dense_scores) if s is not None]
    fused = sum(1.0 / (rrf_k + full_ranks(s)) for s in channels)
    return fused / fused.max()


def inherited_score(candidate: Candidate, relevance_of: dict[str, float], config: RankingConfig) -> float:
    if candidate.in_seed_pool or not candidate.paths:
        return candidate.relevance
    return max(config.hop_decay ** p.hop * relevance_of[p.seed] for p in candidate.paths)


def relationship_score(candidate: Candidate, relevance_of: dict[str, float], config: RankingConfig) -> float:
    evidence = {}
    for path in candidate.paths:
        contribution = config.edge_weights[path.edge] * config.hop_decay ** (path.hop - 1) * relevance_of[path.seed]
        evidence[(path.seed, path.edge)] = max(evidence.get((path.seed, path.edge), 0.0), contribution)
    propagated = min(config.propagation_cap, sum(evidence.values()))
    return config.seed_weight * candidate.relevance + config.propagation_weight * propagated


def novelty_score(candidate: Candidate, relevance_of: dict[str, float], config: RankingConfig) -> float:
    """Expansion may insert newly discovered code but never reorders the hybrid seed pool."""
    if candidate.in_seed_pool:
        return candidate.relevance
    return relationship_score(candidate, relevance_of, config)


GATES = ("agree", "margin", "agree_or_margin", "agree_and_margin")


def file_of(symbol_id: str) -> str:
    return symbol_id.split("::", 1)[0]


def file_margin(ordered_ids: list[str], relevance: dict[str, float]) -> float:
    """Gap between the best symbol of hybrid's top file and the best symbol of the runner-up file."""
    best: dict[str, float] = {}
    for symbol_id in ordered_ids:
        best.setdefault(file_of(symbol_id), relevance[symbol_id])
        if len(best) == 2:
            first, second = best.values()
            return first - second
    return 1.0


def hybrid_is_confident(margin: float, channels_agree: bool, gate: str, threshold: float) -> bool:
    """Gate for expansion: a confident hybrid ranking is returned as is, without asking the language server."""
    if gate == "agree":
        return channels_agree
    if gate == "margin":
        return margin >= threshold
    if gate == "agree_or_margin":
        return channels_agree or margin >= threshold
    if gate == "agree_and_margin":
        return channels_agree and margin >= threshold
    raise ValueError(f"unknown gate {gate!r}; expected one of {GATES}")


def rank(candidates: list[Candidate], relevance_of: dict[str, float], config: RankingConfig,
         strategy: str) -> list[tuple[str, float]]:
    scorer = {"inherited": inherited_score, "relationship": relationship_score, "novelty": novelty_score}[strategy]
    scored = [(c.symbol_id, scorer(c, relevance_of, config)) for c in candidates]
    return sorted(scored, key=lambda item: (-item[1], item[0]))
