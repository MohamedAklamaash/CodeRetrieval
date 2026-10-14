import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from code_retrieval.config import Config
from code_retrieval.dense import DenseRetriever, EmbeddingCache, SentenceTransformerProvider
from code_retrieval.documents import dense_text, lexical_tokens, member_signatures
from code_retrieval.expansion import Expansion, LspExpander
from code_retrieval.indexing import RepositoryIndex, index_repository
from code_retrieval.lexical import LexicalRetriever
from code_retrieval.lsp_session import LanguageServerSession
from code_retrieval.ranking import (Candidate, file_margin, file_of, full_ranks, hybrid_is_confident,
                                    hybrid_relevance, rank)
from code_retrieval.seeding import seed

METHODS = ("bm25", "dense", "hybrid", "lsp", "lsp-ranked", "lsp-novelty", "lsp-gated")


@dataclass
class QueryOutcome:
    ranked: list[tuple[str, float]]
    candidates: dict[str, Candidate]
    expansion: Expansion | None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidatePool:
    """Everything the rankers need, so ranking variants can be re-scored offline without the server."""
    relevance: dict[str, float]
    lexical_rank: dict[str, int]
    dense_rank: dict[str, int]
    seed_pool: list[str]
    expansion: Expansion | None
    timings: dict[str, float]
    file_margin: float = 0.0
    channels_agree: bool = False
    gate_open: bool | None = None


class Retriever:
    def __init__(self, repo: Path, config: Config, provider: SentenceTransformerProvider | None = None,
                 lsp_log: Path | None = None):
        self.config = config
        self.timings: dict[str, float] = {}
        started = time.perf_counter()
        self.index: RepositoryIndex = index_repository(repo)
        self.timings["index_seconds"] = time.perf_counter() - started

        members = member_signatures(self.index)
        started = time.perf_counter()
        self.lexical = LexicalRetriever(
            [lexical_tokens(s, members, config.lexical.field_weights) for s in self.index.symbols],
            config.lexical.k1, config.lexical.b)
        self.timings["lexical_build_seconds"] = time.perf_counter() - started

        self.dense = None
        self.newly_embedded = 0
        if config.dense.enabled:
            provider = provider or load_provider(config)
            started = time.perf_counter()
            vectors, self.newly_embedded = EmbeddingCache(Path(config.dense.cache_path)).embed(
                provider, [dense_text(s, members) for s in self.index.symbols])
            self.dense = DenseRetriever(provider, vectors)
            self.timings["dense_build_seconds"] = time.perf_counter() - started

        self.session: LanguageServerSession | None = None
        self.lsp_log = lsp_log

    def start_language_server(self):
        if self.session is None:
            self.session = LanguageServerSession(self.index, self.config.lsp, self.lsp_log)
            self.timings["lsp_startup_seconds"] = self.session.startup_seconds

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def score_channels(self, query: str, channels: tuple[str, ...]) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        timings = {}
        lexical_scores = dense_scores = None
        if "lexical" in channels:
            started = time.perf_counter()
            lexical_scores = self.lexical.score_all(query)
            timings["lexical_seconds"] = time.perf_counter() - started
        if "dense" in channels and self.dense is not None:
            started = time.perf_counter()
            dense_scores = self.dense.score_all(query)
            timings["dense_seconds"] = time.perf_counter() - started
        return lexical_scores, dense_scores, timings

    def build_pool(self, query: str, expand: bool, gated: bool = False) -> CandidatePool:
        lexical_scores, dense_scores, timings = self.score_channels(query, ("lexical", "dense"))
        seeding = self.config.seeding
        seeds = seed(lexical_scores, dense_scores, seeding.channel_depth, seeding.rrf_k, seeding.pool_size)
        relevance = hybrid_relevance(lexical_scores, dense_scores, seeding.rrf_k)
        ids = [s.id for s in self.index.symbols]
        seed_pool = [ids[s.symbol_index] for s in seeds]
        position = {sid: i for i, sid in enumerate(ids)}
        relevance_of_pool = {sid: float(relevance[position[sid]]) for sid in seed_pool}
        margin = file_margin(sorted(seed_pool, key=lambda sid: (-relevance_of_pool[sid], sid)), relevance_of_pool)
        channels_agree = (lexical_scores is not None and dense_scores is not None
                          and file_of(ids[int(np.argmax(lexical_scores))]) == file_of(ids[int(np.argmax(dense_scores))]))
        gate_open = None
        if gated:
            ranking = self.config.ranking
            gate_open = not hybrid_is_confident(margin, channels_agree, ranking.gate, ranking.gate_threshold)
            expand = expand and gate_open

        expansion = None
        if expand:
            self.start_language_server()
            expandable = [sid for sid in seed_pool if self.index.by_id[sid].kind != "module"]
            expansion = LspExpander(self.session, self.index, self.config.expansion).expand(expandable, relevance)
            timings["expansion_seconds"] = expansion.seconds

        touched = set(seed_pool) | (set(expansion.paths) if expansion else set())
        lexical_ranks = channel_ranks(lexical_scores, ids, touched, position)
        dense_ranks = channel_ranks(dense_scores, ids, touched, position)
        return CandidatePool(
            relevance={sid: float(relevance[position[sid]]) for sid in touched},
            lexical_rank=lexical_ranks, dense_rank=dense_ranks, seed_pool=seed_pool,
            expansion=expansion, timings=timings, file_margin=margin, channels_agree=channels_agree,
            gate_open=gate_open)

    def query(self, query: str, method: str) -> QueryOutcome:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        started = time.perf_counter()
        if method in ("bm25", "dense"):
            channel = "lexical" if method == "bm25" else "dense"
            lexical_scores, dense_scores, timings = self.score_channels(query, (channel,))
            scores = lexical_scores if method == "bm25" else dense_scores
            if scores is None:
                raise RuntimeError("dense retrieval is disabled in this configuration")
            ids = [s.id for s in self.index.symbols]
            order = np.lexsort((np.arange(len(scores)), -scores))[:self.config.seeding.pool_size]
            ranked = [(ids[i], float(scores[i])) for i in order]
            candidates = {sid: Candidate(sid, score, True, ()) for sid, score in ranked}
            timings["total_seconds"] = time.perf_counter() - started
            return QueryOutcome(ranked, candidates, None, timings)

        pool = self.build_pool(query, expand=method != "hybrid", gated=method == "lsp-gated")
        candidates = pool_candidates(pool)
        strategy = {"hybrid": "inherited", "lsp": "inherited", "lsp-ranked": "relationship", "lsp-novelty": "novelty",
                    "lsp-gated": "relationship"}[method]
        ranked = rank(list(candidates.values()), pool.relevance, self.config.ranking, strategy)
        pool.timings["total_seconds"] = time.perf_counter() - started
        return QueryOutcome(ranked, candidates, pool.expansion, pool.timings)


def channel_ranks(scores: np.ndarray | None, ids: list[str], touched: set[str], position: dict[str, int]) -> dict[str, int]:
    if scores is None:
        return {}
    ranks = full_ranks(scores)
    return {sid: int(ranks[position[sid]]) for sid in touched}


def pool_candidates(pool: CandidatePool, max_hop: int | None = None, edges: set[str] | None = None) -> dict[str, Candidate]:
    seed_set = set(pool.seed_pool)
    paths = {}
    if pool.expansion:
        for sid, discovered in pool.expansion.paths.items():
            kept = tuple(p for p in discovered
                         if (max_hop is None or p.hop <= max_hop) and (edges is None or p.edge in edges))
            if kept:
                paths[sid] = kept
    ids = sorted(seed_set | set(paths))
    return {sid: Candidate(sid, pool.relevance[sid], sid in seed_set, paths.get(sid, ())) for sid in ids}


def load_provider(config: Config) -> SentenceTransformerProvider:
    dense = config.dense
    return SentenceTransformerProvider(dense.model_id, dense.document_max_tokens, dense.query_max_tokens,
                                       dense.query_prefix, dense.threads, dense.trust_remote_code)
