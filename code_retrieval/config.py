import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class LexicalConfig:
    k1: float = 1.2
    b: float = 0.75
    field_weights: dict[str, int] = field(default_factory=lambda: {
        "name": 3, "qualified_name": 2, "path": 1, "signature": 1, "docstring": 1, "body": 1})


@dataclass
class DenseConfig:
    enabled: bool = True
    model_id: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_max_tokens: int = 128
    query_max_tokens: int = 512
    threads: int = 8
    trust_remote_code: bool = False
    cache_path: str = "data/cache/embeddings.sqlite"


@dataclass
class SeedingConfig:
    channel_depth: int = 100
    rrf_k: int = 60
    pool_size: int = 100


@dataclass
class LspConfig:
    command: list[str] = field(default_factory=lambda: ["basedpyright-langserver", "--stdio"])
    excludes: list[str] = field(default_factory=lambda: ["**/tests/**", "**/node_modules", "**/.*"])
    initialize_timeout: float = 60.0


@dataclass
class ExpansionConfig:
    seeds: int = 10
    max_depth: int = 1
    edges: list[str] = field(default_factory=lambda: ["called_by", "calls", "referenced_by", "inherits_from"])
    max_neighbors_per_edge: int = 8
    max_requests: int = 60
    max_candidates: int = 200
    request_timeout: float = 10.0
    query_timeout: float = 60.0


@dataclass
class RankingConfig:
    seed_weight: float = 1.0
    propagation_weight: float = 0.5
    hop_decay: float = 0.5
    propagation_cap: float = 1.0
    edge_weights: dict[str, float] = field(default_factory=lambda: {
        "called_by": 1.0, "calls": 0.8, "referenced_by": 0.7, "inherits_from": 0.6})
    gate: str = "agree_or_margin"
    gate_threshold: float = 0.1


@dataclass
class Config:
    lexical: LexicalConfig = field(default_factory=LexicalConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    lsp: LspConfig = field(default_factory=LspConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def apply_overrides(target, overrides: dict, location: str = ""):
    known = {f.name for f in fields(target)}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(f"unknown config key {location}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            apply_overrides(current, value, f"{location}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: Path | None) -> Config:
    config = Config()
    if path is not None:
        apply_overrides(config, tomllib.loads(path.read_text()))
    return config
