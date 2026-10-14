import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    model_id: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerProvider:
    def __init__(self, model_id: str, document_max_tokens: int, query_max_tokens: int,
                 query_prefix: str, threads: int, trust_remote_code: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device="cpu", trust_remote_code=trust_remote_code)
        self.document_max_tokens = document_max_tokens
        self.query_max_tokens = query_max_tokens
        self.query_prefix = query_prefix

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.model.max_seq_length = self.document_max_tokens
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.model.max_seq_length = self.query_max_tokens
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)


class EmbeddingCache:
    """Vectors keyed by sha256(model_id, text), so unchanged symbols are shared across commits and repos."""

    commit_every = 512

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)")

    @staticmethod
    def key(model_id: str, text: str) -> str:
        return hashlib.sha256(f"{model_id}\0{text}".encode()).hexdigest()

    def embed(self, provider: EmbeddingProvider, texts: list[str]) -> tuple[np.ndarray, int]:
        keys = [self.key(provider.model_id, t) for t in texts]
        found = {}
        for start in range(0, len(keys), 900):
            chunk = keys[start:start + 900]
            rows = self.connection.execute(
                f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(chunk))})", chunk)
            found.update((k, np.frombuffer(v, dtype=np.float16)) for k, v in rows)
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in found}.items())
        for start in range(0, len(missing), self.commit_every):
            chunk = missing[start:start + self.commit_every]
            vectors = provider.encode_documents([t for _, t in chunk]).astype(np.float16)
            self.connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                        [(k, v.tobytes()) for (k, _), v in zip(chunk, vectors)])
            self.connection.commit()
            found.update((k, v) for (k, _), v in zip(chunk, vectors))
        return np.stack([found[k] for k in keys]).astype(np.float32), len(missing)


class DenseRetriever:
    def __init__(self, provider: EmbeddingProvider, document_vectors: np.ndarray):
        self.provider = provider
        self.document_vectors = document_vectors

    def score_all(self, query: str) -> np.ndarray:
        return self.document_vectors @ self.provider.encode_query(query)
