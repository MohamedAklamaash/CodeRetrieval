import bm25s
import numpy as np

from code_retrieval.documents import tokenize


class LexicalRetriever:
    def __init__(self, documents: list[list[str]], k1: float, b: float):
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(documents, show_progress=False)
        self.vocabulary = set(self.model.vocab_dict)
        self.document_count = len(documents)

    def score_all(self, query: str) -> np.ndarray:
        query_tokens = sorted({t for t in tokenize(query) if t in self.vocabulary})
        if not query_tokens:
            return np.zeros(self.document_count, dtype=np.float32)
        return self.model.get_scores(query_tokens)
