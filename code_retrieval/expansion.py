import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from code_retrieval.config import ExpansionConfig
from code_retrieval.indexing import RepositoryIndex
from code_retrieval.lsp_client import LspError, LspTimeout, ServerCrashed, UnsupportedCapability, uri_to_path
from code_retrieval.lsp_session import LanguageServerSession
from code_retrieval.symbols import Symbol, innermost_symbol

EDGES_FOR_KIND = {
    "function": ("called_by", "calls", "referenced_by"),
    "method": ("called_by", "calls", "referenced_by"),
    "class": ("referenced_by", "inherits_from", "called_by"),
    "module": (),
}
BASE_CLASS_LIST = re.compile(r"\(([^()]*)\)\s*:?\s*$")
BASE_NAME = re.compile(r"\b[A-Za-z_][\w.]*\b(?!\s*=)")


@dataclass(frozen=True)
class DiscoveryPath:
    seed: str
    parent: str
    edge: str
    hop: int


@dataclass
class Expansion:
    paths: dict[str, list[DiscoveryPath]] = field(default_factory=dict)
    requests: int = 0
    events: Counter = field(default_factory=Counter)
    dropped_outside_index: int = 0
    truncated_neighbors: int = 0
    seconds: float = 0.0
    request_log: list[dict] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "requests": self.requests, "events": dict(self.events),
            "dropped_outside_index": self.dropped_outside_index,
            "truncated_neighbors": self.truncated_neighbors, "seconds": round(self.seconds, 4),
            "discovered": len(self.paths),
        }


class BudgetExhausted(Exception):
    pass


class LspExpander:
    def __init__(self, session: LanguageServerSession, index: RepositoryIndex, config: ExpansionConfig):
        self.session = session
        self.index = index
        self.config = config
        self.position_of = {s.id: i for i, s in enumerate(index.symbols)}

    def expand(self, seed_ids: list[str], query_relevance: np.ndarray) -> Expansion:
        result = Expansion()
        started = time.perf_counter()
        deadline = started + self.config.query_timeout
        visited: set[tuple[str, str]] = set()
        frontier = [(seed_id, seed_id) for seed_id in seed_ids[:self.config.seeds]]
        known = set(seed_ids)
        seed_rank = {seed_id: rank for rank, seed_id in enumerate(seed_ids, start=1)}
        try:
            for hop in range(1, self.config.max_depth + 1):
                next_frontier = []
                for node_id, seed_id in frontier:
                    node = self.index.by_id[node_id]
                    for edge in EDGES_FOR_KIND[node.kind]:
                        if edge not in self.config.edges or (node_id, edge) in visited:
                            continue
                        visited.add((node_id, edge))
                        self._context = {"node": node_id, "seed_rank": seed_rank[seed_id], "hop": hop, "edge": edge}
                        neighbors = self._neighbors(node, edge, result, deadline)
                        for neighbor_id in self._select(neighbors, node_id, query_relevance, result):
                            if neighbor_id == seed_id:
                                continue
                            result.paths.setdefault(neighbor_id, []).append(
                                DiscoveryPath(seed=seed_id, parent=node_id, edge=edge, hop=hop))
                            if neighbor_id not in known:
                                known.add(neighbor_id)
                                next_frontier.append((neighbor_id, seed_id))
                            if len(result.paths) >= self.config.max_candidates:
                                result.events["candidate_budget_exhausted"] += 1
                                raise BudgetExhausted
                frontier = sorted(next_frontier, key=lambda item: -query_relevance[self.position_of[item[0]]])
                frontier = frontier[:self.config.seeds]
        except BudgetExhausted:
            pass
        result.seconds = time.perf_counter() - started
        return result

    def _select(self, neighbors: list[str], source_id: str, query_relevance: np.ndarray,
                result: Expansion) -> list[str]:
        unique = sorted({n for n in neighbors if n != source_id})
        if len(unique) > self.config.max_neighbors_per_edge:
            result.truncated_neighbors += len(unique) - self.config.max_neighbors_per_edge
        ranked = sorted(unique, key=lambda n: (-float(query_relevance[self.position_of[n]]), n))
        return ranked[:self.config.max_neighbors_per_edge]

    def _request(self, method: str, params: dict, result: Expansion, deadline: float):
        if result.requests >= self.config.max_requests:
            result.events["request_budget_exhausted"] += 1
            raise BudgetExhausted
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            result.events["query_timeout"] += 1
            raise BudgetExhausted
        result.requests += 1
        started = time.perf_counter()
        outcome = "ok"
        try:
            return self.session.client.request(method, params, min(self.config.request_timeout, remaining))
        except LspError as error:
            outcome = type(error).__name__
            raise
        finally:
            result.request_log.append({**self._context, "method": method, "outcome": outcome,
                                       "seconds": round(time.perf_counter() - started, 4)})

    def _neighbors(self, node: Symbol, edge: str, result: Expansion, deadline: float) -> list[str]:
        self.session.ensure_open(node)
        try:
            if edge == "referenced_by":
                locations = self._request("textDocument/references", {
                    **self.session.position(node), "context": {"includeDeclaration": False}}, result, deadline)
                return self._map_locations(((loc["uri"], loc["range"]) for loc in locations or []), result)
            if edge in ("called_by", "calls"):
                items = self._request("textDocument/prepareCallHierarchy", self.session.position(node),
                                      result, deadline) or []
                method, side = (("callHierarchy/incomingCalls", "from") if edge == "called_by"
                                else ("callHierarchy/outgoingCalls", "to"))
                calls = []
                for item in items[:1]:
                    calls.extend(self._request(method, {"item": item}, result, deadline) or [])
                return self._map_locations(((c[side]["uri"], c[side]["selectionRange"]) for c in calls), result)
            if edge == "inherits_from":
                return self._base_classes(node, result, deadline)
        except UnsupportedCapability:
            result.events[f"unsupported:{edge}"] += 1
        except LspTimeout:
            result.events[f"timeout:{edge}"] += 1
        except ServerCrashed:
            result.events["server_crashed"] += 1
            raise BudgetExhausted
        except LspError as error:
            result.events[f"error:{edge}:{str(error)[:60]}"] += 1
        return []

    def _base_classes(self, node: Symbol, result: Expansion, deadline: float) -> list[str]:
        """No Python server implements typeHierarchy, so supertypes come from `definition` on each base name."""
        first_line = node.signature.splitlines()[0]
        match = BASE_CLASS_LIST.search(first_line)
        if not match:
            return []
        name_offset = first_line.index(node.name)
        definitions = []
        for base in BASE_NAME.finditer(match.group(1)):
            last_part_offset = match.start(1) + base.start() + base.group().rfind(".") + 1
            character = node.lsp_character + (last_part_offset - name_offset)
            locations = self._request("textDocument/definition",
                                      self.session.position(node, node.lsp_line, character), result, deadline)
            if isinstance(locations, dict):
                locations = [locations]
            definitions.extend((loc["uri"], loc["range"]) for loc in locations or [])
        return self._map_locations(definitions, result)

    def _map_locations(self, locations, result: Expansion) -> list[str]:
        mapped = []
        for uri, span in locations:
            path = uri_to_path(uri)
            try:
                relpath = path.relative_to(self.index.root).as_posix()
            except ValueError:
                result.dropped_outside_index += 1
                continue
            symbol = innermost_symbol(self.index.by_path.get(relpath, []), span["start"]["line"] + 1)
            if symbol is None:
                result.dropped_outside_index += 1
            else:
                mapped.append(symbol.id)
        return mapped
