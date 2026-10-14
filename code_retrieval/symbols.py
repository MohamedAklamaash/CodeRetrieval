from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Symbol:
    id: str
    path: str
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    lsp_line: int
    lsp_character: int
    signature: str
    docstring: str
    source: str
    parent_id: str | None
    file_hash: str

    def to_record(self, with_source: bool = False) -> dict:
        record = asdict(self)
        if not with_source:
            del record["source"]
        return record


def make_symbol_id(path: str, qualified_name: str, kind: str, occurrence: int) -> str:
    base = f"{path}::{qualified_name}::{kind}"
    return base if occurrence == 0 else f"{base}#{occurrence}"


def innermost_symbol(symbols: list[Symbol], line: int) -> Symbol | None:
    containing = [s for s in symbols if s.start_line <= line <= s.end_line]
    if not containing:
        return None
    return min(containing, key=lambda s: s.end_line - s.start_line)
