import re
from collections import defaultdict

from code_retrieval.indexing import RepositoryIndex
from code_retrieval.symbols import Symbol

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset("""
a an and are as at be but by for from has have if in into is it its of on or that the this to was were will with
self cls def class return none true false not import none pass else elif raise try except
""".split())
MAX_BODY_CHARS = 6000


def tokenize(text: str) -> list[str]:
    tokens = []
    for identifier in IDENTIFIER.findall(text):
        lowered = identifier.lower()
        parts = [p.lower() for chunk in identifier.split("_") for p in CAMEL_PART.findall(chunk)]
        if len(parts) > 1 and lowered not in STOPWORDS:
            tokens.append(lowered)
        tokens.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return tokens


def member_signatures(index: RepositoryIndex) -> dict[str, list[str]]:
    members = defaultdict(list)
    for symbol in index.symbols:
        if symbol.parent_id and symbol.kind == "method":
            members[symbol.parent_id].append(symbol.signature)
    return members


def body_text(symbol: Symbol, members: dict[str, list[str]]) -> str:
    if symbol.kind == "class":
        return "\n".join(members.get(symbol.id, []))
    return symbol.source[:MAX_BODY_CHARS]


def lexical_tokens(symbol: Symbol, members: dict[str, list[str]], field_weights: dict[str, int]) -> list[str]:
    fields = {
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "path": symbol.path,
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "body": body_text(symbol, members),
    }
    tokens = []
    for field_name, weight in field_weights.items():
        tokens.extend(tokenize(fields[field_name]) * weight)
    return tokens


def dense_text(symbol: Symbol, members: dict[str, list[str]]) -> str:
    header = f"{symbol.path} :: {symbol.qualified_name} ({symbol.kind})"
    parts = [header, symbol.signature, symbol.docstring[:600], body_text(symbol, members)[:1500]]
    return "\n".join(p for p in parts if p)
