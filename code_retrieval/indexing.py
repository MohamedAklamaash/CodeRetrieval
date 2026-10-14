import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node, Parser

from retriever.parser_setup import get_languages
from code_retrieval.symbols import Symbol, make_symbol_id

SKIPPED_DIRS = {".git", ".hg", ".tox", ".venv", "venv", "node_modules", "build", "dist", "__pycache__", "tests"}
MAX_FILE_BYTES = 600_000


@dataclass
class RepositoryIndex:
    root: Path
    symbols: list[Symbol]
    by_id: dict[str, Symbol] = field(init=False)
    by_path: dict[str, list[Symbol]] = field(init=False)
    skipped_files: dict[str, str] = field(default_factory=dict)
    unparsable_files: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.by_id = {s.id: s for s in self.symbols}
        self.by_path = defaultdict(list)
        for s in self.symbols:
            self.by_path[s.path].append(s)

    def file_hashes(self) -> dict[str, str]:
        return {path: symbols[0].file_hash for path, symbols in self.by_path.items()}


def is_test_path(relpath: str) -> bool:
    parts = Path(relpath).parts
    filename = parts[-1]
    if filename == "conftest.py" or filename.startswith("test_") or filename.endswith("_test.py"):
        return True
    # A top-level "testing/" dir holds tests (pytest), but nested ones like sympy/testing are library code.
    return "tests" in parts[:-1] or parts[0] == "testing"


def iter_python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                path = Path(dirpath) / filename
                relpath = path.relative_to(root).as_posix()
                if not is_test_path(relpath):
                    yield path, relpath


def utf16_column(line_bytes: bytes, byte_column: int) -> int:
    return len(line_bytes[:byte_column].decode("utf-8", errors="replace").encode("utf-16-le")) // 2


def node_text(code: bytes, node: Node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def docstring_of(code: bytes, body: Node | None) -> str:
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_children and first.named_children[0].type == "string":
        return node_text(code, first.named_children[0]).strip("\"' \n\r\t")
    return ""


class PythonExtractor:
    def __init__(self):
        self.parser = Parser(get_languages()["python"])

    def extract(self, code: bytes, relpath: str) -> list[Symbol]:
        tree = self.parser.parse(code)
        if tree.root_node.has_error and not tree.root_node.named_children:
            raise SyntaxError(f"tree-sitter could not parse {relpath}")
        file_hash = hashlib.sha256(code).hexdigest()
        lines = code.split(b"\n")
        occurrences = Counter()
        symbols: list[Symbol] = []

        def add(qualified_name, name, kind, outer: Node, name_node: Node | None, signature, docstring, source, parent_id):
            key = (qualified_name, kind)
            symbol_id = make_symbol_id(relpath, qualified_name, kind, occurrences[key])
            occurrences[key] += 1
            if name_node is None:
                lsp_line, lsp_character = 0, 0
            else:
                lsp_line = name_node.start_point[0]
                lsp_character = utf16_column(lines[lsp_line], name_node.start_point[1])
            symbols.append(Symbol(
                id=symbol_id, path=relpath, name=name, qualified_name=qualified_name, kind=kind,
                start_line=outer.start_point[0] + 1, end_line=outer.end_point[0] + 1,
                lsp_line=lsp_line, lsp_character=lsp_character,
                signature=signature, docstring=docstring, source=source,
                parent_id=parent_id, file_hash=file_hash,
            ))
            return symbol_id

        module_name = relpath.removesuffix(".py").replace("/", ".")
        top_level_code = [node_text(code, child) for child in tree.root_node.named_children
                          if self._definition(child) is None]
        module_id = add(module_name, module_name.rsplit(".", 1)[-1], "module", tree.root_node, None, "",
                        docstring_of(code, tree.root_node), "\n".join(top_level_code), None)

        def visit(container: Node, prefix: str, parent_id: str, in_class: bool):
            for child in container.named_children:
                definition = self._definition(child)
                if definition is None:
                    continue
                name_node = definition.child_by_field_name("name")
                body = definition.child_by_field_name("body")
                name = node_text(code, name_node)
                qualified_name = f"{prefix}.{name}" if prefix else name
                header_end = body.start_byte if body is not None else definition.end_byte
                signature = code[definition.start_byte:header_end].decode("utf-8", errors="replace").strip()
                if definition.type == "class_definition":
                    kind = "class"
                else:
                    kind = "method" if in_class else "function"
                symbol_id = add(qualified_name, name, kind, child, name_node, signature,
                                docstring_of(code, body), node_text(code, child), parent_id)
                if body is not None:
                    visit(body, qualified_name, symbol_id, definition.type == "class_definition")

        visit(tree.root_node, "", module_id, False)
        return symbols

    @staticmethod
    def _definition(node: Node) -> Node | None:
        if node.type in ("class_definition", "function_definition"):
            return node
        if node.type == "decorated_definition":
            return node.child_by_field_name("definition")
        return None


def index_repository(root: Path) -> RepositoryIndex:
    root = root.resolve()
    extractor = PythonExtractor()
    symbols: list[Symbol] = []
    skipped: dict[str, str] = {}
    unparsable: list[str] = []
    for path, relpath in iter_python_files(root):
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped[relpath] = f"size {size} > {MAX_FILE_BYTES}"
            continue
        try:
            symbols.extend(extractor.extract(path.read_bytes(), relpath))
        except SyntaxError:
            unparsable.append(relpath)
    return RepositoryIndex(root=root, symbols=symbols, skipped_files=skipped, unparsable_files=unparsable)
