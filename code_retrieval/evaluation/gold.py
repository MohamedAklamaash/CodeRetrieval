import re
from dataclasses import dataclass, field

from code_retrieval.indexing import RepositoryIndex, is_test_path
from code_retrieval.symbols import innermost_symbol

FILE_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Gold:
    files: list[str]
    symbols: list[str]
    module_level: list[str] = field(default_factory=list)
    unmappable: list[str] = field(default_factory=list)
    files_outside_index: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return self.__dict__.copy()


def changed_pre_image_lines(patch: str) -> dict[str, set[int]]:
    """Old-file line numbers touched by each file's hunks; pure insertions anchor to the preceding old line."""
    changed: dict[str, set[int]] = {}
    current = None
    old_line = 0
    for line in patch.splitlines():
        header = FILE_HEADER.match(line)
        if header:
            current = header.group(1)
            changed.setdefault(current, set())
            continue
        hunk = HUNK_HEADER.match(line)
        if hunk:
            old_line = int(hunk.group(1))
            continue
        if current is None or line.startswith(("---", "+++", "index ", "new file", "deleted file")):
            continue
        if line.startswith("-"):
            changed[current].add(old_line)
            old_line += 1
        elif line.startswith("+"):
            changed[current].add(max(old_line - 1, 1))
        elif line.startswith(" "):
            old_line += 1
    return changed


def derive_gold(patch: str, index: RepositoryIndex) -> Gold:
    gold = Gold(files=[], symbols=[])
    for path, lines in changed_pre_image_lines(patch).items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        gold.files.append(path)
        file_symbols = index.by_path.get(path)
        if not file_symbols:
            gold.files_outside_index.append(path)
            continue
        for line in sorted(lines):
            symbol = innermost_symbol(file_symbols, line)
            if symbol is None:
                gold.unmappable.append(f"{path}:{line}")
            elif symbol.kind == "module":
                if symbol.id not in gold.module_level:
                    gold.module_level.append(symbol.id)
            elif symbol.id not in gold.symbols:
                gold.symbols.append(symbol.id)
    return gold
