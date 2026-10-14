import time
from pathlib import Path

from code_retrieval.config import LspConfig
from code_retrieval.indexing import RepositoryIndex
from code_retrieval.lsp_client import LspClient
from code_retrieval.symbols import Symbol


def basedpyright_settings(root: Path, excludes: list[str]) -> dict:
    extra_paths = [d for d in ("src", "lib") if (root / d).is_dir()]
    analysis = {
        "diagnosticMode": "openFilesOnly",
        "typeCheckingMode": "off",
        "autoSearchPaths": True,
        "useLibraryCodeForTypes": False,
        "extraPaths": extra_paths,
        "exclude": excludes,
    }
    return {"basedpyright": {"analysis": analysis}, "python": {"analysis": analysis}}


class LanguageServerSession:
    """A persistent server for one workspace whose open documents track the index's file hashes."""

    def __init__(self, index: RepositoryIndex, config: LspConfig, log_path: Path | None = None):
        self.index = index
        self.config = config
        started = time.perf_counter()
        self.client = LspClient(config.command, index.root,
                                basedpyright_settings(index.root, config.excludes), log_path)
        self.client.initialize(config.initialize_timeout)
        self.startup_seconds = time.perf_counter() - started
        self.synced_hashes: dict[str, str] = {}

    @property
    def capabilities(self) -> dict:
        return self.client.server_capabilities

    def refresh(self, index: RepositoryIndex):
        """Pushes changed files to the server and closes deleted ones after a re-index."""
        current = index.file_hashes()
        for path in list(self.synced_hashes):
            if path not in current:
                self.client.close_document(index.root / path)
                del self.synced_hashes[path]
        for path, file_hash in current.items():
            if path in self.synced_hashes and self.synced_hashes[path] != file_hash:
                self.client.sync_document(index.root / path, (index.root / path).read_text(errors="replace"))
                self.synced_hashes[path] = file_hash
        self.index = index

    def ensure_open(self, symbol: Symbol):
        if self.synced_hashes.get(symbol.path) != symbol.file_hash:
            path = self.index.root / symbol.path
            self.client.sync_document(path, path.read_text(errors="replace"))
            self.synced_hashes[symbol.path] = symbol.file_hash

    def position(self, symbol: Symbol, line: int | None = None, character: int | None = None) -> dict:
        return {
            "textDocument": {"uri": (self.index.root / symbol.path).as_uri()},
            "position": {"line": symbol.lsp_line if line is None else line,
                         "character": symbol.lsp_character if character is None else character},
        }

    def close(self):
        self.client.shutdown()
