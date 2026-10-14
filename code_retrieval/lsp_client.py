import json
import shutil
import subprocess
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from urllib.parse import unquote, urlparse

CAPABILITY_FOR_METHOD = {
    "textDocument/definition": "definitionProvider",
    "textDocument/references": "referencesProvider",
    "textDocument/prepareCallHierarchy": "callHierarchyProvider",
    "callHierarchy/incomingCalls": "callHierarchyProvider",
    "callHierarchy/outgoingCalls": "callHierarchyProvider",
    "textDocument/prepareTypeHierarchy": "typeHierarchyProvider",
}


class LspError(Exception):
    pass


class UnsupportedCapability(LspError):
    pass


class LspTimeout(LspError):
    pass


class ServerCrashed(LspError):
    pass


def uri_to_path(uri: str) -> Path:
    return Path(unquote(urlparse(uri).path))


class LspClient:
    """JSON-RPC 2.0 over stdio, one long-lived server process per workspace."""

    def __init__(self, command: list[str], root: Path, settings: dict, log_path: Path | None = None):
        if shutil.which(command[0]) is None:
            raise LspError(f"language server executable not found on PATH: {command[0]}")
        self.root = root.resolve()
        self.settings = settings
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=self.root)
        self.next_id = 1
        self.pending: dict[int, Future] = {}
        self.write_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.server_capabilities: dict = {}
        self.open_documents: dict[str, int] = {}
        self.log = open(log_path, "ab") if log_path else None
        threading.Thread(target=self._read_messages, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        for line in self.process.stderr:
            if self.log:
                self.log.write(line)

    def _read_messages(self):
        stream = self.process.stdout
        while True:
            headers = {}
            while True:
                line = stream.readline()
                if not line:
                    self._fail_pending(ServerCrashed(f"server exited with code {self.process.poll()}"))
                    return
                line = line.decode("utf-8").strip()
                if not line:
                    break
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            body = stream.read(int(headers["content-length"]))
            self._dispatch(json.loads(body))

    def _dispatch(self, message: dict):
        if "method" in message and "id" in message:
            self._answer_server_request(message)
        elif "id" in message:
            with self.state_lock:
                future = self.pending.pop(message["id"], None)
            if future is None:
                return
            if "error" in message:
                future.set_exception(LspError(f"{message['error'].get('code')}: {message['error'].get('message')}"))
            else:
                future.set_result(message.get("result"))

    def _answer_server_request(self, message: dict):
        if message["method"] == "workspace/configuration":
            result = [self._setting(item.get("section", "")) for item in message["params"]["items"]]
        else:
            result = None
        self._write({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _setting(self, section: str):
        value = self.settings
        for part in filter(None, section.split(".")):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _fail_pending(self, error: Exception):
        with self.state_lock:
            pending, self.pending = self.pending, {}
        for future in pending.values():
            future.set_exception(error)

    def _write(self, message: dict):
        body = json.dumps(message).encode("utf-8")
        with self.write_lock:
            if self.process.poll() is not None:
                raise ServerCrashed(f"server exited with code {self.process.returncode}")
            self.process.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            self.process.stdin.flush()

    def request(self, method: str, params: dict, timeout: float):
        capability = CAPABILITY_FOR_METHOD.get(method)
        if capability and self.server_capabilities and not self.server_capabilities.get(capability):
            raise UnsupportedCapability(f"server does not advertise {capability} for {method}")
        future: Future = Future()
        with self.state_lock:
            request_id = self.next_id
            self.next_id += 1
            self.pending[request_id] = future
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            with self.state_lock:
                self.pending.pop(request_id, None)
            self.notify("$/cancelRequest", {"id": request_id})
            raise LspTimeout(f"{method} exceeded {timeout:.1f}s") from None

    def notify(self, method: str, params: dict):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def initialize(self, timeout: float):
        root_uri = self.root.as_uri()
        result = self.request("initialize", {
            "processId": None,
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
            "capabilities": {
                "workspace": {"configuration": True, "workspaceFolders": True},
                "textDocument": {
                    "synchronization": {"didSave": False},
                    "definition": {"linkSupport": False},
                    "references": {},
                    "callHierarchy": {},
                    "typeHierarchy": {},
                },
            },
        }, timeout)
        self.server_capabilities = result.get("capabilities", {})
        self.notify("initialized", {})
        self.notify("workspace/didChangeConfiguration", {"settings": self.settings})
        return result

    def sync_document(self, path: Path, text: str):
        """Opens the file, or pushes new contents if it changed since it was opened."""
        uri = path.as_uri()
        version = self.open_documents.get(uri)
        if version is None:
            self.notify("textDocument/didOpen", {"textDocument": {
                "uri": uri, "languageId": "python", "version": 1, "text": text}})
            self.open_documents[uri] = 1
        else:
            self.open_documents[uri] = version + 1
            self.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": version + 1},
                "contentChanges": [{"text": text}]})

    def close_document(self, path: Path):
        uri = path.as_uri()
        if self.open_documents.pop(uri, None) is not None:
            self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def shutdown(self, timeout: float = 5.0):
        try:
            if self.process.poll() is None:
                self.request("shutdown", {}, timeout)
                self.notify("exit", {})
                self.process.wait(timeout=timeout)
        except (LspError, subprocess.TimeoutExpired):
            pass
        finally:
            if self.process.poll() is None:
                self.process.kill()
            if self.log:
                self.log.close()
