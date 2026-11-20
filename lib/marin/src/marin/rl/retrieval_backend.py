# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve retrieval in the immutable image interpreter and its pinned dependencies."""

import argparse
import base64
import importlib
import json
import threading
from argparse import Namespace
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


class OpenResearcherIndex:
    """Keep the source's searcher and tokenizer loaded for the life of one run."""

    def __init__(self, root: Path, state: dict, document_cache_size: int):
        # These optional image dependencies are pinned by the service's immutable image.
        backend = importlib.import_module("browsecomp.retrieval")
        self.searcher = backend.BM25Searcher(Namespace(index_path=str(root / "index/bm25")))
        self.tokenizer = backend.AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        self.state = state
        self._document = lru_cache(maxsize=document_cache_size)(
            lambda docid: json.dumps(self.searcher.get_document(docid), ensure_ascii=False).encode()
        )

    def search(self, query: str) -> list[dict]:
        result = []
        for candidate in self.searcher.search(query, self.state["top_k"]):
            tokens = self.tokenizer.encode(candidate["text"], add_special_tokens=False)
            snippet = (
                self.tokenizer.decode(tokens[: self.state["snippet_max_tokens"]], skip_special_tokens=True)
                if len(tokens) > self.state["snippet_max_tokens"]
                else candidate["text"]
            )
            result.append({"docid": candidate["docid"], "score": candidate["score"], "snippet": snippet})
        return result

    def document(self, docid: str) -> bytes:
        return self._document(docid)


def retrieval_handler(index: OpenResearcherIndex, state: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, result: object) -> None:
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self.respond({"state": state})

        def do_POST(self) -> None:
            if self.path not in {"/search", "/document"}:
                self.send_error(404)
                return
            length = int(self.headers["Content-Length"])
            if not 0 < length <= 1024 * 1024:
                self.send_error(413)
                return
            request = json.loads(self.rfile.read(length))
            if request["state"] != state:
                self.send_error(409)
                return
            arguments = request["arguments"]
            if self.path == "/search":
                if request["name"] != "search" or set(arguments) != {"query"} or not isinstance(arguments["query"], str):
                    self.send_error(400)
                    return
                self.respond(index.search(arguments["query"]))
                return
            if (
                request["name"] != "get_document"
                or set(arguments) != {"docid"}
                or not isinstance(arguments["docid"], str)
            ):
                self.send_error(400)
                return
            offset = request["offset"]
            content = index.document(arguments["docid"])
            if type(offset) is not int or not 0 <= offset <= len(content):
                self.send_error(400)
                return
            chunk = content[offset : offset + CHUNK_BYTES]
            self.respond({"data": base64.b64encode(chunk).decode(), "done": offset + len(chunk) == len(content)})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return Handler


class RetrievalServer(ThreadingHTTPServer):
    """Bound concurrent requests to the run's explicit CPU service allocation."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], concurrency: int):
        self.slots = threading.BoundedSemaphore(concurrency)
        super().__init__(address, handler)

    def process_request(self, request, client_address) -> None:
        self.slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_bytes())
    with importlib.import_module("browsecomp.tool").source_diagnostics():
        index = OpenResearcherIndex(args.root, state, args.concurrency)
    with RetrievalServer(("0.0.0.0", args.port), retrieval_handler(index, state), args.concurrency) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
