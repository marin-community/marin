# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guest entry point for the original BM25 retrieval and snippet protocol."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from transformers import AutoTokenizer

from .scorers.searchers.bm25_searcher import BM25Searcher

ROOT = Path("/opt/browsecomp")


def main() -> None:
    request = json.load(sys.stdin)
    # A contract from a different index, tokenizer or protocol must never use this image.
    state = json.loads((ROOT / "state.json").read_text())
    if request["state"] != state:
        raise ValueError("BrowseComp provider state differs from its pinned image")
    searcher = BM25Searcher(Namespace(index_path=str(ROOT / "index/bm25")))
    name, arguments = request["name"], request["arguments"]
    if name == "get_document" and state["include_get_document"]:
        result = searcher.get_document(arguments["docid"])
    elif name == "search":
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizer", local_files_only=True)
        result = []
        # Match the pinned upstream searcher/tools.py register_tools implementation.
        for candidate in searcher.search(arguments["query"], state["top_k"]):
            tokens = tokenizer.encode(candidate["text"], add_special_tokens=False)
            snippet = (
                tokenizer.decode(tokens[: state["snippet_max_tokens"]], skip_special_tokens=True)
                if len(tokens) > state["snippet_max_tokens"]
                else candidate["text"]
            )
            result.append({"docid": candidate["docid"], "score": candidate["score"], "snippet": snippet})
    else:
        raise ValueError("Tool is not enabled by the pinned BrowseComp contract")
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
