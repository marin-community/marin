# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image-only dependencies for the original offline BM25 retrieval protocol."""

from argparse import Namespace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .scorers.searchers.bm25_searcher import BM25Searcher


def retrieve(request: dict[str, Any], state: dict[str, Any], root: Path) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Run the source BM25 query and apply its original snippet procedure."""
    searcher = BM25Searcher(Namespace(index_path=str(root / "index/bm25")))
    name, arguments = request["name"], request["arguments"]
    if name == "get_document" and state["include_get_document"]:
        return searcher.get_document(arguments["docid"])
    if name != "search":
        raise ValueError("Tool is not enabled by the pinned BrowseComp contract")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
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
    return result
