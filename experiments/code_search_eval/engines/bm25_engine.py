# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["bm25s>=0.2", "numpy"]
# ///

"""BM25 lexical index over line-window chunks. A strong sparse baseline that isolates
"good ranked lexical retrieval" from "dense embeddings": if BM25 already matches the
dense engine, the embeddings are not buying much on this corpus.
"""

import sys

sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])

import bm25s

from experiments.code_search_eval.common import Hit, collect_chunks, read_chunk_meta, run_engine_cli, write_chunk_meta


def build_index(repo_root: str, index_dir: str) -> None:
    meta, corpus = collect_chunks(repo_root)
    if not corpus:
        raise ValueError(f"no indexable chunks under {repo_root}")
    tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(index_dir)
    write_chunk_meta(index_dir, meta)


def query_index(repo_root: str, index_dir: str, queries: list[dict], k: int) -> list[dict]:
    retriever = bm25s.BM25.load(index_dir, load_corpus=False)
    meta = read_chunk_meta(index_dir)
    results = []
    for q in queries:
        qt = bm25s.tokenize(q["query"], stopwords="en", show_progress=False)
        idx, scores = retriever.retrieve(qt, k=min(k, len(meta)), show_progress=False)
        hits = []
        for rank in range(idx.shape[1]):
            m = meta[int(idx[0, rank])]
            hits.append(Hit(m["file"], m["start_line"], m["end_line"], float(scores[0, rank])).to_json())
        results.append({"query_id": q["query_id"], "hits": hits})
    return results


if __name__ == "__main__":
    run_engine_cli(build_index, query_index)
