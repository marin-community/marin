# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastembed>=0.3", "numpy"]
# ///

"""Dense retrieval over line-window chunks with a local embedding model. This is the
reference implementation of the "automatic semantic code index" the context-efficiency
report recommends: build once, embed the query, cosine-rank chunks.

Embeddings come from ``fastembed`` (ONNX on CPU — no torch, no GPU), so the index
builds locally on a dev box. The model is set by ``CSE_EMBED_MODEL``; the default is a
small general model, and ``build``/``query`` must use the same one (its name is
persisted in the index).
"""

import os
import sys

sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])

import numpy as np
from fastembed import TextEmbedding

from experiments.code_search_eval.common import Hit, collect_chunks, read_chunk_meta, run_engine_cli, write_chunk_meta

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 256


def _model_name(index_dir: str) -> str:
    marker = os.path.join(index_dir, "model.txt")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            return fh.read().strip()
    return os.environ.get("CSE_EMBED_MODEL", DEFAULT_MODEL)


def _embed_passages(model: TextEmbedding, texts: list[str]) -> np.ndarray:
    vecs = np.asarray(list(model.embed(texts, batch_size=EMBED_BATCH)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs


def build_index(repo_root: str, index_dir: str) -> None:
    name = os.environ.get("CSE_EMBED_MODEL", DEFAULT_MODEL)
    meta, texts = collect_chunks(repo_root)
    if not texts:
        raise ValueError(f"no indexable chunks under {repo_root}")
    model = TextEmbedding(model_name=name)
    vecs = _embed_passages(model, texts)
    np.save(os.path.join(index_dir, "embeddings.npy"), vecs)
    write_chunk_meta(index_dir, meta)
    with open(os.path.join(index_dir, "model.txt"), "w", encoding="utf-8") as fh:
        fh.write(name)


def query_index(repo_root: str, index_dir: str, queries: list[dict], k: int) -> list[dict]:
    vecs = np.load(os.path.join(index_dir, "embeddings.npy"))
    meta = read_chunk_meta(index_dir)
    model = TextEmbedding(model_name=_model_name(index_dir))
    q_texts = [q["query"] for q in queries]
    q_vecs = np.asarray(list(model.query_embed(q_texts)), dtype=np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8
    sims = q_vecs @ vecs.T  # (n_queries, n_chunks) cosine, both L2-normalized
    results = []
    for i, q in enumerate(queries):
        top = np.argsort(-sims[i])[:k]
        hits = [
            Hit(meta[j]["file"], meta[j]["start_line"], meta[j]["end_line"], float(sims[i, j])).to_json() for j in top
        ]
        results.append({"query_id": q["query_id"], "hits": hits})
    return results


if __name__ == "__main__":
    run_engine_cli(build_index, query_index)
