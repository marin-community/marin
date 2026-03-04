# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.embed_everything.exp3049_match_halves import (
    HalfDocument,
    evaluate_retrieval,
    split_into_halves,
)


def test_split_into_halves():
    docs = [
        {"id": "doc-0", "text": "AAAABBBB"},
        {"id": "doc-1", "text": "CCCCDDDD"},
    ]
    halves = split_into_halves(docs)
    assert len(halves) == 4
    assert halves[0] == HalfDocument(doc_id="doc-0", half="a", text="AAAA")
    assert halves[1] == HalfDocument(doc_id="doc-0", half="b", text="BBBB")
    assert halves[2] == HalfDocument(doc_id="doc-1", half="a", text="CCCC")
    assert halves[3] == HalfDocument(doc_id="doc-1", half="b", text="DDDD")


def test_evaluate_retrieval_perfect_embeddings():
    """When matching halves have identical embeddings, retrieval accuracy should be 1.0."""
    # 4 docs → 8 halves. Each pair (a,b) shares the same embedding.
    halves = []
    embeddings_list = []
    rng = np.random.default_rng(42)

    for i in range(4):
        vec = rng.normal(0, 1, 16).astype(np.float32)
        vec /= np.linalg.norm(vec)
        halves.append(HalfDocument(doc_id=f"doc-{i}", half="a", text=f"text-{i}-a"))
        halves.append(HalfDocument(doc_id=f"doc-{i}", half="b", text=f"text-{i}-b"))
        # Both halves get the same embedding
        embeddings_list.append(vec)
        embeddings_list.append(vec + rng.normal(0, 0.001, 16).astype(np.float32))

    embeddings = np.array(embeddings_list)
    # Re-normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    accuracy = evaluate_retrieval(embeddings, halves, windows=[1])
    # With near-identical embeddings for each pair, top-1 accuracy should be perfect
    assert accuracy["top-1"] == 1.0


def test_evaluate_retrieval_random_embeddings():
    """Random embeddings should have low top-1 retrieval accuracy."""
    rng = np.random.default_rng(42)
    n_docs = 50
    halves = []
    for i in range(n_docs):
        halves.append(HalfDocument(doc_id=f"doc-{i}", half="a", text=f"a-{i}"))
        halves.append(HalfDocument(doc_id=f"doc-{i}", half="b", text=f"b-{i}"))

    embeddings = rng.normal(0, 1, (n_docs * 2, 32)).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    accuracy = evaluate_retrieval(embeddings, halves, windows=[1])
    # With random embeddings and 100 halves, top-1 accuracy should be ~1/99 ≈ 0.01
    assert accuracy["top-1"] < 0.1
