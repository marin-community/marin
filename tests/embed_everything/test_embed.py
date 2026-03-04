# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest

from experiments.embed_everything.embed import embed_documents


def _write_jsonl(path: str, docs: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


@pytest.mark.slow
def test_embed_documents_produces_embeddings(tmp_path):
    """Embeds documents and saves .npz with correct shapes."""
    docs = [
        {"id": f"doc-{i}", "text": f"This is test document number {i}.", "label": f"label_{i % 3}"} for i in range(10)
    ]
    input_path = str(tmp_path / "input" / "docs.jsonl")
    _write_jsonl(input_path, docs)

    output_path = str(tmp_path / "output")
    result = embed_documents(
        output_path=output_path,
        input_path=input_path,
        model_name="DatologyAI/luxical-one",
        batch_size=4,
    )

    assert os.path.exists(result.path)
    data = np.load(result.path)
    assert data["embeddings"].shape[0] == 10
    assert data["embeddings"].shape[1] > 0
    assert len(data["ids"]) == 10
    assert len(data["labels"]) == 10
