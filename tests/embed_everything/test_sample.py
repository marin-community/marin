# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

from experiments.embed_everything.sample import sample_documents


def _write_jsonl(path: str, docs: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def _make_docs(source: str, n: int) -> list[dict]:
    return [{"id": f"{source}-{i}", "text": f"Document {i} from {source}.", "source": source} for i in range(n)]


def test_sample_documents_basic(tmp_path):
    """Samples the requested number of docs per source."""
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _write_jsonl(str(source_a / "data.jsonl"), _make_docs("a", 50))
    _write_jsonl(str(source_b / "data.jsonl"), _make_docs("b", 50))

    output_path = str(tmp_path / "output")
    result = sample_documents(
        output_path=output_path,
        source_paths={"label_a": str(source_a / "*.jsonl"), "label_b": str(source_b / "*.jsonl")},
        n_per_source=10,
        seed=42,
    )

    assert os.path.exists(result.path)
    with open(result.path) as f:
        docs = [json.loads(line) for line in f]

    assert len(docs) == 20
    labels = {d["label"] for d in docs}
    assert labels == {"label_a", "label_b"}
    assert sum(1 for d in docs if d["label"] == "label_a") == 10
    assert sum(1 for d in docs if d["label"] == "label_b") == 10


def test_sample_documents_caps_at_available(tmp_path):
    """If a source has fewer docs than requested, returns all available."""
    source = tmp_path / "small_source"
    _write_jsonl(str(source / "data.jsonl"), _make_docs("small", 3))

    output_path = str(tmp_path / "output")
    result = sample_documents(
        output_path=output_path,
        source_paths={"small": str(source / "*.jsonl")},
        n_per_source=100,
        seed=42,
    )

    with open(result.path) as f:
        docs = [json.loads(line) for line in f]

    assert len(docs) == 3


def test_sample_documents_deterministic(tmp_path):
    """Same seed produces same sample."""
    source = tmp_path / "source"
    _write_jsonl(str(source / "data.jsonl"), _make_docs("det", 50))

    out1 = str(tmp_path / "out1")
    out2 = str(tmp_path / "out2")

    r1 = sample_documents(output_path=out1, source_paths={"det": str(source / "*.jsonl")}, n_per_source=10, seed=42)
    r2 = sample_documents(output_path=out2, source_paths={"det": str(source / "*.jsonl")}, n_per_source=10, seed=42)

    with open(r1.path) as f:
        docs1 = [json.loads(line) for line in f]
    with open(r2.path) as f:
        docs2 = [json.loads(line) for line in f]

    assert [d["id"] for d in docs1] == [d["id"] for d in docs2]
