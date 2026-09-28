# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The content-type step end to end on a local pool: order, columns, and the pin checks."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData

from experiments.datakit.cluster.quality.fast_transformer import domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.content_type import predict_content_types
from experiments.datakit.cluster.quality.fast_transformer.quality_model import ContentTypePin
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import normalize_embeddings

LABELS = ("prose", "code", "other")
DIM = 8
SHARD = "part-00000-of-00001.parquet"


@pytest.fixture(autouse=True)
def marin_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))


def make_classifier(root: Path, labels: tuple[str, ...] = LABELS) -> ContentTypePin:
    rng = np.random.default_rng(0)
    path = root / "models" / "mlp.npz"
    path.parent.mkdir(parents=True)
    weights = {
        "w1": rng.standard_normal((DIM, 512)),
        "b1": np.zeros(512),
        "w2": rng.standard_normal((512, 256)) * 0.05,
        "b2": np.zeros(256),
        "w3": rng.standard_normal((256, len(labels))),
        "b3": np.zeros(len(labels)),
    }
    np.savez(path, labels=np.array(labels), **{k: v.astype(np.float32) for k, v in weights.items()})
    return ContentTypePin(
        name="mlp",
        model_path="models/mlp.npz",
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        labels=labels,
    )


def make_source(root: Path, ids: list[str], embedding_ids: list[str]) -> tuple[NormalizedData, str, np.ndarray]:
    text_dir, embedding_dir = root / "normalized", root / "harrier"
    text_dir.mkdir()
    embedding_dir.mkdir()
    pq.write_table(pa.table({"id": ids, "text": [f"text of {i}" for i in ids]}), text_dir / SHARD)
    rows = np.random.default_rng(1).integers(-127, 127, (len(embedding_ids), DIM), dtype=np.int8)
    embeddings = pa.FixedSizeListArray.from_arrays(pa.array(rows.ravel(), type=pa.int8()), DIM)
    pq.write_table(pa.table({"id": embedding_ids, "embedding": embeddings}), embedding_dir / SHARD)
    normalized = NormalizedData(main_output_dir=str(text_dir), dup_output_dir="", counters={})
    return normalized, str(embedding_dir), rows


def run(tmp_path: Path, normalized: NormalizedData, embedding_dir: str, pin: ContentTypePin):
    return predict_content_types(
        str(tmp_path / "out"), normalized=normalized, embedding_dir=embedding_dir, classifier=pin, max_workers=1
    )


def test_types_follow_the_normalized_order_with_the_full_distribution(tmp_path):
    pin = make_classifier(tmp_path)
    normalized, embedding_dir, rows = make_source(tmp_path, ["c", "a", "b"], ["c", "a", "b"])

    artifact = run(tmp_path, normalized, embedding_dir, pin)

    out = pq.read_table(tmp_path / "out" / SHARD).to_pydict()
    model, _ = domain_mlp.load(str(tmp_path / "models" / "mlp.npz"))
    expected = domain_mlp.predict_probabilities(model, normalize_embeddings(rows))
    assert out["id"] == ["c", "a", "b"]
    assert np.asarray(out["content_type_probs"]) == pytest.approx(expected, abs=1e-6)
    assert out["content_type"] == [LABELS[i] for i in expected.argmax(axis=1)]
    assert out["content_type_prob"] == pytest.approx(expected.max(axis=1).tolist(), abs=1e-6)
    assert artifact.labels == list(LABELS)
    assert artifact.counters["content_type/docs_typed"] == 3


def test_an_embedding_shard_in_another_order_fails_the_source(tmp_path):
    pin = make_classifier(tmp_path)
    normalized, embedding_dir, _ = make_source(tmp_path, ["a", "b"], ["b", "a"])

    with pytest.raises(Exception, match="do not carry the normalized shard's ids"):
        run(tmp_path, normalized, embedding_dir, pin)


def test_weights_that_are_not_the_pinned_ones_are_refused(tmp_path):
    pin = replace(make_classifier(tmp_path), model_sha256="0" * 64)
    normalized, embedding_dir, _ = make_source(tmp_path, ["a"], ["a"])

    with pytest.raises(Exception, match="digests to"):
        run(tmp_path, normalized, embedding_dir, pin)


def test_a_head_with_other_labels_is_refused(tmp_path):
    pin = replace(make_classifier(tmp_path), labels=("code", "prose", "other"))
    normalized, embedding_dir, _ = make_source(tmp_path, ["a"], ["a"])

    with pytest.raises(Exception, match="emits"):
        run(tmp_path, normalized, embedding_dir, pin)
