# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The bucket step end to end on a local pool: order, calibration, and the co-partition guard."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.bucket import bucket_quality_scores
from experiments.datakit.cluster.quality.fast_transformer.calibrate import apply_calibration
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    CALIBRATION_FILE,
    QualityPin,
    calibration_sha256,
)

KNOTS = {
    "default": {"xk": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "yk": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    # Code is graded generously: a raw 0.5 lands in the top bucket.
    "types": {"code": {"xk": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0], "yk": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}},
}
BASENAMES = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]


@pytest.fixture(autouse=True)
def local_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    with set_current_client(LocalClient()):
        yield


def write_shard(directory: Path, name: str, columns: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), directory / name)


def make_source(root: Path, shards: dict[str, list[str]]) -> NormalizedData:
    main = root / "normalized" / "outputs" / "main"
    for name, ids in shards.items():
        write_shard(main, name, {"id": ids, "text": [f"text of {i}" for i in ids]})
    return NormalizedData(
        main_output_dir=str(main), dup_output_dir=str(root / "normalized" / "outputs" / "dups"), counters={}
    )


def run_bucket(tmp_path: Path, normalized: NormalizedData, pin: QualityPin, max_workers: int = 2):
    return bucket_quality_scores(
        str(tmp_path / "out"),
        source="src",
        normalized=normalized,
        scores_dir=str(tmp_path / "scores"),
        content_type_dir=str(tmp_path / "types"),
        quality_model=pin,
        max_workers=max_workers,
    )


def make_pin(root: Path, knots: dict = KNOTS) -> QualityPin:
    model_dir = root / "models" / "pin"
    model_dir.mkdir(parents=True)
    (model_dir / "m.eqx").write_bytes(b"weights")
    (model_dir / "m_remap.json").write_bytes(b"{}")
    (model_dir / "m_meta.json").write_bytes(b"{}")
    (model_dir / CALIBRATION_FILE).write_text(json.dumps(knots))
    return QualityPin(
        name="pin",
        model_path="models/pin",
        model_sha256="unused",
        calibration_sha256=calibration_sha256(str(model_dir)),
        tokenizer="tok",
    )


def test_bucket_writes_the_normalized_order_with_per_type_buckets(tmp_path):
    """Scores and types are stored in other orders; the output follows the normalized shard."""
    normalized = make_source(tmp_path, {BASENAMES[0]: ["b", "a", "c"], BASENAMES[1]: ["z"]})
    scores = tmp_path / "scores"
    write_shard(scores, BASENAMES[0], {"id": ["a", "c", "b"], "score": np.array([0.5, 0.95, 0.15], dtype=np.float32)})
    write_shard(scores, BASENAMES[1], {"id": ["z"], "score": np.array([0.5], dtype=np.float32)})
    types = tmp_path / "types"
    write_shard(types, BASENAMES[0], {"id": ["c", "a", "b"], "content_type": ["prose", "code", "other"]})
    write_shard(types, BASENAMES[1], {"id": ["z"], "content_type": ["prose"]})

    artifact = run_bucket(tmp_path, normalized, make_pin(tmp_path))

    first = pq.read_table(tmp_path / "out" / BASENAMES[0]).to_pydict()
    assert first["id"] == ["b", "a", "c"], "rows follow the normalized shard, not the score or type shard"
    assert first["source"] == ["src"] * 3
    assert first["content_type"] == ["other", "code", "prose"]
    assert first["raw_score"] == pytest.approx([0.15, 0.5, 0.95])
    expected = apply_calibration(np.array(first["raw_score"]), np.array(first["content_type"], dtype=object), KNOTS)
    assert first["score"] == pytest.approx(expected.tolist())
    # ``other`` has no curve of its own and falls back to the default; code does not.
    assert first["quality_bucket"] == [0, 4, 4]
    assert first["quality_bucket"] == np.digitize(expected, BUCKET_EDGES).tolist()
    second = pq.read_table(tmp_path / "out" / BASENAMES[1]).to_pydict()
    assert second["id"] == ["z"] and second["quality_bucket"] == [2]

    assert artifact.main_output_dir == str(tmp_path / "out")
    assert artifact.samples_output_dir is None
    assert artifact.counters["quality/docs_bucketed"] == 4
    assert artifact.counters["quality/shards"] == 2


def test_a_document_without_a_score_fails_the_source(tmp_path):
    normalized = make_source(tmp_path, {BASENAMES[0]: ["a", "b"]})
    scores = tmp_path / "scores"
    write_shard(scores, BASENAMES[0], {"id": ["a"], "score": np.array([0.5], dtype=np.float32)})
    types = tmp_path / "types"
    write_shard(types, BASENAMES[0], {"id": ["a", "b"], "content_type": ["prose", "prose"]})

    with pytest.raises(Exception, match="have no row"):
        run_bucket(tmp_path, normalized, make_pin(tmp_path), max_workers=1)


def test_a_leaf_missing_a_shard_fails_before_any_task_runs(tmp_path):
    normalized = make_source(tmp_path, {BASENAMES[0]: ["a"], BASENAMES[1]: ["b"]})
    scores = tmp_path / "scores"
    write_shard(scores, BASENAMES[0], {"id": ["a"], "score": np.array([0.5], dtype=np.float32)})
    types = tmp_path / "types"
    for name in BASENAMES:
        write_shard(types, name, {"id": ["a"], "content_type": ["prose"]})

    with pytest.raises(ValueError, match="not co-partitioned"):
        run_bucket(tmp_path, normalized, make_pin(tmp_path))


def test_a_calibration_that_is_not_the_pinned_one_is_refused(tmp_path):
    normalized = make_source(tmp_path, {BASENAMES[0]: ["a"]})
    scores = tmp_path / "scores"
    write_shard(scores, BASENAMES[0], {"id": ["a"], "score": np.array([0.5], dtype=np.float32)})
    types = tmp_path / "types"
    write_shard(types, BASENAMES[0], {"id": ["a"], "content_type": ["prose"]})
    pin = make_pin(tmp_path)
    (tmp_path / "models" / "pin" / CALIBRATION_FILE).write_text(json.dumps(KNOTS["default"]))

    with pytest.raises(Exception, match="different calibration"):
        run_bucket(tmp_path, normalized, pin, max_workers=1)
