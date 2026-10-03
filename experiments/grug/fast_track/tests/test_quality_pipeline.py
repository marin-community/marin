# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from levanter.store.cache import SerialCacheWriter, TreeCache
from marin.execution.artifact import write_artifact
from marin.execution.lazy import StepContext
from marin.execution.lazy import run as run_artifacts
from numpy.typing import NDArray

from experiments.grug.fast_track.contracts import ResolvedTrainingBudget
from experiments.grug.fast_track.quality import (
    LabelledEmbedding,
    LabelSplit,
    PoolRequirements,
    QualityScorer,
    RidgeHeadConfig,
    label_split,
)
from experiments.grug.fast_track.quality_cli import main as quality_cli
from experiments.grug.fast_track.quality_pipeline import (
    PinnedFile,
    QualityBundle,
    QualityConfig,
    QualityData,
    QualitySpec,
    QualityTrainingSource,
    SelectionMethod,
    build_quality_data,
    prepare_quality_data,
)


@dataclass(frozen=True)
class _NegativeScorer:
    def scores(self, embeddings: NDArray) -> NDArray[np.float64]:
        return -np.asarray(embeddings, dtype=np.float64)[:, 0]


class _NegativeHeadConfig:
    def __init__(self, revision: str = "test-v1", split_seed: int = 7):
        self.revision = revision
        self.split_seed = split_seed
        self.runtime_state = object()

    @property
    def identity(self) -> Mapping[str, str]:
        return {"implementation": "test-negative", "revision": self.revision}

    def fit(self, rows: Sequence[LabelledEmbedding]) -> QualityScorer:
        assert rows
        assert all(label_split(row.duplicate_group, self.split_seed) is LabelSplit.TRAIN for row in rows)
        return _NegativeScorer()


@dataclass(frozen=True)
class _WrongShapeScorer:
    def scores(self, embeddings: NDArray) -> NDArray[np.float64]:
        return np.zeros((len(embeddings), 1), dtype=np.float64)


@dataclass(frozen=True)
class _WrongShapeHeadConfig:
    @property
    def identity(self) -> Mapping[str, str]:
        return {"implementation": "test-wrong-shape", "revision": "test-v1"}

    def fit(self, rows: Sequence[LabelledEmbedding]) -> QualityScorer:
        return _WrongShapeScorer()


def _pin(path) -> PinnedFile:
    return PinnedFile(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def quality_config(tmp_path) -> QualityConfig:
    labels_path = tmp_path / "labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source": "labels",
                    "id": str(i),
                    "duplicate_group": f"label-{i}",
                    "embedding": [float(i)],
                    "label": float(i),
                }
                for i in range(100)
            ]
        ),
        labels_path,
    )
    cache_path = tmp_path / "source-cache"
    tokens = [np.full(length, i + 1, dtype="<i4") for i, length in enumerate((10, 30, 20, 40))]
    with SerialCacheWriter(str(cache_path), {"input_ids": np.zeros(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": values} for values in tokens])
    pool_path = tmp_path / "pool.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source": "pool",
                    "id": str(i),
                    "duplicate_group": f"pool-{i}",
                    "embedding": [float(i)],
                    "token_count": len(values),
                    "content_type": "text",
                    "language": "en",
                    "cache_path": str(cache_path),
                    "cache_row": i,
                    "token_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "incumbent_score": float(3 - i),
                }
                for i, values in enumerate(tokens)
            ]
        ),
        pool_path,
    )
    bundle = QualityBundle(
        tokenizer="passthrough",
        embedding_revision="embedding-v1",
        embedding_scale=1.0,
        label_revision="glm-v1",
        incumbent_revision="incumbent-v1",
        baseline_recipe="baseline-v1",
        sampling_method="production-weighted-hash",
        pool_seed=3,
        split_seed=7,
        labels=(_pin(labels_path),),
        pool=(_pin(pool_path),),
        quality_bin_edges=(0.0, 1.5, 3.0),
        requirements=PoolRequirements(
            {"pool": 1.0},
            0.01,
            ("low", "high"),
            2,
            4,
            0.5,
        ),
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json())
    return QualityConfig(
        QualitySpec(_pin(bundle_path), SelectionMethod.CANDIDATE, RidgeHeadConfig(0.01), fraction=0.4, tie_seed=0),
        str(tmp_path / "selection"),
    )


def test_quality_preparation_scores_selects_and_reloads_exact_token_documents(quality_config):
    result = prepare_quality_data(quality_config)
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})
    rows = cache.get_batch_sync([0])

    np.testing.assert_array_equal(rows[0]["input_ids"], np.full(40, 4))
    with open(result.report_path) as stream:
        report = json.load(stream)
    assert report["pool"]["quality_bin_documents"] == {"low": 2, "high": 2}
    assert report["selection"]["requested_tokens"] == 40
    assert report["selection"]["selected_tokens"] == 40
    assert report["selected_documents"] == 1
    assert report["token_overlap"] == 0


@pytest.mark.parametrize("method", [SelectionMethod.INCUMBENT, SelectionMethod.RANDOM])
def test_quality_controls_materialize_their_selection(quality_config, method):
    result = prepare_quality_data(replace(quality_config, spec=replace(quality_config.spec, selection_method=method)))
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})
    rows = cache.get_batch_sync(range(len(cache)))

    assert {int(row["input_ids"][0]) for row in rows} == {1, 2}
    assert sum(len(row["input_ids"]) for row in rows) == 40
    with open(result.report_path) as stream:
        report = json.load(stream)
    assert report["selection_method"] == method.value
    assert report["token_overlap"] == 1.0


def test_quality_cache_shuffles_documents_before_block_shuffle(quality_config):
    result = prepare_quality_data(replace(quality_config, spec=replace(quality_config.spec, fraction=1.0)))
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})
    rows = cache.get_batch_sync(range(len(cache)))

    assert [int(row["input_ids"][0]) for row in rows] == [3, 1, 2, 4]
    with open(result.report_path) as stream:
        report = json.load(stream)
    with open(report["selected_ids_path"]) as stream:
        assert [json.loads(line)["id"] for line in stream] == ["2", "0", "1", "3"]


def test_quality_training_rejects_cutoff_overshoot_as_pool_capacity(quality_config, tmp_path):
    spec = replace(quality_config.spec, fraction=0.1)
    result = prepare_quality_data(replace(quality_config, spec=spec))
    artifact_path = str(tmp_path / "artifact")
    write_artifact(result, artifact_path)
    step = build_quality_data(spec, version="test-dev")
    step = replace(step, override_path=artifact_path)

    with pytest.raises(ValueError, match="too small"):
        QualityTrainingSource(step).data_config(
            ctx=StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(step,)),
            validation=(),
            tokenizer="passthrough",
            budget=ResolvedTrainingBudget(1, 20, 1),
        )


def test_quality_preparation_rejects_changed_token_join(quality_config, tmp_path):
    cache_path = tmp_path / "source-cache"
    with SerialCacheWriter(str(cache_path), {"input_ids": np.zeros(0, dtype=np.int32)}) as writer:
        writer.write_batch([{"input_ids": np.full(length, 99, dtype=np.int32)} for length in (10, 30, 20, 40)])

    with pytest.raises(ValueError, match="token reference differs"):
        prepare_quality_data(quality_config)
    assert not (tmp_path / "selection" / "tokens" / "shard_ledger.json").exists()


def test_quality_candidate_identity_changes_artifact_path(quality_config):
    baseline = build_quality_data(quality_config.spec, version="test-dev")
    candidate = build_quality_data(replace(quality_config.spec, head=RidgeHeadConfig(0.5)), version="test-dev")
    incumbent = build_quality_data(
        replace(quality_config.spec, selection_method=SelectionMethod.INCUMBENT), version="test-dev"
    )
    custom_head = build_quality_data(
        replace(quality_config.spec, head=_NegativeHeadConfig(revision="test-v1")), version="test-dev"
    )
    revised_custom_head = build_quality_data(
        replace(quality_config.spec, head=_NegativeHeadConfig(revision="test-v2")), version="test-dev"
    )

    assert len({baseline.name, candidate.name, incumbent.name, custom_head.name, revised_custom_head.name}) == 5
    assert baseline.fingerprint() != candidate.fingerprint()
    assert custom_head.fingerprint() != revised_custom_head.fingerprint()


def test_quality_custom_head_uses_fixed_training_split_and_changes_selection(quality_config, tmp_path, monkeypatch):
    spec = replace(quality_config.spec, head=_NegativeHeadConfig(split_seed=7))
    step = build_quality_data(spec, version="test-dev")
    prefix = str(tmp_path / "artifacts")
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    (result,) = run_artifacts(step, max_concurrent=1)
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})
    rows = cache.get_batch_sync(range(len(cache)))
    with open(result.report_path) as stream:
        report = json.load(stream)
    with open(report["selected_ids_path"]) as stream:
        selected_ids = {json.loads(line)["id"] for line in stream}

    assert selected_ids == {"0", "1"}
    assert {int(row["input_ids"][0]) for row in rows} == {1, 2}
    assert report["head"] == spec.head.identity
    assert report["development"]["documents"] > 0

    spec.head.revision = "mutated-after-build"
    run_config = step.build_config(StepContext.for_run(output_path=step.path(prefix), prefix=prefix, deps=step.deps))
    with pytest.raises(ValueError, match="identity changed after artifact construction"):
        step.run(run_config)


def test_quality_custom_head_must_return_one_score_per_embedding(quality_config):
    spec = replace(quality_config.spec, head=_WrongShapeHeadConfig())

    with pytest.raises(ValueError, match="one finite score per embedding"):
        prepare_quality_data(replace(quality_config, spec=spec))


def test_quality_cli_artifact_loads_through_python_api(quality_config, tmp_path):
    ridge_head = RidgeHeadConfig(0.01)
    spec = replace(quality_config.spec, head=ridge_head)
    prefix = str(tmp_path / "artifacts")
    version = "2026.10.03"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.grug.fast_track.quality_cli",
            "--bundle",
            spec.bundle.path,
            "--bundle-sha256",
            spec.bundle.sha256,
            "--fraction",
            str(spec.fraction),
            "--regularization",
            str(ridge_head.regularization),
            "--tie-seed",
            str(spec.tie_seed),
            "--version",
            version,
            "--prepare-only",
            "--run",
        ],
        env={**os.environ, "MARIN_PREFIX": prefix, "MARIN_FINGERPRINT_STRICT": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    step = build_quality_data(spec, version=version)
    result = QualityData.raw_load(step.path(prefix))
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})

    assert result.requested_tokens == 40
    np.testing.assert_array_equal(cache.get_batch_sync([0])[0]["input_ids"], np.full(40, 4))


def test_quality_training_cli_requires_run_id(quality_config):
    spec = quality_config.spec
    result = CliRunner().invoke(
        quality_cli,
        [
            "--bundle",
            spec.bundle.path,
            "--bundle-sha256",
            spec.bundle.sha256,
            "--version",
            "2026.10.03",
            "--run",
        ],
    )

    assert result.exit_code == 2
    assert "--run-id is required when training" in result.output
