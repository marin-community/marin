# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest
from marin.processing.tokenize.data_configs import ExistingTokenizedCacheConfig
from marin.training.training import TrainLmOnPodConfig

import experiments.domain_phase_mix.launch_300m_checkpoint_features_canary as canary
import experiments.domain_phase_mix.launch_300m_checkpoint_features_full_swarm as full_swarm
import experiments.domain_phase_mix.launch_300m_mde_vertex_experts as mde_vertex
import experiments.domain_phase_mix.launch_mde_uncheatable_token_features_sharded_300m as token_sharded
import experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level as top_level
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    extract_mde_uncheatable_token_features_300m as token_extract,
)
from experiments.domain_phase_mix.launch_300m_generative_smooth_proxy_evals import (
    REQUEST_FEATURES_PARQUET as TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
)
from experiments.domain_phase_mix.launch_300m_generative_smooth_proxy_evals import (
    _write_request_features as write_teacher_forced_request_features,
)
from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import (
    REQUEST_FEATURES_PARQUET as MCQ_REQUEST_FEATURES_PARQUET,
)
from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import (
    _write_request_features as write_mcq_request_features,
)
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DOMAIN_NAMES,
    PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
    build_top_level_domain_steps,
    build_top_level_domains,
)


class DummyInstance:
    def __init__(self, context: str, target: str):
        self.arguments = (context, target)


def _matrix_csv(tmp_path: Path) -> Path:
    path = tmp_path / "raw_metric_matrix_300m.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_proportional",
                "registry_run_key": "300m_6b:signal:source:baseline_proportional",
                "source_experiment": "source",
                "cohort": "signal",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/source/baseline_proportional-123456",
                "expected_checkpoint_step": 22887,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_build_state_row_for_proportional_checkpoint(monkeypatch, tmp_path: Path) -> None:
    def fake_exact_hf_checkpoint(checkpoint_root: str, expected_step: int) -> str:
        return f"{checkpoint_root}/hf/step-{expected_step}"

    monkeypatch.setattr(canary, "_exact_hf_checkpoint", fake_exact_hf_checkpoint)

    row = canary.build_state_row(
        matrix_csv=_matrix_csv(tmp_path),
        run_name="baseline_proportional",
        text_bundle_keys=("paloma", "uncheatable"),
        text_dataset_names=("paloma/c4_en", "uncheatable_eval/github_python"),
        max_docs_per_dataset=8,
        max_eval_instances=4,
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
    )

    assert row.run_name == "baseline_proportional"
    assert row.launch_decision == "launch"
    assert row.hf_checkpoint_latest.endswith("/hf/step-22887")
    assert row.uses_east5_checkpoint
    assert row.text_dataset_count == 2


def test_build_state_row_from_values_does_not_need_local_matrix(monkeypatch) -> None:
    def fake_exact_hf_checkpoint(checkpoint_root: str, expected_step: int) -> str:
        return f"{checkpoint_root}/hf/step-{expected_step}"

    monkeypatch.setattr(canary, "_exact_hf_checkpoint", fake_exact_hf_checkpoint)

    row = canary._build_state_row_from_values(
        run_name="baseline_proportional",
        registry_key="registry",
        source_experiment="source",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/source/baseline_proportional-123456",
        expected_checkpoint_step=22887,
        text_bundle_keys=("paloma",),
        text_dataset_names=("paloma/c4_en",),
        max_docs_per_dataset=8,
        max_eval_instances=4,
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
    )

    assert row.launch_decision == "launch"
    assert row.registry_key == "registry"
    assert row.hf_checkpoint_latest.endswith("/hf/step-22887")


def test_build_state_row_rejects_non_east5_checkpoint(monkeypatch, tmp_path: Path) -> None:
    matrix_path = _matrix_csv(tmp_path)
    frame = pd.read_csv(matrix_path)
    frame.loc[0, "checkpoint_root"] = "gs://marin-us-central1/checkpoints/source/baseline_proportional-123456"
    frame.to_csv(matrix_path, index=False)
    monkeypatch.setattr(canary, "_exact_hf_checkpoint", lambda root, step: f"{root}/hf/step-{step}")

    row = canary.build_state_row(
        matrix_csv=matrix_path,
        run_name="baseline_proportional",
        text_bundle_keys=("paloma",),
        text_dataset_names=("paloma/c4_en",),
        max_docs_per_dataset=8,
        max_eval_instances=4,
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
    )

    assert row.launch_decision == "defer_checkpoint_not_east5"
    assert not row.eligible


def test_request_cache_check_rejects_empty_files(tmp_path: Path) -> None:
    cache = tmp_path / "requests.jsonl"
    cache.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="request cache is empty"):
        canary._require_nonempty_request_cache(str(cache))


def test_build_feature_steps_use_three_surfaces() -> None:
    spec = canary.CheckpointFeatureCanarySpec(
        run_name="baseline_proportional",
        registry_key="registry",
        source_experiment="source",
        cohort="signal",
        checkpoint_root="gs://marin-us-east5/checkpoints/source/baseline_proportional-123456",
        expected_checkpoint_step=22887,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/source/baseline_proportional-123456/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        uses_east5_checkpoint=True,
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        text_bundle_key="paloma",
        text_dataset_count=1,
        text_dataset_names="paloma/c4_en",
        max_docs_per_dataset=8,
        max_eval_instances=4,
        eligible=True,
        launch_decision="launch",
        step_name="checkpoint_feature_canary/baseline_proportional",
    )

    steps, outputs = canary.build_feature_steps(
        name_prefix="pinlin/test_checkpoint_features",
        spec=spec,
        datasets={},
        teacher_forced_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/teacher/requests.jsonl",
        mcq_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/mcq/requests.jsonl",
    )

    assert len(steps) == 3
    assert set(outputs) == {
        canary.TEXT_FEATURE_SURFACE,
        canary.TEACHER_FORCED_SURFACE,
        canary.MCQ_SURFACE,
    }
    assert all("us-east5" in str(step.config) or canary.TEXT_FEATURE_SURFACE in step.name for step in steps)


def test_teacher_forced_request_features_parquet_roundtrips(tmp_path: Path) -> None:
    rows = [
        {
            "metric_prefix": "teacher_forced/gsm8k_5shot_answer_hash",
            "target": " #### 42",
            "instance": DummyInstance("Question: ...\nAnswer:", " #### 42"),
        }
    ]

    write_teacher_forced_request_features(
        output_path=str(tmp_path),
        eval_key="teacher_canary",
        checkpoint_root="gs://marin-us-east5/checkpoint/hf/step-22887",
        request_rows=rows,
        loglikelihoods=[(-3.0, True)],
    )

    table = pq.read_table(os.path.join(tmp_path, TEACHER_FORCED_REQUEST_FEATURES_PARQUET))
    record = table.to_pylist()[0]
    assert record["request_id"]
    assert record["metric_prefix"] == "teacher_forced/gsm8k_5shot_answer_hash"
    assert record["nll"] == 3.0
    assert record["greedy"] is True


def test_mcq_request_features_parquet_roundtrips(tmp_path: Path) -> None:
    rows = [
        {
            "task_alias": "sciq_5shot",
            "task_name": "sciq",
            "doc_id": 7,
            "choice_idx": 2,
            "choice": "choice",
            "context": "context",
            "target": " choice",
            "choice_bytes": 6,
            "target_bytes": 7,
            "is_gold": True,
            "gold_indices": "2",
            "multi_gold": False,
        }
    ]

    write_mcq_request_features(
        output_path=str(tmp_path),
        eval_key="mcq_canary",
        checkpoint_root="gs://marin-us-east5/checkpoint/hf/step-22887",
        request_rows=rows,
        loglikelihoods=[(-2.5, False)],
    )

    table = pq.read_table(os.path.join(tmp_path, MCQ_REQUEST_FEATURES_PARQUET))
    record = table.to_pylist()[0]
    assert record["request_id"]
    assert record["task_alias"] == "sciq_5shot"
    assert record["choice_idx"] == 2
    assert record["nll"] == 2.5
    assert record["greedy"] is False


def test_full_swarm_state_rows_support_uncapped_request_evals(monkeypatch, tmp_path: Path) -> None:
    matrix = tmp_path / "raw_metric_matrix_300m.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "run_00001",
                "registry_run_key": "300m_6b:signal:source:run_00001",
                "source_experiment": "source",
                "cohort": "signal",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/source/run_00001-abc123",
                "expected_checkpoint_step": 22887,
            },
            {
                "run_name": "run_00002",
                "registry_run_key": "300m_6b:signal:source:run_00002",
                "source_experiment": "source",
                "cohort": "signal",
                "checkpoint_root": "gs://marin-us-east5/checkpoints/source/run_00002-def456",
                "expected_checkpoint_step": 22887,
            },
        ]
    ).to_csv(matrix, index=False)
    monkeypatch.setattr(canary, "_exact_hf_checkpoint", lambda root, step: f"{root}/hf/step-{step}")
    monkeypatch.setattr(full_swarm, "_build_state_row_from_values", canary._build_state_row_from_values)

    rows = full_swarm._state_rows_from_matrix(
        matrix_csv=matrix,
        text_bundle_keys=("paloma",),
        text_dataset_names=("paloma/c4_en",),
        max_docs_per_dataset=512,
        max_eval_instances=None,
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        allow_partial=True,
    )

    assert [row.run_name for row in rows] == ["run_00001", "run_00002"]
    assert all(row.launch_decision == "launch" for row in rows)
    assert all(row.max_docs_per_dataset == 512 for row in rows)
    assert all(row.max_eval_instances is None for row in rows)


def test_full_swarm_state_rows_remap_marin_checkpoint_to_target_region(monkeypatch, tmp_path: Path) -> None:
    matrix = tmp_path / "raw_metric_matrix_300m.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_stratified",
                "registry_run_key": "300m_6b:signal:source:baseline_stratified",
                "source_experiment": "source",
                "cohort": "signal",
                "checkpoint_root": "gs://marin-us-central1/checkpoints/source/baseline_stratified-abc123",
                "expected_checkpoint_step": 22887,
            }
        ]
    ).to_csv(matrix, index=False)
    monkeypatch.setattr(canary, "_exact_hf_checkpoint", lambda root, step: f"{root}/hf/step-{step}")
    monkeypatch.setattr(full_swarm, "_build_state_row_from_values", canary._build_state_row_from_values)

    rows = full_swarm._state_rows_from_matrix(
        matrix_csv=matrix,
        text_bundle_keys=("paloma",),
        text_dataset_names=("paloma/c4_en",),
        max_docs_per_dataset=512,
        max_eval_instances=None,
        default_tpu_type="v5p-8",
        default_tpu_region="us-east5",
        default_tpu_zone="us-east5-a",
        allow_partial=True,
    )

    assert rows[0].checkpoint_root.startswith("gs://marin-us-east5/")
    assert rows[0].hf_checkpoint_latest.startswith("gs://marin-us-east5/")
    assert rows[0].launch_decision == "launch"


def test_full_swarm_feature_steps_use_unique_surface_keys() -> None:
    rows = [
        canary.CheckpointFeatureCanarySpec(
            run_name="run_00001",
            registry_key="registry:run_00001",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/source/run_00001-abc123",
            expected_checkpoint_step=22887,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/source/run_00001-abc123/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            uses_east5_checkpoint=True,
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            text_bundle_key="paloma",
            text_dataset_count=1,
            text_dataset_names="paloma/c4_en",
            max_docs_per_dataset=512,
            max_eval_instances=None,
            eligible=True,
            launch_decision="launch",
            step_name="checkpoint_feature_full_swarm/run_00001",
        ),
        canary.CheckpointFeatureCanarySpec(
            run_name="run_00002",
            registry_key="registry:run_00002",
            source_experiment="source",
            cohort="signal",
            checkpoint_root="gs://marin-us-east5/checkpoints/source/run_00002-def456",
            expected_checkpoint_step=22887,
            hf_checkpoint_latest="gs://marin-us-east5/checkpoints/source/run_00002-def456/hf/step-22887",
            hf_checkpoint_latest_step=22887,
            has_exact_hf_checkpoint=True,
            uses_east5_checkpoint=True,
            launch_tpu_type="v5p-8",
            launch_tpu_region="us-east5",
            launch_tpu_zone="us-east5-a",
            text_bundle_key="paloma",
            text_dataset_count=1,
            text_dataset_names="paloma/c4_en",
            max_docs_per_dataset=512,
            max_eval_instances=None,
            eligible=True,
            launch_decision="launch",
            step_name="checkpoint_feature_full_swarm/run_00002",
        ),
    ]

    steps, outputs = full_swarm.build_full_swarm_feature_steps(
        name_prefix="pinlin/test_checkpoint_features_full_swarm",
        state_rows=rows,
        datasets={},
        teacher_forced_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/teacher/requests.jsonl",
        mcq_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/mcq/requests.jsonl",
    )

    assert len(steps) == 6
    assert set(outputs) == {
        "run_00001::raw_text_loss_features",
        "run_00001::teacher_forced_request_features",
        "run_00001::mcq_request_features",
        "run_00002::raw_text_loss_features",
        "run_00002::teacher_forced_request_features",
        "run_00002::mcq_request_features",
    }


def test_full_swarm_live_launch_rejects_local_matrix_without_state_csv(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)

    with pytest.raises(ValueError, match="--state-csv gs://"):
        full_swarm._validate_live_state_source(
            dry_run=False,
            state_csv=None,
            matrix_csv="experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
        )


def test_full_swarm_live_launch_accepts_gcs_state_csv(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)

    full_swarm._validate_live_state_source(
        dry_run=False,
        state_csv="gs://marin-us-east5/pinlin/state.csv",
        matrix_csv="experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
    )


def test_full_swarm_live_launch_rejects_local_state_csv(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)

    with pytest.raises(ValueError, match="state-csv to be a gs:// path"):
        full_swarm._validate_live_state_source(
            dry_run=False,
            state_csv="experiments/domain_phase_mix/metric_registry/state.csv",
            matrix_csv="gs://marin-us-east5/pinlin/matrix.csv",
        )


def _scored_documents(path: Path, *, offset: float) -> None:
    pd.DataFrame.from_records(
        [
            {
                "request_id": "doc_a",
                "dataset_name": "uncheatable_eval/github_python",
                "score_byte_start": 0,
                "score_byte_end": 4,
                "per_byte_loss": [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0],
                "token_byte_starts": [0, 2],
                "token_byte_ends": [2, 4],
            },
            {
                "request_id": "doc_b",
                "dataset_name": "paloma/c4_en",
                "score_byte_start": 0,
                "score_byte_end": 4,
                "per_byte_loss": [10.0, 11.0, 12.0, 13.0],
                "token_byte_starts": [0, 2],
                "token_byte_ends": [2, 4],
            },
        ]
    ).to_parquet(path, index=False)


def test_sharded_token_feature_extraction_writes_per_run_progress_artifacts(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.parquet"
    run_path = tmp_path / "run.parquet"
    _scored_documents(reference_path, offset=0.0)
    _scored_documents(run_path, offset=1.0)
    feature_index = tmp_path / "feature_surface_index.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_proportional",
                "surface": token_extract.TEXT_SURFACE,
                "artifact_name": token_extract.RAW_ARTIFACT,
                "artifact_uri": str(reference_path),
                "size_bytes": reference_path.stat().st_size,
            },
            {
                "run_name": "run_00001",
                "surface": token_extract.TEXT_SURFACE,
                "artifact_name": token_extract.RAW_ARTIFACT,
                "artifact_uri": str(run_path),
                "size_bytes": run_path.stat().st_size,
            },
        ]
    ).to_csv(feature_index, index=False)

    selected_dir = tmp_path / "selected"
    token_extract.select_token_sketch(
        token_extract.SelectTokenSketchConfig(
            output_path=str(selected_dir),
            feature_index=str(feature_index),
            sample_tokens_per_dataset=2,
            batch_size=1,
        )
    )
    run_dir = tmp_path / "run_output"
    token_extract.extract_run_features(
        token_extract.ExtractRunFeaturesConfig(
            output_path=str(run_dir),
            feature_index=str(feature_index),
            selected_tokens_path=str(selected_dir / token_extract.SELECTED_TOKENS_FILE),
            run_name="run_00001",
            batch_size=1,
            progress_every_batches=1,
        )
    )

    manifest = token_extract.read_json(str(run_dir / token_extract.RUN_MANIFEST_DIR / "run_00001.json"))
    assert manifest["documents"] == 1
    assert manifest["selected_tokens_found"] == 2
    assert Path(manifest["token_path"]).exists()
    assert Path(manifest["document_path"]).exists()


def test_token_sketch_selection_keeps_lowest_hash_tokens_per_dataset(monkeypatch) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "request_id": f"doc_{doc_index}",
                "dataset_name": "uncheatable_eval/github_python",
                "score_byte_start": 0,
                "score_byte_end": 4,
                "per_byte_loss": [1.0, 2.0, 3.0, 4.0],
                "token_byte_starts": [0, 2],
                "token_byte_ends": [2, 4],
            }
            for doc_index in range(20)
        ]
    )

    token_record_creations = 0

    def numeric_hash(token_key: str) -> int:
        request_id, token_index = token_key.split(":")
        doc_index = int(request_id.removeprefix("doc_"))
        return doc_index * 2 + int(token_index)

    def bounded_token_record(**kwargs):
        nonlocal token_record_creations
        token_record_creations += 1
        if token_record_creations > 4:
            raise AssertionError("token sketch selector materialized too many candidate records")
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(token_extract, "stable_hash", numeric_hash)
    monkeypatch.setattr(token_extract, "TokenRecord", bounded_token_record)

    selected = token_extract.choose_token_sketch([frame], sample_tokens_per_dataset=3)

    assert selected["token_key"].tolist() == ["doc_0:0", "doc_0:1", "doc_1:0"]


def test_mde_vertex_feature_index_records_surface_size_bytes(tmp_path: Path) -> None:
    text_dir = tmp_path / "text"
    teacher_dir = tmp_path / "teacher"
    mcq_dir = tmp_path / "mcq"
    text_dir.mkdir()
    teacher_dir.mkdir()
    mcq_dir.mkdir()
    text_artifact = text_dir / "scored_documents.parquet"
    teacher_artifact = teacher_dir / mde_vertex.TEACHER_FORCED_REQUEST_FEATURES_PARQUET
    mcq_artifact = mcq_dir / mde_vertex.MCQ_REQUEST_FEATURES_PARQUET
    text_artifact.write_bytes(b"text")
    teacher_artifact.write_bytes(b"teacher")
    mcq_artifact.write_bytes(b"mcq")

    run_spec = mde_vertex.MdeVertexRunSpec(
        run_order=0,
        run_id=900_000,
        run_name="mde_vertex_cap1_dolma3_wikipedia",
        domain_name="dolma3_wikipedia",
        domain_tokens=10,
        train_tokens=10,
        realized_train_tokens=10,
        num_train_steps=1,
        expected_checkpoint_step=1,
        materialized_epochs=1.0,
        is_control=False,
        target_budget=None,
        data_seed=900_000,
        trainer_seed=None,
        simulated_epoch_subset_seed=None,
        phase_weights={"phase_0": {"dolma3_wikipedia": 1.0}, "phase_1": {"dolma3_wikipedia": 1.0}},
    )
    output_dir = tmp_path / "index"
    mde_vertex.collect_mde_vertex_feature_index(
        mde_vertex.CollectMdeVertexFeatureIndexConfig(
            output_path=str(output_dir),
            run_specs_json=mde_vertex.json.dumps([mde_vertex.asdict(run_spec)], sort_keys=True),
            surface_output_paths={
                mde_vertex._surface_key(run_spec.run_name, mde_vertex.TEXT_FEATURE_SURFACE): str(text_dir),
                mde_vertex._surface_key(run_spec.run_name, mde_vertex.TEACHER_FORCED_SURFACE): str(teacher_dir),
                mde_vertex._surface_key(run_spec.run_name, mde_vertex.MCQ_SURFACE): str(mcq_dir),
            },
            checkpoint_paths={mde_vertex._slug(run_spec.run_name): "gs://marin-us-east5/checkpoints/step-1"},
        )
    )

    frame = pd.read_csv(output_dir / mde_vertex.FEATURE_INDEX_FILE)
    text_row = frame.loc[frame["surface"].eq(mde_vertex.TEXT_FEATURE_SURFACE)].iloc[0]
    assert int(text_row["size_bytes"]) == text_artifact.stat().st_size
    assert set(frame["size_bytes"]) == {
        text_artifact.stat().st_size,
        teacher_artifact.stat().st_size,
        mcq_artifact.stat().st_size,
    }


def test_sharded_token_feature_launcher_builds_parallel_per_run_steps(tmp_path: Path) -> None:
    feature_index = tmp_path / "feature_surface_index.csv"
    pd.DataFrame.from_records(
        [
            {
                "run_name": "baseline_proportional",
                "surface": token_extract.TEXT_SURFACE,
                "artifact_name": token_extract.RAW_ARTIFACT,
                "artifact_uri": "gs://marin-us-east5/features/baseline/scored_documents.parquet",
                "size_bytes": 1,
            },
            {
                "run_name": "run_00001",
                "surface": token_extract.TEXT_SURFACE,
                "artifact_name": token_extract.RAW_ARTIFACT,
                "artifact_uri": "gs://marin-us-east5/features/run_00001/scored_documents.parquet",
                "size_bytes": 1,
            },
            {
                "run_name": "run_00002",
                "surface": token_extract.TEXT_SURFACE,
                "artifact_name": token_extract.RAW_ARTIFACT,
                "artifact_uri": "gs://marin-us-east5/features/run_00002/scored_documents.parquet",
                "size_bytes": 1,
            },
        ]
    ).to_csv(feature_index, index=False)

    worker = token_sharded._resource_config(
        cpu=1.0,
        ram="8g",
        disk="20g",
        region="us-east5",
        zone="us-east5-a",
    )
    shard_steps, collect_step, run_names = token_sharded.build_steps(
        name_prefix="pinlin/test_mde_token_shards",
        feature_index=str(feature_index),
        reference_run="baseline_proportional",
        sample_tokens_per_dataset=2,
        dataset_prefix="uncheatable_eval/",
        batch_size=1,
        progress_every_batches=1,
        max_runs=None,
        include_run_names=["run_00001", "run_00002"],
        worker_resources=worker,
    )

    assert run_names == ["run_00001", "run_00002"]
    assert len(shard_steps) == 3
    assert collect_step.config.run_output_paths.keys() == {"run_00001", "run_00002"}
    assert all(step.resources == worker for step in shard_steps)


def test_mde_vertex_run_specs_default_shape_and_epoch_policy() -> None:
    specs = mde_vertex.build_run_specs()

    cap1 = [spec for spec in specs if not spec.is_control]
    controls = [spec for spec in specs if spec.is_control]
    assert len(specs) == 41
    assert len(cap1) == len(DOMAIN_NAMES) == 39
    assert [spec.domain_name for spec in cap1] == list(DOMAIN_NAMES)
    assert {spec.domain_name for spec in controls} == set(mde_vertex.CONTROL_DOMAINS)
    assert all(spec.target_budget is None for spec in specs)
    assert all(spec.materialized_epochs <= 1.0 for spec in cap1)
    assert all(spec.materialized_epochs > 1.0 for spec in controls)

    for spec in specs:
        assert spec.expected_checkpoint_step == spec.num_train_steps - 1
        for phase_name, weights in spec.phase_weights.items():
            assert set(weights) == set(DOMAIN_NAMES), phase_name
            assert weights[spec.domain_name] == 1.0
            assert sum(value > 0 for value in weights.values()) == 1
            assert sum(weights.values()) == 1.0


INCOMPLETE_EAST5_MERGED_CACHE_DOMAINS = {
    "dolma3_arxiv",
    "dolma3_finemath_3plus",
    "dolma3_wikipedia",
    "dolmino_common_crawl_hq",
    "dolmino_olmocr_pdfs_hq",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolmino_synth_instruction",
    "dolmino_synth_math",
    "dolmino_synth_qa",
    "dolmino_synth_thinking",
}


def test_top_level_domains_reuse_only_complete_east5_merged_runtime_caches() -> None:
    domains = build_top_level_domains(runtime_cache_region="us-east5")
    prebuilt_domains = set(PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["us-east5"])

    assert [domain.name for domain in domains] == list(DOMAIN_NAMES)
    assert prebuilt_domains == set(DOMAIN_NAMES) - INCOMPLETE_EAST5_MERGED_CACHE_DOMAINS
    for domain in domains:
        if domain.name not in prebuilt_domains:
            continue
        assert len(domain.components) == 1
        cache_config = domain.components[0].step_fn()
        assert isinstance(cache_config, ExistingTokenizedCacheConfig), domain.name
        assert cache_config.cache_path.startswith(
            "gs://marin-us-east5/tokenized/merged/dolma3_dolmino_top_level/"
        ), domain.name


def test_incomplete_east5_merged_caches_fall_back_to_existing_source_caches(monkeypatch) -> None:
    monkeypatch.setattr(
        top_level,
        "_resolve_finished_gcs_cache_path",
        lambda cache_pattern: cache_pattern.replace("*", "testhash").rstrip("/"),
    )
    domains = {domain.name: domain for domain in build_top_level_domains(runtime_cache_region="us-east5")}

    for domain_name in INCOMPLETE_EAST5_MERGED_CACHE_DOMAINS:
        domain = domains[domain_name]
        assert domain.components
        for component in domain.components:
            component_output = component.step_fn()
            assert isinstance(component_output, ExistingTokenizedCacheConfig), domain_name
            assert "raw/" not in component_output.cache_path


def test_east5_top_level_domain_prep_steps_are_empty_when_prebuilt_caches_exist() -> None:
    assert build_top_level_domain_steps(runtime_cache_region="us-east5") == {}


def test_mde_vertex_runtime_cache_preflight_rejects_incomplete_status(monkeypatch) -> None:
    monkeypatch.setitem(
        mde_vertex.PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
        "test-region",
        {
            "complete": "gs://marin-us-east5/tokenized/merged/complete",
            "incomplete": "gs://marin-us-east5/tokenized/merged/incomplete",
        },
    )

    with pytest.raises(ValueError, match="incomplete"):
        mde_vertex.validate_prebuilt_runtime_caches(
            "test-region",
            status_reader=lambda path: "SUCCESS" if path.rsplit("/", maxsplit=1)[-1] == "complete" else None,
        )


def test_mde_vertex_runtime_cache_preflight_accepts_success_status(monkeypatch) -> None:
    monkeypatch.setitem(
        mde_vertex.PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
        "test-region",
        {
            "plain": "gs://marin-us-east5/tokenized/merged/plain",
            "json": "gs://marin-us-east5/tokenized/merged/json",
        },
    )

    mde_vertex.validate_prebuilt_runtime_caches(
        "test-region",
        status_reader=lambda path: "SUCCESS" if path.endswith("plain") else '{"status": "SUCCESS"}',
    )


def test_mde_vertex_training_graph_disables_simulated_epoching_and_eval_harness() -> None:
    artifacts = mde_vertex.build_launch_artifacts(
        name_prefix="pinlin/test_mde_vertex",
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        include_domains=("dolma3_wikipedia",),
        skip_controls=True,
        include_features=False,
        include_token_features=False,
        include_dense_compaction=False,
        text_bundle_keys=("paloma",),
        max_docs_per_dataset=1,
        max_eval_instances=1,
        teacher_forced_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/teacher/requests.jsonl",
        mcq_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/mcq/requests.jsonl",
        sample_tokens_per_dataset=2,
        token_batch_size=1,
        token_progress_every_batches=1,
        dense_dtype="float32",
        worker_cpu=1.0,
        worker_ram="8g",
        worker_disk="20g",
    )

    assert len(artifacts.run_specs) == 1
    assert len(artifacts.training_steps) == 1
    config = artifacts.training_steps[0].config
    assert isinstance(config, TrainLmOnPodConfig)
    spec = artifacts.run_specs[0]
    assert int(config.train_config.trainer.num_train_steps) == spec.num_train_steps
    assert int(config.train_config.hf_save_steps) == spec.num_train_steps
    assert config.train_config.eval_harness is None
    assert config.env_vars is not None
    assert config.env_vars["MARIN_PREFIX"] == "gs://marin-us-east5"
    assert config.env_vars[SKIP_EVAL_HARNESS_ENV_VAR] == "1"


def test_mde_vertex_feature_graph_depends_on_training_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(mde_vertex, "build_text_feature_datasets", lambda _bundles: {})

    artifacts = mde_vertex.build_launch_artifacts(
        name_prefix="pinlin/test_mde_vertex_features",
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        include_domains=("dolma3_wikipedia",),
        skip_controls=True,
        include_features=True,
        include_token_features=False,
        include_dense_compaction=False,
        text_bundle_keys=("paloma",),
        max_docs_per_dataset=1,
        max_eval_instances=1,
        teacher_forced_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/teacher/requests.jsonl",
        mcq_request_cache_uri="gs://marin-us-east5/raw/eval-datasets/mcq/requests.jsonl",
        sample_tokens_per_dataset=2,
        token_batch_size=1,
        token_progress_every_batches=1,
        dense_dtype="float32",
        worker_cpu=1.0,
        worker_ram="8g",
        worker_disk="20g",
    )

    assert len(artifacts.feature_steps) == 3
    assert artifacts.feature_index_step is not None
    checkpoint_paths = artifacts.feature_index_step.config.checkpoint_paths
    assert list(checkpoint_paths) == ["mde_vertex_cap1_dolma3_wikipedia"]
    assert checkpoint_paths["mde_vertex_cap1_dolma3_wikipedia"].step == artifacts.training_steps[0]
