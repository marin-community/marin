# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import experiments.domain_phase_mix.launch_300m_checkpoint_features_canary as canary
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
