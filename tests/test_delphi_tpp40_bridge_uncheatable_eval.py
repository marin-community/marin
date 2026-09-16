# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval


def test_frozen_bridge_contract_and_deployments() -> None:
    contract = bridge_eval._load_acceptance_contract()

    assert contract["bridge"]["run_orders"] == list(bridge_eval.BRIDGE_RUN_ORDERS)
    assert bridge_eval.BRIDGE_SIDES["east5"].training_tpu_type == "v5p-8"
    assert bridge_eval.BRIDGE_SIDES["europe"].training_tpu_type == "v6e-8"
    assert {side.evaluator_zone for side in bridge_eval.BRIDGE_SIDES.values()} == {
        "us-east5-b",
        "europe-west4-a",
    }
    assert bridge_eval.EVALUATOR_TPU_TYPE == "v6e-8"


def test_uncheatable_component_contract_is_exact() -> None:
    assert bridge_eval.EXPECTED_UNCHEATABLE_NAMES == (
        "uncheatable_eval/wikipedia_english",
        "uncheatable_eval/github_python",
        "uncheatable_eval/github_cpp",
        "uncheatable_eval/bbc_news",
        "uncheatable_eval/arxiv_physics",
        "uncheatable_eval/arxiv_computer_science",
        "uncheatable_eval/ao3_english",
    )


def test_uncheatable_metrics_requires_every_component() -> None:
    raw_metrics = {
        f"eval/{name}/bpb": float(index + 1) for index, name in enumerate(bridge_eval.EXPECTED_UNCHEATABLE_NAMES)
    }
    raw_metrics["eval/uncheatable_eval/macro_bpb"] = 4.0

    component_metrics, macro_bpb = bridge_eval._uncheatable_metrics(raw_metrics)

    assert tuple(component_metrics) == bridge_eval.EXPECTED_UNCHEATABLE_NAMES
    assert macro_bpb == 4.0

    del raw_metrics[f"eval/{bridge_eval.EXPECTED_UNCHEATABLE_NAMES[-1]}/bpb"]
    with pytest.raises(ValueError, match="Missing Uncheatable metric"):
        bridge_eval._uncheatable_metrics(raw_metrics)


def test_uncheatable_metrics_rejects_reported_macro_mismatch() -> None:
    raw_metrics = {f"eval/{name}/bpb": 1.0 for name in bridge_eval.EXPECTED_UNCHEATABLE_NAMES}
    raw_metrics["eval/uncheatable_eval/macro_bpb"] = 1.01

    with pytest.raises(ValueError, match="macro mismatch"):
        bridge_eval._uncheatable_metrics(raw_metrics)


def test_uncheatable_metrics_reconstructs_levanter_float32_macro() -> None:
    source_values = [
        0.90123456,
        0.90234567,
        0.90345678,
        0.90456789,
        0.90567891,
        0.90678912,
        0.90789123,
    ]
    component_values = [float(value) for value in np.asarray(source_values, dtype=np.float32)]
    values = np.asarray(component_values, dtype=np.float32)
    reported_macro = float(np.mean(values, where=np.ones(values.shape, dtype=bool)))
    float64_macro = sum(component_values) / len(component_values)
    assert abs(float64_macro - reported_macro) > 1e-9
    raw_metrics = {
        f"eval/{name}/bpb": value
        for name, value in zip(bridge_eval.EXPECTED_UNCHEATABLE_NAMES, component_values, strict=True)
    }
    raw_metrics["eval/uncheatable_eval/macro_bpb"] = reported_macro

    _, observed_macro = bridge_eval._uncheatable_metrics(raw_metrics)

    assert observed_macro == reported_macro


def test_uncheatable_metrics_matches_levanter_masked_reduction() -> None:
    component_values = np.asarray(
        [0.90123456, 0.90234567, 0.90345678, 0.90456789, 0.90567891, 0.90678912, 0.90789123],
        dtype=np.float32,
    )
    full_tag_values = np.concatenate(
        (np.asarray([8.0], dtype=np.float32), component_values, np.asarray([9.0], dtype=np.float32))
    )
    mask = np.asarray([False, True, True, True, True, True, True, True, False])
    reported_macro = float(np.mean(full_tag_values, where=mask))
    raw_metrics = {
        f"eval/{name}/bpb": float(value)
        for name, value in zip(bridge_eval.EXPECTED_UNCHEATABLE_NAMES, component_values, strict=True)
    }
    raw_metrics["eval/uncheatable_eval/macro_bpb"] = reported_macro

    _, observed_macro = bridge_eval._uncheatable_metrics(raw_metrics)

    assert observed_macro == reported_macro


def test_uncheatable_metrics_rejects_missing_or_nonnumeric_macro() -> None:
    raw_metrics: dict[str, object] = {f"eval/{name}/bpb": 1.0 for name in bridge_eval.EXPECTED_UNCHEATABLE_NAMES}
    with pytest.raises(ValueError, match="Missing reported Uncheatable macro"):
        bridge_eval._uncheatable_metrics(raw_metrics)

    raw_metrics["eval/uncheatable_eval/macro_bpb"] = None
    with pytest.raises(ValueError, match="Non-numeric reported Uncheatable macro"):
        bridge_eval._uncheatable_metrics(raw_metrics)


def test_checkpoint_metadata_requires_exact_complete_permanent_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-21855"
    checkpoint.mkdir()
    assert bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855) is None

    metadata = {"step": 21855, "timestamp": "2026-08-30T00:00:00+00:00", "is_temporary": False}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="without tensor payload"):
        bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855)

    (checkpoint / "manifest.ocdbt").write_text("manifest")
    metadata_result = bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855)
    assert metadata_result is not None
    observed_metadata, observed_sha256 = metadata_result
    assert observed_metadata == metadata
    assert len(observed_sha256) == 64


def test_checkpoint_metadata_rejects_wrong_step_and_temporary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-21855"
    checkpoint.mkdir()
    (checkpoint / "manifest.ocdbt").write_text("manifest")
    (checkpoint / "metadata.json").write_text(
        json.dumps({"step": 21854, "timestamp": "2026-08-30T00:00:00+00:00", "is_temporary": False})
    )
    with pytest.raises(ValueError, match="metadata step changed"):
        bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855)

    (checkpoint / "metadata.json").write_text(
        json.dumps({"step": 21855, "timestamp": "2026-08-30T00:00:00+00:00", "is_temporary": True})
    )
    with pytest.raises(ValueError, match="marked temporary"):
        bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855)


def test_completed_result_is_bound_to_current_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-21855"
    checkpoint.mkdir()
    (checkpoint / "manifest.ocdbt").write_text("manifest")
    metadata = {"step": 21855, "timestamp": "2026-08-30T00:00:00+00:00", "is_temporary": False}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    metadata_result = bridge_eval._checkpoint_metadata(str(checkpoint), expected_step=21855)
    assert metadata_result is not None
    _, metadata_sha256 = metadata_result

    eval_output = tmp_path / "eval"
    eval_output.mkdir()
    component_metrics = {name: 1.0 for name in bridge_eval.EXPECTED_UNCHEATABLE_NAMES}
    (eval_output / bridge_eval.RESULT_FILE).write_text(
        json.dumps(
            {
                "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
                "side": "europe",
                "run_order": 2,
                "checkpoint_step": 21855,
                "checkpoint_path": str(checkpoint),
                "checkpoint_metadata_sha256": metadata_sha256,
                "validation_payload_sha256": "1" * 64,
                "total_trainable_params": 1,
                "model_config_sha256": "0" * 64,
                "component_bpb": component_metrics,
                "macro_bpb": 1.0,
            }
        )
    )
    record = bridge_eval.EvalRecord(
        side="europe",
        run_order=2,
        run_name="fit_002",
        checkpoint_step=21855,
        training_output_path=str(tmp_path),
        checkpoint_path=str(checkpoint),
        checkpoint_ready=True,
        eval_output_path=str(eval_output),
        eval_already_succeeded=True,
        validation_payload_sha256="1" * 64,
    )

    bridge_eval._validate_completed_result(record)

    metadata["timestamp"] = "2026-08-30T01:00:00+00:00"
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint metadata changed"):
        bridge_eval._validate_completed_result(record)
