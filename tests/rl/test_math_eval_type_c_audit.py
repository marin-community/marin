# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import yaml

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.type_c_audit import (
    audit_native_attempts,
    audit_native_history,
    validate_checkpoint_receipts,
    validate_saved_source,
)


def fixture(minibatches=1, runner="standard"):
    output = {
        "terminal_manifest_uri": "s3://fixture/run/terminal.json",
        "checkpoint_root": "s3://fixture/run/checkpoints",
        "resolved_config_uri": "s3://fixture/run/resolved.json",
    }
    request = {
        "run_id": "fixture",
        "attempt_id": "attempt",
        "completion_mode": "checkpoint",
        "seed": 17,
        "runtime": {"commit": "a" * 40},
        "output": output,
    }
    request["config_yaml"] = yaml.safe_dump(
        {
            "entrypoint": runner,
            "trainer": {
                "train_batch_size": 64 * minibatches,
                "policy_mini_batch_size": 64,
                "max_steps": 96 // minibatches,
                "eval_interval": 32 // minibatches,
                "update_epochs_per_batch": 1,
                **({"fully_async": {"first_token_admission": True}} if runner == "fully_async" else {}),
            },
        }
    )
    native_step = 96 // minibatches
    receipt_uri = "s3://fixture/run/receipts/attempt.json"
    digest = audit.canonical_sha({"schema_version": 2, "request": request, "receipt_uri": receipt_uri})
    checkpoint = {
        "global_step": native_step,
        "checkpoint_path": output["checkpoint_root"] + f"/global_step_{native_step}",
        "files": [],
    }
    receipt = {
        "schema_version": 1,
        "run_id": "fixture",
        "attempt_id": "attempt",
        "request_fingerprint": digest,
        "completion_mode": "checkpoint",
        "global_step": native_step,
        "checkpoint": checkpoint,
    }
    envelope = {"schema_version": 2, "request": request, "execution": {"max_retries": 0}}
    terminal = envelope | {
        "response": {
            "state": "succeeded",
            "iris_job_state": "succeeded",
            "failure": None,
            "runtime": request["runtime"],
            "run_id": "fixture",
            "attempt_id": "attempt",
            "training": {
                "global_step": native_step,
                "checkpoint": checkpoint,
                "receipt_uri": receipt_uri,
                "resolved_config_uri": output["resolved_config_uri"],
            },
        }
    }
    completion = {
        "mode": "checkpoint",
        "run_id": "fixture",
        "attempt_id": "attempt",
        "request_fingerprint": digest,
        "receipt_uri": receipt_uri,
    }
    resolved = {"hydra_args": [f"++trainer.completion.{key}={value}" for key, value in completion.items()]}
    return envelope, terminal, deepcopy(terminal), receipt, resolved


@pytest.mark.parametrize("minibatches", [1, 2, 4, 8, 16])
def test_checkpoint_step_is_not_optimizer_update_count(minibatches):
    result = validate_checkpoint_receipts(*fixture(minibatches), minibatches=minibatches)
    assert result["global_step"] == 96 // minibatches
    assert result["saved_successful_updates_validation_required"]
    assert result["native_checkpoint_file_validation_required"]


@pytest.mark.parametrize("poison", ["step", "batch", "async", "attempt", "counter", "digest"])
def test_receipts_reject_wrong_schedule_identity_or_checkpoint(poison):
    e, t, a, r, c = fixture(2)
    if poison == "step":
        r["global_step"] = 96
    elif poison in {"batch", "async"}:
        cfg = yaml.safe_load(e["request"]["config_yaml"])
        if poison == "batch":
            cfg["trainer"]["train_batch_size"] = 64
        else:
            cfg["entrypoint"] = "fully_async"
        e["request"]["config_yaml"] = yaml.safe_dump(cfg)
    elif poison == "attempt":
        a["response"]["attempt_id"] = "other"
    elif poison == "counter":
        t["response"]["training"]["global_step"] = 96
    else:
        r["request_fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        validate_checkpoint_receipts(e, t, a, r, c, minibatches=2)


@pytest.mark.parametrize("minibatches,offset", [(1, 9), (2, 6)])
def test_saved_source_uses_real_batch_geometry_and_successful_updates(minibatches, offset):
    contract = {
        "algorithm": "torch-randperm-seed-plus-epoch-v1",
        "seed": 17,
        "dataset_sha256": "a" * 64,
        "rows": 1918,
        "prompts_per_step": 64 * minibatches,
        "updates_per_batch": minibatches,
    }
    state = {
        "successful_policy_updates": 96,
        "source_order": {
            "contract": contract,
            "loader_batch_size": 64 * minibatches,
            "loader_workers": 0,
            "completed_step": 96 // minibatches,
            "epoch": 3,
            "step_in_epoch": offset,
        },
    }
    assert (
        validate_saved_source(state, seed=17, minibatches=minibatches, dataset_sha256="a" * 64) == state["source_order"]
    )
    state["successful_policy_updates"] = 95
    with pytest.raises(ValueError):
        validate_saved_source(state, seed=17, minibatches=minibatches, dataset_sha256="a" * 64)


@pytest.fixture
def native_history_capture():
    selected, scalars = [], []
    for step in range(1, 97):
        metrics = {
            "policy/by_update/0/" + key: 1.0
            for key in (
                "optimizer_step_succeeded",
                "grad_norm_valid",
                "stale/statistics_valid",
                "stale/finite_fraction",
                "stale/quantiles_valid",
                "stale/p999_valid",
                "raw_grad_norm",
                "grad_norm_reduced",
            )
        }
        metrics.update(
            {
                "policy/by_update/0/stale/quantiles_overflow": 0,
                "policy/by_update/0/grad_cosine_valid": int(step > 1),
                "consumed/uid_digest_u52": 3970228113034015,
            }
        )
        selected.append({"global_step": step, **metrics})
        scalars.extend({"step": step, "metric": key, "value": value} for key, value in metrics.items())
    capture = {"results": {"scalars": scalars, "events": []}}
    return capture, selected


def test_native_history_keeps_integer_hashes_outside_float_tolerance(native_history_capture):
    capture, selected = native_history_capture
    assert audit_native_history(capture, selected, minibatches=1)["exact_integer_digest_joins"] == 96
    # This historical transport error is tiny relatively, but a hash is never approximate.
    selected[0]["consumed/uid_digest_u52"] = 3970228113034014.5
    with pytest.raises(ValueError, match="exact integers"):
        audit_native_history(capture, selected, minibatches=1)


@pytest.mark.parametrize("poison", [None, "validity", "norm", "successful"])
def test_zero_gradient_diagnostic_retains_validity_and_success_requirements(native_history_capture, poison):
    capture, selected = native_history_capture
    changes = {
        2: {"raw_grad_norm": 0.0, "grad_norm_reduced": 0.0, "grad_cosine_valid": 0},
        3: {"grad_cosine_valid": 0},
    }
    if poison == "validity":
        changes[3]["grad_cosine_valid"] = 1
    elif poison == "norm":
        changes[2]["grad_norm_reduced"] = 1e-12
    elif poison == "successful":
        changes[2]["optimizer_step_succeeded"] = 0
    for step, fields in changes.items():
        for field, value in fields.items():
            key = "policy/by_update/0/" + field
            selected[step - 1][key] = value
            for row in capture["results"]["scalars"]:
                if row["step"] == step and row["metric"] == key:
                    row["value"] = value
    with pytest.raises(ValueError):
        audit_native_history(capture, selected, minibatches=1)
    if poison is not None:
        with pytest.raises(ValueError):
            audit_native_history(capture, selected, minibatches=1, require_nonzero_gradients=False)
    else:
        result = audit_native_history(capture, selected, minibatches=1, require_nonzero_gradients=False)
        assert result["zero_gradient_updates"] == 1
        assert result["optimizer_updates"] == 96
        assert result["maximum_postclip_relative_norm_error"] == 0.0


@pytest.mark.parametrize("missing_sibling", [False, True])
def test_physical_attempt_cost_retains_failed_startup_and_missing_sibling(missing_sibling):
    job = "/owner/native"
    failed = {
        "attempt_id": 0,
        "attempt_uid": "failed-worker",
        "state": "TASK_STATE_FAILED",
        "exit_code": 1,
        "error": "Artifact inventory changed before Ray startup",
        "started_at": {"epoch_ms": "1000"},
        "finished_at": {"epoch_ms": "2000"},
    }
    success = {
        "attempt_id": 1,
        "attempt_uid": "final-worker",
        "state": "TASK_STATE_SUCCEEDED",
        "exit_code": 0,
        "started_at": {"epoch_ms": "3000"},
        "finished_at": {"epoch_ms": "5000"},
    }
    sibling = {**failed, "attempt_uid": "bounced-sibling"}
    readback = {
        "job": {
            "job_id": job,
            "state": "JOB_STATE_SUCCEEDED",
            "resources": {"device": {"gpu": {"variant": "H100", "count": 8}}},
        },
        "tasks": [
            {
                "task_id": job + "/0",
                "state": "TASK_STATE_SUCCEEDED",
                "attempts": ([] if missing_sibling else [sibling]) + [{**success, "attempt_uid": "final-sibling"}],
            },
            {"task_id": job + "/1", "state": "TASK_STATE_SUCCEEDED", "attempts": [failed, success]},
        ],
    }
    proof = audit_native_attempts(readback, expected_job_id=job)
    assert proof["lifecycle_clean"] is False
    assert proof["physical_cost_complete"] is not missing_sibling
    assert proof["known_task_h100_hours"] == pytest.approx((5 if missing_sibling else 6) * 8 / 3600)
    assert proof["missing_attempts"] == ([{"task_id": job + "/0", "attempt_id": 0}] if missing_sibling else [])
    assert len(proof["failed_attempts"]) == (1 if missing_sibling else 2)
    assert (proof["total_task_h100_hours"] is None) is missing_sibling
    # The same final checkpoint cannot stand in for a failed physical final attempt.
    readback["tasks"][1]["attempts"][-1]["state"] = "TASK_STATE_FAILED"
    with pytest.raises(ValueError):
        audit_native_attempts(readback, expected_job_id=job)


def test_explicit_async_receipt_preserves_default_sync_validation():
    receipts = fixture(1, runner="fully_async")
    with pytest.raises(ValueError):
        validate_checkpoint_receipts(*receipts, minibatches=1)
    result = validate_checkpoint_receipts(*receipts, minibatches=1, runner="fully_async")
    assert result["global_step"] == 96
    assert result["saved_successful_updates_validation_required"]


def test_async_saved_source_preserves_arithmetic_metadata_without_exposure_claim():
    state = {
        "successful_policy_updates": 96,
        "source_order": {
            "contract": {
                "algorithm": "torch-randperm-seed-plus-epoch-v1",
                "seed": 17,
                "dataset_sha256": "a" * 64,
                "rows": 1918,
                "prompts_per_step": 64,
                "updates_per_batch": 1,
            },
            "loader_batch_size": 1,
            "loader_workers": 0,
            "completed_step": 96,
            "epoch": 3,
            "step_in_epoch": 9,
        },
    }
    with pytest.raises(ValueError):
        validate_saved_source(state, seed=17, minibatches=1, dataset_sha256="a" * 64)
    result = validate_saved_source(state, seed=17, minibatches=1, dataset_sha256="a" * 64, runner="fully_async")
    assert result["loader_batch_size"] == 1
    assert result["epoch"] == 3  # Arithmetic source metadata; actual consumed epochs require native vectors.
