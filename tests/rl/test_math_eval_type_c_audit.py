# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import yaml

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.type_c_audit import validate_checkpoint_receipts, validate_saved_source


def fixture(minibatches=1):
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
            "entrypoint": "standard",
            "trainer": {
                "train_batch_size": 64 * minibatches,
                "policy_mini_batch_size": 64,
                "max_steps": 96 // minibatches,
                "eval_interval": 32 // minibatches,
                "update_epochs_per_batch": 1,
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
