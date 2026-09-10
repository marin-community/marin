# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind saved optimizer progress to the request's explicit runner clock."""

import hashlib
import io

import torch
from omegaconf import OmegaConf

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.checkpoint_state import MAX_TRAINER_STATE_BYTES

CLOCK_FIELDS = {
    "entrypoint",
    "trainer.max_steps",
    "trainer.ckpt_interval",
    "trainer.train_batch_size",
    "trainer.policy_mini_batch_size",
    "trainer.update_epochs_per_batch",
}


def request_clock(request, *, expected_entrypoint, optimizer_updates, minibatches):
    """Resolve a qualified request; the caller audits actual native invocation."""
    if (
        expected_entrypoint not in {"standard", "fully_async"}
        or type(optimizer_updates) is not int
        or optimizer_updates <= 0
        or type(minibatches) is not int
        or not 1 <= minibatches <= 16
        or optimizer_updates % minibatches
    ):
        raise ValueError("Unsupported runner or optimizer schedule")
    for override in request["overrides"]:
        key = override.split("=", 1)[0].lstrip("+~")
        if any(key == field or field.startswith(key + ".") for field in CLOCK_FIELDS):
            raise ValueError("Checkpoint clock must be explicit in request YAML, without clock overrides")
    config = OmegaConf.to_container(OmegaConf.create(request["config_yaml"]), resolve=True)
    trainer = config["trainer"]
    step = optimizer_updates if expected_entrypoint == "fully_async" else optimizer_updates // minibatches
    mini = trainer["policy_mini_batch_size"]
    batch = trainer["train_batch_size"]
    if (
        request["completion_mode"] != "checkpoint"
        or config["entrypoint"] != expected_entrypoint
        or trainer["max_steps"] != step
        or type(trainer["ckpt_interval"]) is not int
        or not 1 <= trainer["ckpt_interval"] <= step
        or trainer["update_epochs_per_batch"] != 1
        or type(mini) is not int
        or mini <= 0
        or type(batch) is not int
        or batch != mini * minibatches
    ):
        raise ValueError("Request runner, final checkpoint or batch geometry differs")
    role = request["topology"]["role_plan"]
    if role["train_batch_size"] != batch or role["policy_mini_batch_size"] != mini:
        raise ValueError("Native topology differs from the request clock geometry")
    return {
        "entrypoint": expected_entrypoint,
        "native_step_unit": "successful_optimizer_update" if expected_entrypoint == "fully_async" else "rollout_batch",
        "global_step": step,
        "checkpoint_interval": trainer["ckpt_interval"],
        "optimizer_updates": optimizer_updates,
        "minibatches": minibatches,
        "policy_mini_batch_size": mini,
        "train_batch_size": batch,
    }


def checkpoint_progress_for_request(
    training, trainer_state_bytes, *, training_sha256, expected_entrypoint, optimizer_updates, minibatches
):
    """Decode owned state after its terminal and bytes have independent bindings.

    Request runner identity must also match the independently observed native
    entrypoint. This proof does not establish controller or numerical coverage.
    Legacy checkpoint_progress and historical receipts remain unchanged.
    """
    if audit.canonical_sha(training) != training_sha256:
        raise ValueError("Training terminal differs from its independently audited hash")
    request, response = training["request"], training["response"]
    clock = request_clock(
        request, expected_entrypoint=expected_entrypoint, optimizer_updates=optimizer_updates, minibatches=minibatches
    )
    step = clock["global_step"]
    checkpoint = response["training"]["checkpoint"]
    digest = hashlib.sha256(trainer_state_bytes).hexdigest()
    if (
        not 0 < len(trainer_state_bytes) <= MAX_TRAINER_STATE_BYTES
        or digest != checkpoint["trainer_state_sha256"]
        or response["state"] != "succeeded"
        or response["run_id"] != request["run_id"]
        or response["attempt_id"] != request["attempt_id"]
        or response["training"]["global_step"] != step
        or checkpoint["global_step"] != step
    ):
        raise ValueError("Saved trainer bytes or native checkpoint identity differs")
    state = torch.load(io.BytesIO(trainer_state_bytes), map_location="cpu", weights_only=False)
    config = state["config"]
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    trainer = config["trainer"]
    if (
        type(state["successful_policy_updates"]) is not int
        or state["successful_policy_updates"] != optimizer_updates
        or state["global_step"] != step
        or trainer["max_steps"] != step
        or trainer["ckpt_interval"] != clock["checkpoint_interval"]
        or trainer["update_epochs_per_batch"] != 1
        or trainer["seed"] != request["seed"]
        or trainer["policy_mini_batch_size"] != clock["policy_mini_batch_size"]
        or trainer["train_batch_size"] != clock["train_batch_size"]
    ):
        raise ValueError("Saved successful updates or effective native clock differs")
    result = {
        "schema": "math_eval_request_checkpoint_progress_v1",
        "training_manifest_sha256": training_sha256,
        "training_request_sha256": audit.canonical_sha(request),
        "trainer_state_sha256": digest,
        "saved_config_sha256": audit.canonical_sha(config),
        "training_seed": request["seed"],
        **clock,
    }
    return result | {"progress_sha256": audit.canonical_sha(result)}


def validate_request_progress(
    progress, training, *, training_sha256, expected_progress_sha256, expected_entrypoint, optimizer_updates, minibatches
):
    """Validate a separately audited request-aware progress receipt for export."""
    body = {key: value for key, value in progress.items() if key != "progress_sha256"}
    request = training["request"]
    clock = request_clock(
        request, expected_entrypoint=expected_entrypoint, optimizer_updates=optimizer_updates, minibatches=minibatches
    )
    if (
        audit.canonical_sha(training) != training_sha256
        or progress["schema"] != "math_eval_request_checkpoint_progress_v1"
        or progress["progress_sha256"] != expected_progress_sha256
        or audit.canonical_sha(body) != expected_progress_sha256
        or progress["training_manifest_sha256"] != training_sha256
        or progress["training_request_sha256"] != audit.canonical_sha(request)
        or progress["trainer_state_sha256"] != training["response"]["training"]["checkpoint"]["trainer_state_sha256"]
        or progress["training_seed"] != request["seed"]
        or any(progress[key] != value for key, value in clock.items())
    ):
        raise ValueError("Request-aware progress differs from its qualified evidence")
