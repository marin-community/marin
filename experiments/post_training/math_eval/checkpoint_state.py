# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate successful optimizer updates from native rollout checkpoint steps."""

import hashlib
import io

import torch
from omegaconf import OmegaConf

from experiments.post_training import async_rl_audit as audit

MAX_TRAINER_STATE_BYTES = 16 * 1024**2


def checkpoint_progress(training, trainer_state_bytes, *, training_sha256, optimizer_updates, minibatches):
    """Decode owned state only after binding its bytes to an audited checkpoint.

    The terminal's expected hash must come from independent native training
    qualification. This proves actual saved progress, not numerical training
    correctness or controller completion by itself.
    """
    if audit.canonical_sha(training) != training_sha256:
        raise ValueError("Training terminal differs from its independently audited hash")
    if (
        type(optimizer_updates) is not int
        or optimizer_updates <= 0
        or type(minibatches) is not int
        or not 1 <= minibatches <= 16
        or optimizer_updates % minibatches
    ):
        raise ValueError("Optimizer updates must fit the declared minibatch schedule")
    native_step = optimizer_updates // minibatches
    request, response = training["request"], training["response"]
    checkpoint = response["training"]["checkpoint"]
    digest = hashlib.sha256(trainer_state_bytes).hexdigest()
    if (
        not 0 < len(trainer_state_bytes) <= MAX_TRAINER_STATE_BYTES
        or digest != checkpoint["trainer_state_sha256"]
        or response["state"] != "succeeded"
        or response["training"]["global_step"] != native_step
        or checkpoint["global_step"] != native_step
    ):
        raise ValueError("Saved trainer bytes or native checkpoint step differ")
    # This is the owned hash-bound trainer pickle produced by the qualified
    # native run; arbitrary caller-supplied pickle bytes are not accepted.
    state = torch.load(io.BytesIO(trainer_state_bytes), map_location="cpu", weights_only=False)
    config = state["config"]
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    trainer = config["trainer"]
    mini = trainer["policy_mini_batch_size"]
    batch = trainer["train_batch_size"]
    if (
        type(state["successful_policy_updates"]) is not int
        or state["successful_policy_updates"] != optimizer_updates
        or state["global_step"] != native_step
        or type(mini) is not int
        or mini <= 0
        or type(batch) is not int
        or batch != mini * minibatches
        or trainer["max_steps"] != native_step
        or trainer["seed"] != request["seed"]
    ):
        raise ValueError("Saved successful updates or actual minibatch geometry differ")
    result = {
        "schema": "math_eval_checkpoint_progress_v1",
        "training_manifest_sha256": training_sha256,
        "trainer_state_sha256": digest,
        "saved_config_sha256": audit.canonical_sha(config),
        "training_seed": request["seed"],
        "global_step": native_step,
        "optimizer_updates": optimizer_updates,
        "minibatches": minibatches,
        "policy_mini_batch_size": mini,
        "train_batch_size": batch,
    }
    return result | {"progress_sha256": audit.canonical_sha(result)}
