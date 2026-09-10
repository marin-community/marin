# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io

import pytest
import torch
from omegaconf import OmegaConf

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.checkpoint_request_progress import (
    checkpoint_progress_for_request,
    validate_request_progress,
)


def evidence(entrypoint, updates, step):
    trainer = {
        "seed": 17,
        "max_steps": step,
        "ckpt_interval": step,
        "policy_mini_batch_size": 64,
        "train_batch_size": 128,
        "update_epochs_per_batch": 1,
    }
    state = {"global_step": step, "successful_policy_updates": updates, "config": {"trainer": trainer}}
    request = {
        "run_id": "owned-run",
        "completion_mode": "checkpoint",
        "attempt_id": "owned-attempt",
        "seed": 17,
        "config_yaml": OmegaConf.to_yaml({"entrypoint": entrypoint, "trainer": trainer}),
        "overrides": ["++generator.chat_template_kwargs.enable_thinking=false"],
        "topology": {"role_plan": {"train_batch_size": 128, "policy_mini_batch_size": 64}},
    }
    return {"request": request, "response": {"state": "succeeded", "run_id": "owned-run", "attempt_id": "owned-attempt"}}, state


def bind_state(training, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    raw = stream.getvalue()
    training["response"]["training"] = {
        "global_step": state["global_step"],
        "checkpoint": {"global_step": state["global_step"], "trainer_state_sha256": hashlib.sha256(raw).hexdigest()},
    }
    return raw


def decode(training, raw, entrypoint, updates):
    return checkpoint_progress_for_request(
        training,
        raw,
        training_sha256=audit.canonical_sha(training),
        expected_entrypoint=entrypoint,
        optimizer_updates=updates,
        minibatches=2,
    )


@pytest.mark.parametrize(
    "entrypoint,updates,step,unit",
    [("standard", 8, 4, "rollout_batch"), ("fully_async", 8, 8, "successful_optimizer_update"),
     ("standard", 96, 48, "rollout_batch"), ("fully_async", 96, 96, "successful_optimizer_update")],
)
def test_owned_saved_state_uses_the_declared_runner_clock(entrypoint, updates, step, unit):
    training, state = evidence(entrypoint, updates, step)
    progress = decode(training, bind_state(training, state), entrypoint, updates)
    assert progress["global_step"] == step and progress["optimizer_updates"] == updates
    assert progress["native_step_unit"] == unit
    validate_request_progress(
        progress, training, training_sha256=audit.canonical_sha(training),
        expected_progress_sha256=progress["progress_sha256"], expected_entrypoint=entrypoint,
        optimizer_updates=updates, minibatches=2,
    )
    changed = copy.deepcopy(progress)
    changed["saved_config_sha256"] = "0" * 64
    changed["progress_sha256"] = audit.canonical_sha({k: v for k, v in changed.items() if k != "progress_sha256"})
    with pytest.raises(ValueError, match="qualified evidence"):
        validate_request_progress(
            changed, training, training_sha256=audit.canonical_sha(training),
            expected_progress_sha256=progress["progress_sha256"], expected_entrypoint=entrypoint,
            optimizer_updates=updates, minibatches=2,
        )


@pytest.mark.parametrize("entrypoint,wrong_step", [("fully_async", 4), ("standard", 8)])
def test_rejects_using_the_other_runner_clock(entrypoint, wrong_step):
    training, state = evidence(entrypoint, 8, wrong_step)
    with pytest.raises(ValueError, match="Request runner"):
        decode(training, bind_state(training, state), entrypoint, 8)


def test_observed_entrypoint_must_match_request():
    training, state = evidence("standard", 8, 4)
    with pytest.raises(ValueError, match="Request runner"):
        decode(training, bind_state(training, state), "fully_async", 8)


@pytest.mark.parametrize("override", ["++entrypoint=standard", "trainer.max_steps=4", "~trainer.ckpt_interval", "trainer={max_steps:4}"])
def test_rejects_clock_overrides_before_decoding(override):
    training, state = evidence("fully_async", 8, 8)
    training["request"]["overrides"].append(override)
    with pytest.raises(ValueError, match="clock overrides"):
        decode(training, bind_state(training, state), "fully_async", 8)


@pytest.mark.parametrize("field,value", [("successful_policy_updates", 7), ("global_step", 4), ("train_batch_size", 64)])
def test_hash_bound_saved_state_must_prove_actual_updates_and_geometry(field, value):
    training, state = evidence("fully_async", 8, 8)
    if field == "train_batch_size":
        state["config"]["trainer"][field] = value
    else:
        state[field] = value
    with pytest.raises(ValueError, match="Saved"):
        decode(training, bind_state(training, state), "fully_async", 8)


def test_bad_state_bytes_rejected_before_pickle_decoder(monkeypatch):
    training, state = evidence("fully_async", 8, 8)
    raw = bind_state(training, state)
    def forbidden(*args, **kwargs):
        pytest.fail("Unbound bytes reached pickle decoder")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="Saved trainer bytes"):
        decode(training, raw + b"corruption", "fully_async", 8)


def test_periodic_checkpoint_seven_does_not_change_the_final_update_eight_clock():
    training, state = evidence("fully_async", 8, 8)
    cfg = OmegaConf.create(training["request"]["config_yaml"])
    cfg.trainer.ckpt_interval = 7
    training["request"]["config_yaml"] = OmegaConf.to_yaml(cfg)
    state["config"]["trainer"]["ckpt_interval"] = 7
    progress = decode(training, bind_state(training, state), "fully_async", 8)
    assert progress["global_step"] == progress["optimizer_updates"] == 8
    assert progress["checkpoint_interval"] == 7
    state["config"]["trainer"]["ckpt_interval"] = 8
    with pytest.raises(ValueError, match="effective native clock"):
        decode(training, bind_state(training, state), "fully_async", 8)
