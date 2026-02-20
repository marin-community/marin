# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import levanter.grug_native.checkpoint as grug_checkpoint


@dataclass
class _TrainState:
    step: object


class _NoStepState:
    pass


class _RecordingCheckpointer:
    def __init__(self):
        self.on_step_calls: list[tuple[object, bool]] = []
        self.wait_calls = 0

    def on_step(self, step_info, force: bool = False):
        self.on_step_calls.append((step_info, force))

    def wait_until_finished(self):
        self.wait_calls += 1


def test_save_checkpoint_on_step_noop_when_checkpointer_is_none():
    grug_checkpoint.save_checkpoint_on_step(None, _TrainState(step=0))


def test_save_checkpoint_on_step_forwards_step_force_and_payload():
    checkpointer = _RecordingCheckpointer()
    train_state = _TrainState(step=jnp.array(7, dtype=jnp.int32))

    grug_checkpoint.save_checkpoint_on_step(checkpointer, train_state, force=True)

    assert len(checkpointer.on_step_calls) == 1
    step_info, force = checkpointer.on_step_calls[0]
    assert force is True
    assert step_info.step == 7
    assert step_info.state.saveable_state["train_state"] is train_state


def test_wait_for_checkpoints_noop_when_checkpointer_is_none():
    grug_checkpoint.wait_for_checkpoints(None)


def test_wait_for_checkpoints_calls_wait_once():
    checkpointer = _RecordingCheckpointer()
    grug_checkpoint.wait_for_checkpoints(checkpointer)
    assert checkpointer.wait_calls == 1


def test_save_checkpoint_on_step_raises_clear_error_if_step_missing():
    checkpointer = _RecordingCheckpointer()
    with pytest.raises(ValueError, match="must define a 'step' attribute"):
        grug_checkpoint.save_checkpoint_on_step(checkpointer, _NoStepState())


def test_save_checkpoint_on_step_raises_clear_error_if_step_not_int_like():
    checkpointer = _RecordingCheckpointer()
    with pytest.raises(TypeError, match="must be convertible to int"):
        grug_checkpoint.save_checkpoint_on_step(checkpointer, _TrainState(step=object()))


def test_maybe_restore_checkpoint_forwards_args_and_sets_discover_latest(monkeypatch):
    calls = {}
    sentinel_state = {"x": 1}

    def fake_load_checkpoint(state, checkpoint_path, discover_latest, axis_mapping, mesh, allow_partial):
        calls["state"] = state
        calls["checkpoint_path"] = checkpoint_path
        calls["discover_latest"] = discover_latest
        calls["axis_mapping"] = axis_mapping
        calls["mesh"] = mesh
        calls["allow_partial"] = allow_partial
        return {"restored": True}

    monkeypatch.setattr(grug_checkpoint, "load_checkpoint", fake_load_checkpoint)

    out = grug_checkpoint.maybe_restore_checkpoint(
        sentinel_state,
        checkpoint_path="gs://my-ckpt",
        axis_mapping={"embed": "model"},
        mesh=None,
        allow_partial=True,
    )

    assert out == {"restored": True}
    assert calls["state"] is sentinel_state
    assert calls["checkpoint_path"] == "gs://my-ckpt"
    assert calls["discover_latest"] is True
    assert calls["axis_mapping"] == {"embed": "model"}
    assert calls["mesh"] is None
    assert calls["allow_partial"] is True
