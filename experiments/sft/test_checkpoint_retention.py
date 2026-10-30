# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig, discover_latest_checkpoint, load_checkpoint_or_initialize
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig
from levanter.testing.helpers import use_test_mesh
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path
from rigging.filesystem.storage_path import StoragePath

from experiments.sft.launcher import HFModel, SFTSpec


@pytest.fixture
def clock(monkeypatch):
    real_datetime = datetime.datetime

    class Clock(real_datetime):
        current = real_datetime(2026, 1, 1)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(datetime, "datetime", Clock)
    return Clock


@pytest.fixture
def sft_trainer(tmp_path, monkeypatch):
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    spec = SFTSpec(
        name="checkpoints/test-sft",
        version="2026.10.29",
        model=HFModel(model_ref="unused-local-model"),
        chat_template="unused",
        datasets=[],
        optimizer=AdamConfig(),
        num_train_steps=5001,
    )

    def build(checkpointer=None):
        configured = replace(spec, checkpointer=checkpointer) if checkpointer is not None else spec
        pod = configured.model.build_train_config(
            StepContext.for_run(str(tmp_path / "run"), str(tmp_path)),
            configured,
            LmDataConfig(),
            ResourceConfig.with_cpu(),
            num_train_steps=5001,
        )
        return apply_output_path(pod.train_config, pod.output_path).trainer

    return build


def _state(step):
    # Small checkpoint payload with weights, optimizer moments, and a progress counter; no training.
    return {
        "model": jnp.array([step, step + 1], dtype=jnp.float32),
        "opt_state": jnp.array([step / 2], dtype=jnp.float32),
        "step": jnp.array(step, dtype=jnp.int32),
    }


def _metadata(root):
    return {
        path.parent.name: json.loads(path.read_text()) for path in (StoragePath(root) / "step-*/metadata.json").glob()
    }


def test_sft_default_rolls_temporary_checkpoints_resumes_and_keeps_final(sft_trainer, clock):
    trainer = sft_trainer()
    config = trainer.checkpointer
    checkpointer = config.create("test-run")
    checkpointer.on_step(tree=_state(0), step=0)

    # Cross the launcher's former 5,000-step permanent interval.
    for step in (4999, 5000):
        clock.current += datetime.timedelta(minutes=30)
        checkpointer.on_step(tree=_state(step), step=step)
        checkpointer.wait_until_finished()
        assert _metadata(config.base_path) == {}
        assert list(_metadata(config.temporary_base_path)) == [f"step-{step}"]
        assert _metadata(config.temporary_base_path)[f"step-{step}"]["is_temporary"] is True

    # A fresh launch must restore the complete payload from scratch when no permanent save exists.
    restarted_trainer = sft_trainer()
    paths = restarted_trainer.checkpoint_search_paths("test-run")
    with use_test_mesh():
        restored = load_checkpoint_or_initialize(lambda: _state(0), paths)()
    for name, expected in _state(5000).items():
        np.testing.assert_array_equal(restored[name], expected)

    restarted = restarted_trainer.checkpointer.create("test-run")
    clock.current += datetime.timedelta(minutes=30)
    restarted.on_step(tree=_state(5001), step=5001)
    restarted.wait_until_finished()
    assert list(_metadata(config.temporary_base_path)) == ["step-5001"]

    # Trainer's terminal hook forces a permanent save even if that step was just saved temporarily.
    restarted.on_step(tree=_state(5001), step=5001, force=True)
    restarted.wait_until_finished()
    final = _metadata(config.base_path)
    assert list(final) == ["step-5001"]
    assert final["step-5001"]["is_temporary"] is False
    assert _metadata(config.temporary_base_path) == {}
    assert discover_latest_checkpoint(*paths) == str(Path(config.base_path) / "step-5001")
    with use_test_mesh():
        final_state = load_checkpoint_or_initialize(lambda: _state(0), paths)()
    for name, expected in _state(5001).items():
        np.testing.assert_array_equal(final_state[name], expected)


def test_sft_explicit_keep_retains_intermediate_permanent_checkpoints(sft_trainer, clock):
    trainer = sft_trainer(
        CheckpointerConfig(
            save_interval=datetime.timedelta(minutes=30),
            keep=[{"every": 5000}],
            keep_last_temporary_checkpoints=2,
        )
    )
    config = trainer.checkpointer
    checkpointer = config.create("test-run")
    checkpointer.on_step(tree=_state(0), step=0)
    for step in (4997, 4998, 4999):
        clock.current += datetime.timedelta(minutes=30)
        checkpointer.on_step(tree=_state(step), step=step)
        checkpointer.wait_until_finished()
    assert set(_metadata(config.temporary_base_path)) == {"step-4998", "step-4999"}

    checkpointer.on_step(tree=_state(5000), step=5000)
    checkpointer.wait_until_finished()
    checkpointer.on_step(tree=_state(5001), step=5001, force=True)
    checkpointer.wait_until_finished()
    permanent = _metadata(config.base_path)
    assert set(permanent) == {"step-5000", "step-5001"}
    assert all(not metadata["is_temporary"] for metadata in permanent.values())
    assert _metadata(config.temporary_base_path) == {}
