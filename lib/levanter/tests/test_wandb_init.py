# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from levanter.tracker.wandb import WandbConfig
from levanter.utils import jax_utils


class FakeDistributedClient:
    def __init__(self, process_count: int):
        self.values = {}
        self.barrier = threading.Barrier(process_count)

    def key_value_set(self, key, value):
        self.values[key] = value

    def wait_at_barrier(self, _key, timeout_in_ms):
        self.barrier.wait(timeout=timeout_in_ms / 1000)

    def blocking_key_value_get(self, key, timeout_in_ms):
        del timeout_in_ms
        return self.values[key]


def _run_two_process_wandb_init(monkeypatch, wandb_init):
    process = threading.local()
    client = FakeDistributedClient(process_count=2)

    monkeypatch.setattr("levanter.tracker.wandb.jax.process_count", lambda: 2)
    monkeypatch.setattr("levanter.tracker.wandb.jax.process_index", lambda: process.index)
    monkeypatch.setattr("levanter.tracker.wandb.wandb.init", wandb_init)
    monkeypatch.setattr(jax_utils.jax_distributed.global_state, "client", client)

    def initialize(process_index):
        process.index = process_index
        try:
            WandbConfig(save_code=False).init("run-id")
        except Exception as e:
            return e
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(initialize, range(2)))


def test_wandb_init_primary_failure_reaches_every_process(monkeypatch):
    def wandb_init(**kwargs):
        if kwargs["mode"] != "disabled":
            raise RuntimeError("No API key configured")
        return SimpleNamespace(step=0)

    errors = _run_two_process_wandb_init(monkeypatch, wandb_init)

    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "No API key configured"
    assert isinstance(errors[1], RuntimeError)
    assert str(errors[1]) == "W&B initialization failed on process 0: RuntimeError: No API key configured"


def test_wandb_init_secondary_failure_reaches_every_process(monkeypatch):
    primary_run = SimpleNamespace(step=0, project="project", name="name", tags=[], id="run-id", group=None)

    def wandb_init(**kwargs):
        if kwargs["mode"] == "disabled":
            raise OSError("W&B directory is not writable")
        return primary_run

    errors = _run_two_process_wandb_init(monkeypatch, wandb_init)

    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "W&B initialization failed on process 1: OSError: W&B directory is not writable"
    assert isinstance(errors[1], OSError)
    assert str(errors[1]) == "W&B directory is not writable"


def test_wandb_init_preserves_local_error_when_coordination_fails(monkeypatch):
    initialization_error = ValueError("No API key configured")
    coordination_error = OSError("distributed client stopped")

    monkeypatch.setattr("levanter.tracker.wandb.jax.process_count", lambda: 2)
    monkeypatch.setattr("levanter.tracker.wandb.jax.process_index", lambda: 0)

    def wandb_init(**_kwargs):
        raise initialization_error

    def allgather(_value):
        raise coordination_error

    monkeypatch.setattr("levanter.tracker.wandb.wandb.init", wandb_init)
    monkeypatch.setattr("levanter.tracker.wandb.jax_utils.multihost_allgather_sync", allgather)

    with pytest.raises(ValueError, match="No API key configured") as exc_info:
        WandbConfig(save_code=False).init("run-id")

    assert exc_info.value.__cause__ is coordination_error
