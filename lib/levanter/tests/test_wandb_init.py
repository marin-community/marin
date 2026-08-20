# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.tracker.wandb import WandbConfig


def test_wandb_init_primary_failure_is_broadcast(monkeypatch):
    config = WandbConfig(save_code=False)
    failure = RuntimeError("No API key configured")
    broadcasts = []

    monkeypatch.setattr("levanter.tracker.wandb.jax.process_count", lambda: 2)
    monkeypatch.setattr("levanter.tracker.wandb.jax.process_index", lambda: 0)

    def wandb_init(**_kwargs):
        raise failure

    monkeypatch.setattr("levanter.tracker.wandb.wandb.init", wandb_init)

    def broadcast(value, *, is_source):
        broadcasts.append((value, is_source))
        return value

    monkeypatch.setattr("levanter.tracker.wandb.jax_utils.multihost_broadcast_sync", broadcast)

    with pytest.raises(RuntimeError, match="No API key configured"):
        config.init("run-id")

    assert list(broadcasts[0][0].values()) == ["RuntimeError: No API key configured"]
    assert broadcasts[0][1] is True


def test_wandb_init_non_primary_exits_before_local_init_when_primary_fails(monkeypatch):
    config = WandbConfig(save_code=False)
    wandb_init_called = False

    monkeypatch.setattr("levanter.tracker.wandb.jax.process_count", lambda: 2)
    monkeypatch.setattr("levanter.tracker.wandb.jax.process_index", lambda: 1)

    def wandb_init(**_kwargs):
        nonlocal wandb_init_called
        wandb_init_called = True
        raise AssertionError("non-primary W&B initialization should not run after a primary failure")

    monkeypatch.setattr("levanter.tracker.wandb.wandb.init", wandb_init)
    monkeypatch.setattr(
        "levanter.tracker.wandb.jax_utils.multihost_broadcast_sync",
        lambda value, **_kwargs: {next(iter(value)): "UsageError: No API key configured"},
    )

    with pytest.raises(RuntimeError, match="W&B initialization failed on process 0"):
        config.init("run-id")

    assert not wandb_init_called
