# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.post_training.long_loop.sft_runtime import state_rebind, train
from experiments.post_training.long_loop.timing import EventLog


def test_rebind_preserves_signature_and_deletes_original_state(tmp_path, monkeypatch):
    (tmp_path / "memory.current").write_text("100")
    (tmp_path / "memory.max").write_text("1000")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = {
        "w": jax.device_put(jnp.arange(16).reshape(4, 4), sharding),
        "m": jnp.ones((2, 3, 4)),
        "weak_scalar": jnp.array(3),
        "replicated_weak_scalar": jax.device_put(jnp.array(4), replicated),
    }
    batch = {"tokens": jnp.arange(4)}
    original_signature = state_rebind.signature((state, batch))
    expected = jax.tree.map(lambda x: np.asarray(x).copy(), state)
    original = jax.tree.leaves(state)
    expected_shards = sum(len(x.addressable_shards) for x in original)
    expected_bytes = sum(shard.data.size * shard.data.dtype.itemsize for x in original for shard in x.addressable_shards)
    observations = []
    restore = state_rebind.restore_state

    def checked_restore(snapshot):
        assert all(leaf.is_deleted() for leaf in original)
        return restore(snapshot)

    monkeypatch.setattr(state_rebind, "restore_state", checked_restore)
    actual = state_rebind.rebind_first_state(
        state, batch, cgroup_path=tmp_path, observe=lambda phase, sample: observations.append((phase, sample))
    )
    assert state_rebind.signature((actual, batch)) == original_signature
    assert [phase for phase, _ in observations] == ["before_snapshot", "after_snapshot"]
    assert observations[-1][1]["snapshot_shards"] == expected_shards
    assert observations[-1][1]["snapshot_bytes"] == expected_bytes
    for got, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, wanted)


def test_rebind_logs_unavailable_cgroup_without_preventing_restore(tmp_path):
    state = {"w": jnp.ones((2, 3))}
    before = state_rebind.signature(state)
    observations = []
    result = state_rebind.rebind_first_state(
        state, {}, cgroup_path=tmp_path, observe=lambda phase, sample: observations.append((phase, sample))
    )
    assert state_rebind.signature(result) == before
    assert [sample["cgroup_error"] for _, sample in observations] == ["FileNotFoundError", "FileNotFoundError"]
    assert all(sample["task_scope_verified"] is False for _, sample in observations)

    (tmp_path / "memory.current").write_text("999")
    (tmp_path / "memory.max").write_text("1000")
    state = {"w": jnp.ones((2, 3))}
    observations = []
    result = state_rebind.rebind_first_state(
        state, {}, cgroup_path=tmp_path, observe=lambda phase, sample: observations.append((phase, sample))
    )
    np.testing.assert_array_equal(result["w"], np.ones((2, 3)))
    assert all(sample["cgroup_current_bytes"] == 999 for _, sample in observations)


def test_post_rebind_barrier_records_wait_and_uses_configured_timeout(tmp_path, monkeypatch):
    timing = EventLog(tmp_path / "events.jsonl", "run", "sft", "attempt")
    calls = []

    def barrier(barrier_id, *, timeout):
        events = [json.loads(line) for line in timing.path.read_text().splitlines()]
        assert [(event["kind"], event["event"]) for event in events] == [("post_rebind_barrier", "start")]
        calls.append((barrier_id, timeout))

    monkeypatch.setattr(train, "barrier_sync_named", barrier)
    config = state_rebind.HostRebindConfig(post_rebind_barrier_timeout_seconds=42.0)
    train._wait_after_first_step_host_rebind(timing, config, step=5)

    assert calls == [("after_first_step_host_rebind", 42.0)]
    events = [json.loads(line) for line in timing.path.read_text().splitlines()]
    assert [(event["kind"], event["event"], event["step"]) for event in events] == [
        ("post_rebind_barrier", "start", 5),
        ("post_rebind_barrier", "end", 5),
    ]
