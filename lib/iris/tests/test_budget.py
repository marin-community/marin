# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.controller.budget import UserTask, compute_effective_band, interleave_by_user, resource_value
from iris.rpc import cluster_pb2
from iris.rpc.proto_utils import PRIORITY_BAND_VALUES, priority_band_name, priority_band_value

PRODUCTION = cluster_pb2.PRIORITY_BAND_PRODUCTION
INTERACTIVE = cluster_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = cluster_pb2.PRIORITY_BAND_BATCH

GiB = 1024**3


def test_resource_value_cpu_only():
    # 4 cores, 16 GiB RAM, no accelerators
    assert resource_value(cpu_millicores=4000, memory_bytes=16 * GiB, accelerator_count=0) == 5 * 4 + 16


def test_resource_value_gpu():
    # 8 cores, 64 GiB, 4 GPUs
    assert resource_value(cpu_millicores=8000, memory_bytes=64 * GiB, accelerator_count=4) == 1000 * 4 + 64 + 5 * 8


def test_resource_value_tpu():
    # 96 cores, 320 GiB, 8 TPU chips
    expected = 1000 * 8 + 320 + 5 * 96
    assert resource_value(cpu_millicores=96000, memory_bytes=320 * GiB, accelerator_count=8) == expected


def test_resource_value_zero_resources():
    assert resource_value(cpu_millicores=0, memory_bytes=0, accelerator_count=0) == 0


def test_resource_value_truncates_fractional():
    # 1500 millicores = 1 core (truncated), 1.5 GiB = 1 GiB (truncated)
    assert resource_value(cpu_millicores=1500, memory_bytes=int(1.5 * GiB), accelerator_count=0) == 5 * 1 + 1


def test_interleave_by_user_single_user():
    tasks = [UserTask("alice", "t1"), UserTask("alice", "t2"), UserTask("alice", "t3")]
    result = interleave_by_user(tasks, user_spend={})
    assert result == ["t1", "t2", "t3"]


def test_interleave_by_user_two_users_equal_spend():
    tasks = [UserTask("alice", "a1"), UserTask("alice", "a2"), UserTask("bob", "b1"), UserTask("bob", "b2")]
    result = interleave_by_user(tasks, user_spend={"alice": 100, "bob": 100})
    # Equal spend: stable sort by user name, then round-robin
    assert result == ["a1", "b1", "a2", "b2"] or result == ["b1", "a1", "b2", "a2"]
    assert len(result) == 4


def test_interleave_by_user_spend_ordering():
    tasks = [
        UserTask("alice", "a1"),
        UserTask("alice", "a2"),
        UserTask("bob", "b1"),
        UserTask("bob", "b2"),
    ]
    # Bob has spent less, so his tasks should come first in each round
    result = interleave_by_user(tasks, user_spend={"alice": 8000, "bob": 1000})
    assert result == ["b1", "a1", "b2", "a2"]


def test_interleave_by_user_unequal_task_counts():
    tasks = [UserTask("alice", "a1"), UserTask("alice", "a2"), UserTask("alice", "a3"), UserTask("bob", "b1")]
    result = interleave_by_user(tasks, user_spend={"alice": 0, "bob": 0})
    # Round 0: a1, b1; Round 1: a2; Round 2: a3
    assert result[0] in ("a1", "b1")
    assert result[1] in ("a1", "b1")
    assert "a2" in result
    assert "a3" in result
    assert len(result) == 4


def test_interleave_by_user_empty():
    assert interleave_by_user([], user_spend={}) == []


def test_interleave_by_user_missing_spend_defaults_to_zero():
    tasks = [UserTask("alice", "a1"), UserTask("bob", "b1")]
    # Alice has no spend entry → defaults to 0, Bob has 5000
    result = interleave_by_user(tasks, user_spend={"bob": 5000})
    assert result == ["a1", "b1"]


def test_effective_band_over_budget_becomes_batch():
    """INTERACTIVE task becomes BATCH when user exceeds budget."""
    spend = {"alice": 10000}
    limits = {"alice": 5000}
    assert compute_effective_band(INTERACTIVE, "alice", spend, limits) == BATCH


def test_effective_band_within_budget_keeps_band():
    """INTERACTIVE task stays INTERACTIVE when user is within budget."""
    spend = {"alice": 3000}
    limits = {"alice": 5000}
    assert compute_effective_band(INTERACTIVE, "alice", spend, limits) == INTERACTIVE


def test_effective_band_production_never_downgraded():
    """PRODUCTION tasks are never downgraded, even when over budget."""
    spend = {"alice": 10000}
    limits = {"alice": 5000}
    assert compute_effective_band(PRODUCTION, "alice", spend, limits) == PRODUCTION


def test_effective_band_zero_budget_means_unlimited():
    """budget_limit=0 means no down-weighting regardless of spend."""
    spend = {"alice": 999999}
    limits = {"alice": 0}
    assert compute_effective_band(INTERACTIVE, "alice", spend, limits) == INTERACTIVE


def test_effective_band_no_budget_row_means_unlimited():
    """User with no budget row is treated as unlimited."""
    spend = {"alice": 999999}
    limits = {}  # no row for alice
    assert compute_effective_band(INTERACTIVE, "alice", spend, limits) == INTERACTIVE


def test_effective_band_batch_stays_batch_when_over_budget():
    """Already-BATCH task stays BATCH (max of requested and BATCH)."""
    spend = {"alice": 10000}
    limits = {"alice": 5000}
    assert compute_effective_band(BATCH, "alice", spend, limits) == BATCH


def test_priority_band_name_roundtrip():
    for band in PRIORITY_BAND_VALUES:
        name = priority_band_name(band)
        assert priority_band_value(name) == band


def test_priority_band_values_are_ordered():
    """Proto enum values are ordered: PRODUCTION < INTERACTIVE < BATCH."""
    assert cluster_pb2.PRIORITY_BAND_PRODUCTION < cluster_pb2.PRIORITY_BAND_INTERACTIVE
    assert cluster_pb2.PRIORITY_BAND_INTERACTIVE < cluster_pb2.PRIORITY_BAND_BATCH
