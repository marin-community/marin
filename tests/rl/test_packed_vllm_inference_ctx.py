# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.rl.environments.inference_ctx import InferenceRequestKind
from marin.rl.environments.inference_ctx.packed_vllm import (
    PackedDispatchPlan,
    choose_packed_dispatch_plan,
    choose_packed_target_replica_indices,
    merge_packed_completion_shards,
    split_packed_prompt_batch,
)
from marin.rl.environments.inference_ctx.packed_vllm_protocol import PackedReplicaStatus


def test_split_packed_prompt_batch_preserves_contiguous_order():
    shards = split_packed_prompt_batch(["p0", "p1", "p2", "p3", "p4"], num_replicas=2)

    assert shards == [
        ([0, 1, 2], ["p0", "p1", "p2"]),
        ([3, 4], ["p3", "p4"]),
    ]


def test_merge_packed_completion_shards_restores_original_order():
    completions = merge_packed_completion_shards(
        [
            ([2, 3], ["c2", "c3"]),
            ([0, 1], ["c0", "c1"]),
        ],
        total_prompts=4,
    )

    assert completions == ["c0", "c1", "c2", "c3"]


def test_merge_packed_completion_shards_rejects_missing_entries():
    with pytest.raises(ValueError, match="left gaps"):
        merge_packed_completion_shards(
            [
                ([0], ["c0"]),
                ([2], ["c2"]),
            ],
            total_prompts=3,
        )


def test_choose_packed_dispatch_plan_prefers_shared_pending_activation():
    statuses = [
        PackedReplicaStatus(worker_index=0, active_weight_id=5, pending_weight_id=6, busy=False),
        PackedReplicaStatus(worker_index=1, active_weight_id=5, pending_weight_id=6, busy=False),
    ]

    assert choose_packed_dispatch_plan(statuses) == PackedDispatchPlan(
        dispatch_weight_id=6,
        activate_weight_id=6,
    )


def test_choose_packed_dispatch_plan_keeps_shared_active_weight_when_pending_mismatches():
    statuses = [
        PackedReplicaStatus(worker_index=0, active_weight_id=7, pending_weight_id=8, busy=False),
        PackedReplicaStatus(worker_index=1, active_weight_id=7, pending_weight_id=9, busy=False),
    ]

    assert choose_packed_dispatch_plan(statuses) == PackedDispatchPlan(dispatch_weight_id=7)


def test_choose_packed_dispatch_plan_supports_single_replica_eval_subset():
    statuses = [
        PackedReplicaStatus(worker_index=1, active_weight_id=9, pending_weight_id=10, busy=False),
    ]

    assert choose_packed_dispatch_plan(statuses) == PackedDispatchPlan(
        dispatch_weight_id=10,
        activate_weight_id=10,
    )


def test_choose_packed_dispatch_plan_rejects_active_weight_divergence():
    statuses = [
        PackedReplicaStatus(worker_index=0, active_weight_id=7, pending_weight_id=None, busy=False),
        PackedReplicaStatus(worker_index=1, active_weight_id=8, pending_weight_id=None, busy=False),
    ]

    with pytest.raises(RuntimeError, match="disagree on active weights"):
        choose_packed_dispatch_plan(statuses)


def test_choose_packed_target_replica_indices_uses_both_replicas_for_train_when_eval_idle():
    replica_indices = choose_packed_target_replica_indices(
        request_kind=InferenceRequestKind.TRAIN,
        reserved_request_kinds={0: None, 1: None},
        eval_waiters=0,
    )

    assert replica_indices == (0, 1)


def test_choose_packed_target_replica_indices_pins_eval_to_replica_one():
    replica_indices = choose_packed_target_replica_indices(
        request_kind=InferenceRequestKind.EVAL,
        reserved_request_kinds={0: InferenceRequestKind.TRAIN, 1: None},
        eval_waiters=1,
    )

    assert replica_indices == (1,)


def test_choose_packed_target_replica_indices_routes_train_to_replica_zero_while_eval_waits():
    replica_indices = choose_packed_target_replica_indices(
        request_kind=InferenceRequestKind.TRAIN,
        reserved_request_kinds={0: None, 1: None},
        eval_waiters=1,
    )

    assert replica_indices == (0,)
