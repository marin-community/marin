# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import jax.numpy as jnp

import levanter.grug._moe.source_push_inbox as source_push_inbox
import levanter.grug._moe.source_push_combine as source_push_combine
import levanter.grug._moe.source_push_forward as source_push_forward
import levanter.grug._moe.source_push_inbox_blackwell as source_push_inbox_blackwell
import levanter.grug._moe.source_push_mlp as source_push_mlp
import levanter.grug._moe.source_push_w2_return as source_push_w2_return
import levanter.grug._moe.source_push_plan as source_push_plan
from levanter.grug._moe.ep_common import _clip_receiver_group_sizes
from levanter.grug._moe.source_push_inbox_profiles import (
    SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
    SOURCE_PUSH_PROFILE_STABLE_216,
    SOURCE_PUSH_PROFILES,
    source_push_profile_defaults,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_inbox.py"
REPRO_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "repro_source_push_inbox_queue.py"
W2_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_w2_return.py"
COMBINE_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_combine.py"
FORWARD_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_forward.py"
FORWARD_PUBLIC_COMPARE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_forward_public_compare.py"
)
MLP_FWD_BWD_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_mlp_fwd_bwd.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_source_push_inbox", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
source_push_cli = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(source_push_cli)
COMBINE_SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_source_push_combine", COMBINE_SCRIPT_PATH)
assert COMBINE_SCRIPT_SPEC is not None
source_push_combine_cli = importlib.util.module_from_spec(COMBINE_SCRIPT_SPEC)
assert COMBINE_SCRIPT_SPEC.loader is not None
COMBINE_SCRIPT_SPEC.loader.exec_module(source_push_combine_cli)
FORWARD_SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_source_push_forward", FORWARD_SCRIPT_PATH)
assert FORWARD_SCRIPT_SPEC is not None
source_push_forward_cli = importlib.util.module_from_spec(FORWARD_SCRIPT_SPEC)
assert FORWARD_SCRIPT_SPEC.loader is not None
FORWARD_SCRIPT_SPEC.loader.exec_module(source_push_forward_cli)
FORWARD_PUBLIC_COMPARE_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "bench_source_push_forward_public_compare",
    FORWARD_PUBLIC_COMPARE_SCRIPT_PATH,
)
assert FORWARD_PUBLIC_COMPARE_SCRIPT_SPEC is not None
source_push_forward_public_compare_cli = importlib.util.module_from_spec(FORWARD_PUBLIC_COMPARE_SCRIPT_SPEC)
assert FORWARD_PUBLIC_COMPARE_SCRIPT_SPEC.loader is not None
FORWARD_PUBLIC_COMPARE_SCRIPT_SPEC.loader.exec_module(source_push_forward_public_compare_cli)
MLP_FWD_BWD_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "bench_source_push_mlp_fwd_bwd",
    MLP_FWD_BWD_SCRIPT_PATH,
)
assert MLP_FWD_BWD_SCRIPT_SPEC is not None
source_push_mlp_fwd_bwd_cli = importlib.util.module_from_spec(MLP_FWD_BWD_SCRIPT_SPEC)
assert MLP_FWD_BWD_SCRIPT_SPEC.loader is not None
MLP_FWD_BWD_SCRIPT_SPEC.loader.exec_module(source_push_mlp_fwd_bwd_cli)

DIAGNOSTIC_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_inbox_diagnostics.py"
)
DIAGNOSTIC_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "bench_source_push_inbox_diagnostics",
    DIAGNOSTIC_SCRIPT_PATH,
)
assert DIAGNOSTIC_SCRIPT_SPEC is not None
source_push_diagnostic_cli = importlib.util.module_from_spec(DIAGNOSTIC_SCRIPT_SPEC)
assert DIAGNOSTIC_SCRIPT_SPEC.loader is not None
DIAGNOSTIC_SCRIPT_SPEC.loader.exec_module(source_push_diagnostic_cli)


@pytest.mark.parametrize(
    ("profile", "expected_routing", "expected_send_pipeline_depth"),
    [
        (SOURCE_PUSH_PROFILE_STABLE_216, "roughly_balanced", 1),
    ],
)
def test_source_push_profile_applies_current_best_candidate_defaults(
    profile, expected_routing, expected_send_pipeline_depth
):
    args = source_push_cli.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            profile,
        ]
    )

    config, settings = source_push_inbox.source_push_inbox_profile(profile)
    config.validate()
    assert args.routing == config.routing == expected_routing
    assert args.send_pipeline_depth == config.send_pipeline_depth == expected_send_pipeline_depth
    assert args.n_groups_per_job == config.n_groups_per_job == 2
    assert args.send_worker_programs_per_peer == config.send_worker_programs_per_peer == 2
    assert args.worker_programs_per_peer == config.worker_programs_per_peer == 32
    assert args.repeat_runs == settings.repeat_runs == 48


def test_source_push_profile_allows_explicit_overrides():
    args = source_push_cli.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--routing",
            "uniform",
            "--send-pipeline-depth",
            "2",
            "--send-worker-programs-per-peer",
            "1",
            "--worker-programs-per-peer",
            "16",
            "--repeat-runs",
            "3",
        ]
    )

    assert args.routing == "uniform"
    assert args.send_pipeline_depth == 2
    assert args.send_worker_programs_per_peer == 1
    assert args.worker_programs_per_peer == 16
    assert args.repeat_runs == 3
    assert args.n_groups_per_job == 2


def test_source_push_profile_defaults_are_copied():
    defaults = source_push_profile_defaults(SOURCE_PUSH_PROFILE_STABLE_216)
    defaults["routing"] = "uniform"

    fresh_defaults = source_push_profile_defaults(SOURCE_PUSH_PROFILE_STABLE_216)

    assert fresh_defaults["routing"] == "roughly_balanced"


def test_source_push_profile_returns_typed_config_and_run_settings():
    config, settings = source_push_inbox.source_push_inbox_profile(SOURCE_PUSH_PROFILE_STABLE_216)

    config.validate()
    assert config.routing == "roughly_balanced"
    assert config.n_groups_per_job == 2
    assert config.send_pipeline_depth == 1
    assert config.send_worker_programs_per_peer == 2
    assert config.worker_programs_per_peer == 32
    assert settings.warmup == 2
    assert settings.steps == 7
    assert settings.repeat_runs == 48
    assert not settings.check
    assert settings.separate_compile
    assert settings.progress_events


def test_source_push_profile_exposes_named_candidates():
    assert SOURCE_PUSH_PROFILES == (
        "none",
        SOURCE_PUSH_PROFILE_STABLE_216,
        SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072,
    )


def test_blackwell_source_push_profile_uses_target_shape():
    config, settings = source_push_inbox.source_push_inbox_profile(SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072)

    config.validate()
    assert config.tokens_per_rank == 65536
    assert config.hidden_dim == 3072
    assert config.intermediate_dim == 3072
    assert config.experts_per_rank == 32
    assert config.topk == 4
    assert config.entries_per_rank == 576
    assert config.inbox_slots == 24
    assert config.routing == "roughly_balanced"
    assert config.send_worker_programs_per_peer == 4
    assert config.send_pipeline_depth == 1
    assert settings.repeat_runs == 48


def test_blackwell_source_push_profile_uses_staged_strategy():
    assert (
        source_push_inbox_blackwell.source_push_inbox_architecture(SOURCE_PUSH_PROFILE_BLACKWELL_65K_D3072_I3072)
        == source_push_inbox_blackwell.SourcePushInboxArchitecture.BLACKWELL
    )
    assert (
        source_push_inbox_blackwell.source_push_inbox_architecture(SOURCE_PUSH_PROFILE_STABLE_216)
        == source_push_inbox_blackwell.SourcePushInboxArchitecture.HOPPER
    )
    assert (
        source_push_inbox_blackwell.BLACKWELL_SOURCE_PUSH_STRATEGY
        == source_push_inbox_blackwell.BlackwellSourcePushStrategy.STAGED_COPY_LOCAL_W13
    )
    assert (
        source_push_inbox_blackwell.BLACKWELL_PEER_REF_SUPPORT
        == source_push_inbox_blackwell.BlackwellPeerRefSupport.UNSUPPORTED_IN_WARPGROUP_LOWERING
    )


def test_blackwell_source_push_records_tuned_w13_config():
    config = source_push_inbox_blackwell.BLACKWELL_TARGET_W13_TUNING_CONFIG

    assert config.tile_m == 128
    assert config.tile_n == 128
    assert config.tile_k == 64
    assert config.max_concurrent_steps == 6
    assert config.collective
    assert config.grid_tile_width == 1
    assert config.grid_minor_dim == source_push_inbox_blackwell.BlackwellGridMinorDim.M
    assert config.epilogue_tile_n == 64

    w2_config = source_push_inbox_blackwell.BLACKWELL_TARGET_W2_TUNING_CONFIG
    assert w2_config.tile_n * (2 if w2_config.collective else 1) == 128
    assert w2_config.tile_k == config.tile_k
    assert w2_config.epilogue_tile_n == 64


def test_blackwell_source_push_performance_gate_uses_repeat_medians():
    w13_only_gate = source_push_inbox_blackwell.blackwell_performance_gate(
        baseline_useful_tflops_per_rank=1824.22,
        inbox_useful_tflops_per_rank=1596.34,
    )
    materialized_swiglu_gate = source_push_inbox_blackwell.blackwell_performance_gate(
        baseline_useful_tflops_per_rank=1534.20,
        inbox_useful_tflops_per_rank=1596.34,
    )

    assert w13_only_gate.passes
    assert materialized_swiglu_gate.passes
    assert w13_only_gate.required_useful_tflops_per_rank == pytest.approx(1094.532)
    assert materialized_swiglu_gate.required_useful_tflops_per_rank == pytest.approx(920.52)
    assert w13_only_gate.achieved_fraction == pytest.approx(1596.34 / 1824.22)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_useful_tflops_per_rank": 0.0, "inbox_useful_tflops_per_rank": 1.0},
        {"baseline_useful_tflops_per_rank": 1.0, "inbox_useful_tflops_per_rank": -1.0},
        {"baseline_useful_tflops_per_rank": 1.0, "inbox_useful_tflops_per_rank": 1.0, "required_fraction": 0.0},
    ],
)
def test_blackwell_source_push_performance_gate_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        source_push_inbox_blackwell.blackwell_performance_gate(**kwargs)


def test_disabled_modes_are_not_public_cli_choices():
    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--receiver-schedule", "ready_scan"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--inbox-storage", "alias"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--implementation", "send_only"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--hidden-output-mode", "full"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--hidden-compute-mode", "store_zero"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--num-send-sms", "2"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--num-sms", "32"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--diagnostic-variant", "semaphore_only"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--variants", "semaphore_only"])


def test_removed_experimental_modes_are_not_config_fields():
    for kwargs in (
        {"metadata_mode": "remote_slot"},
        {"receiver_schedule": "slot_group"},
        {"direct_self_compute": True},
        {"lowering_semantics": "warpgroup"},
        {"output_mode": "debug"},
        {"implementation": "send_only"},
        {"hidden_output_mode": "full"},
        {"hidden_compute_mode": "store_zero"},
        {"num_send_sms": 2},
        {"num_sms": 32},
    ):
        with pytest.raises(TypeError):
            source_push_inbox.PushInboxConfig(**kwargs)


def test_removed_send_pipeline_depths_are_rejected_by_config_validation():
    for kwargs in ({"send_pipeline_depth": 3},):
        with pytest.raises(ValueError):
            source_push_inbox.PushInboxConfig(**kwargs).validate()


def test_compact_routing_inputs_match_synthetic_queue_metadata():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=4,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=8,
        topk=2,
    )

    synthetic_inputs = source_push_inbox._make_routing_inputs(config)
    compact_inputs = source_push_inbox._make_compact_routing_inputs(config)

    assert np.array_equal(compact_inputs.send_meta, synthetic_inputs.send_meta)
    assert np.array_equal(compact_inputs.recv_meta, synthetic_inputs.recv_meta)
    assert compact_inputs.queue_stats["input_mode"] == "compact_routing"
    assert compact_inputs.queue_stats["dropped_entries_total"] == 0
    assert (
        compact_inputs.queue_stats["compact_pack_rows_total"] == config.ep_size * config.tokens_per_rank * config.topk
    )
    assert not np.all(compact_inputs.x[0, 0, 0, 0, :] == compact_inputs.x[0, 0, 0, 0, 0])


def test_source_push_plan_offsets_match_accepted_count_prefixes():
    selected_experts = np.asarray(
        [
            [[0, 0], [1, 2], [2, 3], [3, 3]],
            [[0, 1], [1, 1], [2, 2], [3, 3]],
        ],
        dtype=np.int32,
    )
    combine_weights = np.arange(selected_experts.size, dtype=np.float32).reshape(selected_experts.shape) / 100.0

    plan = source_push_plan.build_source_push_plan(
        jnp.asarray(selected_experts),
        jnp.asarray(combine_weights),
        ep_size=2,
        experts_per_rank=2,
        block_m=2,
        capacity_factor=2.0,
        entries_per_dst=3,
    )

    expected_counts = np.asarray(
        [
            [[2, 1], [2, 3]],
            [[1, 3], [2, 2]],
        ],
        dtype=np.int32,
    )
    expected_rows_per_local_expert = np.asarray([[3, 4], [4, 5]], dtype=np.int32)
    expected_expert_base = np.asarray([[0, 3], [0, 4]], dtype=np.int32)
    expected_src_base = np.asarray(
        [
            [[0, 0], [2, 1]],
            [[0, 0], [2, 3]],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(np.asarray(plan.counts_by_src_dst_expert), expected_counts)
    np.testing.assert_array_equal(np.asarray(plan.rows_per_local_expert), expected_rows_per_local_expert)
    np.testing.assert_array_equal(np.asarray(plan.expert_base), expected_expert_base)
    np.testing.assert_array_equal(np.asarray(plan.src_base_by_expert), expected_src_base)
    assert int(np.asarray(plan.dropped_routes)) == 0

    assignment_ids = np.asarray(plan.assignment_ids)
    token_ids = np.asarray(plan.token_ids)
    route_slots = np.asarray(plan.route_slots)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    send_meta = np.asarray(plan.send_meta)
    for src, dst_ordinal, entry, row in np.argwhere(valid_mask):
        assignment_id = int(assignment_ids[src, dst_ordinal, entry, row])
        token = int(token_ids[src, dst_ordinal, entry, row])
        route_slot = int(route_slots[src, dst_ordinal, entry, row])
        dst = (src + dst_ordinal) % 2
        expert = int(local_experts[src, dst_ordinal, entry])

        assert token == assignment_id // selected_experts.shape[-1]
        assert route_slot == assignment_id % selected_experts.shape[-1]
        assert int(selected_experts[src, token, route_slot]) == dst * 2 + expert
        assert int(send_meta[src, dst_ordinal, entry, source_push_plan.SOURCE_PUSH_META_SRC_RANK]) == src
        assert int(send_meta[src, dst_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_EXPERT]) == expert
        assert int(send_meta[src, dst_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_ROW_START]) == int(
            local_row_starts[src, dst_ordinal, entry]
        )


def test_source_push_plan_uses_stable_expert_order_and_masks_padding():
    selected_experts = np.asarray(
        [
            [[3, 0], [2, 0], [3, 1], [0, 2], [2, 1], [0, 3]],
            [[3, 0], [2, 0], [3, 1], [0, 2], [2, 1], [0, 3]],
        ],
        dtype=np.int32,
    )
    combine_weights = np.ones_like(selected_experts, dtype=np.float32)

    plan = source_push_plan.build_source_push_plan(
        jnp.asarray(selected_experts),
        jnp.asarray(combine_weights),
        ep_size=2,
        experts_per_rank=2,
        block_m=3,
        capacity_factor=2.0,
        entries_per_dst=3,
    )

    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    local_experts = np.asarray(plan.local_experts)
    local_row_starts = np.asarray(plan.local_row_starts)
    send_meta = np.asarray(plan.send_meta)

    src = 0
    dst0_ordinal = source_push_plan.dst_ordinal(src, 0, ep_size=2)
    dst1_ordinal = source_push_plan.dst_ordinal(src, 1, ep_size=2)

    np.testing.assert_array_equal(assignment_ids[src, dst0_ordinal, 0], [1, 3, 6])
    np.testing.assert_array_equal(assignment_ids[src, dst0_ordinal, 1], [10, -1, -1])
    np.testing.assert_array_equal(assignment_ids[src, dst0_ordinal, 2], [5, 9, -1])
    np.testing.assert_array_equal(valid_mask[src, dst0_ordinal, 0], [True, True, True])
    np.testing.assert_array_equal(valid_mask[src, dst0_ordinal, 1], [True, False, False])
    np.testing.assert_array_equal(valid_mask[src, dst0_ordinal, 2], [True, True, False])
    np.testing.assert_array_equal(local_experts[src, dst0_ordinal], [0, 0, 1])
    np.testing.assert_array_equal(local_row_starts[src, dst0_ordinal], [0, 3, 0])
    np.testing.assert_array_equal(
        send_meta[src, dst0_ordinal, :, source_push_plan.SOURCE_PUSH_META_VALID_ROWS],
        [3, 1, 2],
    )

    np.testing.assert_array_equal(assignment_ids[src, dst1_ordinal, 0], [2, 7, 8])
    np.testing.assert_array_equal(assignment_ids[src, dst1_ordinal, 1], [0, 4, 11])
    np.testing.assert_array_equal(assignment_ids[src, dst1_ordinal, 2], [-1, -1, -1])
    np.testing.assert_array_equal(valid_mask[src, dst1_ordinal, 0], [True, True, True])
    np.testing.assert_array_equal(valid_mask[src, dst1_ordinal, 1], [True, True, True])
    np.testing.assert_array_equal(valid_mask[src, dst1_ordinal, 2], [False, False, False])
    np.testing.assert_array_equal(local_experts[src, dst1_ordinal], [0, 1, -1])
    np.testing.assert_array_equal(local_row_starts[src, dst1_ordinal], [0, 0, 0])
    np.testing.assert_array_equal(
        send_meta[src, dst1_ordinal, :, source_push_plan.SOURCE_PUSH_META_VALID_ROWS],
        [3, 3, 0],
    )


def test_source_push_plan_capacity_clipping_matches_ep_reference_and_keeps_prefix_assignments():
    selected_experts = np.asarray(
        [
            [[0, 0], [0, 1], [0, 1], [0, 1]],
            [[0, 0], [1, 1], [0, 1], [0, 1]],
        ],
        dtype=np.int32,
    )
    combine_weights = np.ones_like(selected_experts, dtype=np.float32)
    group_sizes = np.stack(
        [np.bincount(source.reshape(-1), minlength=4).astype(np.int32) for source in selected_experts],
        axis=0,
    )
    expected_counts = np.asarray(
        _clip_receiver_group_sizes(
            jnp.asarray(group_sizes),
            local_expert_size=2,
            receiver_capacity=2,
        )
    ).reshape(2, 2, 2)

    plan = source_push_plan.build_source_push_plan(
        jnp.asarray(selected_experts),
        jnp.asarray(combine_weights),
        ep_size=2,
        experts_per_rank=2,
        block_m=2,
        capacity_factor=0.25,
        entries_per_dst=1,
    )

    np.testing.assert_array_equal(np.asarray(plan.counts_by_src_dst_expert), expected_counts)
    assert int(np.asarray(plan.dropped_routes)) == selected_experts.size - int(np.sum(expected_counts))

    src = 0
    dst = 0
    dst_ordinal = source_push_plan.dst_ordinal(src, dst, ep_size=2)
    accepted = plan.assignment_ids[src, dst_ordinal, 0]

    np.testing.assert_array_equal(np.asarray(accepted), np.asarray([0, 1], dtype=np.int32))
    assert not np.any(np.asarray(plan.valid_mask[1]))


def test_source_push_plan_rejects_queue_capacity_overflow():
    selected_experts = np.zeros((2, 4, 1), dtype=np.int32)
    combine_weights = np.ones_like(selected_experts, dtype=np.float32)

    with pytest.raises(ValueError, match="source-push queue capacity overflow"):
        source_push_plan.build_source_push_plan(
            jnp.asarray(selected_experts),
            jnp.asarray(combine_weights),
            ep_size=2,
            experts_per_rank=1,
            block_m=2,
            capacity_factor=1.0,
            entries_per_dst=1,
        )


def test_source_push_plan_inputs_use_source_padded_row_starts():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=5,
        topk=2,
        capacity_factor=1.25,
    )

    host_inputs = source_push_inbox._make_source_push_plan_inputs(config)
    valid_rows = host_inputs.send_meta[..., 3]
    live_entries = valid_rows > 0
    live_mask = source_push_inbox._hidden_live_row_mask(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )

    assert not host_inputs.use_exact_expert_major
    assert host_inputs.queue_stats["input_mode"] == "source_push_plan"
    assert host_inputs.queue_stats["row_start_mode"] == source_push_inbox.ROW_START_MODE_SOURCE_PADDED
    assert host_inputs.queue_stats["row_layout"] == source_push_inbox.ROW_LAYOUT_SOURCE_PADDED_EXPERT_MAJOR
    assert int(np.sum(live_mask)) == int(np.sum(live_entries) * config.block_m)
    assert int(np.sum(live_entries) * config.block_m) > int(np.sum(valid_rows))
    assert host_inputs.queue_stats["plan_padded_rows_total"] == int(np.sum(live_entries) * config.block_m)
    assert host_inputs.queue_stats["plan_layout_rows_total"] == host_inputs.queue_stats["plan_padded_rows_total"]
    assert host_inputs.queue_stats["plan_layout_padding_rows_total"] == int(
        host_inputs.queue_stats["plan_layout_rows_total"] - np.sum(valid_rows)
    )

    src = 1
    dst = 0
    dst_ordinal = source_push_inbox._dst_ordinal(config, src, dst)
    first_live_entry = int(np.flatnonzero(live_entries[src, dst_ordinal])[0])
    row_start = host_inputs.send_meta[src, dst_ordinal, first_live_entry, 2]

    assert row_start >= config.block_m


def test_exact_source_push_plan_inputs_use_count_derived_row_starts():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=1,
        inbox_slots=1,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=4,
        topk=1,
        capacity_factor=1.25,
    )

    host_inputs = source_push_inbox._make_exact_source_push_plan_inputs(config)
    valid_rows = host_inputs.send_meta[..., 3]
    live_mask = source_push_inbox._hidden_live_row_mask(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )

    assert host_inputs.use_exact_expert_major
    assert host_inputs.queue_stats["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert host_inputs.queue_stats["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert host_inputs.queue_stats["plan_layout_padding_rows_total"] == 0
    assert host_inputs.queue_stats["plan_layout_rows_total"] == int(np.sum(valid_rows))
    assert int(np.sum(live_mask)) == int(np.sum(valid_rows))

    src = 1
    dst = 0
    dst_ordinal = source_push_inbox._dst_ordinal(config, src, dst)
    entry = 0
    recv_src_ordinal = source_push_inbox._recv_src_ordinal(config, dst, src)

    assert host_inputs.send_meta[src, dst_ordinal, entry, 2] == 0
    np.testing.assert_array_equal(
        host_inputs.recv_meta[dst, recv_src_ordinal, entry],
        host_inputs.send_meta[src, dst_ordinal, entry],
    )
    assert host_inputs.expert_base[dst, 0] + host_inputs.src_base_by_expert[dst, src, 0] == config.block_m


def test_exact_source_push_plan_inputs_reject_tail_blocks():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=5,
        topk=1,
        capacity_factor=1.25,
    )

    with pytest.raises(ValueError, match="block_m-aligned live blocks"):
        source_push_inbox._make_exact_source_push_plan_inputs(config)


def test_exact_reference_hidden_masks_tail_rows_before_next_source_slice():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=1,
        inbox_slots=1,
        hidden_dim=2,
        intermediate_dim=1,
        block_m=2,
        block_k=1,
        block_n=1,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=2,
        tokens_per_rank=2,
        topk=1,
    )
    x_host = np.zeros((config.ep_size, config.traffic_fanout, config.entries_per_rank, config.block_m, 2))
    send_meta = np.zeros(
        (config.ep_size, config.traffic_fanout, config.entries_per_rank, source_push_inbox.META_FIELDS),
        dtype=np.int32,
    )
    expert_base = np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base_by_expert = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base_by_expert[0, 1, 0] = 1
    w_host = np.ones((config.ep_size, config.experts_per_rank, config.hidden_dim, 2 * config.intermediate_dim))

    dst = 0
    src0_dst_ordinal = source_push_inbox._dst_ordinal(config, 0, dst)
    src1_dst_ordinal = source_push_inbox._dst_ordinal(config, 1, dst)
    send_meta[0, src0_dst_ordinal, 0, :] = (0, 0, 0, 1)
    send_meta[1, src1_dst_ordinal, 0, :] = (1, 0, 0, 2)
    x_host[0, src0_dst_ordinal, 0, 0, :] = 1.0
    x_host[0, src0_dst_ordinal, 0, 1, :] = 100.0
    x_host[1, src1_dst_ordinal, 0, :, :] = 2.0

    hidden = source_push_inbox._reference_hidden(
        config,
        x_host,
        send_meta,
        w_host,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=True,
    )
    live_mask = source_push_inbox._hidden_live_row_mask(
        config,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=True,
    )

    src0_expected = 2.0 * (1.0 / (1.0 + np.exp(-2.0))) * 2.0
    src1_expected = 4.0 * (1.0 / (1.0 + np.exp(-4.0))) * 4.0
    np.testing.assert_allclose(hidden[dst, 0, 0], src0_expected)
    np.testing.assert_allclose(hidden[dst, 1, 0], src1_expected)
    np.testing.assert_array_equal(live_mask[dst, :3], [True, True, True])
    assert not np.any(hidden[dst, 3:, :])


def test_source_push_reference_hidden_uses_metadata_row_start():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=3,
        inbox_slots=1,
        hidden_dim=2,
        intermediate_dim=2,
        block_m=2,
        block_k=1,
        block_n=1,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=2,
        tokens_per_rank=2,
        topk=1,
    )
    x_host = np.zeros((config.ep_size, config.traffic_fanout, config.entries_per_rank, config.block_m, 2))
    send_meta = np.zeros(
        (config.ep_size, config.traffic_fanout, config.entries_per_rank, source_push_inbox.META_FIELDS),
        dtype=np.int32,
    )
    w_host = np.ones((config.ep_size, config.experts_per_rank, config.hidden_dim, 2 * config.intermediate_dim))

    src = 0
    dst = 1
    dst_ordinal = source_push_inbox._dst_ordinal(config, src, dst)
    entry = 1
    metadata_row_start = 4
    queue_order_row_start = (
        source_push_inbox._recv_src_ordinal(config, dst, src) * config.entries_per_rank + entry
    ) * (config.block_m)
    assert metadata_row_start != queue_order_row_start

    x_host[src, dst_ordinal, entry, :, :] = 1.0
    send_meta[src, dst_ordinal, entry, :] = (src, 0, metadata_row_start, config.block_m)

    hidden = source_push_inbox._reference_hidden(config, x_host, send_meta, w_host)
    live_mask = source_push_inbox._hidden_live_row_mask(config, send_meta)

    assert np.any(hidden[dst, metadata_row_start : metadata_row_start + config.block_m, :] != 0)
    assert np.all(hidden[dst, queue_order_row_start : queue_order_row_start + config.block_m, :] == 0)
    np.testing.assert_array_equal(live_mask[dst, metadata_row_start : metadata_row_start + config.block_m], True)
    np.testing.assert_array_equal(
        live_mask[dst, queue_order_row_start : queue_order_row_start + config.block_m], False
    )


def test_source_push_w2_destination_return_reorders_to_source_queue():
    config = source_push_inbox.PushInboxConfig(
        ep_size=3,
        entries_per_rank=1,
        hidden_dim=1,
        intermediate_dim=1,
        block_m=1,
        block_k=1,
        block_n=1,
    )
    return_by_destination = np.zeros((3, 3, 1, 1, 1), dtype=np.float32)
    for dst in range(config.ep_size):
        for src_ordinal in range(config.ep_size):
            return_by_destination[dst, src_ordinal, 0, 0, 0] = 100 * dst + src_ordinal

    source_queue = np.asarray(
        source_push_w2_return.source_queue_from_destination_return(config, return_by_destination)
    )

    for src in range(config.ep_size):
        for dst_ordinal in range(config.ep_size):
            dst = (src + dst_ordinal) % config.ep_size
            recv_src_ordinal = (src - dst) % config.ep_size
            assert source_queue[src, dst_ordinal, 0, 0, 0] == 100 * dst + recv_src_ordinal


def test_source_push_w2_reference_uses_recv_metadata_for_expert_major_rows():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=3,
        intermediate_dim=2,
        block_m=2,
        block_k=1,
        block_n=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=2,
        tokens_per_rank=2,
        topk=1,
    )
    hidden = np.arange(config.ep_size * 8 * config.intermediate_dim, dtype=np.float32).reshape(
        config.ep_size, 8, config.intermediate_dim
    )
    recv_meta = np.zeros(
        (config.ep_size, config.ep_size, config.entries_per_rank, source_push_inbox.META_FIELDS),
        dtype=np.int32,
    )
    w_down = (
        np.arange(
            config.ep_size * config.experts_per_rank * config.intermediate_dim * config.hidden_dim,
            dtype=np.float32,
        ).reshape(config.ep_size, config.experts_per_rank, config.intermediate_dim, config.hidden_dim)
        + 1.0
    )
    dst = 1
    src_ordinal = 1
    entry = 0
    expert = 1
    row_start = 3
    recv_meta[dst, src_ordinal, entry, :] = (0, expert, row_start, config.block_m)

    return_by_destination = np.asarray(
        source_push_w2_return.reference_w2_return_by_destination(config, hidden, recv_meta, w_down)
    )

    expected = hidden[dst, row_start : row_start + config.block_m] @ w_down[dst, expert]
    np.testing.assert_allclose(return_by_destination[dst, src_ordinal, entry], expected)
    assert not np.any(return_by_destination[0])
    assert not np.any(return_by_destination[dst, 0])


def test_source_push_w2_source_plan_inputs_match_reference_layout():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=5,
        topk=2,
        capacity_factor=1.25,
    )

    inputs = source_push_w2_return.make_w2_return_source_plan_inputs(config)
    return_by_destination = source_push_w2_return.reference_w2_return_by_destination(
        config,
        inputs.hidden,
        inputs.recv_meta,
        inputs.w_down,
    )
    source_queue = np.asarray(
        source_push_w2_return.source_queue_from_destination_return(config, return_by_destination)
    )

    assert inputs.queue_stats["w2_input_mode"] == "source_push_plan"
    assert inputs.queue_stats["w2_hidden_input_mode"] == "synthetic"
    assert inputs.hidden.shape == (config.ep_size, config.hidden_rows_per_rank, config.intermediate_dim)
    assert source_queue.shape == (
        config.ep_size,
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
    )
    assert np.count_nonzero(source_queue) > 0


def test_source_push_w2_source_plan_inputs_can_use_w13_reference_hidden():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=5,
        topk=2,
        capacity_factor=1.25,
    )

    inputs = source_push_w2_return.make_w2_return_source_plan_inputs(config, hidden_input_mode="w13_reference")

    assert inputs.queue_stats["w2_hidden_input_mode"] == "w13_reference"
    assert inputs.hidden.shape == (config.ep_size, config.hidden_rows_per_rank, config.intermediate_dim)
    assert np.count_nonzero(inputs.hidden) > 0


def test_source_push_w2_exact_source_plan_inputs_use_count_derived_row_starts():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=1,
        inbox_slots=1,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=4,
        topk=1,
        capacity_factor=1.25,
    )

    inputs = source_push_w2_return.make_w2_return_exact_source_plan_inputs(
        config,
        hidden_input_mode=source_push_w2_return.W2_HIDDEN_INPUT_W13_REFERENCE,
    )
    return_by_destination = source_push_w2_return.reference_w2_return_by_destination(
        config,
        inputs.hidden,
        inputs.recv_meta,
        inputs.w_down,
        inputs.expert_base,
        inputs.src_base_by_expert,
        use_exact_expert_major=inputs.use_exact_expert_major,
    )

    src = 1
    dst = 0
    recv_src_ordinal = source_push_inbox._recv_src_ordinal(config, dst, src)
    entry = 0
    local_row_start = inputs.recv_meta[dst, recv_src_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_ROW_START]
    expert = inputs.recv_meta[dst, recv_src_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_EXPERT]
    exact_row_start = inputs.expert_base[dst, expert] + inputs.src_base_by_expert[dst, src, expert] + local_row_start

    assert inputs.use_exact_expert_major
    assert inputs.queue_stats["w2_input_mode"] == "exact_source_push_plan"
    assert inputs.queue_stats["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert inputs.queue_stats["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert local_row_start == 0
    assert exact_row_start == config.block_m
    assert inputs.hidden[dst, local_row_start, 0] != inputs.hidden[dst, exact_row_start, 0]
    expected = inputs.hidden[dst, exact_row_start : exact_row_start + config.block_m] @ inputs.w_down[dst, expert]
    np.testing.assert_allclose(
        np.asarray(return_by_destination[dst, recv_src_ordinal, entry], dtype=np.float32),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_source_push_w2_plan_reference_matches_destination_reference_for_non_symmetric_ep():
    config = source_push_inbox.PushInboxConfig(
        ep_size=3,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=4,
        intermediate_dim=4,
        block_m=2,
        block_k=2,
        block_n=2,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    inputs = source_push_forward.make_source_push_forward_source_plan_inputs(config)
    hidden = source_push_inbox._reference_hidden(
        config,
        inputs.x,
        inputs.send_meta,
        inputs.w_gate_up,
        inputs.expert_base,
        inputs.src_base_by_expert,
        use_exact_expert_major=False,
    )

    plan_return = source_push_plan.source_push_w2_return(
        jnp.asarray(hidden, dtype=jnp.bfloat16),
        jnp.asarray(inputs.w_down, dtype=jnp.bfloat16),
        inputs.plan,
        expert_base=inputs.expert_base,
        src_base_by_expert=inputs.src_base_by_expert,
    )
    destination_return = source_push_w2_return.reference_w2_return_by_destination(
        config,
        jnp.asarray(hidden, dtype=jnp.bfloat16),
        inputs.recv_meta,
        jnp.asarray(inputs.w_down, dtype=jnp.bfloat16),
    )
    destination_source_queue = source_push_w2_return.source_queue_from_destination_return(config, destination_return)

    np.testing.assert_allclose(
        np.asarray(plan_return, dtype=np.float32),
        np.asarray(destination_source_queue, dtype=np.float32),
        atol=0,
        rtol=0,
    )


def test_source_push_combine_inputs_invert_queue_rows_to_route_slots():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )

    inputs = source_push_combine.make_source_push_combine_source_plan_inputs(config)
    expected = np.asarray(
        source_push_combine.reference_source_push_combine(
            jnp.asarray(inputs.return_y, dtype=jnp.bfloat16),
            inputs.plan,
        ),
        dtype=np.float32,
    )
    valid_routes = np.argwhere(inputs.route_valid_mask)
    src, token, route_slot = valid_routes[0]
    dst_ord = inputs.queue_dst_ord[src, token, route_slot]
    entry = inputs.queue_entry[src, token, route_slot]
    row = inputs.queue_row[src, token, route_slot]

    assert inputs.queue_stats["combine_mode"] == source_push_combine.SOURCE_COMBINE_MODE_DIRECT_GATHER_SUM
    assert inputs.queue_stats["dropped_routes"] == 0
    assert inputs.return_y.shape == (
        config.ep_size,
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
    )
    assert inputs.route_valid_mask.shape == (config.ep_size, config.tokens_per_rank, config.topk)
    assert int(np.sum(inputs.route_valid_mask)) == config.ep_size * config.tokens_per_rank * config.topk
    assert int(np.asarray(inputs.plan.token_ids[src, dst_ord, entry, row])) == int(token)
    assert int(np.asarray(inputs.plan.route_slots[src, dst_ord, entry, row])) == int(route_slot)
    assert np.count_nonzero(expected) > 0


def test_source_push_combine_ignores_invalid_padded_queue_rows():
    selected_experts = np.asarray([[[0, 1], [0, 1], [0, 1]]], dtype=np.int32)
    combine_weights = np.asarray([[[1.0, 0.5], [2.0, 0.25], [3.0, 0.125]]], dtype=np.float32)
    plan = source_push_plan.build_source_push_plan(
        jnp.asarray(selected_experts),
        jnp.asarray(combine_weights),
        ep_size=1,
        experts_per_rank=2,
        block_m=4,
        capacity_factor=2.0,
        entries_per_dst=2,
    )
    return_y = np.zeros((*plan.assignment_ids.shape, 2), dtype=np.float32)
    assignment_ids = np.asarray(plan.assignment_ids)
    valid_mask = np.asarray(plan.valid_mask)
    for src, dst_ord, entry, row in np.argwhere(valid_mask):
        assignment_id = int(assignment_ids[src, dst_ord, entry, row])
        return_y[src, dst_ord, entry, row, :] = [assignment_id + 1.0, 10.0 * (assignment_id + 1.0)]

    expected = np.asarray(source_push_plan.source_push_combine(jnp.asarray(return_y), plan), dtype=np.float32)
    poisoned_return_y = return_y.copy()
    poisoned_return_y[~valid_mask] = 1.0e6
    observed = np.asarray(source_push_plan.source_push_combine(jnp.asarray(poisoned_return_y), plan), dtype=np.float32)

    np.testing.assert_allclose(observed, expected, atol=0, rtol=0)
    assert np.count_nonzero(expected) > 0
    assert np.any(~valid_mask)


def test_source_push_forward_inputs_share_one_plan_across_all_stages():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )

    inputs = source_push_forward.make_source_push_forward_source_plan_inputs(config)
    expected = np.asarray(source_push_forward.reference_source_push_forward(config, inputs), dtype=np.float32)
    valid_routes = np.argwhere(inputs.route_valid_mask)
    src, token, route_slot = valid_routes[0]
    dst_ord = inputs.queue_dst_ord[src, token, route_slot]
    entry = inputs.queue_entry[src, token, route_slot]
    row = inputs.queue_row[src, token, route_slot]

    assert inputs.queue_stats["forward_mode"] == "w13_w2_direct_return_combine"
    assert inputs.queue_stats["combine_mode"] == source_push_combine.SOURCE_COMBINE_MODE_DIRECT_GATHER_SUM
    assert inputs.queue_stats["dropped_routes"] == 0
    assert inputs.x.shape == (
        config.ep_size,
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
    )
    assert inputs.h_route_weights.shape == (
        config.ep_size,
        config.hidden_rows_per_rank,
    )
    expected_h_route_weights = np.asarray(
        source_push_plan.source_push_h_row_route_weights_jax(
            jnp.asarray(inputs.route_combine_weights),
            inputs.plan,
            inputs.send_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=inputs.use_exact_expert_major,
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(inputs.h_route_weights, expected_h_route_weights, atol=0, rtol=0)
    assert expected.shape == (config.ep_size, config.tokens_per_rank, config.hidden_dim)
    assert np.count_nonzero(expected) > 0
    assert int(np.asarray(inputs.plan.token_ids[src, dst_ord, entry, row])) == int(token)
    assert int(np.asarray(inputs.plan.route_slots[src, dst_ord, entry, row])) == int(route_slot)


def test_source_push_forward_device_inputs_from_plan_use_dynamic_jax_arrays():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    host_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )
    dynamic_x = jnp.asarray(raw_inputs.x + 0.25, dtype=jnp.float32)
    dynamic_weights = jnp.asarray(0.125 + raw_inputs.combine_weights * 0.25, dtype=jnp.float32)
    dynamic_w13 = jnp.asarray(raw_inputs.w_gate_up + 0.5, dtype=jnp.float32)
    dynamic_w2 = jnp.asarray(raw_inputs.w_down - 0.25, dtype=jnp.float32)

    device_inputs = source_push_forward.device_source_push_forward_inputs_from_plan(
        config,
        host_inputs,
        dynamic_x,
        dynamic_weights,
        dynamic_w13,
        dynamic_w2,
    )
    expected_packed_x = source_push_plan.pack_source_push_tokens_jax(dynamic_x, host_inputs.plan).astype(jnp.bfloat16)
    expected_h_route_weights = source_push_plan.source_push_h_row_route_weights_jax(
        dynamic_weights,
        host_inputs.plan,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(np.asarray(device_inputs.x), np.asarray(expected_packed_x))
    np.testing.assert_array_equal(np.asarray(device_inputs.h_route_weights), np.asarray(expected_h_route_weights))
    np.testing.assert_array_equal(np.asarray(device_inputs.w_gate_up), np.asarray(dynamic_w13.astype(jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(device_inputs.w_down), np.asarray(dynamic_w2.astype(jnp.bfloat16)))


def test_source_push_forward_plan_inputs_do_not_capture_differentiable_arrays():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    full_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )
    plan_inputs = source_push_forward.make_source_push_forward_plan_inputs(
        config,
        raw_inputs.selected_experts,
    )
    dynamic_x = jnp.asarray(raw_inputs.x + 0.25, dtype=jnp.float32)
    dynamic_weights = jnp.asarray(0.125 + raw_inputs.combine_weights * 0.25, dtype=jnp.float32)
    dynamic_w13 = jnp.asarray(raw_inputs.w_gate_up + 0.5, dtype=jnp.float32)
    dynamic_w2 = jnp.asarray(raw_inputs.w_down - 0.25, dtype=jnp.float32)

    np.testing.assert_array_equal(plan_inputs.send_meta, full_inputs.send_meta)
    np.testing.assert_array_equal(plan_inputs.recv_meta, full_inputs.recv_meta)
    np.testing.assert_array_equal(plan_inputs.expert_base, full_inputs.expert_base)
    np.testing.assert_array_equal(plan_inputs.src_base_by_expert, full_inputs.src_base_by_expert)
    np.testing.assert_array_equal(plan_inputs.queue_dst_ord, full_inputs.queue_dst_ord)
    np.testing.assert_array_equal(plan_inputs.queue_entry, full_inputs.queue_entry)
    np.testing.assert_array_equal(plan_inputs.queue_row, full_inputs.queue_row)
    np.testing.assert_array_equal(plan_inputs.route_valid_mask, full_inputs.route_valid_mask)
    np.testing.assert_array_equal(plan_inputs.x, np.zeros_like(plan_inputs.x))
    np.testing.assert_array_equal(plan_inputs.w_gate_up, np.zeros_like(plan_inputs.w_gate_up))
    np.testing.assert_array_equal(plan_inputs.w_down, np.zeros_like(plan_inputs.w_down))

    device_inputs = source_push_forward.device_source_push_forward_inputs_from_plan(
        config,
        plan_inputs,
        dynamic_x,
        dynamic_weights,
        dynamic_w13,
        dynamic_w2,
    )
    expected_packed_x = source_push_plan.pack_source_push_tokens_jax(dynamic_x, plan_inputs.plan).astype(jnp.bfloat16)
    expected_h_route_weights = source_push_plan.source_push_h_row_route_weights_jax(
        dynamic_weights,
        plan_inputs.plan,
        plan_inputs.send_meta,
        plan_inputs.expert_base,
        plan_inputs.src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=plan_inputs.use_exact_expert_major,
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(np.asarray(device_inputs.x), np.asarray(expected_packed_x))
    np.testing.assert_array_equal(np.asarray(device_inputs.h_route_weights), np.asarray(expected_h_route_weights))
    np.testing.assert_array_equal(np.asarray(device_inputs.w_gate_up), np.asarray(dynamic_w13.astype(jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(device_inputs.w_down), np.asarray(dynamic_w2.astype(jnp.bfloat16)))


def test_source_push_forward_h_reference_matches_mlp_boundary():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )

    route_table = source_push_mlp.source_push_mlp_route_table_from_plan(
        inputs.plan,
        src_base_by_expert=inputs.src_base_by_expert,
    )
    expected = source_push_mlp.source_push_moe_mlp_reference(
        route_table,
        jnp.asarray(raw_inputs.x, dtype=jnp.bfloat16),
        jnp.asarray(raw_inputs.combine_weights, dtype=jnp.float32),
        jnp.asarray(raw_inputs.w_gate_up, dtype=jnp.bfloat16),
        jnp.asarray(raw_inputs.w_down, dtype=jnp.bfloat16),
    )
    observed = source_push_forward.reference_source_push_forward_h(config, inputs)

    assert route_table.expert_capacity <= inputs.queue_stats["hidden_capacity_rows_per_rank"]
    np.testing.assert_allclose(
        np.asarray(observed, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-2,
        atol=3e-4,
    )


def test_source_push_forward_with_h_returns_route_weight_independent_preactivation():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )

    observed_y, observed_h, dropped_routes = source_push_forward.source_push_forward_with_h(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        implementation="reference",
    )
    expected_y, expected_h = source_push_forward.reference_source_push_forward_with_h(config, inputs)

    rescaled_weights = 0.125 + raw_inputs.combine_weights * 0.25
    rescaled_y, rescaled_h, rescaled_dropped_routes = source_push_forward.source_push_forward_with_h(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        rescaled_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        implementation="reference",
    )

    assert int(dropped_routes) == 0
    assert int(rescaled_dropped_routes) == 0
    assert observed_h.shape == (config.ep_size, config.hidden_rows_per_rank, 2 * config.intermediate_dim)
    np.testing.assert_allclose(
        np.asarray(observed_y, dtype=np.float32),
        np.asarray(expected_y, dtype=np.float32),
        rtol=1e-2,
        atol=3e-4,
    )
    np.testing.assert_allclose(np.asarray(observed_h), np.asarray(expected_h), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(rescaled_h), np.asarray(observed_h), atol=0, rtol=0)
    assert not np.allclose(np.asarray(rescaled_y, dtype=np.float32), np.asarray(observed_y, dtype=np.float32))


def test_source_push_w13_h_reference_stores_preactivation_before_swiglu():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=6,
        topk=2,
        capacity_factor=1.25,
    )
    inputs = source_push_forward.make_source_push_forward_source_plan_inputs(config)

    h = source_push_inbox._reference_h_flat(
        config,
        inputs.x,
        inputs.send_meta,
        inputs.w_gate_up,
        inputs.expert_base,
        inputs.src_base_by_expert,
        use_exact_expert_major=inputs.use_exact_expert_major,
    )
    hidden = source_push_inbox._reference_hidden(
        config,
        inputs.x,
        inputs.send_meta,
        inputs.w_gate_up,
        inputs.expert_base,
        inputs.src_base_by_expert,
        use_exact_expert_major=inputs.use_exact_expert_major,
    )

    src, dst_ord, entry = np.argwhere(inputs.send_meta[..., source_push_plan.SOURCE_PUSH_META_VALID_ROWS] > 0)[0]
    dst = (src + dst_ord) % config.ep_size
    expert = inputs.send_meta[src, dst_ord, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_EXPERT]
    row = inputs.send_meta[src, dst_ord, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_ROW_START]
    if inputs.use_exact_expert_major:
        row += inputs.expert_base[dst, expert] + inputs.src_base_by_expert[dst, src, expert]

    gate = h[dst, row, : config.intermediate_dim]
    up = h[dst, row, config.intermediate_dim :]
    expected_activation = gate * (1.0 / (1.0 + np.exp(-gate))) * up

    assert (
        source_push_inbox._hidden_output_shape_for_kernel(
            config,
            source_push_inbox.DIAGNOSTIC_VARIANT_FULL,
            output_preactivation_h=True,
        )
        == config.h_output_shape
    )
    assert h.shape == (config.ep_size, config.hidden_rows_per_rank, 2 * config.intermediate_dim)
    np.testing.assert_allclose(hidden[dst, row], expected_activation, rtol=1e-6, atol=1e-6)
    assert not np.allclose(h[dst, row, : config.intermediate_dim], hidden[dst, row])


def test_source_push_forward_exact_inputs_match_source_padded_reference_for_block_aligned_plan():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=1,
        inbox_slots=1,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=4,
        topk=1,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    source_padded_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )
    exact_inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="exact_source_push_plan",
        use_exact_expert_major=True,
    )
    source_padded_out = np.asarray(
        source_push_forward.reference_source_push_forward(config, source_padded_inputs),
        dtype=np.float32,
    )
    exact_out = np.asarray(source_push_forward.reference_source_push_forward(config, exact_inputs), dtype=np.float32)

    src = 1
    dst = 0
    dst_ordinal = source_push_inbox._dst_ordinal(config, src, dst)
    entry = 0
    local_row_start = exact_inputs.send_meta[
        src, dst_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_ROW_START
    ]
    expert = exact_inputs.send_meta[src, dst_ordinal, entry, source_push_plan.SOURCE_PUSH_META_LOCAL_EXPERT]
    exact_row_start = (
        exact_inputs.expert_base[dst, expert] + exact_inputs.src_base_by_expert[dst, src, expert] + local_row_start
    )

    assert exact_inputs.use_exact_expert_major
    assert exact_inputs.queue_stats["input_mode"] == "exact_source_push_plan"
    assert exact_inputs.queue_stats["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert exact_inputs.queue_stats["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert exact_inputs.queue_stats["plan_layout_padding_rows_total"] == 0
    assert local_row_start == 0
    assert exact_row_start == config.block_m
    np.testing.assert_allclose(exact_out, source_padded_out, atol=1e-5, rtol=1e-5)


def test_source_push_forward_exact_inputs_reject_tail_blocks():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=2,
        block_k=4,
        block_n=4,
        experts_per_rank=1,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=5,
        topk=1,
        capacity_factor=1.25,
    )

    with pytest.raises(ValueError, match="block_m-aligned live blocks"):
        source_push_forward.make_source_push_forward_exact_source_plan_inputs(config)


def test_source_push_forward_real_inputs_match_independent_moe_reference():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=2,
        hidden_dim=4,
        intermediate_dim=4,
        block_m=2,
        block_k=2,
        block_n=2,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=4,
        topk=2,
        capacity_factor=1.25,
    )
    x = np.arange(config.ep_size * config.tokens_per_rank * config.hidden_dim, dtype=np.float32).reshape(
        config.ep_size,
        config.tokens_per_rank,
        config.hidden_dim,
    )
    x = 0.05 + x * 0.001
    selected_experts = np.asarray(
        [
            [[0, 2], [1, 3], [0, 2], [1, 3]],
            [[2, 0], [3, 1], [2, 0], [3, 1]],
        ],
        dtype=np.int32,
    )
    combine_weights = np.asarray(
        [
            [[0.50, 0.25], [0.75, 0.125], [0.375, 0.625], [0.25, 0.50]],
            [[0.20, 0.80], [0.60, 0.40], [0.30, 0.70], [0.90, 0.10]],
        ],
        dtype=np.float32,
    )
    w_gate_up = np.arange(
        config.ep_size * config.experts_per_rank * config.hidden_dim * 2 * config.intermediate_dim,
        dtype=np.float32,
    ).reshape(config.ep_size, config.experts_per_rank, config.hidden_dim, 2 * config.intermediate_dim)
    w_gate_up = 0.01 + w_gate_up * 0.0001
    w_down = np.arange(
        config.ep_size * config.experts_per_rank * config.intermediate_dim * config.hidden_dim,
        dtype=np.float32,
    ).reshape(config.ep_size, config.experts_per_rank, config.intermediate_dim, config.hidden_dim)
    w_down = 0.02 + w_down * 0.0002

    inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
    )
    observed = np.asarray(source_push_forward.reference_source_push_forward(config, inputs), dtype=np.float32)
    observed_from_api, dropped_routes = source_push_forward.source_push_forward(
        config,
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
        implementation="reference",
    )
    expected = _naive_source_push_forward(config, x, selected_experts, combine_weights, w_gate_up, w_down)

    valid_routes = np.argwhere(inputs.route_valid_mask)
    src, token, route_slot = valid_routes[0]
    dst_ord = inputs.queue_dst_ord[src, token, route_slot]
    entry = inputs.queue_entry[src, token, route_slot]
    row = inputs.queue_row[src, token, route_slot]

    assert inputs.queue_stats["input_mode"] == "real_arrays"
    assert inputs.queue_stats["dropped_routes"] == 0
    np.testing.assert_array_equal(inputs.x[src, dst_ord, entry, row], x[src, token])
    np.testing.assert_array_equal(inputs.w_gate_up, w_gate_up)
    np.testing.assert_array_equal(inputs.w_down, w_down)
    assert inputs.route_combine_weights[src, token, route_slot] == combine_weights[src, token, route_slot]
    np.testing.assert_allclose(observed, expected, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(observed_from_api, dtype=np.float32), expected, atol=2e-3, rtol=2e-3)
    assert int(dropped_routes) == 0


def _small_source_push_mlp_gradient_case():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=4,
        inbox_slots=2,
        hidden_dim=4,
        intermediate_dim=4,
        block_m=2,
        block_k=2,
        block_n=2,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=4,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = source_push_forward.make_source_push_forward_source_plan_raw_inputs(config)
    inputs = source_push_forward.make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
    )
    route_table = source_push_mlp.source_push_mlp_route_table_from_plan(
        inputs.plan,
        src_base_by_expert=inputs.src_base_by_expert,
    )
    assert inputs.queue_stats["dropped_routes"] == 0
    return (
        config,
        inputs,
        route_table,
        jnp.asarray(raw_inputs.x, dtype=jnp.float32),
        jnp.asarray(raw_inputs.combine_weights, dtype=jnp.float32),
        jnp.asarray(raw_inputs.w_gate_up, dtype=jnp.float32),
        jnp.asarray(raw_inputs.w_down, dtype=jnp.float32),
    )


def test_source_push_moe_mlp_custom_vjp_gradients_match_reference_mlp_boundary():
    _, _inputs, route_table, x, route_weights, w13, w2 = _small_source_push_mlp_gradient_case()

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_mlp.source_push_moe_mlp_reference(
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
        )
        return jnp.sum(y.astype(jnp.float32))

    def custom_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y = source_push_mlp.source_push_moe_mlp_custom_vjp(
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
        )
        return jnp.sum(y.astype(jnp.float32))

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    custom_value, custom_grads = jax.value_and_grad(custom_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    np.testing.assert_allclose(np.asarray(custom_value), np.asarray(reference_value), atol=1e-5, rtol=1e-5)
    for custom_grad, reference_grad in zip(custom_grads, reference_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(custom_grad),
            np.asarray(reference_grad),
            atol=1e-4,
            rtol=1e-4,
        )
    assert np.any(np.abs(np.asarray(custom_grads[1])) > 1e-8)


def test_source_push_moe_mlp_from_plan_reference_gradients_match_flat_h_reference():
    config, inputs, route_table, x, route_weights, w13, w2 = _small_source_push_mlp_gradient_case()
    expert_base = jnp.asarray(inputs.expert_base, dtype=jnp.int32)

    def reference_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, _ = source_push_mlp.source_push_moe_mlp_reference_with_h_flat(
            route_table,
            expert_base,
            config.hidden_rows_per_rank,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
        )
        return jnp.sum(y.astype(jnp.float32))

    def custom_loss(x_arg, route_weights_arg, w13_arg, w2_arg):
        y, dropped_routes = source_push_mlp.source_push_moe_mlp_from_plan(
            config,
            inputs,
            route_table,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=source_push_mlp.SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        )
        assert int(dropped_routes) == 0
        return jnp.sum(y.astype(jnp.float32))

    reference_value, reference_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )
    custom_value, custom_grads = jax.value_and_grad(custom_loss, argnums=(0, 1, 2, 3))(
        x,
        route_weights,
        w13,
        w2,
    )

    np.testing.assert_allclose(np.asarray(custom_value), np.asarray(reference_value), atol=1e-5, rtol=1e-5)
    for custom_grad, reference_grad in zip(custom_grads, reference_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(custom_grad),
            np.asarray(reference_grad),
            atol=1e-4,
            rtol=1e-4,
        )
    assert np.any(np.abs(np.asarray(custom_grads[1])) > 1e-8)


def test_source_push_mlp_backward_decomposed_matches_flat_h_backward():
    config, inputs, route_table, x, route_weights, w13, w2 = _small_source_push_mlp_gradient_case()
    expert_base = jnp.asarray(inputs.expert_base, dtype=jnp.int32)
    _, h_flat = source_push_mlp.source_push_moe_mlp_reference_with_h_flat(
        route_table,
        expert_base,
        config.hidden_rows_per_rank,
        x,
        route_weights,
        w13,
        w2,
    )
    dy = jnp.linspace(-0.5, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)

    expected = source_push_mlp._source_push_moe_mlp_backward_from_h_flat(
        route_table,
        expert_base,
        x,
        route_weights,
        w13,
        w2,
        h_flat,
        dy,
    )
    timing = source_push_mlp_fwd_bwd_cli._time_source_push_backward_decomposed(
        route_table,
        expert_base,
        x,
        route_weights,
        w13,
        w2,
        h_flat,
        dy,
        warmup=0,
        steps=1,
        repeat_runs=1,
    )

    for observed, expected_grad in zip(timing.output, expected, strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected_grad), atol=0, rtol=0)
    assert set(timing.stage_steady_state_times) == set(source_push_mlp_fwd_bwd_cli.BACKWARD_STAGES)


def _naive_source_push_forward(
    config: source_push_inbox.PushInboxConfig,
    x: np.ndarray,
    selected_experts: np.ndarray,
    combine_weights: np.ndarray,
    w_gate_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    out = np.zeros((config.ep_size, config.tokens_per_rank, config.hidden_dim), dtype=np.float32)
    for src in range(config.ep_size):
        for token in range(config.tokens_per_rank):
            for route_slot in range(config.topk):
                global_expert = int(selected_experts[src, token, route_slot])
                dst = global_expert // config.experts_per_rank
                local_expert = global_expert % config.experts_per_rank
                w13 = x[src, token] @ w_gate_up[dst, local_expert]
                gate = w13[: config.intermediate_dim]
                up = w13[config.intermediate_dim :]
                hidden = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                hidden = np.asarray(jnp.asarray(hidden, dtype=jnp.bfloat16), dtype=np.float32)
                w_down_bf16 = np.asarray(jnp.asarray(w_down[dst, local_expert], dtype=jnp.bfloat16), dtype=np.float32)
                route_out = hidden @ w_down_bf16
                route_out = np.asarray(jnp.asarray(route_out, dtype=jnp.bfloat16), dtype=np.float32)
                weighted = route_out * combine_weights[src, token, route_slot]
                out[src, token] += np.asarray(jnp.asarray(weighted, dtype=jnp.bfloat16), dtype=np.float32)
    return out


def test_source_push_package_private_runner_returns_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_inbox.run_source_push_inbox(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_inbox"
    assert rows[0]["repeat_runs"] == 1
    assert rows[0]["rounded_w13_tflops_per_rank"] is None
    assert rows[0]["useful_w13_tflops_per_rank"] is None


def test_source_push_w2_runner_returns_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_w2_return.run_source_push_w2_return_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_w2_return"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_w2_runner_tags_copy_to_source_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_w2_return.run_source_push_w2_return_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
        return_mode=source_push_w2_return.W2_RETURN_MODE_SEPARATE_COPY,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["implementation"] == "source_push_w2_return_copy"
    assert rows[0]["copy_to_source"]


def test_source_push_w2_runner_tags_direct_to_source_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_w2_return.run_source_push_w2_return_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
        return_mode=source_push_w2_return.W2_RETURN_MODE_DIRECT_REMOTE,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["implementation"] == "source_push_w2_return_direct"
    assert rows[0]["return_mode"] == "direct_remote"
    assert rows[0]["direct_to_source"]


def test_source_push_combine_runner_returns_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_combine.run_source_push_combine_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_combine"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_forward_runner_returns_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_forward.run_source_push_forward_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_forward"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_forward_adds_summary_rows_for_target_gate_reporting():
    def repeat(stage, steady_state_time, **metrics):
        row_type = "repeat" if stage == "total" else "stage_repeat"
        return {
            "kernel": "source_push_forward",
            "implementation": "source_push_forward" if stage == "total" else f"source_push_forward_{stage}",
            "row_type": row_type,
            "stage": stage,
            "execution_mode": "staged_host_sync",
            "config": {"routing": "roughly_balanced"},
            "queue_stats": {"plan_row_efficiency": 0.9},
            "repeat_runs": 2,
            "steady_state_time": steady_state_time,
            "error_type": None,
            **metrics,
        }

    rows = [
        repeat("total", 3.0, rounded_forward_tflops_per_rank=90.0, useful_forward_tflops_per_rank=81.0),
        repeat("w13", 1.0, useful_w13_tflops_per_rank=200.0),
        repeat("w2_return", 1.5, w2_tflops_per_rank=120.0),
        repeat("combine", 0.5, combine_gbps_per_rank=1000.0),
        repeat("total", 5.0, rounded_forward_tflops_per_rank=50.0, useful_forward_tflops_per_rank=45.0),
        repeat("w13", 2.0, useful_w13_tflops_per_rank=100.0),
        repeat("w2_return", 2.5, w2_tflops_per_rank=80.0),
        repeat("combine", 0.7, combine_gbps_per_rank=800.0),
    ]

    observed = source_push_forward._add_forward_summary_rows(rows)

    summaries = [row for row in observed if row["row_type"] == "summary"]
    assert [row["stage"] for row in summaries] == ["total", "w13", "w2_return", "combine"]
    total_summary = summaries[0]
    assert total_summary["median_steady_state_time"] == 4.0
    assert total_summary["median_rounded_forward_tflops_per_rank"] == 70.0
    assert total_summary["median_useful_forward_tflops_per_rank"] == 63.0
    assert total_summary["p90_steady_state_time"] == 4.8
    assert total_summary["p95_steady_state_time"] == 4.9
    assert total_summary["plan_row_efficiency"] == 0.9

    w13_summary = summaries[1]
    assert w13_summary["median_useful_w13_tflops_per_rank"] == 150.0
    assert w13_summary["min_useful_w13_tflops_per_rank"] == 100.0
    assert w13_summary["slow_useful_w13_threshold"] == 160.0
    assert w13_summary["slow_useful_w13_repeats"] == 1
    assert w13_summary["slow_useful_w13_fraction"] == 0.5

    assert summaries[2]["median_w2_tflops_per_rank"] == 100.0
    assert summaries[3]["median_combine_gbps_per_rank"] == 900.0


def test_source_push_repro_wrapper_imports_active_bench_cli():
    result = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_w2_bench_cli_imports():
    result = subprocess.run(
        [sys.executable, str(W2_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_combine_bench_cli_imports():
    result = subprocess.run(
        [sys.executable, str(COMBINE_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_forward_bench_cli_imports():
    result = subprocess.run(
        [sys.executable, str(FORWARD_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_forward_public_compare_bench_cli_imports():
    result = subprocess.run(
        [sys.executable, str(FORWARD_PUBLIC_COMPARE_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_mlp_fwd_bwd_bench_cli_imports():
    result = subprocess.run(
        [sys.executable, str(MLP_FWD_BWD_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_diagnostic_runner_tags_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_inbox.run_source_push_inbox_diagnostic(
        config,
        diagnostic_variant="semaphore_only",
        warmup=0,
        steps=1,
        repeat_runs=1,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_inbox_diagnostic"
    assert rows[0]["implementation"] == "source_push_inbox_diagnostic:semaphore_only"
    assert rows[0]["diagnostic_variant"] == "semaphore_only"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_cli_runs_every_send_pipeline_depth_sweep_value(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_inbox(config, **kwargs):
        calls.append((config.send_pipeline_depth, kwargs["repeat_runs"]))
        return [{"send_pipeline_depth": config.send_pipeline_depth, "repeat_runs": kwargs["repeat_runs"]}]

    monkeypatch.setattr(source_push_cli, "run_source_push_inbox", fake_run_source_push_inbox)

    source_push_cli.main(
        [
            "--sweep-send-pipeline-depth",
            "1,2",
            "--repeat-runs",
            "1",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(1, 1), (2, 1)]
    assert [row["send_pipeline_depth"] for row in rows] == [1, 2]


def test_source_push_cli_selects_source_push_plan_input_mode(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_inbox_source_plan(config, **kwargs):
        calls.append((config.capacity_factor, kwargs["repeat_runs"]))
        return [{"input_mode": "source_push_plan", "capacity_factor": config.capacity_factor}]

    monkeypatch.setattr(source_push_cli, "run_source_push_inbox_source_plan", fake_run_source_push_inbox_source_plan)

    source_push_cli.main(
        [
            "--input-mode",
            "source_push_plan",
            "--capacity-factor",
            "1.0",
            "--git-sha",
            "abc123",
            "--repeat-runs",
            "1",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(1.0, 1)]
    assert rows == [{"capacity_factor": 1.0, "git_sha": "abc123", "input_mode": "source_push_plan"}]


def test_source_push_diagnostic_cli_runs_requested_variants(monkeypatch, capsys):
    calls = []
    repeat_rows = [
        {
            "config": {"entries_per_rank": 2},
            "queue_stats": {},
            "repeat_run": 0,
            "repeat_runs": 3,
            "steady_state_time": 1.0,
            "w13_tflops_per_rank": 2.0,
            "rounded_w13_tflops_per_rank": 2.0,
            "useful_w13_tflops_per_rank": 1.5,
            "send_gbps_per_rank": 3.0,
            "compile_time": 4.0,
            "lower_compile_time": 5.0,
            "first_run_time": 6.0,
            "error_type": None,
            "error": None,
        },
        {
            "config": {"entries_per_rank": 2},
            "queue_stats": {},
            "repeat_run": 1,
            "repeat_runs": 3,
            "steady_state_time": 2.0,
            "w13_tflops_per_rank": 1.0,
            "rounded_w13_tflops_per_rank": 1.0,
            "useful_w13_tflops_per_rank": 0.75,
            "send_gbps_per_rank": 1.5,
            "compile_time": 4.0,
            "lower_compile_time": 5.0,
            "first_run_time": 6.0,
            "error_type": None,
            "error": None,
        },
    ]

    def fake_run_source_push_inbox_diagnostic(config, **kwargs):
        calls.append((kwargs["diagnostic_variant"], kwargs["repeat_runs"], kwargs["input_mode"]))
        return [
            {
                **row,
                "kernel": "source_push_inbox_diagnostic",
                "implementation": f"source_push_inbox_diagnostic:{kwargs['diagnostic_variant']}",
                "diagnostic_variant": kwargs["diagnostic_variant"],
                "diagnostic_input_mode": kwargs["input_mode"],
            }
            for row in repeat_rows
        ]

    monkeypatch.setattr(
        source_push_diagnostic_cli,
        "run_source_push_inbox_diagnostic",
        fake_run_source_push_inbox_diagnostic,
    )

    source_push_diagnostic_cli.main(
        [
            "--variants",
            "semaphore_only,copy_release_only",
            "--repeat-runs",
            "3",
            "--input-mode",
            "source_push_plan",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [("semaphore_only", 3, "source_push_plan"), ("copy_release_only", 3, "source_push_plan")]
    assert [row["row_type"] for row in rows] == ["repeat", "repeat", "summary", "repeat", "repeat", "summary"]
    assert [row["diagnostic_variant"] for row in rows] == [
        "semaphore_only",
        "semaphore_only",
        "semaphore_only",
        "copy_release_only",
        "copy_release_only",
        "copy_release_only",
    ]
    assert [row.get("median_rounded_w13_tflops_per_rank") for row in rows if row["row_type"] == "summary"] == [
        1.5,
        1.5,
    ]
    assert [row.get("median_useful_w13_tflops_per_rank") for row in rows if row["row_type"] == "summary"] == [
        1.125,
        1.125,
    ]
    assert [row.get("diagnostic_input_mode") for row in rows if row["row_type"] == "summary"] == [
        "source_push_plan",
        "source_push_plan",
    ]
    assert [row.get("git_sha") for row in rows] == ["abc123"] * len(rows)
    assert [row.get("p90_steady_state_time") for row in rows if row["row_type"] == "summary"] == [
        1.9,
        1.9,
    ]
    assert [row.get("p95_steady_state_time") for row in rows if row["row_type"] == "summary"] == [
        1.95,
        1.95,
    ]
    assert [row.get("slow_useful_w13_repeats") for row in rows if row["row_type"] == "summary"] == [2, 2]
    assert [row.get("slow_useful_w13_fraction") for row in rows if row["row_type"] == "summary"] == [1.0, 1.0]


def test_source_push_combine_cli_passes_profile_defaults(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_combine_source_plan(config, **kwargs):
        calls.append((config.routing, config.entries_per_rank, kwargs["repeat_runs"]))
        return [{"kernel": "source_push_combine", "repeat_runs": kwargs["repeat_runs"]}]

    monkeypatch.setattr(
        source_push_combine_cli,
        "run_source_push_combine_source_plan",
        fake_run_source_push_combine_source_plan,
    )

    source_push_combine_cli.main(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--repeat-runs",
            "3",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [("roughly_balanced", 288, 3)]
    assert rows == [{"git_sha": "abc123", "kernel": "source_push_combine", "repeat_runs": 3}]


def test_source_push_forward_cli_passes_profile_defaults(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_forward_source_plan(config, **kwargs):
        calls.append((config.routing, config.entries_per_rank, kwargs["repeat_runs"], kwargs["execution_mode"]))
        return [{"kernel": "source_push_forward", "repeat_runs": kwargs["repeat_runs"]}]

    monkeypatch.setattr(
        source_push_forward_cli,
        "run_source_push_forward_source_plan",
        fake_run_source_push_forward_source_plan,
    )

    source_push_forward_cli.main(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--repeat-runs",
            "3",
            "--execution-mode",
            "staged_host_sync",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [("roughly_balanced", 288, 3, "staged_host_sync")]
    assert rows == [{"git_sha": "abc123", "kernel": "source_push_forward", "repeat_runs": 3}]


def test_source_push_forward_public_compare_cli_passes_profile_defaults(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_forward_public_compare(
        config,
        *,
        source_push_implementation,
        source_push_execution_mode,
        public_implementations,
        public_call_mode,
    ):
        calls.append(
            (
                config.routing,
                config.entries_per_rank,
                source_push_implementation,
                source_push_execution_mode,
                public_implementations,
                public_call_mode,
            )
        )
        return [
            {
                "kernel": "source_push_forward_public_compare",
                "public_implementation": public_implementations[0],
            }
        ]

    monkeypatch.setattr(
        source_push_forward_public_compare_cli,
        "run_source_push_forward_public_compare",
        fake_run_source_push_forward_public_compare,
    )

    source_push_forward_public_compare_cli.main(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--source-push-implementation",
            "reference",
            "--source-push-execution-mode",
            "staged_host_sync",
            "--public-implementations",
            "pallas_mgpu_source_push,ring",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [
        (
            "roughly_balanced",
            288,
            "reference",
            "staged_host_sync",
            ("pallas_mgpu_source_push", "ring"),
            "direct",
        )
    ]
    assert rows == [
        {
            "git_sha": "abc123",
            "kernel": "source_push_forward_public_compare",
            "public_implementation": "pallas_mgpu_source_push",
        }
    ]


def test_source_push_mlp_fwd_bwd_cli_passes_profile_defaults(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_mlp_fwd_bwd(
        config,
        *,
        backends,
        modes,
        warmup,
        steps,
        repeat_runs,
        outer_jit,
        separate_compile,
        debug_exceptions,
    ):
        calls.append(
            (
                config.routing,
                config.entries_per_rank,
                backends,
                modes,
                warmup,
                steps,
                repeat_runs,
                outer_jit,
                separate_compile,
                debug_exceptions,
            )
        )
        return [
            {
                "kernel": "source_push_mlp_fwd_bwd",
                "backend": backends[0],
                "mode": modes[0],
            }
        ]

    monkeypatch.setattr(
        source_push_mlp_fwd_bwd_cli,
        "run_source_push_mlp_fwd_bwd",
        fake_run_source_push_mlp_fwd_bwd,
    )

    source_push_mlp_fwd_bwd_cli.main(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--backends",
            "source_push_pallas_mgpu,ring",
            "--modes",
            "forward_backward",
            "--repeat-runs",
            "3",
            "--outer-jit",
            "false",
            "--separate-compile",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [
        (
            "roughly_balanced",
            288,
            ("source_push_pallas_mgpu", "ring"),
            ("forward_backward",),
            2,
            7,
            3,
            "false",
            True,
            False,
        )
    ]
    assert rows == [
        {
            "backend": "source_push_pallas_mgpu",
            "git_sha": "abc123",
            "kernel": "source_push_mlp_fwd_bwd",
            "mode": "forward_backward",
        }
    ]


def test_source_push_mlp_fwd_bwd_cli_accepts_forward_decomposed_mode(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_mlp_fwd_bwd(
        config,
        *,
        backends,
        modes,
        warmup,
        steps,
        repeat_runs,
        outer_jit,
        separate_compile,
        debug_exceptions,
    ):
        calls.append((backends, modes))
        return [
            {
                "kernel": "source_push_mlp_fwd_bwd",
                "backend": backends[0],
                "mode": modes[0],
                "stage": "pack_inputs",
            }
        ]

    monkeypatch.setattr(
        source_push_mlp_fwd_bwd_cli,
        "run_source_push_mlp_fwd_bwd",
        fake_run_source_push_mlp_fwd_bwd,
    )

    source_push_mlp_fwd_bwd_cli.main(
        [
            "--backends",
            "source_push_pallas_mgpu",
            "--modes",
            "forward_decomposed",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(("source_push_pallas_mgpu",), ("forward_decomposed",))]
    assert rows == [
        {
            "backend": "source_push_pallas_mgpu",
            "git_sha": "abc123",
            "kernel": "source_push_mlp_fwd_bwd",
            "mode": "forward_decomposed",
            "stage": "pack_inputs",
        }
    ]


def test_source_push_mlp_fwd_bwd_cli_accepts_raw_token_forward_decomposed_mode(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_mlp_fwd_bwd(
        config,
        *,
        backends,
        modes,
        warmup,
        steps,
        repeat_runs,
        outer_jit,
        separate_compile,
        debug_exceptions,
    ):
        calls.append((backends, modes))
        return [
            {
                "kernel": "source_push_mlp_fwd_bwd",
                "backend": backends[0],
                "mode": modes[0],
                "stage": "prepare_inputs",
            }
        ]

    monkeypatch.setattr(
        source_push_mlp_fwd_bwd_cli,
        "run_source_push_mlp_fwd_bwd",
        fake_run_source_push_mlp_fwd_bwd,
    )

    source_push_mlp_fwd_bwd_cli.main(
        [
            "--backends",
            "source_push_pallas_mgpu",
            "--modes",
            "forward_decomposed_raw_tokens",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(("source_push_pallas_mgpu",), ("forward_decomposed_raw_tokens",))]
    assert rows == [
        {
            "backend": "source_push_pallas_mgpu",
            "git_sha": "abc123",
            "kernel": "source_push_mlp_fwd_bwd",
            "mode": "forward_decomposed_raw_tokens",
            "stage": "prepare_inputs",
        }
    ]


def test_source_push_mlp_fwd_bwd_cli_accepts_backward_decomposed_mode(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_mlp_fwd_bwd(
        config,
        *,
        backends,
        modes,
        warmup,
        steps,
        repeat_runs,
        outer_jit,
        separate_compile,
        debug_exceptions,
    ):
        calls.append((backends, modes))
        return [
            {
                "kernel": "source_push_mlp_fwd_bwd",
                "backend": backends[0],
                "mode": modes[0],
                "stage": "dy_route",
            }
        ]

    monkeypatch.setattr(
        source_push_mlp_fwd_bwd_cli,
        "run_source_push_mlp_fwd_bwd",
        fake_run_source_push_mlp_fwd_bwd,
    )

    source_push_mlp_fwd_bwd_cli.main(
        [
            "--backends",
            "source_push_pallas_mgpu",
            "--modes",
            "backward_decomposed",
            "--git-sha",
            "abc123",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(("source_push_pallas_mgpu",), ("backward_decomposed",))]
    assert rows == [
        {
            "backend": "source_push_pallas_mgpu",
            "git_sha": "abc123",
            "kernel": "source_push_mlp_fwd_bwd",
            "mode": "backward_decomposed",
            "stage": "dy_route",
        }
    ]
