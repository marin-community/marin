# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

import levanter.grug._moe.source_push_inbox as source_push_inbox
import levanter.grug._moe.source_push_combine as source_push_combine
import levanter.grug._moe.source_push_forward as source_push_forward
import levanter.grug._moe.source_push_w2_return as source_push_w2_return
import levanter.grug._moe.source_push_plan as source_push_plan
from levanter.grug._moe.source_push_inbox_profiles import (
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


def test_source_push_profile_exposes_single_stable_candidate():
    assert SOURCE_PUSH_PROFILES == ("none", SOURCE_PUSH_PROFILE_STABLE_216)


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

    assert inputs.queue_stats["combine_mode"] == "route_buffer_gather_sum"
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
    assert inputs.queue_stats["combine_mode"] == "route_buffer_gather_sum"
    assert inputs.queue_stats["dropped_routes"] == 0
    assert inputs.x.shape == (
        config.ep_size,
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
    )
    assert expected.shape == (config.ep_size, config.tokens_per_rank, config.hidden_dim)
    assert np.count_nonzero(expected) > 0
    assert int(np.asarray(inputs.plan.token_ids[src, dst_ord, entry, row])) == int(token)
    assert int(np.asarray(inputs.plan.route_slots[src, dst_ord, entry, row])) == int(route_slot)


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
    ):
        calls.append(
            (
                config.routing,
                config.entries_per_rank,
                source_push_implementation,
                source_push_execution_mode,
                public_implementations,
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
            "ring,ragged_all_to_all",
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
            ("ring", "ragged_all_to_all"),
        )
    ]
    assert rows == [
        {
            "git_sha": "abc123",
            "kernel": "source_push_forward_public_compare",
            "public_implementation": "ring",
        }
    ]
