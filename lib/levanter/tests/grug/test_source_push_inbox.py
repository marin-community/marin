# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
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
    assert host_inputs.queue_stats["row_start_mode"] == "source_padded_row_start"
    assert int(np.sum(live_mask)) == int(np.sum(live_entries) * config.block_m)
    assert int(np.sum(live_entries) * config.block_m) > int(np.sum(valid_rows))
    assert host_inputs.queue_stats["plan_padded_rows_total"] == int(np.sum(live_entries) * config.block_m)

    src = 1
    dst = 0
    dst_ordinal = source_push_inbox._dst_ordinal(config, src, dst)
    first_live_entry = int(np.flatnonzero(live_entries[src, dst_ordinal])[0])
    row_start = host_inputs.send_meta[src, dst_ordinal, first_live_entry, 2]

    assert row_start >= config.block_m


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

    assert inputs.queue_stats["forward_mode"] == "w13_w2_return_copy_combine"
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
        copy_to_source=True,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["implementation"] == "source_push_w2_return_copy"
    assert rows[0]["copy_to_source"]


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

    def fake_run_source_push_inbox_diagnostic(config, **kwargs):
        calls.append((kwargs["diagnostic_variant"], kwargs["repeat_runs"], kwargs["compact_routing"]))
        return [
            {
                "kernel": "source_push_inbox_diagnostic",
                "implementation": f"source_push_inbox_diagnostic:{kwargs['diagnostic_variant']}",
                "diagnostic_variant": kwargs["diagnostic_variant"],
                "config": {"entries_per_rank": config.entries_per_rank},
                "queue_stats": {},
                "repeat_run": 0,
                "repeat_runs": kwargs["repeat_runs"],
                "steady_state_time": 1.0,
                "w13_tflops_per_rank": 2.0,
                "send_gbps_per_rank": 3.0,
                "compile_time": 4.0,
                "lower_compile_time": 5.0,
                "first_run_time": 6.0,
                "error_type": None,
                "error": None,
            }
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
            "--compact-routing",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [("semaphore_only", 3, True), ("copy_release_only", 3, True)]
    assert [row["row_type"] for row in rows] == ["repeat", "summary", "repeat", "summary"]
    assert [row["diagnostic_variant"] for row in rows] == [
        "semaphore_only",
        "semaphore_only",
        "copy_release_only",
        "copy_release_only",
    ]


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
