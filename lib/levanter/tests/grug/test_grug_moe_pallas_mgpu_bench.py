# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.pallas_mgpu import (
    MoeMgpuConfig,
    _expert_group_peer_copy_order,
    _max_blocks_touched_by_unaligned_range,
    _require_chunked_balanced_counts,
    _static_workqueue_phase_indices,
)
from levanter.grug._moe.source_push_inbox_profiles import source_push_profile_defaults


def _load_bench_module():
    levanter_root = Path(__file__).parents[2]
    bench_path = levanter_root / "scripts" / "bench" / "bench_grug_moe_pallas_mgpu.py"
    spec = importlib.util.spec_from_file_location("bench_grug_moe_pallas_mgpu", bench_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_dependencies(bench, **kwargs):
    values = {
        "needs_permute": False,
        "needs_permute_up": False,
        "needs_w2": False,
        "needs_backward_prereq": False,
        "needs_combine_bwd": False,
        "needs_w2_bwd": False,
        "needs_w13_bwd": False,
    }
    values.update(kwargs)
    return bench.PallasStageDependencies(**values)


def _valid_chunked_cli_args() -> list[str]:
    return [
        "bench_grug_moe_pallas_mgpu.py",
        "--routing",
        "balanced",
        "--dispatch-chunked-permute-up",
        "--dispatch-expert-group-size",
        "2",
        "--tokens-per-rank",
        "128",
        "--hidden-dim",
        "128",
        "--intermediate-dim",
        "128",
        "--experts-per-rank",
        "2",
        "--topk",
        "2",
        "--ep-size",
        "2",
        "--block-m",
        "64",
        "--block-n",
        "128",
        "--block-k",
        "64",
        "--dispatch-chunk-copy-tile",
        "128",
    ]


def test_pallas_mgpu_benchmark_stage_dependencies_isolate_independent_stages():
    bench = _load_bench_module()

    assert bench._pallas_stage_dependencies(frozenset({"permute"})) == _stage_dependencies(
        bench,
        needs_permute=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"w13"})) == _stage_dependencies(
        bench,
        needs_permute=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"up_down_baseline"})) == _stage_dependencies(bench)
    assert bench._pallas_stage_dependencies(frozenset({"up_down_candidate"})) == _stage_dependencies(bench)
    assert bench._pallas_stage_dependencies(frozenset({"up_down_compare_split"})) == _stage_dependencies(bench)
    assert bench._pallas_stage_dependencies(frozenset({"staged_forward"})) == _stage_dependencies(bench)


def test_pallas_mgpu_benchmark_cli_defaults_match_kernel_config(monkeypatch):
    bench = _load_bench_module()
    default_config = bench.MoeMgpuConfig()
    monkeypatch.setattr("sys.argv", ["bench_grug_moe_pallas_mgpu.py"])

    args = bench._parse_args()

    assert args.max_concurrent_steps == default_config.max_concurrent_steps
    assert args.grid_block_n == default_config.grid_block_n
    assert args.dispatch_copy_schedule == default_config.dispatch_copy_schedule
    assert args.dispatch_peer_phase_order == default_config.dispatch_peer_phase_order
    assert args.dispatch_swizzle_expert_groups == default_config.dispatch_swizzle_expert_groups
    assert args.dispatch_chunk_copy_tile == default_config.dispatch_chunk_copy_tile
    assert args.dispatch_chunk_copy_rows == default_config.dispatch_chunk_copy_rows
    assert args.dispatch_chunk_vectorized_copy_rows == default_config.dispatch_chunk_vectorized_copy_rows
    assert args.dispatch_user_kernel_permute_up == default_config.dispatch_user_kernel_permute_up
    assert args.dispatch_static_workqueue_init_recv_x == default_config.dispatch_static_workqueue_init_recv_x
    assert args.dispatch_static_workqueue_init_hidden == default_config.dispatch_static_workqueue_init_hidden
    assert args.dispatch_static_workqueue_destination_pull == default_config.dispatch_static_workqueue_destination_pull
    assert (
        args.dispatch_static_workqueue_destination_pull_n_group
        == default_config.dispatch_static_workqueue_destination_pull_n_group
    )
    assert args.dispatch_fuse_metadata == default_config.dispatch_fuse_metadata
    assert args.dispatch_chunked_permute_up == default_config.dispatch_chunked_permute_up
    assert args.dispatch_split_wg_permute_up == default_config.dispatch_split_wg_permute_up
    assert args.dispatch_split_wg_overlap_permute_up == default_config.dispatch_split_wg_overlap_permute_up
    assert args.combine_bwd_block_n == default_config.combine_bwd_block_n
    assert args.dx_unpermute_block_n == default_config.dx_unpermute_block_n
    assert args.candidate_timeout_seconds is None
    assert args.pallas_mgpu_tuning_profile == "none"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            "hopper_queue_ngroups2_210",
            {
                "routing": "uniform",
                "send_pipeline_depth": 2,
                "repeat_runs": 48,
            },
        ),
        (
            "hopper_queue_roughly_balanced_ngroups2_210",
            {
                "routing": "roughly_balanced",
                "send_pipeline_depth": 1,
                "repeat_runs": 48,
            },
        ),
    ],
)
def test_source_push_profile_applies_current_best_candidate_defaults(profile, expected):
    args = source_push_inbox.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            profile,
        ]
    )

    common_expected = {
        "implementation": "m_n_slots",
        "queue_mode": "routing",
        "traffic_pattern": "all_to_all",
        "peer_loop": "grid_switch",
        "lowering_semantics": "lane",
        "metadata_mode": "static_recv",
        "output_mode": "perf",
        "hidden_output_mode": "queue",
        "tokens_per_rank": 32768,
        "hidden_dim": 2560,
        "intermediate_dim": 1280,
        "experts_per_rank": 32,
        "topk": 4,
        "entries_per_rank": 288,
        "inbox_slots": 12,
        "num_send_sms": 2,
        "num_sms": 32,
        "block_m": 64,
        "block_k": 128,
        "block_n": 128,
        "n_group": 1,
        "n_groups_per_job": 2,
        "receiver_schedule": "fixed_wait",
        "slot_order": "current",
        "warmup": 2,
        "steps": 7,
        "check": False,
        "separate_compile": True,
        "progress_events": True,
    }
    common_expected.update(expected)
    for field, value in common_expected.items():
        assert getattr(args, field) == value


def test_source_push_profile_allows_explicit_overrides():
    args = source_push_inbox.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            "hopper_queue_roughly_balanced_ngroups2_210",
            "--routing",
            "uniform",
            "--send-pipeline-depth",
            "2",
            "--repeat-runs",
            "3",
        ]
    )

    assert args.routing == "uniform"
    assert args.send_pipeline_depth == 2
    assert args.repeat_runs == 3
    assert args.n_groups_per_job == 2
    assert args.output_mode == "perf"


def test_source_push_profile_defaults_are_copied():
    defaults = source_push_profile_defaults("hopper_queue_ngroups2_210")
    defaults["routing"] = "roughly_balanced"

    fresh_defaults = source_push_profile_defaults("hopper_queue_ngroups2_210")

    assert fresh_defaults["routing"] == "uniform"


def test_source_push_profile_returns_typed_config_and_run_settings():
    config, settings = source_push_inbox.source_push_inbox_profile("hopper_queue_roughly_balanced_ngroups2_210")

    config.validate()
    assert config.routing == "roughly_balanced"
    assert config.hidden_output_mode == "queue"
    assert config.n_groups_per_job == 2
    assert config.send_pipeline_depth == 1
    assert settings.warmup == 2
    assert settings.steps == 7
    assert settings.repeat_runs == 48
    assert not settings.check
    assert settings.separate_compile
    assert settings.progress_events


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
    assert rows[0]["error"] == "ValueError: ep_size must be greater than 1, got 1"


def test_destination_pull_dense_hopper_config_constructor():
    config = MoeMgpuConfig.destination_pull_dense_hopper(max_rows_per_src_expert=512)

    assert config.block_m == 64
    assert config.block_n == 128
    assert config.block_k == 128
    assert config.grid_block_n == 4
    assert config.dispatch_expert_group_size == 32
    assert config.dispatch_static_workqueue_permute_up
    assert config.dispatch_static_workqueue_dense_balanced_compute
    assert config.dispatch_static_workqueue_destination_pull
    assert config.dispatch_static_workqueue_destination_pull_n_group == 2
    assert config.dispatch_static_workqueue_max_rows_per_src_expert == 512


def test_pallas_mgpu_benchmark_profile_applies_dense_hopper_defaults(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--pallas-mgpu-tuning-profile",
            "destination_pull_dense_balanced_hopper",
            "--routing",
            "balanced",
            "--ep-size",
            "8",
            "--tokens-per-rank",
            "32768",
            "--topk",
            "4",
            "--experts-per-rank",
            "32",
        ],
    )

    args = bench._parse_args()

    assert args.block_m == 64
    assert args.block_n == 128
    assert args.block_k == 128
    assert args.grid_block_n == 4
    assert args.dispatch_expert_group_size == 32
    assert args.dispatch_static_workqueue_permute_up
    assert args.dispatch_static_workqueue_dense_balanced_compute
    assert args.dispatch_static_workqueue_destination_pull
    assert args.dispatch_static_workqueue_destination_pull_n_group == 2
    assert args.dispatch_static_workqueue_max_rows_per_src_expert == 512


def test_pallas_mgpu_benchmark_profile_keeps_explicit_tile_overrides(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--pallas-mgpu-tuning-profile",
            "destination_pull_dense_balanced_hopper",
            "--routing",
            "balanced",
            "--ep-size",
            "8",
            "--tokens-per-rank",
            "32768",
            "--topk",
            "4",
            "--experts-per-rank",
            "32",
            "--block-k",
            "256",
            "--dispatch-static-workqueue-destination-pull-n-group",
            "1",
        ],
    )

    args = bench._parse_args()

    assert args.block_k == 256
    assert args.dispatch_static_workqueue_destination_pull_n_group == 1
    assert args.dispatch_static_workqueue_permute_up
    assert args.dispatch_static_workqueue_dense_balanced_compute
    assert args.dispatch_static_workqueue_destination_pull
    assert args.dispatch_static_workqueue_max_rows_per_src_expert == 512


def test_pallas_mgpu_benchmark_cli_accepts_comma_separated_stage_subset(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--implementations",
            "none",
            "--include-pallas-stages",
            "--pallas-stages",
            "combine_bwd,w2_bwd",
            "w13_bwd",
            "dx_unpermute_vector",
        ],
    )

    args = bench._parse_args()

    assert args.pallas_stages == ["combine_bwd", "w2_bwd", "w13_bwd", "dx_unpermute_vector"]


def test_pallas_mgpu_benchmark_cli_accepts_peer_phase_order(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--ep-size",
            "4",
            "--dispatch-peer-phase-order",
            "0,1,3,2",
            "--dispatch-swizzle-expert-groups",
        ],
    )

    args = bench._parse_args()

    assert args.dispatch_peer_phase_order == (0, 1, 3, 2)
    assert args.dispatch_swizzle_expert_groups


def test_destination_pull_rejects_non_wgmma_block_m():
    with pytest.raises(ValueError, match="block_m to be a multiple of 64"):
        MoeMgpuConfig(
            block_m=32,
            dispatch_static_workqueue_permute_up=True,
            dispatch_static_workqueue_dense_balanced_compute=True,
            dispatch_static_workqueue_destination_pull=True,
        )


def test_baseline_permute_up_config_clears_destination_pull():
    bench = _load_bench_module()
    config = MoeMgpuConfig(
        dispatch_static_workqueue_permute_up=True,
        dispatch_static_workqueue_dense_balanced_compute=True,
        dispatch_static_workqueue_destination_pull=True,
    )

    baseline_config = bench._baseline_permute_up_config(config)

    assert not baseline_config.dispatch_static_workqueue_permute_up
    assert not baseline_config.dispatch_static_workqueue_dense_balanced_compute
    assert not baseline_config.dispatch_static_workqueue_destination_pull


def test_pallas_mgpu_benchmark_cli_rejects_invalid_peer_phase_order(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--ep-size",
            "4",
            "--dispatch-peer-phase-order",
            "0,1,1,2",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_rejects_duplicate_implementations(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--implementations",
            "pallas_mgpu",
            "pallas_mgpu",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_threads_config_only_to_pallas_mgpu():
    bench = _load_bench_module()
    config = bench.MoeMgpuConfig(dispatch_static_workqueue_permute_up=True)

    assert bench._config_for_moe_call("pallas_mgpu", config) is config
    assert bench._config_for_moe_call("ragged_all_to_all", config) is None
    assert bench._config_for_moe_call("ring", config) is None


def test_pallas_mgpu_config_rejects_mixed_permute_up_experiments():
    bench = _load_bench_module()

    with pytest.raises(ValueError, match="standalone permute_up experiment"):
        bench.MoeMgpuConfig(
            dispatch_user_kernel_permute_up=True,
            dispatch_static_workqueue_permute_up=True,
        )


@pytest.mark.parametrize(
    "stage_args",
    [
        ["permute,w13,permute"],
        ["permute", "w13", "permute"],
    ],
)
def test_pallas_mgpu_benchmark_cli_rejects_duplicate_stage_subset(monkeypatch, stage_args):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--implementations",
            "none",
            "--include-pallas-stages",
            "--pallas-stages",
            *stage_args,
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_accepts_candidate_timeout(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--candidate-timeout-seconds",
            "30",
        ],
    )

    args = bench._parse_args()

    assert args.candidate_timeout_seconds == 30.0


@pytest.mark.parametrize("timeout", ["0", "-1"])
def test_pallas_mgpu_benchmark_cli_rejects_non_positive_candidate_timeout(monkeypatch, timeout):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--candidate-timeout-seconds",
            timeout,
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_candidate_timeout_disabled_is_noop():
    bench = _load_bench_module()

    with bench._candidate_timeout(None):
        value = 1 + 1

    assert value == 2


def test_pallas_mgpu_benchmark_candidate_timeout_rejects_non_positive_value():
    bench = _load_bench_module()

    with pytest.raises(ValueError, match="must be positive"):
        with bench._candidate_timeout(0.0):
            pass


def test_pallas_mgpu_benchmark_cli_rejects_unknown_pallas_stage(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--implementations",
            "none",
            "--include-pallas-stages",
            "--pallas-stages",
            "combine_bwd,not_a_stage",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_rejects_mixed_permute_up_experiments(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--dispatch-user-kernel-permute-up",
            "--dispatch-static-workqueue-permute-up",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_expected_result_count_tracks_implementations_and_stages():
    bench = _load_bench_module()

    assert (
        bench._expected_result_count(
            implementations=["ragged_all_to_all", "pallas_mgpu"],
            include_pallas_stages=False,
            pallas_stages=None,
        )
        == 2
    )
    assert (
        bench._expected_result_count(
            implementations=["none"],
            include_pallas_stages=True,
            pallas_stages=frozenset({"permute_metadata", "w13"}),
        )
        == 2
    )
    assert (
        bench._expected_result_count(
            implementations=["pallas_mgpu"],
            include_pallas_stages=True,
            pallas_stages=frozenset({"permute_metadata", "w13"}),
        )
        == 3
    )
    assert bench._expected_result_count(
        implementations=["none"],
        include_pallas_stages=True,
        pallas_stages=None,
    ) == len(bench._PALLAS_STAGE_CHOICES)


def test_pallas_mgpu_benchmark_cli_rejects_chunked_permute_up_without_balanced_routing(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--implementations",
            "none",
            "--include-pallas-stages",
            "--pallas-stages",
            "permute_up",
            "--dispatch-chunked-permute-up",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_rejects_split_wg_without_chunked(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--routing",
            "balanced",
            "--dispatch-split-wg-permute-up",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_rejects_overlap_without_split_wg(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_grug_moe_pallas_mgpu.py",
            "--routing",
            "balanced",
            "--dispatch-chunked-permute-up",
            "--dispatch-split-wg-overlap-permute-up",
        ],
    )

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_pallas_mgpu_benchmark_cli_accepts_valid_chunked_shape(monkeypatch):
    bench = _load_bench_module()
    monkeypatch.setattr("sys.argv", _valid_chunked_cli_args())

    args = bench._parse_args()

    assert args.dispatch_chunked_permute_up


@pytest.mark.parametrize(
    "overrides",
    [
        ["--tokens-per-rank", "65"],
        ["--tokens-per-rank", "192", "--experts-per-rank", "3"],
        ["--tokens-per-rank", "96"],
        ["--hidden-dim", "192"],
        ["--hidden-dim", "192", "--block-k", "128", "--dispatch-chunk-copy-tile", "64"],
        ["--intermediate-dim", "192"],
        ["--block-m", "0"],
    ],
)
def test_pallas_mgpu_benchmark_cli_rejects_invalid_chunked_shapes(monkeypatch, overrides):
    bench = _load_bench_module()
    monkeypatch.setattr("sys.argv", _valid_chunked_cli_args() + overrides)

    with pytest.raises(SystemExit):
        bench._parse_args()


def test_expert_group_peer_copy_order_rotates_peers_inside_expert_groups():
    dst_ranks = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    local_experts = jnp.array([0, 2, 0, 2, 1, 3, 1, 3], dtype=jnp.int32)
    local_pos = jnp.zeros((8,), dtype=jnp.int32)

    order = _expert_group_peer_copy_order(
        dst_ranks,
        local_experts,
        local_pos,
        rank=jnp.array(1, dtype=jnp.int32),
        ep_size=4,
        local_experts=4,
        expert_group_size=2,
    )

    assert order.tolist() == [2, 4, 6, 0, 3, 5, 7, 1]


def test_static_workqueue_phase_indices_support_peer_and_expert_swizzles():
    peer_order = jnp.array([0, 1, 3, 2], dtype=jnp.int32)
    expert_groups = []
    peer_phases = []
    for phase in range(8):
        expert_group, peer_phase = _static_workqueue_phase_indices(
            jnp.asarray(phase, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            ep_size=4,
            expert_groups=2,
            dispatch_copy_schedule="expert_group_peer",
            peer_phase_order=peer_order,
            swizzle_expert_groups=True,
        )
        expert_groups.append(int(expert_group))
        peer_phases.append(int(peer_phase))

    assert expert_groups == [1, 1, 1, 1, 0, 0, 0, 0]
    assert peer_phases == [0, 1, 3, 2, 0, 1, 3, 2]


def test_chunked_balanced_counts_guard_accepts_equal_buckets():
    counts = jnp.full((2, 4), 16, dtype=jnp.int32)

    _require_chunked_balanced_counts(counts, rows_per_source_expert=16)


def test_chunked_balanced_counts_guard_rejects_unequal_buckets():
    counts = jnp.array([[16, 16, 17, 16], [16, 16, 16, 16]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="synthetic balanced-routing diagnostic"):
        _require_chunked_balanced_counts(counts, rows_per_source_expert=16)


@pytest.mark.parametrize(
    ("max_rows", "block_m", "expected"),
    [
        (1, 64, 1),
        (64, 64, 2),
        (65, 64, 2),
        (128, 64, 3),
    ],
)
def test_max_blocks_touched_by_unaligned_range_covers_compact_layout_tails(max_rows, block_m, expected):
    assert _max_blocks_touched_by_unaligned_range(max_rows, block_m) == expected


def test_pallas_mgpu_benchmark_balanced_routing_assigns_each_source_to_each_expert_equally():
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=16,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=4,
        capacity_factor=1.0,
    )

    selected_experts = bench._make_selected_experts(
        key=jnp.array([0, 0], dtype=jnp.uint32),
        shape=shape,
        routing="balanced",
    )

    per_rank = selected_experts.reshape(shape.ep_size, shape.tokens_per_rank * shape.topk)
    expected = jnp.tile(jnp.arange(shape.ep_size * shape.experts_per_rank, dtype=jnp.int32), 4)
    assert jnp.array_equal(per_rank[0], expected)
    assert jnp.array_equal(per_rank[3], expected)


def test_pallas_mgpu_benchmark_result_row_records_required_schema_and_padding():
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.25,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)

    row = bench._result_row(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        routing="balanced",
        warmup=1,
        steps=2,
        git_sha="test-sha",
        flops=123.0,
        dispatch_bytes=456.0,
        return_bytes=789.0,
        memory_footprint=1024.0,
        compile_time=3.0,
        steady_state_time=0.5,
        error=None,
        dropped_routes=0,
        baseline_implementation="ragged_all_to_all",
        max_abs_diff=0.01,
        mean_abs_diff=0.001,
        matches_baseline=True,
        allclose_rtol=1e-2,
        allclose_atol=0.1,
        candidate_timeout_seconds=60.0,
    )
    serialized = bench.asdict(row)
    required_fields = {
        "kernel",
        "implementation",
        "shape",
        "dtype",
        "backend",
        "device_type",
        "device_count",
        "block_sizes",
        "compile_time",
        "steady_state_time",
        "error",
        "git_sha",
        "xla_flags",
        "backend_env",
    }

    assert required_fields <= serialized.keys()
    assert serialized["kernel"] == "grug_moe_mlp_forward"
    assert serialized["implementation"] == "pallas_mgpu"
    assert serialized["shape"] == "T=4,D=128,I=128,E_local=2,K=2,EP=2,capacity_factor=1.25"
    assert serialized["dtype"] == "bfloat16"
    assert serialized["backend"] == jax.devices()[0].platform
    assert serialized["device_type"] == getattr(jax.devices()[0], "device_kind", "unknown")
    assert serialized["device_count"] == 2
    assert serialized["compile_time"] == 3.0
    assert serialized["steady_state_time"] == 0.5
    assert serialized["error"] is None
    assert serialized["git_sha"] == "test-sha"
    assert serialized["xla_flags"] == os.environ.get("XLA_FLAGS", "")
    assert json.loads(serialized["backend_env"]) == {
        key: os.environ[key]
        for key in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "NCCL_DEBUG")
        if key in os.environ
    }
    assert serialized["assignments_per_rank"] == 8
    assert serialized["requested_receiver_capacity_per_rank"] == 10
    assert serialized["receiver_capacity_per_rank"] == 16
    assert serialized["receiver_capacity_padding_per_rank"] == 6
    assert serialized["status"] == "ok"
    measurement_key = json.loads(serialized["measurement_key"])
    assert measurement_key["kernel"] == "grug_moe_mlp_forward"
    assert measurement_key["implementation"] == "pallas_mgpu"
    assert measurement_key["shape"] == serialized["shape"]
    assert measurement_key["dtype"] == "bfloat16"
    assert measurement_key["backend"] == serialized["backend"]
    assert measurement_key["device_count"] == 2
    assert measurement_key["block_sizes"] == serialized["block_sizes"]
    assert measurement_key["routing"] == "balanced"
    assert serialized["estimated_flops_per_rank"] == 123.0
    assert serialized["effective_tflops_per_rank"] == pytest.approx(123.0 / 0.5 / 1e12)
    assert serialized["estimated_dispatch_bytes_per_rank"] == 456.0
    assert serialized["estimated_return_bytes_per_rank"] == 789.0
    assert serialized["estimated_memory_footprint_per_rank"] == 1024.0
    assert serialized["candidate_timeout_seconds"] == 60.0
    assert serialized["baseline_implementation"] == "ragged_all_to_all"
    assert serialized["max_abs_diff_vs_baseline"] == 0.01
    assert serialized["mean_abs_diff_vs_baseline"] == 0.001
    assert serialized["allclose_rtol"] == 1e-2
    assert serialized["allclose_atol"] == 0.1
    assert serialized["matches_baseline"] is True
    assert json.loads(serialized["block_sizes"])["capacity_factor"] == 1.25


def test_pallas_mgpu_benchmark_measurement_key_tracks_routing_axis():
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.0,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)

    row_kwargs = dict(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        warmup=1,
        steps=2,
        git_sha="test-sha",
        flops=123.0,
        dispatch_bytes=456.0,
        return_bytes=789.0,
        memory_footprint=1024.0,
        compile_time=3.0,
        steady_state_time=0.5,
        error=None,
        dropped_routes=0,
        baseline_implementation="ragged_all_to_all",
        max_abs_diff=0.01,
        mean_abs_diff=0.001,
        matches_baseline=True,
        allclose_rtol=1e-2,
        allclose_atol=0.1,
    )

    balanced = bench._result_row(routing="balanced", **row_kwargs)
    uniform = bench._result_row(routing="uniform", **row_kwargs)

    assert balanced.measurement_key != uniform.measurement_key
    assert json.loads(balanced.measurement_key)["routing"] == "balanced"
    assert json.loads(uniform.measurement_key)["routing"] == "uniform"


@pytest.mark.parametrize(
    ("error", "matches_baseline", "expected_status"),
    [
        (None, False, "mismatch"),
        ("ValueError: unsupported shape", None, "error"),
        ("ValueError: unsupported shape", False, "error"),
    ],
)
def test_pallas_mgpu_benchmark_result_row_records_failure_status(error, matches_baseline, expected_status):
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.0,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)

    row = bench._result_row(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        routing="balanced",
        warmup=1,
        steps=2,
        git_sha="test-sha",
        flops=123.0,
        dispatch_bytes=456.0,
        return_bytes=789.0,
        memory_footprint=1024.0,
        compile_time=None if error else 3.0,
        steady_state_time=None if error else 0.5,
        error=error,
        dropped_routes=None,
        baseline_implementation="ragged_all_to_all",
        max_abs_diff=0.01,
        mean_abs_diff=0.001,
        matches_baseline=matches_baseline,
        allclose_rtol=1e-2,
        allclose_atol=0.1,
    )

    assert row.status == expected_status


def test_pallas_mgpu_benchmark_emit_result_writes_parseable_stdout_and_jsonl(capsys):
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.0,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)
    row = bench._result_row(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        routing="balanced",
        warmup=1,
        steps=2,
        git_sha="test-sha",
        flops=123.0,
        dispatch_bytes=456.0,
        return_bytes=789.0,
        memory_footprint=1024.0,
        compile_time=3.0,
        steady_state_time=0.5,
        error=None,
        dropped_routes=0,
        baseline_implementation="ragged_all_to_all",
        max_abs_diff=0.01,
        mean_abs_diff=0.001,
        matches_baseline=True,
        allclose_rtol=1e-2,
        allclose_atol=0.1,
    )
    jsonl_handle = io.StringIO()

    bench._emit_result(row, jsonl_handle)

    stdout_row = json.loads(capsys.readouterr().out)
    jsonl_row = json.loads(jsonl_handle.getvalue())
    assert stdout_row == jsonl_row
    assert stdout_row["status"] == "ok"
    assert json.loads(stdout_row["measurement_key"])["implementation"] == "pallas_mgpu"


def test_pallas_mgpu_benchmark_emit_result_rejects_duplicate_measurement_key(capsys):
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.0,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)
    row = bench._result_row(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        routing="balanced",
        warmup=1,
        steps=2,
        git_sha="test-sha",
        flops=123.0,
        dispatch_bytes=456.0,
        return_bytes=789.0,
        memory_footprint=1024.0,
        compile_time=3.0,
        steady_state_time=0.5,
        error=None,
        dropped_routes=0,
        baseline_implementation="ragged_all_to_all",
        max_abs_diff=0.01,
        mean_abs_diff=0.001,
        matches_baseline=True,
        allclose_rtol=1e-2,
        allclose_atol=0.1,
    )
    jsonl_handle = io.StringIO()
    seen_measurement_keys: set[str] = set()

    bench._emit_result(row, jsonl_handle, seen_measurement_keys)
    with pytest.raises(ValueError, match="duplicate benchmark measurement_key emitted"):
        bench._emit_result(row, jsonl_handle, seen_measurement_keys)

    rows = [json.loads(line) for line in jsonl_handle.getvalue().splitlines()]
    assert len(rows) == 1
    assert rows[0]["measurement_key"] == row.measurement_key
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_pallas_mgpu_benchmark_progress_event_records_measurement_key(capsys):
    bench = _load_bench_module()
    shape = bench.BenchShape(
        tokens_per_rank=4,
        hidden_dim=128,
        intermediate_dim=128,
        experts_per_rank=2,
        topk=2,
        ep_size=2,
        capacity_factor=1.0,
    )
    config = bench.MoeMgpuConfig(capacity_factor=shape.capacity_factor)

    bench._emit_progress(
        kernel="grug_moe_mlp_forward",
        implementation="pallas_mgpu",
        shape=shape,
        config=config,
        dtype=jnp.bfloat16,
        routing="balanced",
    )

    progress = json.loads(capsys.readouterr().err)
    assert progress["event"] == "starting"
    assert progress["dtype"] == "bfloat16"
    assert progress["routing"] == "balanced"
    measurement_key = json.loads(progress["measurement_key"])
    assert measurement_key["kernel"] == "grug_moe_mlp_forward"
    assert measurement_key["implementation"] == "pallas_mgpu"
    assert measurement_key["shape"] == progress["shape"]
    assert measurement_key["block_sizes"] == progress["block_sizes"]


def test_pallas_mgpu_benchmark_stage_dependencies_keep_required_fused_dispatch():
    bench = _load_bench_module()

    assert bench._pallas_stage_dependencies(frozenset({"w2"})) == _stage_dependencies(
        bench,
        needs_permute_up=True,
        needs_w2=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"down_unpermute"})) == _stage_dependencies(
        bench,
        needs_permute_up=True,
    )
    assert bench._pallas_stage_dependencies(None) == _stage_dependencies(
        bench,
        needs_permute=True,
        needs_permute_up=True,
        needs_w2=True,
        needs_backward_prereq=True,
        needs_combine_bwd=True,
        needs_w2_bwd=True,
        needs_w13_bwd=True,
    )


def test_pallas_mgpu_benchmark_stage_dependencies_chain_backward_stages():
    bench = _load_bench_module()

    assert bench._pallas_stage_dependencies(frozenset({"combine_bwd"})) == _stage_dependencies(
        bench,
        needs_backward_prereq=True,
        needs_combine_bwd=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"w13_bwd"})) == _stage_dependencies(
        bench,
        needs_backward_prereq=True,
        needs_combine_bwd=True,
        needs_w2_bwd=True,
        needs_w13_bwd=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"dx_unpermute_vector"})) == _stage_dependencies(
        bench,
        needs_backward_prereq=True,
        needs_combine_bwd=True,
        needs_w2_bwd=True,
        needs_w13_bwd=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"dx_pull_combine_vector"})) == _stage_dependencies(
        bench,
        needs_backward_prereq=True,
        needs_combine_bwd=True,
        needs_w2_bwd=True,
        needs_w13_bwd=True,
    )
    assert bench._pallas_stage_dependencies(frozenset({"manual_backward_pipeline"})) == _stage_dependencies(bench)
    assert bench._pallas_stage_dependencies(frozenset({"manual_forward_backward_pipeline"})) == _stage_dependencies(
        bench
    )


def test_pallas_mgpu_benchmark_gradient_diff_helpers():
    bench = _load_bench_module()

    baseline = (
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jnp.array([[3.0, 4.0]], dtype=jnp.float32),
    )
    actual = (
        jnp.array([1.25, 2.0], dtype=jnp.float32),
        jnp.array([[2.5, 4.25]], dtype=jnp.float32),
    )

    max_abs_diff, mean_abs_diff = bench._grad_diff_stats(actual, baseline)

    assert max_abs_diff == 0.5
    assert mean_abs_diff == 0.25
    assert bench._grads_allclose(actual, baseline, rtol=0.0, atol=0.5)
    assert not bench._grads_allclose(actual, baseline, rtol=0.0, atol=0.49)
