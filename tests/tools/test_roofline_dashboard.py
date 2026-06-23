# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from marin.tools.roofline.formulas import (
    attention_context_tokens,
    attention_training_flops,
    expert_shard_devices,
    formula_estimates,
    fused_expert_muon_ns_flops,
    fused_expert_muon_ns_part_flops,
    local_activation_shard_devices,
    long_attention_layers,
    moe_activation_collective_bytes,
    ring_collective_wire_factor,
)
from marin.tools.roofline.hardware import load_hardware_registry
from marin.tools.roofline.model_spec import grug_moe_d2560_may_spec
from marin.tools.roofline.report import build_report
from marin.tools.roofline.serve import _render_html
from marin.tools.roofline.types import ObservedTimeBasis
from marin.tools.roofline.wandb_ingest import normalize_wandb_run

MAY208_RUN_ID = "GM2560-MAY208-B16-R2D1E8-GMUONH3-CHUNKLOCAL-PROF-N2-cw-20260620-1811"


@pytest.mark.parametrize(
    "value",
    [
        MAY208_RUN_ID,
        f"marin-community/marin_moe/{MAY208_RUN_ID}",
        f"https://wandb.ai/marin-community/marin_moe/runs/{MAY208_RUN_ID}",
    ],
)
def test_normalize_wandb_run_accepts_id_path_and_url(value: str) -> None:
    ref = normalize_wandb_run(value)

    assert ref is not None
    assert ref.path == f"marin-community/marin_moe/{MAY208_RUN_ID}"
    assert ref.url == f"https://wandb.ai/marin-community/marin_moe/runs/{MAY208_RUN_ID}"


def test_hardware_registry_loads_required_editable_presets() -> None:
    registry = load_hardware_registry()

    assert {"coreweave_h100", "b200", "tpu_v4", "tpu_v5p", "tpu_v5e", "tpu_v6e"} <= set(registry)
    assert registry["coreweave_h100"].default_compute_efficiency["muon_ns"] == 0.8
    assert registry["coreweave_h100"].provenance is not None


def test_grug_moe_fused_expert_muonh3_formula_matches_expected_scale() -> None:
    flops = fused_expert_muon_ns_flops(grug_moe_d2560_may_spec())

    assert flops == pytest.approx(2.43e15, rel=0.01)


def test_grug_moe_preset_uses_llama3_vocab_size() -> None:
    assert grug_moe_d2560_may_spec().vocab_size == 128_256


def test_vocab_sized_rows_use_llama3_vocab_size() -> None:
    model = grug_moe_d2560_may_spec()
    estimates = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}
    local_shard_devices = local_activation_shard_devices(model)

    assert estimates["token_embedding_output_projection"].flops == pytest.approx(
        2.0 * model.global_batch_tokens * model.hidden_dim * 128_256 / local_shard_devices
    )
    assert estimates["xent"].flops == pytest.approx(4.0 * model.global_batch_tokens * 128_256 / local_shard_devices)


def test_attention_formula_covers_training_forward_and_backward_without_remat() -> None:
    model = grug_moe_d2560_may_spec()
    long_layers = 6
    context_tokens = (model.num_layers - long_layers) * (model.sliding_window // 2) + long_layers * model.sliding_window
    forward_flops = 4.0 * model.global_batch_tokens * model.hidden_dim * context_tokens
    local_shard_devices = local_activation_shard_devices(model)

    estimates = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}

    assert model.long_attention_every == 4
    assert model.remat == "none"
    assert long_attention_layers(model) == long_layers
    assert attention_context_tokens(model) == context_tokens
    assert attention_training_flops(model) == pytest.approx(3.0 * forward_flops)
    assert estimates["attention_fa4"].flops == pytest.approx(3.0 * forward_flops / local_shard_devices)
    assert estimates["attention_fa4"].formula == (
        "4 * local_tokens * num_heads * head_dim * "
        "[(layers - long_attention_layers) * sliding_window/2 + "
        "long_attention_layers * sliding_window] * "
        "attention_train_forward_equivalents(remat) / local_activation_shard_devices"
    )


def test_moe_formula_covers_training_forward_and_backward_without_remat() -> None:
    model = grug_moe_d2560_may_spec()
    forward_flops = (
        6.0 * model.num_layers * model.global_batch_tokens * model.top_k * model.hidden_dim * model.intermediate_dim
    )
    local_shard_devices = local_activation_shard_devices(model)
    estimates = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}

    assert model.remat == "none"
    assert estimates["moe_expert"].flops == pytest.approx(3.0 * forward_flops / local_shard_devices)


def test_expert_all_to_all_formula_covers_layers_without_remat() -> None:
    model = grug_moe_d2560_may_spec()
    forward_bytes = model.num_layers * model.global_batch_tokens * model.hidden_dim * model.top_k * 2.0
    local_shard_devices = local_activation_shard_devices(model)
    estimates = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}

    assert model.remat == "none"
    assert moe_activation_collective_bytes(model) == pytest.approx(3.0 * forward_bytes)
    assert estimates["expert_all_to_all"].bytes_accessed == pytest.approx(3.0 * forward_bytes / local_shard_devices)
    assert estimates["expert_all_to_all"].formula == (
        "layers * local_tokens * hidden_dim * top_k * bf16_bytes * "
        "training_forward_equivalents(remat) / local_activation_shard_devices"
    )


def test_expert_optimizer_rows_are_per_expert_shard_device() -> None:
    model = grug_moe_d2560_may_spec()
    estimates = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}

    assert expert_shard_devices(model) == model.mesh.replica_dcn * model.mesh.expert * model.mesh.model
    assert estimates["muon_ns_gram"].flops == pytest.approx(
        fused_expert_muon_ns_part_flops(model, "gram") / expert_shard_devices(model),
    )
    assert estimates["fsdp_param_all_gather"].bytes_accessed == pytest.approx(
        model.num_layers
        * model.num_experts
        * model.hidden_dim
        * model.intermediate_dim
        * 6.0
        / expert_shard_devices(model)
    )


def test_expert_all_to_all_ideal_uses_gigabits_without_device_multiplier() -> None:
    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
    )
    row = next(row for row in report.rows if row.semantic_op == "expert_all_to_all")

    assert row.ideal_time == pytest.approx(row.estimated_bytes / (400e9 / 8.0))


def test_grouped_muon_restore_ideal_uses_ring_all_gather_wire_bytes() -> None:
    model = grug_moe_d2560_may_spec()
    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
    )
    row = next(row for row in report.rows if row.semantic_op == "grouped_muon_restore")
    restored_bytes = (
        model.num_layers
        * model.num_experts
        * model.hidden_dim
        * model.intermediate_dim
        * 6.0
        / expert_shard_devices(model)
    )
    expected_wire_bytes = restored_bytes * ring_collective_wire_factor(expert_shard_devices(model))

    assert row.estimated_bytes == pytest.approx(expected_wire_bytes)
    assert row.ideal_time == pytest.approx(expected_wire_bytes / (400e9 / 8.0))


def test_may208_xprof_rows_attribute_tnt_kernel_to_muon_ns_gram(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        wandb_run=MAY208_RUN_ID,
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    gram = rows["muon_ns_gram"]
    assert gram.matched_name is not None
    assert "newton_schulz_grouped_4d" in gram.matched_name
    assert gram.observed_count == 3 * 416
    assert gram.observed_time_basis == ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME


def test_replicated_grug_muon_newton_schulz_attributes_to_muon_ns_polynomial(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_single_xprof_row(
        profile_path,
        op_name=(
            "jit(train_step)/optimizer_update/optimizer/group/muonh/muonh/direction_update/grug_muon/"
            "transform_updates/grug_muon/orthogonalize_2d_replicated/"
            "_zeropower_via_newtonschulz_replicated/newton_schulz_replicated/"
            "iter_0/polynomial/ik,kj->ij/dot_general:"
        ),
        kernel_name="nvjet_sm90_tst_128x160_64x5_4x2_h_bz_NNT",
        total_duration_us=123_254.119,
        occurrences=2080,
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    polynomial = rows["muon_ns_polynomial"]
    assert polynomial.track_summed_observed_time == pytest.approx(123_254.119e-6)
    assert polynomial.observed_count == 2080
    assert "uncategorized" not in rows


def test_moe_all_gather_rows_attribute_to_expert_all_to_all_without_efficiency(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    expert_comm = rows["expert_all_to_all"]
    assert expert_comm.matched_name is not None
    assert "MoEExpertMlp/moe_mlp/shard_map/scatter/all_gather" in expert_comm.matched_name
    assert expert_comm.observed_count == 416
    assert expert_comm.observed_time_basis == ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME
    assert expert_comm.track_summed_observed_time is not None
    assert expert_comm.estimated_bytes > 0.0
    assert not expert_comm.observed_comparable_to_model


def test_moe_gather_where_and_scatter_reduce_scatter_attribute_to_expert_all_to_all(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_xprof_rows(
        profile_path,
        [
            (
                "jit(train_step)/forward_backward/Block/Block._mlp_update/MoEMLP/"
                "MoEExpertMlp/moe_mlp/shard_map/gather/jit(_where)/select_n:",
                "loop_select_fusion_7",
                30_853.0,
                416,
            ),
            (
                "jit(train_step)/forward_backward/Block/Block._mlp_update/MoEMLP/"
                "MoEExpertMlp/moe_mlp/shard_map/scatter/reduce_scatter:",
                "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
                223_883.0,
                416,
            ),
        ],
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    expert_comm = rows["expert_all_to_all"]
    assert expert_comm.track_summed_observed_time == pytest.approx((30_853.0 + 223_883.0) * 1e-6)
    assert expert_comm.observed_count == 832
    assert "expert_activation_reduce_scatter" not in rows
    assert not expert_comm.observed_comparable_to_model


def test_moe_psum_rows_are_observed_only_collective(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_single_xprof_row(
        profile_path,
        op_name=(
            "jit(train_step)/forward_backward/Block/Block._mlp_update/MoEMLP/" "MoEExpertMlp/moe_mlp/shard_map/psum:"
        ),
        kernel_name="ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        total_duration_us=12_000.0,
        occurrences=16,
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    assert rows["expert_backward_psum"].track_summed_observed_time == pytest.approx(12_000e-6)
    assert rows["expert_all_to_all"].track_summed_observed_time is None
    assert rows["moe_expert"].track_summed_observed_time is None


def test_dense_mlp_attributes_entire_dense_mlp_subtree(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_single_xprof_row(
        profile_path,
        op_name=("jit(train_step)/forward_backward/Block/Block._mlp_update/" "DenseMLP.dense_mlp/wi/dot_general:"),
        kernel_name="cutlass_gemm",
        total_duration_us=14_000.0,
        occurrences=16,
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    dense = rows["dense_mlp"]
    assert dense.track_summed_observed_time == pytest.approx(14_000e-6)
    assert dense.observed_count == 16
    assert "uncategorized" not in rows


def test_scatter_add_allreduce_is_observed_only_collective(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_single_xprof_row(
        profile_path,
        op_name="jit(train_step)/forward_backward/transpose(jvp(Transformer))/scatter-add:",
        kernel_name="ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        total_duration_us=82_323.0,
        occurrences=16,
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    rows = {row.semantic_op: row for row in report.rows}
    assert rows["grad_reduce_scatter"].track_summed_observed_time is None
    assert rows["scatter_add_collective"].track_summed_observed_time == pytest.approx(82_323e-6)
    assert rows["scatter_add_collective"].estimated_bytes == 0.0


def test_may208_tnt_kernel_reports_observed_peak_fraction(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    gram = next(row for row in report.rows if row.semantic_op == "muon_ns_gram")
    assert gram.observed_avg_time == pytest.approx(852e-6)
    assert gram.profile_achieved_pct is None


def test_observed_profile_time_keeps_track_summed_basis_separate(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    gram = next(row for row in report.rows if row.semantic_op == "muon_ns_gram")
    assert gram.track_summed_observed_time == pytest.approx(3 * 416 * 852e-6)
    assert gram.critical_path_observed_time is None
    assert gram.observed_time_basis == ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME
    assert any("do not read them as exposed wall-clock time" in warning for warning in report.imports.warnings)


def test_profile_dir_prefers_kernel_stats_over_framework_stats(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_single_xprof_row(
        profile_dir / "kernel_stats.json",
        op_name="train/Block/CausalSelfAttention/fa4_cute_kernel",
        kernel_name="SegmentedFlashAttentionForward",
        total_duration_us=100.0,
        occurrences=2,
    )
    _write_single_xprof_row(
        profile_dir / "framework_op_stats.json",
        op_name="train/Block/CausalSelfAttention/fa4_cute_kernel",
        kernel_name="",
        total_duration_us=10_000.0,
        occurrences=2,
    )

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_dir),
    )

    attention = next(row for row in report.rows if row.semantic_op == "attention_fa4")
    assert attention.track_summed_observed_time == pytest.approx(100e-6)
    assert attention.observed_count == 2


def test_profile_dir_infers_devices_and_top_level_train_steps(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_single_xprof_row(
        profile_dir / "kernel_stats.json",
        op_name="train/Block/CausalSelfAttention/fa4_cute_kernel",
        kernel_name="SegmentedFlashAttentionForward",
        total_duration_us=100.0,
        occurrences=2,
    )
    _write_overview_page(profile_dir / "overview_page.json", device_core_count=8)
    trace_dir = profile_dir / "plugins" / "profile" / "timestamp"
    trace_dir.mkdir(parents=True)
    _write_train_step_trace(trace_dir / "perfetto_trace.json.gz")

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_dir),
    )

    assert report.imports.profile_devices == 8
    assert report.imports.profile_steps == 2


def test_profile_dir_adds_uncategorized_and_unaccounted_rows(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_single_xprof_row(
        profile_dir / "kernel_stats.json",
        op_name="train_step/unknown_expensive_region",
        kernel_name="unmapped_kernel",
        total_duration_us=10_000.0,
        occurrences=5,
    )
    _write_overview_page(profile_dir / "overview_page.json", device_core_count=2)
    trace_dir = profile_dir / "plugins" / "profile" / "timestamp"
    trace_dir.mkdir(parents=True)
    _write_device_activity_trace(trace_dir / "perfetto_trace.json.gz")

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_dir),
    )

    rows = {row.semantic_op: row for row in report.rows}
    uncategorized = rows["uncategorized"]
    unaccounted = rows["unaccounted_for"]

    assert uncategorized.track_summed_observed_time == pytest.approx(10_000e-6)
    assert uncategorized.observed_count == 5
    assert uncategorized.observed_time_basis == ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME
    assert report.unattributed[0]["name"] == "train_step/unknown_expensive_region"

    assert report.imports.profile_devices == 2
    assert report.imports.profile_steps == 1
    assert unaccounted.track_summed_observed_time == pytest.approx(900e-6)
    assert unaccounted.observed_count == 2
    assert unaccounted.observed_avg_time == pytest.approx(450e-6)
    assert unaccounted.observed_time_basis == ObservedTimeBasis.TRACE_EMPTY_TRAIN_STEP_TIME


def test_hyperball_profile_rows_attribute_to_main_table(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_grouped_muonh_kernel_stats(profile_dir / "kernel_stats.json")

    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_dir),
    )

    rows = {row.semantic_op: row for row in report.rows}
    hyperball = rows["muon_hyperball"]

    assert hyperball.kind == "compute"
    assert hyperball.estimated_flops > 0
    assert hyperball.track_summed_observed_time == pytest.approx(24_000e-6)
    assert hyperball.observed_count == 32
    assert hyperball.matched_name is not None
    assert "grouped_muonh/hyperball" in hyperball.matched_name

    html = _render_html(report)
    assert "muon_hyperball" in html
    assert "profile-breakdowns" not in html
    assert "profile breakdown" not in html


def test_dashboard_html_embeds_parseable_report_json(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)
    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    html = _render_html(report)

    report_payload = html.split('<script id="report" type="application/json">', maxsplit=1)[1].split(
        "</script>", maxsplit=1
    )[0]
    assert "&quot;" not in report_payload
    assert '""":' not in html
    assert '"roofline_dashboard.v1"' in html


def test_dashboard_html_exposes_interactive_scenario_controls(tmp_path: Path) -> None:
    profile_path = tmp_path / "kernel_stats.json"
    _write_may208_kernel_stats(profile_path)
    report = build_report(
        hardware_name="coreweave_h100",
        model_preset_name="grug_moe_d2560_may",
        profile=str(profile_path),
    )

    html = _render_html(report)

    assert 'id="hardware-select"' in html
    assert 'id="node-count"' in html
    assert 'id="comm-fabric"' in html
    assert "inter-host fabric" in html
    assert "intra-host fabric" in html
    assert 'id="profile-devices"' in html
    assert 'id="profile-steps"' in html
    assert 'id="critical-path-mode"' in html
    assert "compute + exposed expert all-to-all" in html
    assert "normalized observed" in html
    assert '"b200"' in html
    assert 'data-sort="work"' in html
    assert 'data-sort="estimate"' in html
    assert 'data-sort="track_summed"' in html
    assert "Per-device work" in html
    assert "Per-device FLOPs" in html
    assert "Per-device comm bytes" in html
    assert "Ideal/device ms" in html
    assert "Modeled/device ms" in html
    assert "Actual/device ms" in html
    assert "Track-summed ms" in html
    assert "track-summed /" in html
    assert "manualEfficiencies" in html
    assert 'min="0.001"' in html
    assert 'title="${title}"' in html
    assert "bytes / throughput" in html
    assert "Gbps / 8" in html


def _write_may208_kernel_stats(path: Path) -> None:
    op_names = [
        (
            "optimizer_update/.../grouped_muonh/.../newton_schulz_grouped_4d/"
            f"iter_{iteration}/gram/...ik,...jk->...ij/dot_general"
        )
        for iteration in range(3)
    ]
    payload = {
        "cols": [
            {"id": "rank"},
            {"id": "kernel_name"},
            {"id": "op_name"},
            {"id": "total_duration_us"},
            {"id": "occurrences"},
        ],
        "rows": [
            *[
                {
                    "c": [
                        {"v": iteration + 1},
                        {"v": "nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT"},
                        {"v": op_name},
                        {"v": 416 * 852.0},
                        {"v": 416},
                    ]
                }
                for iteration, op_name in enumerate(op_names)
            ],
            {
                "c": [
                    {"v": 4},
                    {"v": "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"},
                    {
                        "v": (
                            "jit(train_step)/forward_backward/transpose(jvp(Transformer))/forward_backward/"
                            "jvp(Transformer)/checkpoint/Block/Block._mlp_update/MoEMLP/MoEExpertMlp/"
                            "moe_mlp/shard_map/scatter/all_gather:"
                        )
                    },
                    {"v": 416 * 785.736},
                    {"v": 416},
                ]
            },
            {
                "c": [
                    {"v": 5},
                    {"v": "unmapped_kernel"},
                    {"v": "train_step/unknown_expensive_region"},
                    {"v": 12_000.0},
                    {"v": 3},
                ]
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_grouped_muonh_kernel_stats(path: Path) -> None:
    payload = {
        "cols": [
            {"id": "rank"},
            {"id": "kernel_name"},
            {"id": "op_name"},
            {"id": "total_duration_us"},
            {"id": "occurrences"},
        ],
        "rows": [
            {
                "c": [
                    {"v": 1},
                    {"v": "nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT"},
                    {
                        "v": (
                            "optimizer_update/grouped_muonh/grouped_muon_newton_schulz/"
                            "iter_0/gram/...ik,...jk->...ij/dot_general"
                        )
                    },
                    {"v": 16_000.0},
                    {"v": 16},
                ]
            },
            {
                "c": [
                    {"v": 2},
                    {"v": "fusion_add"},
                    {"v": "optimizer_update/grouped_muonh/grouped_muon_newton_schulz/" "iter_0/polynomial/fused_add"},
                    {"v": 8_000.0},
                    {"v": 16},
                ]
            },
            {
                "c": [
                    {"v": 3},
                    {"v": "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"},
                    {"v": "optimizer_update/grouped_muonh/chunk_local_restore/" "shard_map/all_gather:"},
                    {"v": 32_000.0},
                    {"v": 16},
                ]
            },
            {
                "c": [
                    {"v": 4},
                    {"v": "reduce_sum"},
                    {"v": "optimizer_update/grouped_muonh/hyperball/norms/reduce_sum:"},
                    {"v": 12_000.0},
                    {"v": 16},
                ]
            },
            {
                "c": [
                    {"v": 5},
                    {"v": "subtract"},
                    {"v": "optimizer_update/grouped_muonh/hyperball/project/sub:"},
                    {"v": 12_000.0},
                    {"v": 16},
                ]
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_single_xprof_row(
    path: Path,
    *,
    op_name: str,
    kernel_name: str,
    total_duration_us: float,
    occurrences: int,
) -> None:
    payload = {
        "cols": [
            {"id": "rank"},
            {"id": "kernel_name"},
            {"id": "op_name"},
            {"id": "total_duration_us"},
            {"id": "occurrences"},
        ],
        "rows": [
            {
                "c": [
                    {"v": 1},
                    {"v": kernel_name},
                    {"v": op_name},
                    {"v": total_duration_us},
                    {"v": occurrences},
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_xprof_rows(path: Path, rows: list[tuple[str, str, float, int]]) -> None:
    payload = {
        "cols": [
            {"id": "rank"},
            {"id": "kernel_name"},
            {"id": "op_name"},
            {"id": "total_duration_us"},
            {"id": "occurrences"},
        ],
        "rows": [
            {
                "c": [
                    {"v": rank},
                    {"v": kernel_name},
                    {"v": op_name},
                    {"v": total_duration_us},
                    {"v": occurrences},
                ]
            }
            for rank, (op_name, kernel_name, total_duration_us, occurrences) in enumerate(rows, 1)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_overview_page(path: Path, *, device_core_count: int) -> None:
    payload = [
        {
            "cols": [],
            "p": {
                "device_core_count": str(device_core_count),
            },
            "rows": [],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_train_step_trace(path: Path) -> None:
    events = []
    for step, run_id in enumerate(["-1", "-2"]):
        timestamp = 100_000.0 + step * 1_000_000.0
        events.append(
            {
                "name": "CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                "ph": "X",
                "ts": timestamp,
                "dur": 10_000.0,
                "args": {
                    "name": "jit_train_step",
                    "num_addressable_devices": "8",
                    "num_partitions": "16",
                    "run_id": run_id,
                },
            }
        )
        for device_id in range(8):
            events.append(
                {
                    "name": f"[{device_id}] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                    "ph": "X",
                    "ts": timestamp + device_id,
                    "dur": 9_000.0,
                    "args": {
                        "global_device_id": str(device_id),
                        "name": "jit_train_step",
                    },
                }
            )
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _write_device_activity_trace(path: Path) -> None:
    events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": "/device:GPU:0"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 2,
            "tid": 0,
            "args": {"name": "/device:GPU:1"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 701,
            "tid": 0,
            "args": {"name": "/host:CPU"},
        },
        {
            "name": "CommonPjRtLoadedExecutable::Execute (jit_train_step)",
            "ph": "X",
            "pid": 701,
            "tid": 100,
            "ts": 1_000.0,
            "dur": 1_000.0,
            "args": {
                "name": "jit_train_step",
                "run_id": "-1",
            },
        },
        {
            "name": "[0] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
            "ph": "X",
            "pid": 701,
            "tid": 101,
            "ts": 1_000.0,
            "dur": 1_000.0,
            "args": {
                "global_device_id": "0",
                "name": "jit_train_step",
            },
        },
        {
            "name": "[1] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
            "ph": "X",
            "pid": 701,
            "tid": 102,
            "ts": 1_000.0,
            "dur": 1_000.0,
            "args": {
                "global_device_id": "1",
                "name": "jit_train_step",
            },
        },
        {"name": "kernel_a", "ph": "X", "pid": 1, "tid": 10, "ts": 1_000.0, "dur": 400.0, "args": {}},
        {"name": "kernel_b", "ph": "X", "pid": 1, "tid": 10, "ts": 1_600.0, "dur": 200.0, "args": {}},
        {"name": "kernel_c", "ph": "X", "pid": 2, "tid": 20, "ts": 1_100.0, "dur": 500.0, "args": {}},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)
