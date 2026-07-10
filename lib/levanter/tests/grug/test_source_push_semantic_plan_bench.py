# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _bench_module():
    script_path = Path(__file__).parents[2] / "scripts" / "bench" / "bench_source_push_semantic_plan.py"
    spec = importlib.util.spec_from_file_location("bench_source_push_semantic_plan", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_push_semantic_plan_constrains_w13_backward_expert_major_inputs_to_destination_major():
    bench_source_push_semantic_plan = _bench_module()
    mesh = bench_source_push_semantic_plan._make_mesh(1)
    inputs = bench_source_push_semantic_plan.SemanticW13BackwardExpertBenchInputs(
        x_expert=jnp.arange(1 * 2 * 3 * 4, dtype=jnp.float32).reshape(1, 2, 3, 4).astype(jnp.bfloat16),
        dz13=jnp.arange(1 * 2 * 3 * 8, dtype=jnp.float32).reshape(1, 2, 3, 8).astype(jnp.bfloat16),
        w_gate_up=jnp.arange(1 * 2 * 4 * 8, dtype=jnp.float32).reshape(1, 2, 4, 8).astype(jnp.bfloat16),
        valid=jnp.asarray([[[True, False, True], [False, True, True]]]),
    )

    constrained = bench_source_push_semantic_plan._constrain_w13_backward_expert_major_inputs(inputs, mesh)

    assert constrained.x_expert.sharding.spec == bench_source_push_semantic_plan.P(
        bench_source_push_semantic_plan.SOURCE_PUSH_MESH_AXIS,
        None,
        None,
        None,
    )
    assert constrained.dz13.sharding.spec == constrained.x_expert.sharding.spec
    assert constrained.w_gate_up.sharding.spec == constrained.x_expert.sharding.spec
    assert constrained.valid.sharding.spec == bench_source_push_semantic_plan.P(
        bench_source_push_semantic_plan.SOURCE_PUSH_MESH_AXIS,
        None,
        None,
    )
    np.testing.assert_array_equal(np.asarray(constrained.x_expert), np.asarray(inputs.x_expert))
    np.testing.assert_array_equal(np.asarray(constrained.dz13), np.asarray(inputs.dz13))
    np.testing.assert_array_equal(np.asarray(constrained.w_gate_up), np.asarray(inputs.w_gate_up))
    np.testing.assert_array_equal(np.asarray(constrained.valid), np.asarray(inputs.valid))


def test_source_push_semantic_plan_all_modes_excludes_source_driven_lowering_repros():
    bench_source_push_semantic_plan = _bench_module()

    all_modes = bench_source_push_semantic_plan._parse_modes("all")

    assert bench_source_push_semantic_plan.MODE_W13_EXPERT_MAJOR_PACK_PALLAS_DIRECT in all_modes
    assert bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_PALLAS in all_modes
    for mode in bench_source_push_semantic_plan.SOURCE_DRIVEN_EXPLICIT_MODES:
        assert mode not in all_modes


def test_source_push_semantic_plan_source_driven_lowering_repros_remain_explicit_modes():
    bench_source_push_semantic_plan = _bench_module()

    mode = bench_source_push_semantic_plan.MODE_W13_EXPERT_MAJOR_PALLAS

    assert bench_source_push_semantic_plan._parse_modes(mode) == (mode,)


def test_source_push_semantic_plan_current_best_alias_selects_promoted_fwd_bwd_mode():
    bench_source_push_semantic_plan = _bench_module()

    assert bench_source_push_semantic_plan._parse_modes("current_best_fwd_bwd") == (
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_OWNER_SHARDED_Y_PALLAS,
    )
    assert bench_source_push_semantic_plan._parse_modes("current_best_fwd_bwd_with_metadata") == (
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_OWNER_SHARDED_Y_WITH_METADATA_PALLAS,
    )
    assert bench_source_push_semantic_plan._parse_modes("current_best_fwd_bwd_with_pallas_metadata") == (
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_OWNER_SHARDED_Y_WITH_PALLAS_METADATA_PALLAS,
    )


def test_source_push_semantic_plan_integrated_aliases_and_cli_help(capsys):
    bench_source_push_semantic_plan = _bench_module()

    assert bench_source_push_semantic_plan._parse_modes(
        "integrated_forward,integrated_forward_compare,diagnostic_fwd_bwd_duplicate_w2,"
        "diagnostic_fwd_bwd_duplicate_w2_compare,direct_queue_fwd_bwd,"
        "direct_queue_fwd_bwd_with_metadata,direct_queue_fwd_bwd_compare"
    ) == (
        bench_source_push_semantic_plan.MODE_FORWARD_EXPERT_MAJOR_DIRECT_PACK_DIRECT_RETURN_COMBINE_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_EXPERT_MAJOR_DIRECT_PACK_DIRECT_RETURN_COMBINE_COMPARE,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_DIRECT_RETURN_COMBINE_DUPLICATE_W2_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_DIRECT_RETURN_COMBINE_DUPLICATE_W2_COMPARE,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_DIRECT_QUEUE_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_DIRECT_QUEUE_WITH_METADATA_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_EXPERT_MAJOR_SAVED_X_DIRECT_PACK_DIRECT_QUEUE_COMPARE,
    )

    with pytest.raises(SystemExit) as exc_info:
        bench_source_push_semantic_plan.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for alias in (
        "integrated_forward",
        "integrated_forward_compare",
        "diagnostic_fwd_bwd_duplicate_w2",
        "diagnostic_fwd_bwd_duplicate_w2_compare",
        "direct_queue_fwd_bwd",
        "direct_queue_fwd_bwd_with_metadata",
        "direct_queue_fwd_bwd_compare",
    ):
        assert alias in help_text


def test_source_push_semantic_plan_source_padded_direct_aliases_are_distinct_from_inbox_modes():
    bench_source_push_semantic_plan = _bench_module()

    assert bench_source_push_semantic_plan._parse_modes(
        "source_padded_pack,source_padded_direct_w13,source_padded_direct_w13_compare,"
        "source_padded_direct_forward,source_padded_direct_forward_compare,"
        "source_padded_direct_fwd_bwd,source_padded_direct_fwd_bwd_compare,"
        "source_padded_direct_fwd_bwd_with_metadata"
    ) == (
        bench_source_push_semantic_plan.MODE_SOURCE_PADDED_INBOX_PACK_PALLAS,
        bench_source_push_semantic_plan.MODE_W13_SOURCE_PADDED_DIRECT_PACK_PALLAS,
        bench_source_push_semantic_plan.MODE_W13_SOURCE_PADDED_DIRECT_PACK_COMPARE,
        bench_source_push_semantic_plan.MODE_FORWARD_SOURCE_PADDED_DIRECT_PACK_DIRECT_RETURN_COMBINE_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_SOURCE_PADDED_DIRECT_PACK_DIRECT_RETURN_COMBINE_COMPARE,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_SOURCE_PADDED_DIRECT_PACK_DIRECT_QUEUE_PALLAS,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_SOURCE_PADDED_DIRECT_PACK_DIRECT_QUEUE_COMPARE,
        bench_source_push_semantic_plan.MODE_FORWARD_BACKWARD_SOURCE_PADDED_DIRECT_PACK_DIRECT_QUEUE_WITH_METADATA_PALLAS,
    )


def test_source_push_semantic_plan_semantic_permute_w13_alias_and_cli_help(capsys):
    bench_source_push_semantic_plan = _bench_module()

    assert bench_source_push_semantic_plan._parse_modes("semantic_permute_w13") == (
        bench_source_push_semantic_plan.MODE_SEMANTIC_PERMUTE_W13_PALLAS,
        bench_source_push_semantic_plan.MODE_W13_SOURCE_PADDED_INBOX_PALLAS,
        bench_source_push_semantic_plan.MODE_SOURCE_PADDED_INBOX_PACK_PALLAS,
        bench_source_push_semantic_plan.MODE_W13_EXPERT_MAJOR_PREPACKED_PALLAS,
    )

    with pytest.raises(SystemExit) as exc_info:
        bench_source_push_semantic_plan.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "semantic_permute_w13" in capsys.readouterr().out


def test_source_push_semantic_plan_semantic_fused_stage_alias_includes_queue_shaped_w2_backward():
    bench_source_push_semantic_plan = _bench_module()

    assert bench_source_push_semantic_plan._parse_modes("semantic_fused_stages") == (
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_W2_RETURN_PALLAS,
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_W2_BACKWARD_PALLAS,
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_W13_BACKWARD_PALLAS,
    )
    assert bench_source_push_semantic_plan._parse_modes("semantic_fused_w2_backward_pallas") == (
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_W2_BACKWARD_PALLAS,
    )


def test_source_push_semantic_plan_builds_fused_w2_backward_inputs_directly_sharded():
    bench = _bench_module()
    selected = jnp.asarray([[[0], [0]]], dtype=jnp.int32)
    weights = jnp.ones(selected.shape, dtype=jnp.float32)
    plan = bench.build_source_push_semantic_plan_jax(
        selected,
        weights,
        ep_size=1,
        experts_per_rank=1,
        rows_per_src_dst_capacity=2,
        capacity_factor=1.0,
    )
    args = SimpleNamespace(
        ep_size=1,
        tokens_per_rank=2,
        hidden_dim=256,
        intermediate_dim=128,
        experts_per_rank=1,
    )
    mesh = bench._make_mesh(1)

    inputs = bench._make_semantic_fused_w2_backward_inputs(args, plan, mesh)

    assert inputs.dy.shape == (1, 2, 256)
    assert inputs.return_y.shape[-2:] == (64, 256)
    assert inputs.h_expert.shape == (1, 1, 256, 128)
    assert inputs.w_down.shape == (1, 1, 128, 256)
    assert inputs.dy.sharding.spec == bench.P(bench.SOURCE_PUSH_MESH_AXIS, None, None)
    assert inputs.return_y.sharding.spec == bench.P(bench.SOURCE_PUSH_MESH_AXIS, None, None, None, None)
    assert inputs.h_expert.sharding.spec == bench.P(bench.SOURCE_PUSH_MESH_AXIS, None, None, None)


def test_source_push_semantic_plan_source_padded_h_reference_uses_stored_bf16_z():
    bench_source_push_semantic_plan = _bench_module()
    x = jnp.asarray([[[1.1, 0.0]]], dtype=jnp.bfloat16)
    w_gate_up = jnp.asarray([[[[1.1, 1.3], [0.0, 0.0]]]], dtype=jnp.bfloat16)
    z_fp32 = jnp.einsum(
        "h,ho->o",
        x[0, 0].astype(jnp.float32),
        w_gate_up[0, 0].astype(jnp.float32),
    )
    stored_z = z_fp32.astype(jnp.bfloat16).reshape(1, 1, 1, 2)
    gate, up = jnp.split(stored_z.astype(jnp.float32), 2, axis=-1)
    stored_h = jax.nn.silu(gate) * up
    float_gate, float_up = jnp.split(z_fp32, 2, axis=-1)
    float_h = jax.nn.silu(float_gate) * float_up

    metrics = bench_source_push_semantic_plan._source_padded_expert_sample_metrics(
        x,
        w_gate_up,
        stored_z,
        stored_h,
        jnp.ones((1, 1, 1), dtype=jnp.bool_),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        jnp.ones((1, 1, 1), dtype=jnp.bool_),
    )

    assert float(jnp.max(jnp.abs(stored_h.reshape(-1) - float_h.reshape(-1)))) > 0
    assert float(metrics["z_max_abs_diff"]) == 0.0
    assert float(metrics["h_max_abs_diff"]) == 0.0


def test_source_push_semantic_plan_only_reserves_tail_guard_when_requested():
    bench_source_push_semantic_plan = _bench_module()
    plan = SimpleNamespace(rows_per_local_expert=jnp.asarray([[4096, 4249]], dtype=jnp.int32))

    assert bench_source_push_semantic_plan._rows_per_expert_capacity(plan, row_multiple=128) == 4352
    assert (
        bench_source_push_semantic_plan._rows_per_expert_capacity(
            plan,
            row_multiple=128,
            tail_guard_rows=127,
        )
        == 4480
    )


def test_source_push_semantic_plan_direct_return_combine_samples_detect_perturbations():
    bench_source_push_semantic_plan = _bench_module()
    observed_h = jnp.asarray([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=jnp.bfloat16)
    w_down = jnp.asarray(
        [[[[1.0, 0.0, 2.0, -1.0], [0.5, 1.0, -1.0, 2.0]]]],
        dtype=jnp.bfloat16,
    )
    route_rows = jnp.einsum(
        "ri,ih->rh",
        observed_h[0, 0].astype(jnp.float32),
        w_down[0, 0].astype(jnp.float32),
    ).astype(jnp.bfloat16)
    observed_return_y = route_rows.reshape(1, 1, 1, 2, 4)
    route_weight = jnp.asarray([[[1.0], [0.5]]], dtype=jnp.float32)
    observed_y = route_rows.astype(jnp.float32) * route_weight[0, :, 0, None]
    observed_y = observed_y.reshape(1, 2, 4)
    queue_metadata = SimpleNamespace(
        queue_local_expert=jnp.asarray([[[0]]], dtype=jnp.int32),
        queue_local_row_start=jnp.asarray([[[0]]], dtype=jnp.int32),
        queue_dst_ord=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        queue_entry=jnp.zeros((1, 2, 1), dtype=jnp.int32),
        queue_row=jnp.asarray([[[0], [1]]], dtype=jnp.int32),
        route_weight=route_weight,
        route_valid=jnp.ones((1, 2, 1), dtype=jnp.bool_),
    )
    source_row_bases = jnp.zeros((1, 1, 1), dtype=jnp.int32)

    def metrics(return_y, y):
        return bench_source_push_semantic_plan._direct_return_combine_sample_metrics(
            y,
            return_y,
            observed_h,
            w_down,
            source_row_bases,
            queue_metadata,
        )

    exact = metrics(observed_return_y, observed_y)
    assert float(exact["return_y_max_abs_diff"]) == 0.0
    assert float(exact["y_max_abs_diff"]) == 0.0
    assert int(exact["return_y_sampled_element_count"]) == observed_return_y.size
    assert int(exact["y_sampled_element_count"]) == observed_y.size
    assert int(exact["expected_return_y_nonfinite_error_count"]) == 0
    assert int(exact["observed_return_y_nonfinite_error_count"]) == 0
    assert int(exact["expected_y_nonfinite_error_count"]) == 0
    assert int(exact["observed_y_nonfinite_error_count"]) == 0

    mesh = bench_source_push_semantic_plan._make_mesh(1)
    with bench_source_push_semantic_plan.jax.set_mesh(mesh):
        explicit_metrics = bench_source_push_semantic_plan.jax.jit(metrics)(
            bench_source_push_semantic_plan.jax.sharding.reshard(
                observed_return_y,
                bench_source_push_semantic_plan.P(
                    bench_source_push_semantic_plan.SOURCE_PUSH_MESH_AXIS,
                    None,
                    None,
                    None,
                    None,
                ),
            ),
            bench_source_push_semantic_plan.jax.sharding.reshard(
                observed_y,
                bench_source_push_semantic_plan.P(
                    bench_source_push_semantic_plan.SOURCE_PUSH_MESH_AXIS,
                    None,
                    None,
                ),
            ),
        )
    assert float(explicit_metrics["return_y_max_abs_diff"]) == 0.0
    assert float(explicit_metrics["y_max_abs_diff"]) == 0.0

    perturbed_return_y = observed_return_y.at[0, 0, 0, 1, 3].add(jnp.asarray(1.0, dtype=jnp.bfloat16))
    assert float(metrics(perturbed_return_y, observed_y)["return_y_max_abs_diff"]) > 0.0

    perturbed_y = observed_y.at[0, 1, 3].add(1.0)
    assert float(metrics(observed_return_y, perturbed_y)["y_max_abs_diff"]) > 0.0


def test_source_push_semantic_plan_direct_dx_samples_separate_producer_and_combine_errors():
    bench_source_push_semantic_plan = _bench_module()
    dx_route = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 1, 4).astype(jnp.bfloat16)
    observed_return_dx = jnp.stack(
        (
            jnp.stack((dx_route[0, 0, 0], dx_route[1, 0, 0])),
            jnp.stack((dx_route[1, 0, 0], dx_route[0, 0, 0])),
        )
    ).reshape(2, 2, 1, 1, 4)
    queue_metadata = SimpleNamespace(
        local_expert=jnp.zeros((2, 2, 1), dtype=jnp.int32),
        local_row_start=jnp.zeros((2, 2, 1), dtype=jnp.int32),
        valid_rows=jnp.ones((2, 2, 1), dtype=jnp.int32),
        route_dst_ordinal=jnp.asarray([[[0, 1]], [[0, 1]]], dtype=jnp.int32),
        route_entry=jnp.zeros((2, 1, 2), dtype=jnp.int32),
        route_queue_row=jnp.zeros((2, 1, 2), dtype=jnp.int32),
        route_valid=jnp.ones((2, 1, 2), dtype=jnp.bool_),
    )
    source_row_bases = jnp.zeros((2, 2, 1), dtype=jnp.int32)

    exact_producer = bench_source_push_semantic_plan._direct_dx_queue_sample_metrics(
        observed_return_dx,
        dx_route,
        source_row_bases,
        queue_metadata,
    )
    assert float(exact_producer["producer_return_dx_max_abs_diff"]) == 0.0
    assert float(exact_producer["producer_return_dx_dst_ordinal_0_max_abs_diff"]) == 0.0
    assert float(exact_producer["producer_return_dx_dst_ordinal_1_max_abs_diff"]) == 0.0
    assert float(exact_producer["producer_return_dx_expected_ordinal_0_stored_axis_0_max_abs_diff"]) == 0.0
    assert float(exact_producer["producer_return_dx_expected_ordinal_1_stored_axis_1_max_abs_diff"]) == 0.0
    assert int(exact_producer["producer_return_dx_sampled_element_count"]) == 16
    assert int(exact_producer["producer_return_dx_dst_ordinal_0_live_element_count"]) == 8
    assert int(exact_producer["producer_return_dx_dst_ordinal_1_live_element_count"]) == 8

    perturbed_return_dx = observed_return_dx.at[0, 1, 0, 0, 2].add(jnp.asarray(2.0, dtype=jnp.bfloat16))
    producer_metrics = bench_source_push_semantic_plan._direct_dx_queue_sample_metrics(
        perturbed_return_dx,
        dx_route,
        source_row_bases,
        queue_metadata,
    )
    assert float(producer_metrics["producer_return_dx_max_abs_diff"]) == 2.0
    assert float(producer_metrics["producer_return_dx_dst_ordinal_0_max_abs_diff"]) == 0.0
    assert float(producer_metrics["producer_return_dx_dst_ordinal_1_max_abs_diff"]) == 2.0
    assert float(producer_metrics["producer_return_dx_expected_ordinal_1_stored_axis_1_max_abs_diff"]) == 2.0

    observed_dx = jnp.sum(perturbed_return_dx[:, :, 0, 0].astype(jnp.float32), axis=1).reshape(2, 1, 4)
    combine_metrics = bench_source_push_semantic_plan._direct_dx_combine_sample_metrics(
        observed_dx,
        perturbed_return_dx,
        dx_route,
        source_row_bases,
        queue_metadata,
    )
    assert float(combine_metrics["combine_from_observed_queue_dx_max_abs_diff"]) == 0.0
    assert float(combine_metrics["dx_max_abs_diff"]) == 2.0

    trusted_metrics = bench_source_push_semantic_plan._sampled_source_output_metrics(
        "dx_vs_source_gather",
        observed_dx,
        observed_dx,
    )
    assert float(trusted_metrics["dx_vs_source_gather_max_abs_diff"]) == 0.0

    perturbed_dx = observed_dx.at[1, 0, 3].add(1.0)
    combine_error_metrics = bench_source_push_semantic_plan._direct_dx_combine_sample_metrics(
        perturbed_dx,
        perturbed_return_dx,
        dx_route,
        source_row_bases,
        queue_metadata,
    )
    assert float(combine_error_metrics["combine_from_observed_queue_dx_max_abs_diff"]) == 1.0
    assert (
        float(
            bench_source_push_semantic_plan._sampled_source_output_metrics(
                "dx_vs_source_gather",
                perturbed_dx,
                observed_dx,
            )["dx_vs_source_gather_max_abs_diff"]
        )
        == 1.0
    )


def test_source_push_semantic_plan_source_padded_modes_match_references(tmp_path):
    jsonl = tmp_path / "semantic_source_padded.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "64",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "64",
            "--intermediate-dim",
            "64",
            "--rows-per-src-dst-capacity",
            "exact",
            "--routing",
            "random",
            "--modes",
            (
                "source_padded_pack,source_padded_w13,source_padded_w13_compare,"
                "source_padded_forward,source_padded_forward_compare,"
                "source_padded_fwd_bwd,source_padded_fwd_bwd_compare,"
                "source_padded_direct_w13,source_padded_direct_w13_compare,"
                "source_padded_direct_forward,source_padded_direct_forward_compare,"
                "source_padded_direct_fwd_bwd,source_padded_direct_fwd_bwd_compare,"
                "source_padded_direct_fwd_bwd_with_metadata"
            ),
            "--gather-row-block",
            "64",
            "--gather-hidden-block",
            "64",
            "--w2-expert-major-row-block",
            "64",
            "--w2-expert-major-intermediate-block",
            "64",
            "--w2-expert-major-hidden-block",
            "64",
            "--forward-return-row-block",
            "64",
            "--forward-return-hidden-block",
            "64",
            "--backward-row-block",
            "64",
            "--backward-hidden-block",
            "64",
            "--dx-return-hidden-block",
            "64",
            "--w13-backward-row-block",
            "64",
            "--w13-backward-hidden-block",
            "64",
            "--w13-backward-output-block",
            "64",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    assert set(summaries) == {
        "source_padded_inbox_pack_pallas",
        "w13_source_padded_inbox_pallas",
        "w13_source_padded_inbox_compare",
        "forward_source_padded_inbox_direct_return_combine_pallas",
        "forward_source_padded_inbox_direct_return_combine_compare",
        "forward_backward_source_padded_inbox_direct_queue_pallas",
        "forward_backward_source_padded_inbox_direct_queue_compare",
        "w13_source_padded_direct_pack_pallas",
        "w13_source_padded_direct_pack_compare",
        "forward_source_padded_direct_pack_direct_return_combine_pallas",
        "forward_source_padded_direct_pack_direct_return_combine_compare",
        "forward_backward_source_padded_direct_pack_direct_queue_pallas",
        "forward_backward_source_padded_direct_pack_direct_queue_compare",
        "forward_backward_source_padded_direct_pack_direct_queue_with_metadata_pallas",
    }
    for summary in summaries.values():
        assert summary["error_rows"] == 0
        if "median_queue_overflow_entry_error_count" in summary:
            assert summary["median_queue_overflow_entry_error_count"] == 0.0
            assert summary["median_queue_overflow_route_error_count"] == 0.0
        assert summary["median_layout_overflow_row_error_count"] == 0.0

    pack = summaries["source_padded_inbox_pack_pallas"]
    assert pack["implementation"] == "pallas_mgpu"
    assert pack["median_useful_tflops_per_rank"] is None
    assert pack["median_rounded_tflops_per_rank"] is None

    w13_compare = summaries["w13_source_padded_inbox_compare"]
    assert w13_compare["median_z_max_abs_diff"] == 0.0
    assert w13_compare["median_h_max_abs_diff"] == 0.0
    assert w13_compare["median_valid_error_count"] == 0.0
    assert w13_compare["median_source_row_base_error_count"] == 0.0

    forward_compare = summaries["forward_source_padded_inbox_direct_return_combine_compare"]
    assert forward_compare["median_return_y_max_abs_diff"] == 0.0
    assert forward_compare["median_y_max_abs_diff"] == 0.0
    assert forward_compare["median_expected_y_nonfinite_error_count"] == 0.0
    assert forward_compare["median_observed_y_nonfinite_error_count"] == 0.0

    fwd_bwd_compare = summaries["forward_backward_source_padded_inbox_direct_queue_compare"]
    for stage in ("y", "return_y", "dy_route", "dcombine", "dh", "dz", "dw2", "dx_route", "dw13", "dx"):
        assert np.isfinite(fwd_bwd_compare[f"median_{stage}_stage_max_abs_diff"])
        assert fwd_bwd_compare[f"median_expected_{stage}_stage_nonfinite_error_count"] == 0.0
        assert fwd_bwd_compare[f"median_observed_{stage}_stage_nonfinite_error_count"] == 0.0
    for stage in ("y", "return_y", "dy_route", "dcombine", "dh", "dz", "dw2"):
        assert fwd_bwd_compare[f"median_{stage}_stage_max_abs_diff"] == 0.0
    assert fwd_bwd_compare["median_valid_error_count"] == 0.0
    assert fwd_bwd_compare["median_x_residual_valid_error_count"] == 0.0
    assert fwd_bwd_compare["median_source_row_base_error_count"] == 0.0
    assert fwd_bwd_compare["median_layout_overflow_mismatch_error_count"] == 0.0

    direct_w13_compare = summaries["w13_source_padded_direct_pack_compare"]
    assert direct_w13_compare["median_x_max_abs_diff"] == 0.0
    assert direct_w13_compare["median_x_sampled_element_count"] > 0
    for stage in ("z", "h"):
        assert np.isfinite(direct_w13_compare[f"median_{stage}_max_abs_diff"])
        assert direct_w13_compare[f"median_expected_{stage}_nonfinite_error_count"] == 0.0
        assert direct_w13_compare[f"median_observed_{stage}_nonfinite_error_count"] == 0.0
    assert direct_w13_compare["median_valid_error_count"] == 0.0
    assert direct_w13_compare["median_source_row_base_error_count"] == 0.0

    direct_forward_compare = summaries["forward_source_padded_direct_pack_direct_return_combine_compare"]
    assert direct_forward_compare["median_x_max_abs_diff"] == 0.0
    assert direct_forward_compare["median_y_sampled_element_count"] > 0
    for stage in ("z", "h", "return_y", "y"):
        assert np.isfinite(direct_forward_compare[f"median_{stage}_max_abs_diff"])
        assert direct_forward_compare[f"median_expected_{stage}_nonfinite_error_count"] == 0.0
        assert direct_forward_compare[f"median_observed_{stage}_nonfinite_error_count"] == 0.0

    direct_fwd_bwd_compare = summaries["forward_backward_source_padded_direct_pack_direct_queue_compare"]
    assert direct_fwd_bwd_compare["median_x_stage_max_abs_diff"] == 0.0
    assert direct_fwd_bwd_compare["median_dx_stage_sampled_element_count"] > 0
    for stage in ("z", "h", "y", "return_y", "dy_route", "dcombine", "dh", "dz", "dw2", "dx_route", "dw13", "dx"):
        assert np.isfinite(direct_fwd_bwd_compare[f"median_{stage}_stage_max_abs_diff"])
        assert direct_fwd_bwd_compare[f"median_expected_{stage}_stage_nonfinite_error_count"] == 0.0
        assert direct_fwd_bwd_compare[f"median_observed_{stage}_stage_nonfinite_error_count"] == 0.0
    assert direct_fwd_bwd_compare["median_valid_error_count"] == 0.0
    assert direct_fwd_bwd_compare["median_source_row_base_error_count"] == 0.0
    assert direct_fwd_bwd_compare["median_layout_overflow_mismatch_error_count"] == 0.0

    direct_fwd_bwd = summaries["forward_backward_source_padded_direct_pack_direct_queue_pallas"]
    assert direct_fwd_bwd["block_sizes"]["w13_row_block"] == 64
    assert direct_fwd_bwd["block_sizes"]["w2_row_block"] == 64
    assert direct_fwd_bwd["block_sizes"]["return_row_block"] == 64
    assert direct_fwd_bwd["block_sizes"]["source_expand_row_block"] == 64
    assert direct_fwd_bwd["block_sizes"]["w13_backward_row_block"] == 64
    assert direct_fwd_bwd["block_sizes"]["dx_return_row_block"] == 64


def test_source_push_semantic_plan_fused_permute_w13_modes_emit_jsonl_metrics(tmp_path):
    jsonl = tmp_path / "semantic_permute_w13.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "64",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "256",
            "--intermediate-dim",
            "128",
            "--rows-per-src-dst-capacity",
            "exact",
            "--routing",
            "random",
            "--modes",
            "semantic_permute_w13_pallas,semantic_permute_w13_compare",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    assert set(summaries) == {"semantic_permute_w13_pallas", "semantic_permute_w13_compare"}
    for summary in summaries.values():
        assert summary["error_rows"] == 0
        assert summary["median_compile_time"] > 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] > 0
        assert summary["median_rounded_tflops_per_rank"] > 0
        assert summary["median_queue_overflow_entry_error_count"] == 0.0
        assert summary["median_queue_overflow_route_error_count"] == 0.0
        assert summary["median_layout_overflow_row_error_count"] == 0.0
        assert (
            summary["block_sizes"]["source_push_profile"]
            == bench_source_push_semantic_plan.SOURCE_PUSH_PROFILE_STABLE_216
        )
        assert summary["block_sizes"]["block_m"] == bench_source_push_semantic_plan.SOURCE_PADDED_ROW_BLOCK

    compare = summaries["semantic_permute_w13_compare"]
    assert compare["median_z_max_abs_diff"] == 0.0
    assert compare["median_h_max_abs_diff"] == 0.0
    assert compare["median_valid_error_count"] == 0.0
    assert compare["median_layout_overflow_mismatch_error_count"] == 0.0


def test_source_push_semantic_plan_pallas_metadata_current_best_emits_block_size_summary(tmp_path):
    jsonl = tmp_path / "semantic_current_best_pallas_metadata.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            "current_best_fwd_bwd_with_pallas_metadata",
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--dx-return-row-block",
            "2",
            "--dx-return-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--metadata-tile-assignments",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    mode = "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_y_with_pallas_metadata_pallas"
    summary = summaries[mode]

    assert summary["implementation"] == "pallas_mgpu"
    assert summary["error_rows"] == 0
    assert summary["metadata_overflow_routes"] == 0
    assert summary["block_sizes"] == {
        "gather_row_block": 16,
        "gather_hidden_block": 512,
        "w13_row_block": 1,
        "w13_hidden_block": 4,
        "w13_intermediate_block": 4,
        "w2_row_block": 2,
        "w2_intermediate_block": 2,
        "w2_hidden_block": 4,
        "source_expand_row_block": 2,
        "source_expand_hidden_block": 4,
        "w13_backward_row_block": 2,
        "w13_backward_hidden_block": 4,
        "w13_backward_output_block": 4,
        "w13_backward_lowering": "warpgroup",
        "dx_return_row_block": 2,
        "dx_return_hidden_block": 4,
    }
    assert summary["median_steady_state_time"] > 0


def test_source_push_semantic_plan_direct_return_queue_modes_emit_summary_rows(tmp_path):
    jsonl = tmp_path / "semantic_direct_return_queue.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = (
        "forward_return_direct_to_source_pallas,"
        "forward_return_direct_to_source_compare,"
        "forward_combine_source_gather_pallas,"
        "forward_combine_source_gather_compare,"
        "forward_expert_major_direct_return_combine_pallas,"
        "forward_expert_major_direct_return_combine_compare"
    )
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    for mode in modes.split(","):
        assert summaries[mode]["error_rows"] == 0
        assert summaries[mode]["metadata_overflow_routes"] == 0
        assert summaries[mode]["median_steady_state_time"] > 0
    assert summaries["forward_return_direct_to_source_pallas"]["block_sizes"] == {
        "w2_row_block": 2,
        "w2_intermediate_block": 2,
        "w2_hidden_block": 4,
        "return_row_block": 2,
        "return_hidden_block": 4,
    }


def test_source_push_semantic_plan_integrated_forward_and_duplicate_w2_diagnostic_emit_compare_metrics(tmp_path):
    jsonl = tmp_path / "semantic_integrated_direct_return.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            "integrated_forward,integrated_forward_compare,diagnostic_fwd_bwd_duplicate_w2,diagnostic_fwd_bwd_duplicate_w2_compare",
            "--gather-row-block",
            "2",
            "--gather-hidden-block",
            "4",
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--dx-return-row-block",
            "2",
            "--dx-return-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    forward_mode = "forward_expert_major_direct_pack_direct_return_combine_pallas"
    forward_compare_mode = "forward_expert_major_direct_pack_direct_return_combine_compare"
    fwd_bwd_mode = "forward_backward_expert_major_saved_x_direct_pack_direct_return_combine_duplicate_w2_pallas"
    fwd_bwd_compare_mode = (
        "forward_backward_expert_major_saved_x_direct_pack_direct_return_combine_duplicate_w2_compare"
    )

    assert set(summaries) == {forward_mode, forward_compare_mode, fwd_bwd_mode, fwd_bwd_compare_mode}
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None

    assert summaries[forward_mode]["block_sizes"] == {
        "w13_row_block": 1,
        "w13_hidden_block": 4,
        "w13_intermediate_block": 4,
        "w2_row_block": 2,
        "w2_intermediate_block": 2,
        "w2_hidden_block": 4,
        "return_row_block": 2,
        "return_hidden_block": 4,
        "gather_row_block": 2,
        "gather_hidden_block": 4,
    }
    forward_compare = summaries[forward_compare_mode]
    assert np.isfinite(forward_compare["median_y_max_abs_diff"])
    assert np.isfinite(forward_compare["median_y_mean_abs_diff"])
    assert forward_compare["median_return_y_max_abs_diff"] == 0.0
    assert forward_compare["median_return_y_mean_abs_diff"] == 0.0
    assert forward_compare["median_return_y_live_element_count"] > 0
    assert forward_compare["median_return_y_sampled_element_count"] > 0
    assert forward_compare["median_y_sampled_element_count"] > 0
    assert forward_compare["median_expected_y_nonfinite_error_count"] == 0.0
    assert forward_compare["median_observed_y_nonfinite_error_count"] == 0.0
    assert forward_compare["median_expected_return_y_nonfinite_error_count"] == 0.0
    assert forward_compare["median_observed_return_y_nonfinite_error_count"] == 0.0

    fwd_bwd_compare = summaries[fwd_bwd_compare_mode]
    assert np.isfinite(fwd_bwd_compare["median_y_stage_max_abs_diff"])
    assert fwd_bwd_compare["median_dy_route_stage_max_abs_diff"] == 0.0
    assert fwd_bwd_compare["median_dw2_stage_max_abs_diff"] == 0.0
    assert np.isfinite(fwd_bwd_compare["median_dw13_stage_max_abs_diff"])
    assert fwd_bwd_compare["median_expected_y_nonfinite_error_count"] == 0.0
    assert fwd_bwd_compare["median_observed_y_nonfinite_error_count"] == 0.0


def test_source_push_semantic_plan_direct_queue_backward_and_fwd_bwd_emit_compare_metrics(tmp_path):
    jsonl = tmp_path / "semantic_direct_queue_fwd_bwd.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "backward_source_expand_from_saved_return_queue_pallas,"
        "backward_source_expand_from_saved_return_queue_compare,"
        "dx_return_direct_to_source_combine_pallas,"
        "dx_return_direct_to_source_combine_compare,"
        "direct_queue_fwd_bwd,"
        "direct_queue_fwd_bwd_with_metadata,"
        "direct_queue_fwd_bwd_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--gather-row-block",
            "2",
            "--gather-hidden-block",
            "4",
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--dx-return-row-block",
            "2",
            "--dx-return-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    expected_modes = {
        "backward_source_expand_from_saved_return_queue_pallas",
        "backward_source_expand_from_saved_return_queue_compare",
        "dx_return_direct_to_source_combine_pallas",
        "dx_return_direct_to_source_combine_compare",
        "forward_backward_expert_major_saved_x_direct_pack_direct_queue_pallas",
        "forward_backward_expert_major_saved_x_direct_pack_direct_queue_with_metadata_pallas",
        "forward_backward_expert_major_saved_x_direct_pack_direct_queue_compare",
    }

    assert set(summaries) == expected_modes
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0

    source_expand_compare = summaries["backward_source_expand_from_saved_return_queue_compare"]
    assert source_expand_compare["median_dy_route_max_abs_diff"] == 0.0
    assert source_expand_compare["median_dcombine_max_abs_diff"] == 0.0

    dx_compare = summaries["dx_return_direct_to_source_combine_compare"]
    assert dx_compare["median_dx_max_abs_diff"] == 0.0
    assert dx_compare["median_dx_mean_abs_diff"] == 0.0
    assert dx_compare["median_producer_return_dx_max_abs_diff"] == 0.0
    assert dx_compare["median_producer_return_dx_mean_abs_diff"] == 0.0
    assert dx_compare["median_producer_return_dx_sampled_element_count"] > 0
    assert dx_compare["median_combine_from_observed_queue_dx_max_abs_diff"] == 0.0
    assert dx_compare["median_combine_from_observed_queue_dx_mean_abs_diff"] == 0.0
    for ordinal in range(2):
        assert dx_compare[f"median_producer_return_dx_dst_ordinal_{ordinal}_max_abs_diff"] == 0.0
        assert dx_compare[f"median_producer_return_dx_dst_ordinal_{ordinal}_live_element_count"] > 0

    full_mode = "forward_backward_expert_major_saved_x_direct_pack_direct_queue_pallas"
    metadata_mode = "forward_backward_expert_major_saved_x_direct_pack_direct_queue_with_metadata_pallas"
    full_compare = summaries["forward_backward_expert_major_saved_x_direct_pack_direct_queue_compare"]
    assert summaries[full_mode]["median_useful_tflops_per_rank"] is not None
    assert summaries[metadata_mode]["median_useful_tflops_per_rank"] is not None
    assert summaries[metadata_mode]["block_sizes"] == summaries[full_mode]["block_sizes"]
    assert full_compare["median_useful_tflops_per_rank"] is not None
    assert full_compare["median_valid_stage_error_count"] == 0.0
    assert full_compare["median_return_y_stage_live_element_count"] > 0
    for stage in (
        "x_expert_stage",
        "z_stage",
        "h_stage",
        "return_y_stage",
        "y_stage",
        "dy_route_stage",
        "dcombine_stage",
        "dh_stage",
        "dz13_stage",
        "dw2_stage",
        "dx_route_stage",
        "dw13_stage",
        "dx_stage",
    ):
        assert np.isfinite(full_compare[f"median_{stage}_max_abs_diff"])
        assert np.isfinite(full_compare[f"median_{stage}_mean_abs_diff"])
    for key, value in full_compare.items():
        if key.endswith("nonfinite_error_count"):
            assert value == 0.0


def test_source_push_semantic_plan_bench_emits_summary_rows(tmp_path):
    jsonl = tmp_path / "semantic_plan.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            "metadata,gather_x",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == {"metadata", "gather_x"}
    assert summaries["metadata"]["implementation"] == "jax"
    assert summaries["gather_x"]["implementation"] == "jax_reference"
    assert summaries["metadata"]["error_rows"] == 0
    assert summaries["gather_x"]["error_rows"] == 0
    assert summaries["metadata"]["median_steady_state_time"] > 0
    assert summaries["gather_x"]["median_steady_state_time"] > 0


def test_source_push_semantic_plan_bench_emits_metadata_pallas_summary_rows(tmp_path):
    jsonl = tmp_path / "semantic_metadata_pallas.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "3",
            "--capacity-factor",
            "0.5",
            "--modes",
            "metadata_pallas,metadata_tile_pallas",
            "--metadata-tile-assignments",
            "3",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == {"metadata_pallas", "metadata_tile_pallas"}
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["routing_dropped_routes"] > 0
    assert summaries["metadata_pallas"]["metadata_overflow_routes"] == 0
    assert summaries["metadata_tile_pallas"]["metadata_overflow_routes"] is None
    assert summaries["metadata_tile_pallas"]["block_sizes"] == {"tile_assignments": 3}


def test_source_push_semantic_plan_bench_uses_pallas_plan_builder_for_stage_modes(tmp_path):
    jsonl = tmp_path / "semantic_pallas_plan_builder_stage.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "3",
            "--capacity-factor",
            "0.5",
            "--modes",
            "gather_x",
            "--plan-builder",
            "pallas",
            "--metadata-tile-assignments",
            "3",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summary = next(row for row in rows if row["row_type"] == "summary")

    assert summary["mode"] == "gather_x"
    assert summary["error_rows"] == 0
    assert summary["shape"]["plan_builder"] == "pallas"
    assert summary["routing_dropped_routes"] > 0
    assert summary["metadata_overflow_routes"] == 0


def test_source_push_semantic_plan_bench_emits_backward_pallas_summary_rows(tmp_path):
    jsonl = tmp_path / "semantic_backward_pallas.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = (
        "backward_source_expand_pallas,"
        "backward_source_expand_expert_major_pallas,"
        "backward_source_expand_expert_major_compare,"
        "backward_source_expand_from_expert_major_owner_sharded_dcombine_pallas,"
        "backward_source_expand_from_expert_major_owner_sharded_dcombine_compare,"
        "backward_dy_route_source_push_expert_major_pallas,"
        "backward_dy_route_source_push_expert_major_compare,"
        "backward_source_expand_from_expert_major_source_push_pallas,"
        "backward_source_expand_from_expert_major_source_push_compare,"
        "backward_dcombine_source_gather_expert_major_pallas,"
        "backward_dcombine_source_gather_expert_major_compare,"
        "backward_source_expand_from_expert_major_source_gather_pallas,"
        "backward_source_expand_from_expert_major_source_gather_compare,"
        "backward_w2_dh_pallas,"
        "backward_w2_dw2_pallas,"
        "backward_w2_pallas_scaffold,"
        "backward_w13_dx_pair_pallas,"
        "backward_w13_dw13_pallas,"
        "backward_w13_pallas_scaffold,"
        "dx_combine_pallas,"
        "dx_combine_expert_major_pallas,"
        "dx_combine_expert_major_compare"
    )
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--w2-backward-row-block",
            "2",
            "--w2-backward-intermediate-block",
            "2",
            "--w2-backward-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0


def test_source_push_semantic_plan_bench_emits_w2_expert_major_prepacked_rows(tmp_path):
    jsonl = tmp_path / "semantic_w2_expert_major_prepacked.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "w2_expert_major_prepacked,"
        "w2_expert_major_prepacked_pallas,"
        "w2_expert_major_prepacked_compare,"
        "w2_expert_major_prepacked_pallas_assume_zero_invalid,"
        "w2_expert_major_prepacked_assume_zero_invalid_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    assert summaries["w2_expert_major_prepacked"]["implementation"] == "jax_reference"
    assert summaries["w2_expert_major_prepacked_pallas"]["implementation"] == "pallas_mgpu"
    assert summaries["w2_expert_major_prepacked_compare"]["implementation"] == "pallas_mgpu"
    assert summaries["w2_expert_major_prepacked_pallas_assume_zero_invalid"]["implementation"] == "pallas_mgpu"
    assert summaries["w2_expert_major_prepacked_assume_zero_invalid_compare"]["implementation"] == "pallas_mgpu"
    for summary in summaries.values():
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None
    assert summaries["w2_expert_major_prepacked_compare"]["median_route_y_max_abs_diff"] == 0.0
    assert summaries["w2_expert_major_prepacked_assume_zero_invalid_compare"]["median_route_y_max_abs_diff"] == 0.0


def test_source_push_semantic_plan_bench_emits_sharded_forward_return_only_compare_rows(tmp_path):
    jsonl = tmp_path / "semantic_forward_return_sharded.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "forward_return_expert_major_prepacked_sharded_compare,"
        "forward_return_slot_reduce_expert_major_prepacked_sharded_compare,"
        "forward_return_slot_reduce_expert_major_prepacked_owner_sharded_compare,"
        "forward_return_sum_expert_major_prepacked_sharded_compare,"
        "forward_return_sum_expert_major_prepacked_owner_sharded_compare,"
        "forward_return_sum_lookup_expert_major_prepacked_owner_sharded_compare,"
        "forward_return_remote_source_gather_expert_major_prepacked_compare,"
        "dx_return_source_gather_expert_major_prepacked_owner_sharded_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--dx-return-hidden-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        if summary["mode"].startswith("dx_return"):
            assert summary["median_dx_max_abs_diff"] == 0.0
            assert summary["median_expected_dx_nonfinite_error_count"] == 0.0
            assert summary["median_observed_dx_nonfinite_error_count"] == 0.0
        else:
            assert summary["median_y_max_abs_diff"] == 0.0
            assert summary["median_expected_y_nonfinite_error_count"] == 0.0
            assert summary["median_observed_y_nonfinite_error_count"] == 0.0

    materialized = summaries["forward_return_expert_major_prepacked_sharded_compare"]
    assert materialized["median_route_by_slot_max_abs_diff"] == 0.0
    assert materialized["median_expected_route_by_slot_nonfinite_error_count"] == 0.0
    assert materialized["median_observed_route_by_slot_nonfinite_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_reverse_route_source_gather_return_rows(tmp_path):
    jsonl = tmp_path / "semantic_forward_return_reverse_route_source_gather.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "forward_return_reverse_route_source_gather_expert_major_prepacked_pallas,"
        "forward_return_reverse_route_source_gather_expert_major_prepacked_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["block_sizes"] == {"row_block": 2, "hidden_block": 4}

    compare = summaries["forward_return_reverse_route_source_gather_expert_major_prepacked_compare"]
    assert compare["median_y_max_abs_diff"] == 0.0
    assert compare["median_expected_y_nonfinite_error_count"] == 0.0
    assert compare["median_observed_y_nonfinite_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_dx_return_slot_reduce_rows(tmp_path):
    jsonl = tmp_path / "semantic_dx_return_slot_reduce.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "dx_return_slot_reduce_expert_major_prepacked_compare,"
        "dx_return_sum_expert_major_prepacked_compare,"
        "dx_return_source_gather_expert_major_prepacked_compare,"
        "dx_return_remote_source_gather_expert_major_prepacked_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--dx-return-row-block",
            "2",
            "--dx-return-hidden-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_dx_max_abs_diff"] == 0.0
        assert summary["median_expected_dx_nonfinite_error_count"] == 0.0
        assert summary["median_observed_dx_nonfinite_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_backward_w2_expert_major_prepacked_compare_rows(tmp_path):
    jsonl = tmp_path / "semantic_backward_w2_expert_major_prepacked.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "backward_w2_expert_major_prepacked,"
        "backward_w2_expert_major_prepacked_pallas,"
        "backward_w2_expert_major_prepacked_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--w2-expert-major-row-block",
            "1",
            "--w2-expert-major-intermediate-block",
            "1",
            "--w2-expert-major-hidden-block",
            "1",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    assert summaries["backward_w2_expert_major_prepacked"]["implementation"] == "jax_reference"
    assert summaries["backward_w2_expert_major_prepacked_pallas"]["implementation"] == "pallas_mgpu"
    assert summaries["backward_w2_expert_major_prepacked_compare"]["implementation"] == "pallas_mgpu"
    for summary in summaries.values():
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None

    compare = summaries["backward_w2_expert_major_prepacked_compare"]
    assert compare["median_d_weighted_activation_max_abs_diff"] == 0.0
    assert compare["median_dw2_max_abs_diff"] == 0.0


def test_source_push_semantic_plan_bench_emits_w13_backward_expert_major_prepacked_rows(tmp_path):
    jsonl = tmp_path / "semantic_w13_backward_expert_major_prepacked.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = (
        "backward_w13_expert_major_prepacked,"
        "backward_w13_expert_major_prepacked_pallas,"
        "backward_w13_dx_route_expert_major_prepacked_pallas,"
        "backward_w13_dw13_expert_major_prepacked_pallas,"
        "backward_w13_expert_major_prepacked_compare"
    )
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    assert summaries["backward_w13_expert_major_prepacked"]["implementation"] == "jax_reference"
    for mode, summary in summaries.items():
        if mode != "backward_w13_expert_major_prepacked":
            assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None
    compare = summaries["backward_w13_expert_major_prepacked_compare"]
    assert compare["median_dx_route_max_abs_diff"] == 0.0
    assert compare["median_dw13_max_abs_diff"] == 0.0


def test_source_push_semantic_plan_bench_emits_w13_expert_major_pallas_summary_row(tmp_path):
    jsonl = tmp_path / "semantic_w13_expert_major_pallas.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            "w13_expert_major_pallas",
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == {"w13_expert_major_pallas"}
    summary = summaries["w13_expert_major_pallas"]
    assert summary["implementation"] == "pallas_mgpu"
    assert summary["error_rows"] == 0
    assert summary["median_steady_state_time"] > 0
    assert summary["median_useful_tflops_per_rank"] is not None


def test_source_push_semantic_plan_bench_emits_w13_expert_major_pack_rows(tmp_path):
    jsonl = tmp_path / "semantic_w13_expert_major_pack.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            (
                "w13_expert_major_pack,w13_expert_major_pack_pallas_scaffold,"
                "w13_expert_major_pack_pallas_direct,w13_expert_major_pack_pallas_lookup,"
                "w13_expert_major_pack_reshard,w13_expert_major_prepacked,"
                "w13_expert_major_prepacked_pallas,w13_expert_major_prepacked_compare"
            ),
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == {
        "w13_expert_major_pack",
        "w13_expert_major_pack_pallas_scaffold",
        "w13_expert_major_pack_pallas_direct",
        "w13_expert_major_pack_pallas_lookup",
        "w13_expert_major_pack_reshard",
        "w13_expert_major_prepacked",
        "w13_expert_major_prepacked_pallas",
        "w13_expert_major_prepacked_compare",
    }
    pack = summaries["w13_expert_major_pack"]
    pack_pallas_scaffold = summaries["w13_expert_major_pack_pallas_scaffold"]
    pack_pallas_direct = summaries["w13_expert_major_pack_pallas_direct"]
    pack_pallas_lookup = summaries["w13_expert_major_pack_pallas_lookup"]
    pack_reshard = summaries["w13_expert_major_pack_reshard"]
    prepacked = summaries["w13_expert_major_prepacked"]
    prepacked_pallas = summaries["w13_expert_major_prepacked_pallas"]
    compare = summaries["w13_expert_major_prepacked_compare"]
    assert pack["implementation"] == "jax_reference"
    assert pack_pallas_scaffold["implementation"] == "pallas_mgpu"
    assert pack_pallas_direct["implementation"] == "pallas_mgpu"
    assert pack_pallas_lookup["implementation"] == "pallas_mgpu"
    assert pack_reshard["implementation"] == "jax_reference"
    assert prepacked["implementation"] == "jax_reference"
    assert prepacked_pallas["implementation"] == "pallas_mgpu"
    assert compare["implementation"] == "pallas_mgpu"
    assert pack["error_rows"] == 0
    assert pack_pallas_scaffold["error_rows"] == 0
    assert pack_pallas_direct["error_rows"] == 0
    assert pack_pallas_lookup["error_rows"] == 0
    assert pack_reshard["error_rows"] == 0
    assert prepacked["error_rows"] == 0
    assert prepacked_pallas["error_rows"] == 0
    assert compare["error_rows"] == 0
    assert pack["median_useful_tflops_per_rank"] is None
    assert pack_pallas_scaffold["median_useful_tflops_per_rank"] is None
    assert pack_pallas_direct["median_useful_tflops_per_rank"] is None
    assert pack_pallas_lookup["median_useful_tflops_per_rank"] is None
    assert pack_pallas_direct["block_sizes"] == {"row_block": 16, "hidden_block": 512}
    assert pack_pallas_lookup["block_sizes"] == {"row_block": 16, "hidden_block": 512}
    assert pack_reshard["median_useful_tflops_per_rank"] is None
    assert prepacked["median_useful_tflops_per_rank"] is not None
    assert prepacked_pallas["median_useful_tflops_per_rank"] is not None
    assert compare["median_z_max_abs_diff"] == 0.0
    assert compare["median_h_max_abs_diff"] < 1e-4


def test_source_push_semantic_plan_bench_emits_w13_expert_major_compare_metrics(tmp_path):
    jsonl = tmp_path / "semantic_w13_expert_major_compare.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            "w13_expert_major_compare",
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    repeats = [row for row in rows if row["row_type"] == "repeat"]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == {"w13_expert_major_compare"}
    assert summaries["w13_expert_major_compare"]["error_rows"] == 0
    assert repeats[0]["z_max_abs_diff"] == 0.0
    assert repeats[0]["h_max_abs_diff"] == 0.0
    assert repeats[0]["valid_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_forward_expert_major_rows(tmp_path):
    jsonl = tmp_path / "semantic_forward_expert_major.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = (
        "forward_expert_major_pallas,"
        "forward_expert_major_compare,"
        "forward_expert_major_return_sum_pallas,"
        "forward_expert_major_return_sum_compare"
    )
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None

    forward_compare = summaries["forward_expert_major_compare"]
    assert forward_compare["median_z_max_abs_diff"] == 0.0
    assert forward_compare["median_h_max_abs_diff"] == 0.0
    assert forward_compare["median_route_y_w2_only_max_abs_diff"] == 0.0
    assert forward_compare["median_valid_error_count"] == 0.0

    return_compare = summaries["forward_expert_major_return_sum_compare"]
    assert return_compare["median_z_max_abs_diff"] == 0.0
    assert return_compare["median_h_max_abs_diff"] == 0.0
    assert return_compare["median_route_y_w2_only_max_abs_diff"] == 0.0
    assert return_compare["median_y_w2_return_only_max_abs_diff"] == 0.0
    assert return_compare["median_valid_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_forward_backward_expert_major_rows(tmp_path):
    jsonl = tmp_path / "semantic_forward_backward_expert_major.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = (
        "forward_backward_expert_major_pallas,"
        "forward_backward_expert_major_compare,"
        "forward_backward_expert_major_saved_x_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_pallas,"
        "forward_backward_expert_major_saved_x_lookup_pack_pallas,"
        "forward_backward_expert_major_saved_x_lookup_pack_compare,"
        "forward_backward_expert_major_saved_x_direct_pack_source_gather_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_source_gather_compare,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_y_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_y_slot_reduce_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_no_y_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_remote_y_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_remote_y_delayed_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_dx_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_remote_dx_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_y_with_metadata_pallas,"
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_compare,"
        "forward_backward_expert_major_saved_x_compare"
    )
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--w13-expert-major-row-block",
            "1",
            "--w13-expert-major-hidden-block",
            "4",
            "--w13-expert-major-intermediate-block",
            "4",
            "--w2-expert-major-row-block",
            "2",
            "--w2-expert-major-intermediate-block",
            "2",
            "--w2-expert-major-hidden-block",
            "4",
            "--forward-return-row-block",
            "2",
            "--forward-return-hidden-block",
            "4",
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--dx-return-row-block",
            "2",
            "--dx-return-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None

    for mode in (
        "forward_backward_expert_major_compare",
        "forward_backward_expert_major_saved_x_compare",
        "forward_backward_expert_major_saved_x_lookup_pack_compare",
        "forward_backward_expert_major_saved_x_direct_pack_source_gather_compare",
        "forward_backward_expert_major_saved_x_direct_pack_owner_sharded_compare",
    ):
        compare = summaries[mode]
        assert compare["median_y_stage_max_abs_diff"] == 0.0
        assert compare["median_dy_route_stage_max_abs_diff"] == 0.0
        assert compare["median_dh_stage_max_abs_diff"] == 0.0
        assert compare["median_dz13_stage_max_abs_diff"] == 0.0
        assert compare["median_dw2_stage_max_abs_diff"] == 0.0
        assert compare["median_expected_dx_nonfinite_error_count"] == 0.0
        assert compare["median_observed_dx_nonfinite_error_count"] == 0.0


def test_source_push_semantic_plan_bench_emits_composed_pallas_summary_rows(tmp_path):
    jsonl = tmp_path / "semantic_composed_pallas.jsonl"
    bench_source_push_semantic_plan = _bench_module()

    modes = "forward_pallas_scaffold,backward_pallas_scaffold,forward_backward_pallas_scaffold"
    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "2",
            "--tokens-per-rank",
            "4",
            "--topk",
            "2",
            "--experts-per-rank",
            "2",
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "4",
            "--rows-per-src-dst-capacity",
            "exact",
            "--modes",
            modes,
            "--gather-row-block",
            "2",
            "--gather-hidden-block",
            "4",
            "--w13-row-block",
            "1",
            "--w13-hidden-block",
            "4",
            "--w13-intermediate-block",
            "4",
            "--w2-row-block",
            "1",
            "--w2-intermediate-block",
            "4",
            "--w2-hidden-block",
            "4",
            "--backward-row-block",
            "2",
            "--backward-hidden-block",
            "4",
            "--w2-backward-row-block",
            "2",
            "--w2-backward-intermediate-block",
            "2",
            "--w2-backward-hidden-block",
            "4",
            "--w13-backward-row-block",
            "2",
            "--w13-backward-hidden-block",
            "4",
            "--w13-backward-output-block",
            "4",
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}

    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None


def test_source_push_semantic_plan_bench_emits_fused_stage_rows(tmp_path):
    jsonl = tmp_path / "semantic_fused_stages.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "semantic_fused_w2_return_pallas,semantic_fused_w2_return_compare,"
        "semantic_fused_w13_backward_pallas,semantic_fused_w13_backward_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "1",
            "--tokens-per-rank",
            "64",
            "--topk",
            "1",
            "--experts-per-rank",
            "1",
            "--hidden-dim",
            "256",
            "--intermediate-dim",
            "128",
            "--capacity-factor",
            "1.0",
            "--routing",
            "balanced",
            "--modes",
            modes,
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    repeats = {row["mode"]: row for row in rows if row["row_type"] == "repeat"}
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] is not None
        assert summary["median_queue_overflow_route_error_count"] == 0.0
        assert summary["median_layout_overflow_row_error_count"] == 0.0

    w2_compare = repeats["semantic_fused_w2_return_compare"]
    assert w2_compare["y_max_abs_diff"] == 0.0
    assert w2_compare["return_y_max_abs_diff"] == 0.0
    assert w2_compare["valid_error_count"] == 0.0
    w13_backward_compare = repeats["semantic_fused_w13_backward_compare"]
    assert w13_backward_compare["dx_max_abs_diff"] == 0.0
    assert w13_backward_compare["dw13_max_abs_diff"] == 0.0


def test_source_push_semantic_plan_fused_mlp_modes_use_target_shape_flops():
    bench_source_push_semantic_plan = _bench_module()
    hidden_dim = 256
    intermediate_dim = 128
    useful_rows = 7
    rounded_rows = 11
    ep_size = 2
    forward_per_row = (
        2.0 * hidden_dim * intermediate_dim * 2.0 + 2.0 * intermediate_dim * hidden_dim + 8.0 * intermediate_dim
    )

    forward = bench_source_push_semantic_plan._mode_flops_per_rank(
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_MLP_FORWARD_PALLAS,
        useful_rows_total=useful_rows,
        rounded_rows_total=rounded_rows,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        ep_size=ep_size,
    )
    forward_backward = bench_source_push_semantic_plan._mode_flops_per_rank(
        bench_source_push_semantic_plan.MODE_SEMANTIC_FUSED_MLP_FORWARD_BACKWARD_PALLAS,
        useful_rows_total=useful_rows,
        rounded_rows_total=rounded_rows,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        ep_size=ep_size,
    )

    assert forward == (useful_rows * forward_per_row / ep_size, rounded_rows * forward_per_row / ep_size)
    assert forward_backward == (
        useful_rows * 3.0 * forward_per_row / ep_size,
        rounded_rows * 3.0 * forward_per_row / ep_size,
    )


def test_source_push_semantic_plan_bench_emits_full_fused_mlp_rows(tmp_path):
    jsonl = tmp_path / "semantic_fused_mlp.jsonl"
    bench_source_push_semantic_plan = _bench_module()
    modes = (
        "semantic_fused_mlp_forward_pallas,semantic_fused_mlp_forward_compare,"
        "semantic_fused_mlp_forward_backward_pallas,semantic_fused_mlp_forward_backward_compare"
    )

    bench_source_push_semantic_plan.main(
        [
            "--ep-size",
            "1",
            "--tokens-per-rank",
            "2",
            "--topk",
            "1",
            "--experts-per-rank",
            "1",
            "--hidden-dim",
            "256",
            "--intermediate-dim",
            "128",
            "--capacity-factor",
            "1.0",
            "--routing",
            "balanced",
            "--modes",
            modes,
            "--pallas-interpret",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--repeat-runs",
            "1",
            "--jsonl",
            str(jsonl),
        ]
    )

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    repeats = {row["mode"]: row for row in rows if row["row_type"] == "repeat"}
    summaries = {row["mode"]: row for row in rows if row["row_type"] == "summary"}
    assert set(summaries) == set(modes.split(","))
    for summary in summaries.values():
        assert summary["implementation"] == "pallas_mgpu"
        assert summary["error_rows"] == 0
        assert summary["median_steady_state_time"] > 0
        assert summary["median_useful_tflops_per_rank"] > 0
        assert summary["median_rounded_tflops_per_rank"] > 0

    forward = repeats["semantic_fused_mlp_forward_pallas"]
    assert forward["dropped_routes"] == 0.0
    forward_compare = repeats["semantic_fused_mlp_forward_compare"]
    assert forward_compare["y_max_abs_diff"] == 0.0
    assert forward_compare["dropped_routes_error_count"] == 0.0

    forward_backward = repeats["semantic_fused_mlp_forward_backward_pallas"]
    assert forward_backward["dropped_routes"] == 0.0
    forward_backward_compare = repeats["semantic_fused_mlp_forward_backward_compare"]
    for output in ("y", "dx", "d_route_weights", "dw13", "dw2"):
        assert forward_backward_compare[f"{output}_max_abs_diff"] == 0.0
        assert forward_backward_compare[f"expected_{output}_nonfinite_error_count"] == 0.0
        assert forward_backward_compare[f"observed_{output}_nonfinite_error_count"] == 0.0
    assert forward_backward_compare["dropped_routes_error_count"] == 0.0
