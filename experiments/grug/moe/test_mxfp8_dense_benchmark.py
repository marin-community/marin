# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest

from experiments.grug.moe.standalone import bench_mxfp8_dense as benchmark
from experiments.grug.moe.standalone import bench_te_dense as te_benchmark


def test_sample_statistics_reports_median_and_mad():
    median, mad = benchmark.sample_statistics([1.0, 2.0, 3.0, 100.0])

    assert median == 2.5
    assert mad == 1.0


def test_weighted_production_ratio_uses_dense_operation_mix():
    ratio = benchmark.weighted_production_ratio(
        [
            (5, 1.01, 1.0),
            (2, 0.95, 1.0),
        ]
    )

    assert ratio == pytest.approx(6.95 / 7.0)


def test_git_sha_is_required_benchmark_provenance():
    with pytest.raises(SystemExit):
        benchmark.parse_args([])

    args = benchmark.parse_args(["--git-sha", "abc123"])

    assert args.git_sha == "abc123"


def test_custom_call_count_counts_compiled_call_sites():
    hlo = "\n".join(
        [
            'custom-call(), custom_call_target="__cudnn$blockScaledDot"',
            'custom-call(), custom_call_target="unrelated"',
            'custom-call(), custom_call_target="__cudnn$blockScaledDot"',
        ]
    )

    assert benchmark.custom_call_count(hlo, "__cudnn$blockScaledDot") == 2


def test_linear_orientations_match_scaled_matmul_contract():
    q_row = jnp.arange(128, dtype=jnp.uint8).reshape(2, 64)
    s_row_t = jnp.arange(4, dtype=jnp.uint8).reshape(2, 2)
    q_col = q_row + 1
    s_col = jnp.arange(4, dtype=jnp.uint8).reshape(1, 4)

    row_q, row_scale, col_q, col_scale = benchmark.linear_orientations(q_row, s_row_t, q_col, s_col)

    assert row_q.shape == (1, 2, 64)
    assert row_scale.shape == (1, 2, 2)
    assert col_q.shape == (1, 64, 2)
    assert col_scale.shape == (1, 4, 1)
    assert jnp.array_equal(row_q[0], q_row)
    assert jnp.array_equal(row_scale[0].view(jnp.uint8), s_row_t.T)
    assert jnp.array_equal(col_q[0], q_col.T)
    assert jnp.array_equal(col_scale[0].view(jnp.uint8), s_col.T)


def test_cute_producer_is_explicit():
    args = benchmark.parse_args(["--git-sha", "abc123", "--producer", "cute"])

    assert args.producer == "cute"


def test_byte_mismatch_count_compares_fp8_bits():
    lhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float8_e4m3fn)
    rhs = jnp.array([1.0, 2.5, 3.0], dtype=jnp.float8_e4m3fn)

    assert benchmark.byte_mismatch_count(lhs, rhs) == 1


def test_projection_reuse_benchmark_is_explicit():
    args = benchmark.parse_args(["--git-sha", "abc123", "--producer", "cute", "--projection-reuse"])

    assert args.projection_reuse


def test_te_dense_benchmark_requires_provenance():
    with pytest.raises(SystemExit):
        te_benchmark.parse_args([])

    args = te_benchmark.parse_args(["--git-sha", "abc123", "--shape", "kv_5120x1280"])

    assert args.git_sha == "abc123"
    assert args.shape == ["kv_5120x1280"]


def test_te_dense_projection_fusion_is_explicit():
    args = te_benchmark.parse_args(["--git-sha", "abc123", "--projection-fusion"])

    assert args.projection_fusion
