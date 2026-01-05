# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the scaling_laws module.

These tests focus on integration and end-to-end validation with specific expected outputs,
particularly the snapshot test which ensures reproducibility of config generation.
"""

import jax.numpy as jnp
import pandas as pd
import pytest

from marin.scaling_laws.isoflop_analysis import (
    MARIN_TOKENIZER_VOCAB_SIZE,
    IsoFlopSweepConfig,
    IsoFlopTrainArgs,
    candidate_configs,
    compute_total_flops,
    fit_scaling_laws,
    generate_isoflop_train_args,
    parse_isoflop_run_name,
    robust_quad_logx,
    round_flops_to_bucket,
    round_to_power_of_two,
    transform_metrics_for_isoflop,
)

# --- Utility function tests (parametrized) ---


@pytest.mark.parametrize(
    "value,expected",
    [
        # Exact powers unchanged
        (1, 1),
        (2, 2),
        (4, 4),
        (16, 16),
        # Non-powers round up
        (3, 4),
        (5, 8),
        (9, 16),
        # Small/zero values become 1
        (0.5, 1),
        (0, 1),
        # Large values
        (100, 128),
        (1000, 1024),
    ],
)
def test_round_to_power_of_two(value, expected):
    """Test round_to_power_of_two produces correct results."""
    assert round_to_power_of_two(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        # Exact values unchanged
        (1e18, 1e18),
        (1e19, 1e19),
        (3e19, 3e19),
        # Rounds to 1 significant figure
        (1.05e19, 1e19),
        (1.4e19, 1e19),
        (1.5e19, 2e19),
        (2.8e19, 3e19),
        (9.5e19, 1e20),
        # Edge cases
        (0, 0),
    ],
)
def test_round_flops_to_bucket(value, expected):
    """Test round_flops_to_bucket rounds to 1 significant figure."""
    assert round_flops_to_bucket(value) == expected


# --- FLOP computation tests ---


def test_compute_total_flops_linear_in_batch_and_steps():
    """Test that FLOPs scale linearly with batch size and steps."""
    base_flops = compute_total_flops(
        batch=32,
        num_layers=12,
        hidden=512,
        intermediate=2048,
        num_kv_heads=8,
        num_heads=8,
        steps=1000,
        seq_len=4096,
        vocab_size=128256,
    )
    double_batch_flops = compute_total_flops(
        batch=64,
        num_layers=12,
        hidden=512,
        intermediate=2048,
        num_kv_heads=8,
        num_heads=8,
        steps=1000,
        seq_len=4096,
        vocab_size=128256,
    )
    double_steps_flops = compute_total_flops(
        batch=32,
        num_layers=12,
        hidden=512,
        intermediate=2048,
        num_kv_heads=8,
        num_heads=8,
        steps=2000,
        seq_len=4096,
        vocab_size=128256,
    )
    assert abs(double_batch_flops - 2 * base_flops) / base_flops < 0.01
    assert abs(double_steps_flops - 2 * base_flops) / base_flops < 0.01


# --- Run name parsing tests ---


def test_parse_isoflop_run_name():
    """Test parsing isoflop run names extracts correct values."""
    # Standard name
    result = parse_isoflop_run_name("isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt")
    assert result is not None
    assert result["flops"] == 1e19
    assert result["d"] == 2048
    assert result["L"] == 16
    assert result["B"] == 1024
    assert result["experiment_name"] == "nemo-wider-depth-adapt"

    # With hash suffix
    result = parse_isoflop_run_name("isoflop-1e+18-d512-L8-B128-dclm-a1b2c3")
    assert result is not None
    assert result["flops"] == 1e18
    assert result["experiment_name"] == "dclm"

    # Invalid formats return None
    assert parse_isoflop_run_name("not-a-valid-name") is None
    assert parse_isoflop_run_name("") is None


# --- Candidate config tests ---


def test_candidate_configs_within_tolerance():
    """Test that generated configs achieve the target FLOP budget within tolerance."""
    cfg = IsoFlopSweepConfig(flop_tolerance=0.01)
    budget = 1e19
    for candidate in candidate_configs(cfg, budget, MARIN_TOKENIZER_VOCAB_SIZE):
        achieved = compute_total_flops(
            candidate.batch_size,
            candidate.num_layers,
            candidate.hidden_size,
            candidate.intermediate_dim,
            candidate.num_kv_heads,
            candidate.num_heads,
            candidate.train_steps,
            cfg.seq_len,
            MARIN_TOKENIZER_VOCAB_SIZE,
        )
        relative_error = abs(achieved - budget) / budget
        assert relative_error <= cfg.flop_tolerance


# --- Curve fitting tests ---


def test_robust_quad_logx_fits_quadratic():
    """Test that robust_quad_logx recovers known coefficients from synthetic data."""
    x = jnp.array([1e9, 1e10, 1e11, 1e12])
    L = jnp.log10(x)
    # y = 0.1 * L^2 - 2 * L + 20
    y = 0.1 * L**2 - 2 * L + 20

    a, b, c = robust_quad_logx(x, y)

    assert abs(a - 0.1) < 0.01
    assert abs(b - (-2)) < 0.1
    assert abs(c - 20) < 0.5


# --- Snapshot test for config generation ---

# Snapshot of expected output for generate_isoflop_train_args with budget=1e18.
# This captures the configuration generation logic to ensure reproducibility.
EXPECTED_ISOFLOP_CONFIGS_1E18 = [
    {
        "hidden_size": 512,
        "intermediate_dim": 2048,
        "num_layers": 6,
        "num_heads": 4,
        "num_kv_heads": 4,
        "batch_size": 32,
        "train_steps": 32844,
        "learning_rate": 0.003646,
        "beta2": 0.994962,
        "tpu_type": "v5p-8",
        "run_name": "isoflop-1e+18-d512-L6-B32-test-snapshot",
    },
    {
        "hidden_size": 640,
        "intermediate_dim": 2560,
        "num_layers": 7,
        "num_heads": 5,
        "num_kv_heads": 5,
        "batch_size": 16,
        "train_steps": 46274,
        "learning_rate": 0.002063,
        "beta2": 0.997478,
        "tpu_type": "v5p-8",
        "run_name": "isoflop-1e+18-d640-L7-B16-test-snapshot",
    },
    {
        "hidden_size": 768,
        "intermediate_dim": 3072,
        "num_layers": 8,
        "num_heads": 6,
        "num_kv_heads": 6,
        "batch_size": 16,
        "train_steps": 33965,
        "learning_rate": 0.001719,
        "beta2": 0.997478,
        "tpu_type": "v5p-8",
        "run_name": "isoflop-1e+18-d768-L8-B16-test-snapshot",
    },
    {
        "hidden_size": 896,
        "intermediate_dim": 3584,
        "num_layers": 10,
        "num_heads": 7,
        "num_kv_heads": 7,
        "batch_size": 8,
        "train_steps": 48105,
        "learning_rate": 0.001042,
        "beta2": 0.998738,
        "tpu_type": "v5p-8",
        "run_name": "isoflop-1e+18-d896-L10-B8-test-snapshot",
    },
    {
        "hidden_size": 1024,
        "intermediate_dim": 4096,
        "num_layers": 11,
        "num_heads": 8,
        "num_kv_heads": 8,
        "batch_size": 8,
        "train_steps": 37335,
        "learning_rate": 0.000912,
        "beta2": 0.998738,
        "tpu_type": "v5p-8",
        "run_name": "isoflop-1e+18-d1024-L11-B8-test-snapshot",
    },
]


def test_generate_isoflop_train_args_snapshot():
    """Snapshot test: verify generate_isoflop_train_args produces expected configs.

    This test ensures the scaling_laws module produces identical configurations
    for reproducible isoflop sweeps.
    """
    config = IsoFlopSweepConfig(budgets=(1e18,))
    result = generate_isoflop_train_args(
        sweep_config=config,
        experiment_name="test-snapshot",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )

    assert len(result) == len(
        EXPECTED_ISOFLOP_CONFIGS_1E18
    ), f"Expected {len(EXPECTED_ISOFLOP_CONFIGS_1E18)} configs, got {len(result)}"

    for i, (args, expected) in enumerate(zip(result, EXPECTED_ISOFLOP_CONFIGS_1E18, strict=True)):
        assert isinstance(args, IsoFlopTrainArgs)
        c = args.candidate
        actual = {
            "hidden_size": c.hidden_size,
            "intermediate_dim": c.intermediate_dim,
            "num_layers": c.num_layers,
            "num_heads": c.num_heads,
            "num_kv_heads": c.num_kv_heads,
            "batch_size": c.batch_size,
            "train_steps": c.train_steps,
            "learning_rate": round(c.learning_rate, 6),
            "beta2": round(c.beta2, 6),
            "tpu_type": args.tpu_type,
            "run_name": args.run_name,
        }

        for key in expected:
            assert (
                actual[key] == expected[key]
            ), f"Config {i}: {key} mismatch: expected {expected[key]}, got {actual[key]}"


# --- Metrics transformation tests ---

# Sample tracker_metrics.jsonl data extracted from real runs
SAMPLE_METRICS_DATA = [
    # 1e18 budget - 3 runs with U-shaped loss curve
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d1024-L11-B8-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 1024, "num_layers": 11},
            "trainer": {"train_batch_size": 8},
        },
        "summary": {
            "throughput/total_tokens": 1000000000,
            "throughput/total_gflops": 1000000000.0,
            "eval/paloma/c4_en/bpb": 1.25,
            "parameter_count": 400000000,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d768-L8-B16-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 768, "num_layers": 8},
            "trainer": {"train_batch_size": 16},
        },
        "summary": {
            "throughput/total_tokens": 2500000000,
            "throughput/total_gflops": 1000000000.0,
            "eval/paloma/c4_en/bpb": 1.12,
            "parameter_count": 272513792,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d512-L6-B32-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 512, "num_layers": 6},
            "trainer": {"train_batch_size": 32},
        },
        "summary": {
            "throughput/total_tokens": 5000000000,
            "throughput/total_gflops": 1000000000.0,
            "eval/paloma/c4_en/bpb": 1.18,
            "parameter_count": 156508160,
        },
    },
    # 1e19 budget - 3 runs
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d2048-L21-B16-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 2048, "num_layers": 21},
            "trainer": {"train_batch_size": 16},
        },
        "summary": {
            "throughput/total_tokens": 3000000000,
            "throughput/total_gflops": 10000000000.0,
            "eval/paloma/c4_en/bpb": 1.05,
            "parameter_count": 1800000000,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1536-L16-B32-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 1536, "num_layers": 16},
            "trainer": {"train_batch_size": 32},
        },
        "summary": {
            "throughput/total_tokens": 8000000000,
            "throughput/total_gflops": 10000000000.0,
            "eval/paloma/c4_en/bpb": 0.98,
            "parameter_count": 998036992,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1024-L11-B64-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 1024, "num_layers": 11},
            "trainer": {"train_batch_size": 64},
        },
        "summary": {
            "throughput/total_tokens": 20000000000,
            "throughput/total_gflops": 10000000000.0,
            "eval/paloma/c4_en/bpb": 1.02,
            "parameter_count": 400000000,
        },
    },
]


def test_transform_metrics_for_isoflop():
    """Test transformation of raw metrics data to isoflop analysis format."""
    raw_df = pd.DataFrame(SAMPLE_METRICS_DATA)
    metric_key = "eval/paloma/c4_en/bpb"

    result = transform_metrics_for_isoflop(raw_df, metric_key)

    assert len(result) == 6  # 3 runs at 1e18 + 3 runs at 1e19

    # Verify specific values from first row (d1024/L11)
    row0 = result.iloc[0]
    assert row0["tokens"] == 1000000000
    assert row0["loss"] == 1.25
    assert row0["hidden_dim"] == 1024
    assert row0["num_layers"] == 11
    assert row0["batch_size"] == 8
    assert row0["flops"] == 1e18
    assert row0["params"] == 400000000


def test_transform_metrics_filters_low_flops():
    """Test that runs with < 1e18 FLOPs are filtered out."""
    raw_df = pd.DataFrame(
        [
            {
                "run_path": "gs://marin/checkpoints/small-run",
                "config": {
                    "model": {"hidden_dim": 256, "num_layers": 4},
                    "trainer": {"train_batch_size": 8},
                },
                "summary": {
                    "throughput/total_tokens": 1e7,
                    "throughput/total_gflops": 1e6,  # Only 1e15 FLOPs
                    "eval/paloma/c4_en/bpb": 3.0,
                    "parameter_count": 1e7,
                },
            }
        ]
    )

    result = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    assert len(result) == 0


# --- End-to-end integration test ---


def test_end_to_end_analysis_pipeline():
    """Integration test: transform metrics and fit scaling laws end-to-end.

    Uses SAMPLE_METRICS_DATA (simulating real wandb metrics) to verify the full
    pipeline: metrics transformation -> curve fitting -> scaling law extraction.
    """
    raw_df = pd.DataFrame(SAMPLE_METRICS_DATA)

    # Transform metrics
    isoflop_df = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    assert len(isoflop_df) == 6

    # Fit scaling laws
    minima_records, _scaling_fits, _ = fit_scaling_laws(isoflop_df)

    # Should find two minima (one per budget: 1e18 and 1e19)
    assert len(minima_records) == 2
    flops_budgets = {rec.flops for rec in minima_records}
    assert flops_budgets == {1e18, 1e19}

    # Verify fitted minima are near expected optimal points
    # Curve fitting interpolates to find analytical minimum of fitted quadratic
    minima_by_flops = {rec.flops: rec for rec in minima_records}

    # At 1e18: raw data optimal at 2.5B (loss=1.12), fitted minimum ~2.6B
    assert abs(minima_by_flops[1e18].optimal_tokens - 2.6e9) < 0.2e9
    assert abs(minima_by_flops[1e18].loss_at_optimal - 1.12) < 0.01

    # At 1e19: raw data optimal at 8B (loss=0.98), fitted minimum ~8.8B
    assert abs(minima_by_flops[1e19].optimal_tokens - 8.8e9) < 0.2e9
    assert abs(minima_by_flops[1e19].loss_at_optimal - 0.98) < 0.01
