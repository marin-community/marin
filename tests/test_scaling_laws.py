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

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from marin.scaling_laws import ScalingRecipe
from marin.scaling_laws.isoflop_analysis import (
    DEFAULT_SEQ_LEN,
    MARIN_TOKENIZER_VOCAB_SIZE,
    IsoFlopTrainArgs,
    candidate_configs,
    compute_training_flops,
    fit_scaling_laws,
    generate_isoflop_train_args,
    parse_isoflop_run_name,
    robust_quad_logx,
    round_flops_to_bucket,
    round_to_power_of_two,
    solve_for_batch_size,
    solve_for_train_steps,
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


def test_compute_training_flops_linear_in_batch_and_steps():
    """Test that FLOPs scale linearly with batch size and steps."""
    # Build a model config for testing
    model_config = Qwen3Config(
        max_seq_len=4096,
        hidden_dim=512,
        intermediate_dim=2048,
        num_heads=8,
        num_kv_heads=8,
        num_layers=12,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    vocab_size = 128256
    seq_len = 4096

    base_flops = compute_training_flops(model_config, vocab_size, 32, 1000, seq_len)
    double_batch_flops = compute_training_flops(model_config, vocab_size, 64, 1000, seq_len)
    double_steps_flops = compute_training_flops(model_config, vocab_size, 32, 2000, seq_len)

    assert abs(double_batch_flops - 2 * base_flops) / base_flops < 0.01
    assert abs(double_steps_flops - 2 * base_flops) / base_flops < 0.01


def test_solve_for_batch_size_inverts_flop_calculation():
    """Test that solve_for_batch_size correctly inverts compute_training_flops."""
    model_config = Qwen3Config(
        max_seq_len=4096,
        hidden_dim=768,
        intermediate_dim=3072,
        num_heads=12,
        num_kv_heads=12,
        num_layers=12,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    vocab_size = 128256
    seq_len = 4096
    train_steps = 10000
    original_batch_size = 64

    # Compute FLOPs for known batch size
    target_flops = compute_training_flops(model_config, vocab_size, original_batch_size, train_steps, seq_len)

    # Solve for batch size given those FLOPs
    recovered_batch = solve_for_batch_size(model_config, vocab_size, target_flops, train_steps, seq_len)

    # Should recover original batch size (exact float)
    assert abs(recovered_batch - original_batch_size) < 0.01


def test_solve_for_train_steps_inverts_flop_calculation():
    """Test that solve_for_train_steps correctly inverts compute_training_flops."""
    model_config = Qwen3Config(
        max_seq_len=4096,
        hidden_dim=1024,
        intermediate_dim=4096,
        num_heads=8,
        num_kv_heads=8,
        num_layers=16,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    vocab_size = 128256
    seq_len = 4096
    batch_size = 32
    original_steps = 50000

    # Compute FLOPs for known steps
    target_flops = compute_training_flops(model_config, vocab_size, batch_size, original_steps, seq_len)

    # Solve for steps given those FLOPs
    recovered_steps = solve_for_train_steps(model_config, vocab_size, target_flops, batch_size, seq_len)

    # Should recover original steps (exact float)
    assert abs(recovered_steps - original_steps) < 0.01


def test_solvers_consistent_with_each_other():
    """Test that solving for batch and then steps gives consistent results."""
    model_config = Qwen3Config(
        max_seq_len=4096,
        hidden_dim=512,
        intermediate_dim=2048,
        num_heads=8,
        num_kv_heads=8,
        num_layers=8,
        rope=Llama3RotaryEmbeddingsConfig(),
    )
    vocab_size = 128256
    seq_len = 4096
    target_flops = 1e19

    # Pick arbitrary steps, solve for batch
    steps = 20000
    batch = solve_for_batch_size(model_config, vocab_size, target_flops, steps, seq_len)

    # Now with that batch, solve for steps - should get back original
    recovered_steps = solve_for_train_steps(model_config, vocab_size, target_flops, round(batch), seq_len)

    # Allow small error from rounding batch to int
    relative_error = abs(recovered_steps - steps) / steps
    assert relative_error < 0.01


# --- Run name parsing tests ---


def test_parse_isoflop_run_name():
    """Test parsing isoflop run names extracts experiment names."""
    result = parse_isoflop_run_name("isoflop-1e+18-d512-L8-B128-dclm-a1b2c3")
    assert result == "dclm"

    # Invalid formats return None
    assert parse_isoflop_run_name("not-a-valid-name") is None
    assert parse_isoflop_run_name("") is None


# --- Candidate config tests ---


def test_candidate_configs_within_tolerance():
    """Test that generated configs achieve the target FLOP budget within tolerance."""
    recipe = ScalingRecipe(name="test")
    budget = 1e19
    flop_tolerance = 0.01
    seq_len = DEFAULT_SEQ_LEN
    for candidate in candidate_configs(budget, MARIN_TOKENIZER_VOCAB_SIZE, recipe, flop_tolerance=flop_tolerance):
        # Build model config from candidate using recipe
        model_config = recipe.build_model_config(candidate.target_params, MARIN_TOKENIZER_VOCAB_SIZE, seq_len)
        achieved = compute_training_flops(
            model_config,
            MARIN_TOKENIZER_VOCAB_SIZE,
            candidate.batch_size,
            candidate.train_steps,
            seq_len,
        )
        relative_error = abs(achieved - budget) / budget
        assert relative_error <= flop_tolerance


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

# Snapshot of expected output for generate_isoflop_train_args with budget=3e18 training FLOPs.
# Note: compute_training_flops includes the 3x multiplier for training (forward + backward pass),
# matching how FLOPs are tracked in WandB via Levanter's log_performance_stats.
#
# CandidateConfig is now model-agnostic, containing only:
# - batch_size, train_steps, tokens, target_params, flops_budget
EXPECTED_ISOFLOP_CONFIGS_3E18 = [
    {
        "batch_size": 32,
        "train_steps": 32844,
        "flops_budget": 3e18,
    },
    {
        "batch_size": 16,
        "train_steps": 46274,
        "flops_budget": 3e18,
    },
    {
        "batch_size": 16,
        "train_steps": 33965,
        "flops_budget": 3e18,
    },
    {
        "batch_size": 8,
        "train_steps": 48105,
        "flops_budget": 3e18,
    },
    {
        "batch_size": 8,
        "train_steps": 37335,
        "flops_budget": 3e18,
    },
]


def test_generate_isoflop_train_args_snapshot():
    """Snapshot test: verify generate_isoflop_train_args produces expected configs.

    This test ensures the scaling_laws module produces identical configurations
    for reproducible isoflop sweeps. Uses 3e18 training FLOPs budget (which accounts
    for the 3x multiplier for forward + backward pass).

    CandidateConfig is now model-agnostic, so we only check the core compute
    allocation parameters (batch_size, train_steps, flops_budget).
    """
    recipe = ScalingRecipe(name="test-snapshot")
    budgets = (3e18,)
    result = generate_isoflop_train_args(
        budgets=budgets,
        experiment_name="test-snapshot",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
        recipe=recipe,
    )

    assert len(result) == len(
        EXPECTED_ISOFLOP_CONFIGS_3E18
    ), f"Expected {len(EXPECTED_ISOFLOP_CONFIGS_3E18)} configs, got {len(result)}"

    for i, (args, expected) in enumerate(zip(result, EXPECTED_ISOFLOP_CONFIGS_3E18, strict=True)):
        assert isinstance(args, IsoFlopTrainArgs)
        c = args.candidate
        actual = {
            "batch_size": c.batch_size,
            "train_steps": c.train_steps,
            "flops_budget": c.flops_budget,
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
    fit_result = fit_scaling_laws(isoflop_df)

    # Should find two minima (one per budget: 1e18 and 1e19)
    assert len(fit_result.minima_records) == 2
    flops_budgets = {rec.flops for rec in fit_result.minima_records}
    assert flops_budgets == {1e18, 1e19}

    # Verify fitted minima are near expected optimal points
    # Curve fitting interpolates to find analytical minimum of fitted quadratic
    minima_by_flops = {rec.flops: rec for rec in fit_result.minima_records}

    # At 1e18: raw data optimal at 2.5B (loss=1.12), fitted minimum ~2.6B
    assert abs(minima_by_flops[1e18].optimal_tokens - 2.6e9) < 0.2e9
    assert abs(minima_by_flops[1e18].loss_at_optimal - 1.12) < 0.01

    # At 1e19: raw data optimal at 8B (loss=0.98), fitted minimum ~8.8B
    assert abs(minima_by_flops[1e19].optimal_tokens - 8.8e9) < 0.2e9
    assert abs(minima_by_flops[1e19].loss_at_optimal - 0.98) < 0.01
