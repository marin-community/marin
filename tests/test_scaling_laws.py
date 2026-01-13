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

These tests focus on integration and behavioral validation, particularly
the snapshot test which ensures reproducibility of config generation.
"""

import jax.numpy as jnp

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from marin.scaling_laws.isoflop_analysis import (
    DEFAULT_SEQ_LEN,
    CandidateConfig,
    compute_training_flops,
    fit_scaling_laws,
    generate_training_configs,
    robust_quad_logx,
    solve_for_batch_size,
    solve_for_train_steps,
)

# Import the concrete recipe and transform function from experiments
from experiments.isoflop_sweep import Marin2025Recipe, parse_isoflop_run_name, transform_levanter_metrics

# --- FLOP computation tests ---


def test_flop_solvers_are_consistent():
    """Test that FLOP solvers correctly invert the FLOP calculation."""
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

    # Verify solve_for_batch_size inverts compute_training_flops
    original_batch = 64
    train_steps = 10000
    target_flops = compute_training_flops(model_config, vocab_size, original_batch, train_steps, seq_len)
    recovered_batch = solve_for_batch_size(model_config, vocab_size, target_flops, train_steps, seq_len)
    assert abs(recovered_batch - original_batch) < 0.01

    # Verify solve_for_train_steps inverts compute_training_flops
    original_steps = 50000
    batch_size = 32
    target_flops = compute_training_flops(model_config, vocab_size, batch_size, original_steps, seq_len)
    recovered_steps = solve_for_train_steps(model_config, vocab_size, target_flops, batch_size, seq_len)
    assert abs(recovered_steps - original_steps) < 0.01


# --- Run name parsing tests ---


def test_parse_isoflop_run_name():
    """Test parsing isoflop run names extracts experiment names."""
    # New format: isoflop-{budget}-N{params}-B{batch}-{experiment_name}
    assert parse_isoflop_run_name("isoflop-1e+18-N1e+08-B128-nemo-wider-depth-adapt") == "nemo-wider-depth-adapt"
    assert parse_isoflop_run_name("isoflop-1e+18-N1e+08-B128-dclm-a1b2c3") == "dclm"  # hash stripped

    # Legacy format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    assert parse_isoflop_run_name("isoflop-1e+18-d512-L8-B128-dclm-a1b2c3") == "dclm"
    assert parse_isoflop_run_name("isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt") == "nemo-wider-depth-adapt"

    # Invalid formats return None
    assert parse_isoflop_run_name("not-a-valid-name") is None
    assert parse_isoflop_run_name("") is None


# --- Candidate config tests ---


def test_candidate_configs_within_tolerance():
    """Test that generated configs achieve the target FLOP budget within tolerance."""
    recipe = Marin2025Recipe()
    budget = 1e19
    flop_tolerance = 0.01
    seq_len = DEFAULT_SEQ_LEN

    # Generate candidates using the new API
    for model_config in recipe.build_model_configs(budget, seq_len):
        flops_per_token = model_config.flops_per_token(recipe.vocab_size, seq_len)
        tokens = budget / (3 * flops_per_token)
        candidate = recipe.build_candidate_config(model_config, tokens, budget, seq_len)

        if candidate is None:
            continue

        achieved = compute_training_flops(
            candidate.model_config,
            recipe.vocab_size,
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

# Snapshot of expected output for generate_training_configs with budget=3e18 training FLOPs.
EXPECTED_ISOFLOP_CONFIGS_3E18 = [
    {"batch_size": 32, "train_steps": 32844, "flops_budget": 3e18},
    {"batch_size": 16, "train_steps": 46274, "flops_budget": 3e18},
    {"batch_size": 16, "train_steps": 33965, "flops_budget": 3e18},
    {"batch_size": 8, "train_steps": 48105, "flops_budget": 3e18},
    {"batch_size": 8, "train_steps": 37335, "flops_budget": 3e18},
]


def test_generate_training_configs_snapshot():
    """Snapshot test: verify generate_training_configs produces expected configs.

    This ensures reproducibility of the config generation algorithm.
    """
    recipe = Marin2025Recipe()
    result = generate_training_configs(
        budgets=(3e18,),
        recipe=recipe,
    )

    assert len(result) == len(EXPECTED_ISOFLOP_CONFIGS_3E18)

    for i, (candidate, expected) in enumerate(zip(result, EXPECTED_ISOFLOP_CONFIGS_3E18, strict=True)):
        assert isinstance(candidate, CandidateConfig)
        # batch_size and train_steps are now directly on the candidate
        assert candidate.batch_size == expected["batch_size"], f"Config {i}: batch_size mismatch"
        assert candidate.train_steps == expected["train_steps"], f"Config {i}: train_steps mismatch"
        assert candidate.flops_budget == expected["flops_budget"], f"Config {i}: flops_budget mismatch"


# --- End-to-end integration test ---

# Sample tracker_metrics.jsonl data simulating real runs
SAMPLE_METRICS_DATA = [
    # 1e18 budget - 3 runs with U-shaped loss curve
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d1024-L11-B8-nemo",
        "config": {"model": {"hidden_dim": 1024, "num_layers": 11}, "trainer": {"train_batch_size": 8}},
        "summary": {
            "throughput/total_tokens": 1e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.25,
            "parameter_count": 4e8,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d768-L8-B16-nemo",
        "config": {"model": {"hidden_dim": 768, "num_layers": 8}, "trainer": {"train_batch_size": 16}},
        "summary": {
            "throughput/total_tokens": 2.5e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.12,
            "parameter_count": 2.7e8,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d512-L6-B32-nemo",
        "config": {"model": {"hidden_dim": 512, "num_layers": 6}, "trainer": {"train_batch_size": 32}},
        "summary": {
            "throughput/total_tokens": 5e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.18,
            "parameter_count": 1.5e8,
        },
    },
    # 1e19 budget - 3 runs
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d2048-L21-B16-nemo",
        "config": {"model": {"hidden_dim": 2048, "num_layers": 21}, "trainer": {"train_batch_size": 16}},
        "summary": {
            "throughput/total_tokens": 3e9,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 1.05,
            "parameter_count": 1.8e9,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1536-L16-B32-nemo",
        "config": {"model": {"hidden_dim": 1536, "num_layers": 16}, "trainer": {"train_batch_size": 32}},
        "summary": {
            "throughput/total_tokens": 8e9,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 0.98,
            "parameter_count": 1e9,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1024-L11-B64-nemo",
        "config": {"model": {"hidden_dim": 1024, "num_layers": 11}, "trainer": {"train_batch_size": 64}},
        "summary": {
            "throughput/total_tokens": 2e10,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 1.02,
            "parameter_count": 4e8,
        },
    },
]


def test_end_to_end_analysis_pipeline():
    """Integration test: transform metrics and fit scaling laws end-to-end.

    Uses SAMPLE_METRICS_DATA (simulating real wandb metrics) to verify the full
    pipeline: metrics transformation -> curve fitting -> scaling law extraction.
    """
    from marin.scaling_laws import round_flops_to_bucket

    # Transform metrics using the Levanter transform function
    records = transform_levanter_metrics(SAMPLE_METRICS_DATA, "eval/paloma/c4_en/bpb")
    assert len(records) == 6

    # Fit scaling laws
    fit_result = fit_scaling_laws(records)

    # Should find two minima (one per budget: ~1e18 and ~1e19)
    # FLOP values are bucketed by round_flops_to_bucket
    assert len(fit_result.minima_records) == 2

    # Get expected bucketed values
    bucket_1e18 = round_flops_to_bucket(1e18)
    bucket_1e19 = round_flops_to_bucket(1e19)
    assert {rec.flops for rec in fit_result.minima_records} == {bucket_1e18, bucket_1e19}

    # Verify fitted minima are near expected optimal points
    minima_by_flops = {rec.flops: rec for rec in fit_result.minima_records}

    # At ~1e18: raw data optimal at 2.5B tokens (loss=1.12)
    assert abs(minima_by_flops[bucket_1e18].optimal_tokens - 2.6e9) < 0.2e9
    assert abs(minima_by_flops[bucket_1e18].loss_at_optimal - 1.12) < 0.01

    # At ~1e19: raw data optimal at 8B tokens (loss=0.98)
    assert abs(minima_by_flops[bucket_1e19].optimal_tokens - 8.8e9) < 0.2e9
    assert abs(minima_by_flops[bucket_1e19].loss_at_optimal - 0.98) < 0.01
