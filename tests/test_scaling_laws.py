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

"""Unit tests for the scaling_laws module."""

import jax.numpy as jnp
import pandas as pd

from marin.scaling_laws.isoflop_analysis import (
    BETA2_BASE,
    BETA2_BATCH_DIVISOR,
    DEFAULT_BUDGETS,
    HIDDEN_HEAD_RATIO,
    LR_CONSTANT,
    MARIN_TOKENIZER_VOCAB_SIZE,
    MLP_RATIO,
    SEQ_LEN,
    CandidateConfig,
    IsoFlopSweepConfig,
    IsoFlopTrainArgs,
    MinimaRecord,
    candidate_configs,
    compute_total_flops,
    compute_transformer_params,
    fit_scaling_laws,
    generate_isoflop_train_args,
    parse_isoflop_run_name,
    predict_optimal_config,
    robust_quad_logx,
    round_flops_to_bucket,
    round_to_power_of_two,
    transform_metrics_for_isoflop,
)

# --- round_to_power_of_two tests ---


def test_round_to_power_of_two_exact_powers():
    """Test that exact powers of two are unchanged."""
    assert round_to_power_of_two(1) == 1
    assert round_to_power_of_two(2) == 2
    assert round_to_power_of_two(4) == 4
    assert round_to_power_of_two(8) == 8
    assert round_to_power_of_two(16) == 16


def test_round_to_power_of_two_rounds_up():
    """Test that non-powers round up to nearest power of two."""
    assert round_to_power_of_two(3) == 4
    assert round_to_power_of_two(5) == 8
    assert round_to_power_of_two(7) == 8
    assert round_to_power_of_two(9) == 16


def test_round_to_power_of_two_small_values():
    """Test that small/zero values become 1."""
    assert round_to_power_of_two(0.5) == 1
    assert round_to_power_of_two(0.1) == 1
    assert round_to_power_of_two(0) == 1


def test_round_to_power_of_two_large_values():
    """Test rounding for large values."""
    assert round_to_power_of_two(100) == 128
    assert round_to_power_of_two(1000) == 1024
    assert round_to_power_of_two(1025) == 2048


# --- compute_total_flops tests ---


def test_compute_total_flops_larger_model_uses_more_flops():
    """Test that larger models use more FLOPs."""
    small_flops = compute_total_flops(
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
    large_flops = compute_total_flops(
        batch=32,
        num_layers=24,
        hidden=1024,
        intermediate=4096,
        num_kv_heads=16,
        num_heads=16,
        steps=1000,
        seq_len=4096,
        vocab_size=128256,
    )
    assert large_flops > small_flops


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


# --- parse_isoflop_run_name tests ---


def test_parse_isoflop_run_name_basic():
    """Test parsing a standard isoflop run name."""
    result = parse_isoflop_run_name("isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt")
    assert result is not None
    assert result["flops"] == 1e19
    assert result["d"] == 2048
    assert result["L"] == 16
    assert result["B"] == 1024
    assert result["experiment_name"] == "nemo-wider-depth-adapt"


def test_parse_isoflop_run_name_with_hash_suffix():
    """Test parsing run name with hash suffix."""
    result = parse_isoflop_run_name("isoflop-1e+18-d512-L8-B128-dclm-a1b2c3")
    assert result is not None
    assert result["flops"] == 1e18
    assert result["d"] == 512
    assert result["L"] == 8
    assert result["B"] == 128
    assert result["experiment_name"] == "dclm"


def test_parse_isoflop_run_name_invalid_format():
    """Test that invalid formats return None."""
    assert parse_isoflop_run_name("not-a-valid-name") is None
    assert parse_isoflop_run_name("isoflop-missing-parts") is None
    assert parse_isoflop_run_name("") is None


# --- candidate_configs tests ---


def test_candidate_configs_generates_candidates():
    """Test that candidate_configs generates at least one config."""
    cfg = IsoFlopSweepConfig()
    candidates = list(candidate_configs(cfg, 1e19, MARIN_TOKENIZER_VOCAB_SIZE))
    assert len(candidates) > 0


def test_candidate_configs_within_tolerance():
    """Test that generated configs are within FLOP tolerance."""
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


def test_candidate_configs_fields_populated():
    """Test that all candidate fields are properly populated."""
    cfg = IsoFlopSweepConfig()
    candidates = list(candidate_configs(cfg, 1e19, MARIN_TOKENIZER_VOCAB_SIZE))
    assert len(candidates) > 0

    for candidate in candidates:
        assert candidate.hidden_size > 0
        assert candidate.intermediate_dim == candidate.hidden_size * MLP_RATIO
        assert candidate.num_layers > 0
        assert candidate.num_heads > 0
        assert candidate.num_kv_heads > 0
        assert candidate.batch_size >= 8
        assert candidate.train_steps > 0
        assert candidate.learning_rate > 0
        assert 0 < candidate.beta2 < 1
        assert candidate.tokens > 0
        assert candidate.flops_budget == 1e19


# --- robust_quad_logx tests ---


def test_robust_quad_logx_fits_quadratic():
    """Test that robust_quad_logx recovers known coefficients."""
    x = jnp.array([1e9, 1e10, 1e11, 1e12])
    L = jnp.log10(x)
    y = 0.1 * L**2 - 2 * L + 20

    a, b, c = robust_quad_logx(x, y)

    assert abs(a - 0.1) < 0.01
    assert abs(b - (-2)) < 0.1
    assert abs(c - 20) < 0.5


def test_robust_quad_logx_handles_noise():
    """Test that robust_quad_logx handles noisy data."""
    x = jnp.array([1e9, 1e10, 1e11, 1e12, 1e13])
    L = jnp.log10(x)
    y_clean = 0.05 * L**2 - 1.5 * L + 15
    noise = jnp.array([0.01, -0.02, 0.015, -0.01, 0.005])
    y = y_clean + noise

    a, b, _ = robust_quad_logx(x, y)

    assert abs(a - 0.05) < 0.05
    assert abs(b - (-1.5)) < 0.5


# --- predict_optimal_config tests ---


def test_predict_optimal_config_unknown_label_returns_none():
    """Test that unknown labels return None."""
    scaling_fits = {"nemo": (0.5, 1e5)}
    result = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=1e21,
        label="unknown",
    )
    assert result is None


def test_predict_optimal_config_valid_label():
    """Test prediction with a valid label."""
    scaling_fits = {"nemo": (0.5, 1e5)}
    result = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=1e20,
        label="nemo",
    )
    assert result is None or isinstance(result, CandidateConfig)


# --- Constants tests ---


def test_constants_default_budgets():
    """Test that DEFAULT_BUDGETS is valid."""
    assert len(DEFAULT_BUDGETS) > 0
    assert all(b > 0 for b in DEFAULT_BUDGETS)
    assert list(DEFAULT_BUDGETS) == sorted(DEFAULT_BUDGETS)


def test_constants_have_expected_values():
    """Test that constants have expected values."""
    assert SEQ_LEN == 4096
    assert MARIN_TOKENIZER_VOCAB_SIZE == 128256
    assert LR_CONSTANT == 0.33
    assert HIDDEN_HEAD_RATIO == 128
    assert BETA2_BASE == 0.98
    assert BETA2_BATCH_DIVISOR == 128
    assert MLP_RATIO == 4


# --- compute_transformer_params tests ---


def test_compute_transformer_params_returns_positive_int():
    """Test that compute_transformer_params returns a positive integer."""
    params = compute_transformer_params(
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=12,
        vocab_size=128256,
        num_kv_heads=8,
        num_heads=8,
    )
    assert params > 0
    assert isinstance(params, int)


def test_compute_transformer_params_scales_with_hidden_dim():
    """Test that params scale with hidden dimension."""
    small = compute_transformer_params(
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=12,
        vocab_size=128256,
    )
    large = compute_transformer_params(
        hidden_dim=1024,
        intermediate_dim=4096,
        num_layers=12,
        vocab_size=128256,
    )
    assert large > small


def test_compute_transformer_params_scales_with_layers():
    """Test that params scale with number of layers."""
    shallow = compute_transformer_params(
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=6,
        vocab_size=128256,
    )
    deep = compute_transformer_params(
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=12,
        vocab_size=128256,
    )
    assert deep > shallow


# --- generate_isoflop_train_args tests ---


def test_generate_isoflop_train_args_returns_list():
    """Test that generate_isoflop_train_args returns a list of IsoFlopTrainArgs."""
    config = IsoFlopSweepConfig(budgets=(1e18,))
    result = generate_isoflop_train_args(
        sweep_config=config,
        experiment_name="test-experiment",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(arg, IsoFlopTrainArgs) for arg in result)


def test_generate_isoflop_train_args_populates_fields():
    """Test that all required fields are populated."""
    config = IsoFlopSweepConfig(budgets=(1e18,))
    result = generate_isoflop_train_args(
        sweep_config=config,
        experiment_name="test-experiment",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    for args in result:
        assert args.candidate is not None
        assert args.candidate.hidden_size > 0
        assert args.candidate.num_layers > 0

        assert args.model_config is not None
        assert args.model_config.hidden_dim == args.candidate.hidden_size
        assert args.model_config.num_layers == args.candidate.num_layers

        assert args.optimizer_config is not None
        assert args.optimizer_config.learning_rate == args.candidate.learning_rate

        assert args.tpu_type.startswith("v5p-")
        assert "isoflop" in args.run_name
        assert "test-experiment" in args.run_name
        assert len(args.tags) > 0
        assert args.output_path.startswith("checkpoints/isoflop/")


def test_generate_isoflop_train_args_more_budgets_more_configs():
    """Test that more budgets produce more configs."""
    config_single = IsoFlopSweepConfig(budgets=(1e18,))
    config_multi = IsoFlopSweepConfig(budgets=(1e18, 1e19))

    result_single = generate_isoflop_train_args(
        sweep_config=config_single,
        experiment_name="test",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    result_multi = generate_isoflop_train_args(
        sweep_config=config_multi,
        experiment_name="test",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    assert len(result_multi) > len(result_single)


def test_generate_isoflop_train_args_unique_run_names():
    """Test that all run names are unique."""
    config = IsoFlopSweepConfig(budgets=(1e18, 1e19))
    result = generate_isoflop_train_args(
        sweep_config=config,
        experiment_name="test",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    run_names = [args.run_name for args in result]
    assert len(run_names) == len(set(run_names))


def test_generate_isoflop_train_args_includes_experiment_name():
    """Test that experiment name appears in run names."""
    config = IsoFlopSweepConfig(budgets=(1e18,))
    result = generate_isoflop_train_args(
        sweep_config=config,
        experiment_name="my-custom-experiment",
        vocab_size=MARIN_TOKENIZER_VOCAB_SIZE,
    )
    for args in result:
        assert "my-custom-experiment" in args.run_name


# --- Plotting tests ---


def test_create_isoflop_plot_empty_data():
    """Test that create_isoflop_plot handles empty data."""
    from marin.scaling_laws import create_isoflop_plot

    df = pd.DataFrame()
    fig = create_isoflop_plot(df, [], {})
    assert fig is not None


def test_create_isoflop_plot_with_data():
    """Test create_isoflop_plot with sample data."""
    from marin.scaling_laws import create_isoflop_plot

    df = pd.DataFrame(
        {
            "tokens": [1e9, 2e9, 3e9],
            "loss": [2.5, 2.3, 2.2],
            "flops": [1e18, 1e18, 1e18],
            "params": [1e8, 1e8, 1e8],
            "name": ["run1", "run2", "run3"],
            "label": ["nemo", "nemo", "nemo"],
        }
    )
    minima_records = [
        MinimaRecord(
            label="nemo",
            flops=1e18,
            optimal_tokens=2e9,
            loss_at_optimal=2.3,
            hidden_dim=512,
            num_layers=8,
            batch_size=64,
            optimal_params=1e8,
        )
    ]
    fit_curves = {("nemo", 1e18): (0.1, -1.0, 3.0, 1e9, 3e9)}
    fig = create_isoflop_plot(df, minima_records, fit_curves)
    assert fig is not None


def test_create_scaling_plot_empty():
    """Test that create_scaling_plot handles empty data."""
    from marin.scaling_laws import create_scaling_plot

    fig = create_scaling_plot([], {})
    assert fig is not None


def test_create_scaling_plot_with_data():
    """Test create_scaling_plot with sample data."""
    from marin.scaling_laws import create_scaling_plot

    minima_records = [
        MinimaRecord(
            label="nemo",
            flops=1e18,
            optimal_tokens=1e9,
            loss_at_optimal=2.3,
            hidden_dim=512,
            num_layers=8,
            batch_size=64,
            optimal_params=1e8,
        ),
        MinimaRecord(
            label="nemo",
            flops=1e19,
            optimal_tokens=5e9,
            loss_at_optimal=2.1,
            hidden_dim=1024,
            num_layers=16,
            batch_size=128,
            optimal_params=5e8,
        ),
    ]
    scaling_fits = {"nemo": (0.5, 1e5)}
    fig = create_scaling_plot(minima_records, scaling_fits)
    assert fig is not None


# --- Snapshot tests ---

# Snapshot of expected output for generate_isoflop_train_args with budget=1e18.
# This captures the configuration generation logic from experiments/isoflop_sweep.py
# on the main branch to ensure the refactored code produces identical configs.
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

    This test ensures the refactored scaling_laws module produces identical
    configurations to the original experiments/isoflop_sweep.py implementation.
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


# --- round_flops_to_bucket tests ---


def test_round_flops_to_bucket_exact_values():
    """Test that exact significant figures remain unchanged."""
    assert round_flops_to_bucket(1e18) == 1e18
    assert round_flops_to_bucket(1e19) == 1e19
    assert round_flops_to_bucket(3e19) == 3e19
    assert round_flops_to_bucket(6e19) == 6e19
    assert round_flops_to_bucket(1e20) == 1e20


def test_round_flops_to_bucket_rounds_to_one_significant_figure():
    """Test rounding to 1 significant figure."""
    # Small variations should round to nearest integer mantissa
    assert round_flops_to_bucket(1.05e19) == 1e19
    assert round_flops_to_bucket(1.4e19) == 1e19
    assert round_flops_to_bucket(1.5e19) == 2e19
    assert round_flops_to_bucket(2.8e19) == 3e19
    assert round_flops_to_bucket(9.5e19) == 1e20  # Wraps to next power


def test_round_flops_to_bucket_handles_edge_cases():
    """Test edge cases for FLOP bucket rounding."""
    assert round_flops_to_bucket(0) == 0
    assert round_flops_to_bucket(-1e18) == -1e18
    # Very large values
    assert round_flops_to_bucket(5.5e21) == 6e21


# --- fit_scaling_laws tests ---


def test_fit_scaling_laws_empty_dataframe():
    """Test that fit_scaling_laws handles empty dataframe."""
    df = pd.DataFrame()
    minima_records, scaling_fits, fit_curves = fit_scaling_laws(df)
    assert minima_records == []
    assert scaling_fits == {}
    assert fit_curves == {}


def test_fit_scaling_laws_single_budget():
    """Test fit_scaling_laws with data from a single FLOP budget."""
    # Create synthetic data with multiple token counts at one budget
    df = pd.DataFrame(
        {
            "tokens": [1e9, 2e9, 4e9, 8e9, 16e9],
            "loss": [2.5, 2.2, 2.0, 2.1, 2.3],  # U-shaped (optimal around 4e9)
            "flops": [1e18, 1e18, 1e18, 1e18, 1e18],
            "params": [1e8, 1e8, 1e8, 1e8, 1e8],
            "hidden_dim": [512, 512, 512, 512, 512],
            "num_layers": [6, 6, 6, 6, 6],
            "batch_size": [32, 32, 32, 32, 32],
            "name": ["run1", "run2", "run3", "run4", "run5"],
            "label": ["nemo", "nemo", "nemo", "nemo", "nemo"],
        }
    )
    minima_records, scaling_fits, fit_curves = fit_scaling_laws(df)

    # Should find exactly one minimum for the single (label, budget) pair
    assert len(minima_records) == 1
    rec = minima_records[0]
    assert rec.label == "nemo"
    assert rec.flops == 1e18
    # Optimal should be near 4e9 (the minimum loss point)
    assert 1e9 < rec.optimal_tokens < 20e9

    # With only one budget, cannot fit scaling law
    assert "nemo" not in scaling_fits

    # Should have fit curve for (nemo, 1e18)
    assert ("nemo", 1e18) in fit_curves


def test_fit_scaling_laws_multiple_budgets():
    """Test fit_scaling_laws with multiple FLOP budgets to fit scaling law."""
    # Create data with two FLOP budgets
    df = pd.DataFrame(
        {
            "tokens": [
                # Budget 1e18 - optimal around 2e9
                1e9,
                2e9,
                4e9,
                # Budget 1e19 - optimal around 6e9 (more tokens for more compute)
                2e9,
                6e9,
                18e9,
            ],
            "loss": [
                2.3,
                2.0,
                2.2,  # U-shape at 1e18
                2.0,
                1.7,
                1.9,  # U-shape at 1e19
            ],
            "flops": [1e18, 1e18, 1e18, 1e19, 1e19, 1e19],
            "params": [1e8, 1e8, 1e8, 5e8, 5e8, 5e8],
            "hidden_dim": [512, 512, 512, 1024, 1024, 1024],
            "num_layers": [6, 6, 6, 12, 12, 12],
            "batch_size": [32, 32, 32, 64, 64, 64],
            "name": [f"run{i}" for i in range(6)],
            "label": ["nemo"] * 6,
        }
    )
    minima_records, scaling_fits, fit_curves = fit_scaling_laws(df)

    # Should find two minima (one per budget)
    assert len(minima_records) == 2

    # Should have scaling fit for nemo
    assert "nemo" in scaling_fits
    alpha, _A = scaling_fits["nemo"]

    # Scaling law alpha should be positive (more compute -> more tokens)
    assert 0 < alpha < 1

    # Should have fit curves for both budgets
    assert ("nemo", 1e18) in fit_curves
    assert ("nemo", 1e19) in fit_curves


def test_fit_scaling_laws_multiple_labels():
    """Test fit_scaling_laws with multiple dataset labels."""
    df = pd.DataFrame(
        {
            "tokens": [1e9, 2e9, 4e9, 1e9, 2e9, 4e9],
            "loss": [2.5, 2.2, 2.4, 2.3, 2.0, 2.2],
            "flops": [1e18, 1e18, 1e18, 1e18, 1e18, 1e18],
            "params": [1e8] * 6,
            "hidden_dim": [512] * 6,
            "num_layers": [6] * 6,
            "batch_size": [32] * 6,
            "name": [f"run{i}" for i in range(6)],
            "label": ["nemo", "nemo", "nemo", "dclm", "dclm", "dclm"],
        }
    )
    minima_records, _scaling_fits, fit_curves = fit_scaling_laws(df)

    # Should find two minima (one per label at the single budget)
    assert len(minima_records) == 2
    labels = {rec.label for rec in minima_records}
    assert labels == {"nemo", "dclm"}

    # Should have fit curves for both labels
    assert ("nemo", 1e18) in fit_curves
    assert ("dclm", 1e18) in fit_curves


# --- transform_metrics_for_isoflop tests ---


# Sample tracker_metrics.jsonl data extracted from real runs
# Note: throughput/total_gflops is in GFLOPs, multiply by 1e9 to get FLOPs
# For 1e18 FLOPs, we need ~1e9 GFLOPs (1e9 * 1e9 = 1e18)
# Need at least 3 data points per budget to fit a quadratic
# Loss values form a U-shape in log(tokens) space for each budget
SAMPLE_METRICS_DATA = [
    # 1e18 budget - 3 runs with U-shaped loss curve
    # Too few tokens: model is good but undertrained
    # Just right: optimal training
    # Too many tokens: model is too small
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d1024-L11-B8-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 1024, "num_layers": 11},
            "trainer": {"train_batch_size": 8},
        },
        "summary": {
            "throughput/total_tokens": 1000000000,  # 1B tokens (undertrained)
            "throughput/total_gflops": 1000000000.0,  # 1e9 GFLOPs = 1e18 FLOPs
            "eval/paloma/c4_en/bpb": 1.25,  # Higher loss - undertrained
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
            "throughput/total_tokens": 2500000000,  # 2.5B tokens (optimal)
            "throughput/total_gflops": 1000000000.0,  # 1e9 GFLOPs = 1e18 FLOPs
            "eval/paloma/c4_en/bpb": 1.12,  # Lowest loss - optimal
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
            "throughput/total_tokens": 5000000000,  # 5B tokens (overtrained/small model)
            "throughput/total_gflops": 1000000000.0,  # 1e9 GFLOPs = 1e18 FLOPs
            "eval/paloma/c4_en/bpb": 1.18,  # Higher loss - model too small
            "parameter_count": 156508160,
        },
    },
    # 1e19 budget - 3 runs with U-shaped loss curve (more tokens optimal)
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d2048-L21-B16-nemo-wider-depth-adapt",
        "config": {
            "model": {"hidden_dim": 2048, "num_layers": 21},
            "trainer": {"train_batch_size": 16},
        },
        "summary": {
            "throughput/total_tokens": 3000000000,  # 3B tokens (undertrained)
            "throughput/total_gflops": 10000000000.0,  # 1e10 GFLOPs = 1e19 FLOPs
            "eval/paloma/c4_en/bpb": 1.05,  # Higher loss - undertrained
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
            "throughput/total_tokens": 8000000000,  # 8B tokens (optimal)
            "throughput/total_gflops": 10000000000.0,  # 1e10 GFLOPs = 1e19 FLOPs
            "eval/paloma/c4_en/bpb": 0.98,  # Lowest loss - optimal
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
            "throughput/total_tokens": 20000000000,  # 20B tokens (overtrained)
            "throughput/total_gflops": 10000000000.0,  # 1e10 GFLOPs = 1e19 FLOPs
            "eval/paloma/c4_en/bpb": 1.02,  # Higher loss - model too small
            "parameter_count": 400000000,
        },
    },
]


def test_transform_metrics_for_isoflop_basic():
    """Test basic transformation of metrics data."""
    raw_df = pd.DataFrame(SAMPLE_METRICS_DATA)
    metric_key = "eval/paloma/c4_en/bpb"

    result = transform_metrics_for_isoflop(raw_df, metric_key)

    assert len(result) == 6  # 3 runs at 1e18 + 3 runs at 1e19
    assert set(result.columns) == {
        "tokens",
        "loss",
        "flops",
        "params",
        "hidden_dim",
        "num_layers",
        "batch_size",
        "name",
        "label",
    }

    # Check that values are extracted correctly - first row is d1024/L11
    row0 = result.iloc[0]
    assert row0["tokens"] == 1000000000  # 1B tokens
    assert row0["loss"] == 1.25
    assert row0["hidden_dim"] == 1024
    assert row0["num_layers"] == 11
    assert row0["batch_size"] == 8


def test_transform_metrics_for_isoflop_with_label_map():
    """Test transformation with custom label mapping."""
    raw_df = pd.DataFrame(SAMPLE_METRICS_DATA)
    metric_key = "eval/paloma/c4_en/bpb"
    label_map = {"nemo-wider-depth-adapt": "NeMo"}

    result = transform_metrics_for_isoflop(raw_df, metric_key, label_map)

    assert len(result) == 6  # 3 runs at 1e18 + 3 runs at 1e19
    assert all(result["label"] == "NeMo")


def test_transform_metrics_for_isoflop_filters_low_flops():
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
                    "throughput/total_gflops": 1e6,  # Only 1e15 FLOPs (< 1e18)
                    "eval/paloma/c4_en/bpb": 3.0,
                    "parameter_count": 1e7,
                },
            }
        ]
    )

    result = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    assert len(result) == 0


def test_transform_metrics_for_isoflop_empty_input():
    """Test transformation with empty input."""
    raw_df = pd.DataFrame()
    result = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    assert result.empty


def test_transform_metrics_for_isoflop_missing_fields():
    """Test transformation handles missing fields gracefully."""
    raw_df = pd.DataFrame(
        [
            {
                "run_path": "gs://marin/checkpoints/isoflop-1e+18-d512-L6-B32-incomplete",
                "config": {"model": {}, "trainer": {}},
                "summary": {
                    # Missing throughput/total_tokens
                    "throughput/total_gflops": 1000001.0,
                    "eval/paloma/c4_en/bpb": 1.5,
                },
            }
        ]
    )

    result = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    # Should skip the row with missing required fields
    assert len(result) == 0


# --- Integration test: fit_scaling_laws with transform_metrics_for_isoflop ---


def test_end_to_end_analysis_pipeline():
    """Integration test: transform metrics and fit scaling laws."""
    raw_df = pd.DataFrame(SAMPLE_METRICS_DATA)

    # Transform metrics
    isoflop_df = transform_metrics_for_isoflop(raw_df, "eval/paloma/c4_en/bpb")
    assert len(isoflop_df) == 6  # 3 runs at 1e18 + 3 runs at 1e19

    # Fit scaling laws
    minima_records, scaling_fits, _fit_curves = fit_scaling_laws(isoflop_df)

    # With 2 budgets (1e18, 1e19), each with 3 points, we should get 2 minima
    assert len(minima_records) == 2

    # Should have a scaling fit for the label
    assert len(scaling_fits) == 1
    label = next(iter(scaling_fits.keys()))
    alpha, A = scaling_fits[label]

    # Sanity check the scaling law parameters
    assert 0 < alpha < 1  # Typical range for token scaling exponent
    assert A > 0


def test_minima_records_have_scaling_fit_params():
    """Test that minima records are augmented with scaling fit parameters."""
    df = pd.DataFrame(
        {
            "tokens": [1e9, 2e9, 4e9, 2e9, 6e9, 18e9],
            "loss": [2.3, 2.0, 2.2, 2.0, 1.7, 1.9],
            "flops": [1e18, 1e18, 1e18, 1e19, 1e19, 1e19],
            "params": [1e8, 1e8, 1e8, 5e8, 5e8, 5e8],
            "hidden_dim": [512, 512, 512, 1024, 1024, 1024],
            "num_layers": [6, 6, 6, 12, 12, 12],
            "batch_size": [32, 32, 32, 64, 64, 64],
            "name": [f"run{i}" for i in range(6)],
            "label": ["nemo"] * 6,
        }
    )
    minima_records, scaling_fits, _ = fit_scaling_laws(df)

    # All records for a label with a scaling fit should have the params
    for rec in minima_records:
        if rec.label in scaling_fits:
            alpha, A = scaling_fits[rec.label]
            assert rec.scaling_alpha == alpha
            assert rec.scaling_A == A
