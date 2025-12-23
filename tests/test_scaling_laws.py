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
    generate_isoflop_train_args,
    parse_isoflop_run_name,
    predict_optimal_config,
    robust_quad_logx,
    round_to_power_of_two,
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
    fit_curves = {("nemo", 1e18): (0.1, -1.0, 3.0)}
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

    assert len(result) == len(EXPECTED_ISOFLOP_CONFIGS_1E18), (
        f"Expected {len(EXPECTED_ISOFLOP_CONFIGS_1E18)} configs, got {len(result)}"
    )

    for i, (args, expected) in enumerate(zip(result, EXPECTED_ISOFLOP_CONFIGS_1E18)):
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
            assert actual[key] == expected[key], (
                f"Config {i}: {key} mismatch: expected {expected[key]}, got {actual[key]}"
            )
