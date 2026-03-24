# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.ising_tokenizer.base.data import (
    BklIsingConfig,
    CRITICAL_TEMPERATURE_2D,
    SyntheticSplitConfig,
    TrajectoryTokenizerConfig,
    build_synthetic_split,
    decode_event_positions,
    decode_initial_spins,
    decode_wait_times,
    encode_trajectory_tokens,
    simulate_bkl_trajectory,
    temperature_normalization_stats,
)
from experiments.ising_tokenizer.base.launch import IsingSmokeDataConfig, build_local_smoke_config, run_local_smoke
from experiments.ising_tokenizer.base.train import build_loss_weights


def test_bkl_trajectory_and_tokenization_are_deterministic():
    dynamics = BklIsingConfig(lattice_size=4, num_events=8, burn_in_events=8)
    tokenizer = TrajectoryTokenizerConfig(num_dt_bins=16)

    trajectory_a = simulate_bkl_trajectory(config=dynamics, temperature=1.7, rng=np.random.default_rng(0))
    trajectory_b = simulate_bkl_trajectory(config=dynamics, temperature=1.7, rng=np.random.default_rng(0))

    assert np.array_equal(trajectory_a.initial_spins, trajectory_b.initial_spins)
    assert np.array_equal(trajectory_a.event_positions, trajectory_b.event_positions)
    assert np.allclose(trajectory_a.waiting_times, trajectory_b.waiting_times)

    tokens = encode_trajectory_tokens(trajectory_a, dynamics_config=dynamics, tokenizer_config=tokenizer)
    assert tokens.shape == (dynamics.seq_len,)
    assert tokens.min() >= 0
    assert tokens.max() < tokenizer.vocab_size(dynamics.lattice_size)

    decoded_spins = decode_initial_spins(tokens, dynamics_config=dynamics)
    decoded_positions = decode_event_positions(tokens, dynamics_config=dynamics)
    decoded_wait_times = decode_wait_times(tokens, dynamics_config=dynamics, tokenizer_config=tokenizer)
    assert np.array_equal(decoded_spins, trajectory_a.initial_spins)
    assert np.array_equal(decoded_positions, trajectory_a.event_positions)
    assert np.isfinite(decoded_wait_times).all()
    assert np.all(decoded_wait_times > 0.0)


def test_synthetic_split_has_fixed_length_tokens_and_normalized_temperatures():
    dynamics = BklIsingConfig(lattice_size=4, num_events=6, burn_in_events=6)
    tokenizer = TrajectoryTokenizerConfig(num_dt_bins=8)
    split = SyntheticSplitConfig(name="train", temperatures=(1.5, 2.8), num_examples=6, seed=0)
    mean, std = temperature_normalization_stats(split.temperatures)

    dataset = build_synthetic_split(
        split_config=split,
        dynamics_config=dynamics,
        tokenizer_config=tokenizer,
        temperature_mean=mean,
        temperature_std=std,
    )

    assert dataset.tokens.shape == (6, dynamics.seq_len)
    assert dataset.seq_len == dynamics.seq_len
    assert dataset.vocab_size == tokenizer.vocab_size(dynamics.lattice_size)
    assert np.allclose(np.unique(dataset.temperatures), np.asarray([1.5, 2.8], dtype=np.float32))
    assert np.isfinite(dataset.normalized_temperatures).all()


def test_loss_weights_mask_initial_position_tokens_and_eos():
    dynamics = BklIsingConfig(lattice_size=4, num_events=4, burn_in_events=4)
    tokenizer = TrajectoryTokenizerConfig(num_dt_bins=8)
    split = SyntheticSplitConfig(name="train", temperatures=(1.5, 2.8), num_examples=2, seed=0)
    mean, std = temperature_normalization_stats(split.temperatures)
    dataset = build_synthetic_split(
        split_config=split,
        dynamics_config=dynamics,
        tokenizer_config=tokenizer,
        temperature_mean=mean,
        temperature_std=std,
    )

    loss_weight = build_loss_weights(dataset)

    assert np.all(loss_weight[:, : dataset.initial_state_token_count : 2] == 0.0)
    assert np.all(loss_weight[:, 1 : dataset.initial_state_token_count : 2] == 1.0)
    assert np.all(loss_weight[:, dataset.valid_token_count - 1 :] == 0.0)


def test_local_smoke_run_emits_finite_losses(tmp_path):
    output_dir = tmp_path / "ising_smoke"
    summary = run_local_smoke(output_dir=output_dir, num_train_steps=8, train_examples=24, rollout_examples=1)

    assert summary["initial_train_loss"] > 0
    assert summary["final_train_loss"] > 0
    assert summary["final_validation_loss"] > 0
    assert np.isfinite(summary["final_critical_probe_loss"])
    assert "rollout_eval" in summary
    assert "T2.269" in summary["rollout_eval"]["temperatures"]
    assert (output_dir / "metrics.json").exists()


def test_build_local_smoke_config_matches_sequence_and_vocab():
    data_config = IsingSmokeDataConfig(
        dynamics=BklIsingConfig(lattice_size=4, num_events=8, burn_in_events=8),
        tokenizer=TrajectoryTokenizerConfig(num_dt_bins=16),
        critical_probe_split=SyntheticSplitConfig(
            name="critical_probe",
            temperatures=(CRITICAL_TEMPERATURE_2D,),
            num_examples=4,
            seed=2,
        ),
    )

    run_config = build_local_smoke_config(data_config)

    assert run_config.model.max_seq_len == data_config.dynamics.seq_len
    assert run_config.model.vocab_size == data_config.tokenizer.vocab_size(data_config.dynamics.lattice_size)
