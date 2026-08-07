# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.grug.moe.launch_tied_experts import TiedExpertModelSize, TiedExpertPhase, tied_expert_runs


def test_smoke_matrix_can_be_dispatched_in_cost_controlled_waves() -> None:
    first_wave = tied_expert_runs(
        version="dev",
        phase=TiedExpertPhase.SMOKE,
        variant_names=("baseline", "pairwise_unscaled", "pairwise_sqrt", "pairwise_linear"),
    )
    second_wave = tied_expert_runs(
        version="dev",
        phase=TiedExpertPhase.SMOKE,
        variant_names=("middle4_unscaled", "middle4_sqrt", "middle4_linear"),
    )

    assert [run.name.rsplit("/", 1)[-1] for run in first_wave] == [
        "baseline",
        "pairwise_unscaled",
        "pairwise_sqrt",
        "pairwise_linear",
    ]
    assert [run.name.rsplit("/", 1)[-1] for run in second_wave] == [
        "middle4_unscaled",
        "middle4_sqrt",
        "middle4_linear",
    ]
    assert all(run.runtime_args["train_resources"].regions == ["us-central1"] for run in (*first_wave, *second_wave))


def test_variant_filter_rejects_unknown_or_duplicate_runs() -> None:
    with pytest.raises(ValueError, match="unknown smoke tied-expert variants"):
        tied_expert_runs(version="dev", phase=TiedExpertPhase.SMOKE, variant_names=("unknown",))
    with pytest.raises(ValueError, match="contains duplicates"):
        tied_expert_runs(version="dev", phase=TiedExpertPhase.SMOKE, variant_names=("baseline", "baseline"))


def test_full_matrix_keeps_unscaled_and_sqrt_treatments() -> None:
    runs = tied_expert_runs(version="dev", phase=TiedExpertPhase.FULL)

    assert [run.name.rsplit("/", 1)[-1] for run in runs] == [
        "baseline",
        "pairwise_unscaled",
        "pairwise_sqrt",
        "middle4_unscaled",
        "middle4_sqrt",
    ]


def test_d768_matrix_is_matched_untied_and_two_anchor_middle_four_comparison(monkeypatch) -> None:
    monkeypatch.setenv("GRUG_TIED_MODEL", TiedExpertModelSize.D768)
    runs = tied_expert_runs(version="dev", phase=TiedExpertPhase.FULL)
    configs = [json.loads(run.fingerprint_payload()) for run in runs]

    assert [run.name.rsplit("/", 1)[-1] for run in runs] == [
        "baseline",
        "middle4_two_anchor_unscaled",
        "middle4_two_anchor_sqrt",
    ]
    assert [config["model"]["expert_bank_for_layer"] for config in configs] == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 2, 2, 2, 3, 4],
        [0, 1, 2, 2, 2, 2, 3, 4],
    ]
    assert all(config["model"]["hidden_dim"] == 768 for config in configs)
    assert all(config["model"]["num_layers"] == 8 for config in configs)
    assert all(config["batch_size"] == 128 for config in configs)
    assert all(config["steps"] == 8453 for config in configs)
    assert configs[0]["optimizer"]["expert_bank_group_sizes"] == [1] * 8
    assert all(config["optimizer"]["expert_bank_group_sizes"] == [1, 1, 4, 1, 1] for config in configs[1:])
    assert [config["optimizer"]["tied_expert_lr_scale"] for config in configs] == [
        "unscaled",
        "unscaled",
        "sqrt",
    ]
    assert all(run.runtime_args["train_resources"].regions == ["us-central1"] for run in runs)
