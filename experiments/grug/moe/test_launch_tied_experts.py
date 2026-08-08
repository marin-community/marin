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


@pytest.mark.parametrize(
    ("model_size", "hidden_dim", "num_layers", "batch_size", "full_steps", "tied_topology", "group_sizes"),
    [
        (
            TiedExpertModelSize.D1024,
            1024,
            11,
            128,
            16_149,
            [0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5],
            [1, 1, 4, 3, 1, 1],
        ),
        (
            TiedExpertModelSize.D1280,
            1280,
            13,
            256,
            14_315,
            [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6],
            [1, 1, 4, 4, 1, 1, 1],
        ),
    ],
)
def test_larger_matrix_uses_two_anchors_and_bounded_core_groups(
    model_size: TiedExpertModelSize,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    full_steps: int,
    tied_topology: list[int],
    group_sizes: list[int],
) -> None:
    runs = tied_expert_runs(version="dev", model_size=model_size, phase=TiedExpertPhase.FULL)
    configs = [json.loads(run.fingerprint_payload()) for run in runs]

    assert [run.name.rsplit("/", 1)[-1] for run in runs] == [
        "baseline",
        "core_groups_two_anchor_unscaled",
    ]
    assert configs[0]["model"]["expert_bank_for_layer"] == list(range(num_layers))
    assert configs[1]["model"]["expert_bank_for_layer"] == tied_topology
    assert all(config["model"]["hidden_dim"] == hidden_dim for config in configs)
    assert all(config["batch_size"] == batch_size for config in configs)
    assert all(config["steps"] == full_steps for config in configs)
    assert configs[1]["optimizer"]["expert_bank_group_sizes"] == group_sizes
    assert configs[1]["optimizer"]["tied_expert_lr_scale"] == "unscaled"
    assert max(group_sizes) <= 4
    assert all(run.runtime_args["train_resources"].regions == ["us-central1"] for run in runs)
