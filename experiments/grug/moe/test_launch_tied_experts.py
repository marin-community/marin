# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe.launch_tied_experts import TiedExpertPhase, tied_expert_runs


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
