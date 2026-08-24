# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import cast

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_common_branches as branches
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base

CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
CONTINUATION_SHA256 = "9547515728e3c85ed564066cf8cfa36eefa80a7241cde3b32e351004f9afc883"
SELECTED_CANDIDATES = (
    "observed_cap10_best",
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
)


@dataclass(frozen=True)
class _PrefixSpec:
    phase_weights: dict[str, dict[str, float]]
    data_seed: int
    trainer_seed: int


def test_frozen_continuation_design_obeys_runtime_contract() -> None:
    buckets, continuations = branches.load_continuations(
        branches.DEFAULT_CONTINUATION_WEIGHTS,
        CONTINUATION_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )

    assert len(buckets) == 39
    assert len(continuations) == branches.COMMON_CONTINUATION_COUNT
    assert sum(bool(row["fit_budget"]) for row in continuations) == branches.COMMON_FIT_CONTINUATION_COUNT
    assert max(float(row["max_phase_1_materialized_epoch"]) for row in continuations) <= (
        branches.HISTORICAL_PHASE_1_EPOCH_CAP
    )
    assert max(float(row["max_total_materialized_epoch_across_candidate_prefixes"]) for row in continuations) <= (
        branches.HISTORICAL_TOTAL_EPOCH_CAP
    )


def test_branch_panel_crosses_common_fit_rows_and_keeps_controls_outside_budget() -> None:
    buckets, continuations = branches.load_continuations(
        branches.DEFAULT_CONTINUATION_WEIGHTS,
        CONTINUATION_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )
    uniform = {bucket: 1.0 / len(buckets) for bucket in buckets}
    prefixes = []
    prefix_specs = {}
    for candidate_id in SELECTED_CANDIDATES:
        for repeat_seed in (branches.PRIMARY_BRANCH_SEED, branches.STABILITY_BRANCH_SEED):
            prefix = branches.PrefixCheckpoint(
                candidate_id=candidate_id,
                repeat_seed=repeat_seed,
                checkpoint_uri=f"gs://marin-us-east5/{candidate_id}/step-2399",
                provenance_sha256=f"provenance-{candidate_id}-{repeat_seed}",
            )
            prefixes.append(prefix)
            prefix_specs[(candidate_id, repeat_seed)] = cast(
                base.DelphiSwarmRunSpec,
                _PrefixSpec(
                    phase_weights={"phase_0": uniform, "phase_1": uniform},
                    data_seed=930_000 + repeat_seed,
                    trainer_seed=repeat_seed,
                ),
            )

    rows = branches.enrich_branch_rows(
        branches.branch_rows(prefixes=prefixes, prefix_specs=prefix_specs, continuations=continuations),
        prefix_specs,
    )

    assert len(rows) == branches.TOTAL_BRANCH_ROWS == 228
    assert sum(bool(row["fit_budget"]) for row in rows) == 200
    assert sum(row["branch_role"] == "primary_cross" for row in rows) == 212
    assert sum(row["branch_role"] == "prefix_tied_control" for row in rows) == 4
    assert sum(row["branch_role"] == "prefix_seed_stability_sentinel" for row in rows) == 12
    assert len({row["run_name"] for row in rows}) == len(rows)
