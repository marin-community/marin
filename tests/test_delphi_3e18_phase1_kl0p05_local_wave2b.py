# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_common_branches as branches
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_kl0p05_local_wave2b as local_wave2b
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base

CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
LOCAL_WAVE2B_WEIGHTS_SHA256 = "de2e979a14208b9dd3fa4e2a1b7163eb04732844f32c470d8d932210d9cb0582"
LOCAL_WAVE2B_MANIFEST_SHA256 = "706d37d8d4bbebcdfbc4799bc7586d1b72e97444cf852b200895dceefdb97226"
LOCAL_WAVE2B_CONTRACT_SHA256 = "f89166981aafbf6dc5f6c78051f98b78426b01bb6b5536a8c661e0e0236c3e9c"
LOCAL_WAVE2B_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "domain_phase_mix"
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_kl0p05_local_wave2b_20260825"
)


@dataclass(frozen=True)
class _PrefixSpec:
    phase_weights: dict[str, dict[str, float]]
    data_seed: int
    trainer_seed: int


def test_local_wave2b_contract_and_rows_are_frozen() -> None:
    weights_path = LOCAL_WAVE2B_DIR / "continuation_weights.csv"
    manifest_path = LOCAL_WAVE2B_DIR / "manifest.json"
    contract_path = LOCAL_WAVE2B_DIR / "contract.json"
    local_wave2b.verify_selection_artifacts(
        weights_path,
        LOCAL_WAVE2B_WEIGHTS_SHA256,
        manifest_path,
        LOCAL_WAVE2B_MANIFEST_SHA256,
        contract_path,
        LOCAL_WAVE2B_CONTRACT_SHA256,
        "post_wave1_local_redesign",
    )

    buckets, continuations = local_wave2b.load_wave2_continuations(
        weights_path,
        LOCAL_WAVE2B_WEIGHTS_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )

    assert len(buckets) == 39
    assert len(continuations) == local_wave2b.EXPECTED_CONTINUATIONS
    assert sum(bool(row["fit_budget"]) for row in continuations) == local_wave2b.EXPECTED_FIT_CONTINUATIONS
    assert sum(bool(row["referee_holdout"]) for row in continuations) == local_wave2b.EXPECTED_REFEREE_HOLDOUTS
    tied = [row for row in continuations if row["selection_tranche"] == "tied_control"]
    assert len(tied) == local_wave2b.EXPECTED_TIED_CONTROLS
    assert {row["prefix_repeat_seed"] for row in tied} == {0, 1}
    assert len({row["data_seed"] for row in tied}) == local_wave2b.EXPECTED_TIED_CONTROLS


def test_local_wave2b_rows_use_exact_prefix_seed_and_disjoint_namespace() -> None:
    _, continuations = local_wave2b.load_wave2_continuations(
        LOCAL_WAVE2B_DIR / "continuation_weights.csv",
        LOCAL_WAVE2B_WEIGHTS_SHA256,
        branches.DEFAULT_CANDIDATE_WEIGHTS,
        CANDIDATE_SHA256,
    )
    uniform = {bucket: 1.0 / len(base.DOMAIN_NAMES) for bucket in base.DOMAIN_NAMES}
    prefixes = {
        seed: branches.PrefixCheckpoint(
            candidate_id=local_wave2b.TARGET_PREFIX,
            repeat_seed=seed,
            checkpoint_uri=f"gs://marin-us-east5/prefix-seed{seed}/step-2399",
            provenance_sha256=f"provenance-{seed}",
        )
        for seed in local_wave2b.EXPECTED_PREFIX_SEEDS
    }
    prefix_specs = {
        (local_wave2b.TARGET_PREFIX, seed): cast(
            base.DelphiSwarmRunSpec,
            _PrefixSpec(
                phase_weights={"phase_0": uniform, "phase_1": uniform},
                data_seed=930_000 + seed,
                trainer_seed=seed,
            ),
        )
        for seed in local_wave2b.EXPECTED_PREFIX_SEEDS
    }

    rows = local_wave2b.wave2_rows(prefixes, prefix_specs, continuations, local_wave2b.BRANCH_RUN_ID_BASE)

    assert rows[0]["run_id"] == 954_000
    assert rows[-1]["run_id"] == 954_095
    assert len({row["run_name"] for row in rows}) == len(rows)
    assert sum(row["prefix"].repeat_seed == 1 for row in rows) == 4
    assert all(row["prefix"].repeat_seed == 0 for row in rows if row["fit_budget"])
