# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Delphi 3e18 full-canonical DSP optima under epoch caps 2 through 16."""

from __future__ import annotations

from pathlib import Path

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18_20260901"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "909f7240eee5ee2b1ecc2fa88a987fe471b867609eff4993b31a7b765f35d84d"
CAPS = tuple(range(2, 17, 2))
NOMINAL_CANDIDATE_IDS = tuple(f"{target}_cap{cap:02d}" for target in ("uncheatable", "table9_macro") for cap in CAPS)
EXPECTED_ALIAS_MAP = {candidate_id: candidate_id for candidate_id in NOMINAL_CANDIDATE_IDS}
EXPECTED_RUN_COUNT = len(NOMINAL_CANDIDATE_IDS)
RUN_ID_BASE = 7_340_000
MAX_CONCURRENT = EXPECTED_RUN_COUNT

SWEEP_DEFINITION = sweep.SweepDefinition(
    experiment_name=EXPERIMENT_NAME,
    nominal_candidate_ids=NOMINAL_CANDIDATE_IDS,
    expected_alias_map=EXPECTED_ALIAS_MAP,
    expected_run_count=EXPECTED_RUN_COUNT,
    run_id_base=RUN_ID_BASE,
    common_data_seed=sweep.COMMON_DATA_SEED,
    trainer_seed=sweep.TRAINER_SEED,
    run_name_prefix="onephase_fullcanonical_dsp",
    panel_source="full_canonical_dsp_epoch_cap_optimum",
    table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_full_canonical_dsp_epoch_cap_sweep",
    provenance_panel="delphi_3e18_one_phase_full_canonical_dsp_epoch_cap_sweep",
    wandb_tags=(
        "delphi-3e18",
        "one-phase",
        "dsp",
        "full-canonical",
        "whole-run-epoch-cap-sweep",
    ),
)


def parse_args():
    return sweep.parse_sweep_args(
        default_candidate_weights=DEFAULT_CANDIDATE_WEIGHTS,
        expected_candidate_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        max_concurrent=MAX_CONCURRENT,
    )


def main() -> None:
    args, remaining = parse_args()
    sweep.run_sweep(
        args,
        remaining,
        definition=SWEEP_DEFINITION,
        expected_candidate_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        local_artifact_dir=LOCAL_ARTIFACT_DIR,
    )


if __name__ == "__main__":
    main()
