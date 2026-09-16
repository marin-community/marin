# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Delphi 3e18 successor optima at every integer epoch cap through each plateau.

The frozen source table contains both targets at caps 2 through 8. Uncheatable
caps 7 and 8 exactly alias cap 6, so the generic sweep launcher emits the five
distinct Uncheatable policies at caps 2 through 6 and the seven distinct
Table-9 policies at caps 2 through 8. Every run uses a tied one-phase policy,
common random numbers, inline Uncheatable evaluation, and native Table-9
evaluation.
"""

from __future__ import annotations

from pathlib import Path

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_weibull_softplus_epoch_cap_sweep_3e18_20260902"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "6eb8fb151b1966330b1501f2e3a6e37812f44294803b8d5d46b054f6cdc928f0"
CAPS = tuple(range(2, 9))
TARGETS = ("uncheatable", "table9")
NOMINAL_CANDIDATE_IDS = tuple(f"wspu_{target}_cap{cap:02d}" for target in TARGETS for cap in CAPS)
EXPECTED_ALIAS_MAP = {
    **{f"wspu_uncheatable_cap{cap:02d}": f"wspu_uncheatable_cap{min(cap, 6):02d}" for cap in CAPS},
    **{f"wspu_table9_cap{cap:02d}": f"wspu_table9_cap{cap:02d}" for cap in CAPS},
}
EXPECTED_RUN_COUNT = 12
RUN_ID_BASE = 7_350_000
MAX_CONCURRENT = EXPECTED_RUN_COUNT

SWEEP_DEFINITION = sweep.SweepDefinition(
    experiment_name=EXPERIMENT_NAME,
    nominal_candidate_ids=NOMINAL_CANDIDATE_IDS,
    expected_alias_map=EXPECTED_ALIAS_MAP,
    expected_run_count=EXPECTED_RUN_COUNT,
    run_id_base=RUN_ID_BASE,
    common_data_seed=sweep.COMMON_DATA_SEED,
    trainer_seed=sweep.TRAINER_SEED,
    run_name_prefix="onephase_wspu",
    table9_run_name_prefix="t9w",
    panel_source="weibull_softplus_unscaled_epoch_cap_optimum",
    table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_weibull_softplus_epoch_cap_sweep",
    provenance_panel="delphi_3e18_one_phase_weibull_softplus_epoch_cap_sweep",
    wandb_tags=(
        "delphi-3e18",
        "one-phase",
        "weibull-softplus-unscaled",
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
