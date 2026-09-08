# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the WSPU optima for the three Uncheatable components the full optimum worsened.

The frozen candidate table holds the runtime-grid optima of the byte-weighted
bbc_news, ao3_english and wikipedia_english aggregate at epoch caps 4, 6 and 7;
the mixture plateaus at cap 7, and caps 6 and 7 differ by one runtime count.
Every run uses a tied one-phase policy, the common data seed shared with the
other epoch-cap sweeps, inline Uncheatable evaluation and native Table-9
evaluation, on v6e-8 in us-east5-b.
"""

from __future__ import annotations

from pathlib import Path

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_one_phase_wspu_worsened_components_sweep_3e18_20260905"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_one_phase_wspu_worsened_components_sweep_20260905"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "launch_candidate_weights.csv"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "09e3697b0829516de7ae0a93b84e796ced639b11e06a62b500a15a31d3f68057"
CAPS = (4, 6, 7)
NOMINAL_CANDIDATE_IDS = tuple(f"wspu_worsened_cap{cap:02d}" for cap in CAPS)
EXPECTED_ALIAS_MAP = {candidate_id: candidate_id for candidate_id in NOMINAL_CANDIDATE_IDS}
EXPECTED_RUN_COUNT = 3
RUN_ID_BASE = 7_360_000
MAX_CONCURRENT = EXPECTED_RUN_COUNT

SWEEP_DEFINITION = sweep.SweepDefinition(
    experiment_name=EXPERIMENT_NAME,
    nominal_candidate_ids=NOMINAL_CANDIDATE_IDS,
    expected_alias_map=EXPECTED_ALIAS_MAP,
    expected_run_count=EXPECTED_RUN_COUNT,
    run_id_base=RUN_ID_BASE,
    common_data_seed=sweep.COMMON_DATA_SEED,
    trainer_seed=sweep.TRAINER_SEED,
    run_name_prefix="onephase_wspuw",
    table9_run_name_prefix="t9ww",
    panel_source="weibull_softplus_unscaled_worsened_components_epoch_cap_optimum",
    table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_wspu_worsened_components_sweep",
    provenance_panel="delphi_3e18_one_phase_wspu_worsened_components_sweep",
    wandb_tags=(
        "delphi-3e18",
        "one-phase",
        "weibull-softplus-unscaled",
        "worsened-components",
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
