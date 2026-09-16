# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KL ablation of the frozen procedure at 3e18 FLOPs: sixteen proposals with a KL penalty toward proportional.

The fitted surrogates are the frozen ones (calibration-pinned flat kappa-floor link, no cap); only the proposal
objective changes, adding kl x KL(w || proportional) with kl in {0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5},
the coefficients of Olmix's own 3e18 sweep. Uncheatable proposals under cap 6 and Table-9 proposals under cap 8
(both caps inactive at KL 0, and the penalty only lowers repetition). Policies from the standalone implementation
(`mixture_selection.py optimize --kl`, commit 14fda28); `policy_summary.csv` beside the candidate table lists the
penalized objective per policy, and the surrogate's raw prediction is reported separately at collection. The KL-0
controls are the frozen-procedure validation runs at the same data seeds and trainer seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_kl_ablation_3e18_20260908"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
DEFAULT_CANDIDATE_DIR = REFERENCE_OUTPUTS / "delphi_kl_ablation_3e18_20260908"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "53b5f4b4c23cf9540e31cd9d5378b9350c0a61e2652d9d8512afff8e99953f3a"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
KL_TAGS = ("0p005", "0p01", "0p025", "0p05", "0p075", "0p1", "0p2", "0p5")
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
TRAINER_SEED = 0
MAX_CONCURRENT = 16

UNCHEATABLE_CANDIDATE_IDS = tuple(f"lwspu_u_kl{tag}_cap06" for tag in KL_TAGS)
TABLE9_CANDIDATE_IDS = tuple(f"lwspu_t9_kl{tag}_cap08" for tag in KL_TAGS)


def _definition(
    *,
    target: str,
    candidate_ids: tuple[str, ...],
    data_seed: int,
    run_id_base: int,
) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/{target}_seed{data_seed}",
        nominal_candidate_ids=candidate_ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in candidate_ids},
        expected_run_count=len(candidate_ids),
        run_id_base=run_id_base,
        common_data_seed=data_seed,
        trainer_seed=TRAINER_SEED,
        run_name_prefix=f"lwspukl_{target}_seed{data_seed}",
        table9_run_name_prefix=f"t9q{target[0]}",
        panel_source="wspu_kl_ablation",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_wspu_kl_ablation",
        provenance_panel="delphi_3e18_one_phase_wspu_kl_ablation",
        wandb_tags=(
            "delphi-3e18",
            "one-phase",
            "weibull-softplus-unscaled",
            "kappa-floor-link",
            "flat-profile-kappa",
            "calibration-pinned",
            "frozen-procedure",
            "kl-ablation",
            target,
        ),
    )


UNCHEATABLE_DEFINITION = _definition(
    target="uncheatable",
    candidate_ids=UNCHEATABLE_CANDIDATE_IDS,
    data_seed=UNCHEATABLE_DATA_SEED,
    run_id_base=7_404_000,
)
TABLE9_DEFINITION = _definition(
    target="table9",
    candidate_ids=TABLE9_CANDIDATE_IDS,
    data_seed=TABLE9_DATA_SEED,
    run_id_base=7_404_100,
)
VALIDATION_GROUPS = (
    (UNCHEATABLE_DEFINITION, UNCHEATABLE_CANDIDATE_IDS),
    (TABLE9_DEFINITION, TABLE9_CANDIDATE_IDS),
)


ALL_CANDIDATE_IDS = UNCHEATABLE_CANDIDATE_IDS + TABLE9_CANDIDATE_IDS
ALL_DEFINITION = _definition(target="all", candidate_ids=ALL_CANDIDATE_IDS, data_seed=0, run_id_base=0)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the KL-ablation policy table and return the sixteen rows by target."""
    candidates, _ = sweep.load_candidate_mixtures(
        DEFAULT_CANDIDATE_WEIGHTS,
        EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=ALL_DEFINITION,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    groups = []
    for definition, candidate_ids in VALIDATION_GROUPS:
        selected = [by_id[candidate_id] for candidate_id in candidate_ids]
        if len(selected) != definition.expected_run_count:
            raise ValueError(f"Expected {definition.expected_run_count} {definition.experiment_name} candidates")
        groups.append((definition, selected))
    return groups


def build_validation_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture], list[base.DelphiSwarmRunSpec]]]:
    """Bind each target's KL-penalized proposals to its control's data seed."""
    groups = []
    for definition, candidates in selected_candidates():
        run_specs = sweep.build_run_specs(
            template=template,
            candidates=candidates,
            tpu_type=tpu_type,
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            definition=definition,
        )
        groups.append((definition, candidates, run_specs))

    all_specs = [run_spec for _, _, run_specs in groups for run_spec in run_specs]
    if len(all_specs) != MAX_CONCURRENT:
        raise ValueError(f"Expected {MAX_CONCURRENT} ablation rows, found {len(all_specs)}")
    if len({run_spec.run_id for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Validation run IDs are not unique")
    if len({run_spec.run_name for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Validation run names are not unique")
    return groups


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    return sweep.parse_sweep_args(
        default_candidate_weights=DEFAULT_CANDIDATE_WEIGHTS,
        expected_candidate_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        max_concurrent=MAX_CONCURRENT,
    )


def _save_local_manifests(
    groups: list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture], list[base.DelphiSwarmRunSpec]]],
    *,
    candidate_weights: Path,
    analysis_output_path: str,
) -> None:
    for definition, candidates, run_specs in groups:
        target = candidates[0].target
        sweep.save_sweep_manifest(
            sweep.SaveSweepManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / target),
                candidate_weights_path=str(candidate_weights),
                candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                source_panel=base.DEFAULT_SOURCE_PANEL,
                source_panel_sha256=base.SOURCE_PANEL_SHA256,
                analysis_output_path=analysis_output_path,
                candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
                run_specs_json=json.dumps([asdict(run_spec) for run_spec in run_specs], sort_keys=True),
                sweep_definition_json=json.dumps(asdict(definition), sort_keys=True),
            )
        )


def main() -> None:
    args, remaining = parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.argv = [sys.argv[0], *remaining]

    hardware = (args.tpu_type, args.tpu_region, args.tpu_zone)
    expected_hardware = (sweep.TPU_TYPE, sweep.TPU_REGION, sweep.TPU_ZONE)
    if hardware != expected_hardware:
        raise ValueError(f"This launcher is pinned to {expected_hardware}, got {hardware}")
    if args.max_concurrent != MAX_CONCURRENT:
        raise ValueError(f"Release all {MAX_CONCURRENT} ablation rows with --max-concurrent={MAX_CONCURRENT}")
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The ablation must use the frozen KL-policy table")
    if args.expected_candidate_sha256 != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The KL-policy table hash cannot be overridden")

    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    base.completed_adamh_heuristic = sweep.current_completed_adamh_heuristic
    source_specs = base.load_source_panel(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    groups = build_validation_run_specs(
        template=source_specs[0],
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        _save_local_manifests(
            groups,
            candidate_weights=args.candidate_weights,
            analysis_output_path=args.analysis_output_path,
        )
        logger.info("Wrote %d ablation rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for definition, candidates, run_specs in groups:
            artifacts = sweep.build_launch_artifacts(
                run_specs=run_specs,
                candidates=candidates,
                candidate_weights_path=args.candidate_weights,
                candidate_weights_sha256=args.expected_candidate_sha256,
                analysis_output_path=args.analysis_output_path,
                validation_configs=validation_configs,
                definition=definition,
            )
            steps.extend(artifacts.steps)

    if os.getenv("CI") is not None:
        logger.info("Built sixteen KL-ablation trainings and evaluations; skipping launch in CI")
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 KL ablation of the frozen procedure: eight KL coefficients per objective "
            "(Uncheatable cap 6, Table-9 cap 8) at the control data seeds"
        ),
    )


if __name__ == "__main__":
    main()
