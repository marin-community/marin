# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MARINER optima for higher-SNR subsets of OlmoBaseEval Easy, trained at 3e18 FLOPs with three trainer seeds.

Two proposals from `analyze_table9_snr_subset_optima_20260928.py`: the reference procedure refitted and optimized
without a cap over the components whose swarm standard deviation exceeds five (32 of 51) or eight (17 of 51) times
the proportional repeat SD. Both train at the OlmoBaseEval Easy data seed (662009) and trainer seeds 0, 1 and 2, so
they pair with MARINER's full-objective runs at the same seeds; the native evaluation scores every component, so
each run reports the full mean and both subset means. Six runs on v6e-8. Candidate table:
`delphi_table9_snr_subset_3e18_20260928/candidate_weights.csv`.
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

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_table9_snr_subset_3e18_20260928"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
DEFAULT_CANDIDATE_DIR = REFERENCE_OUTPUTS / "delphi_table9_snr_subset_3e18_20260928"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "265f07e192a038a55c1ddab3efa856cec3d30f30733c2682df217c3479616f14"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
TABLE9_DATA_SEED = 662_009
TRAINER_SEEDS = (0, 1, 2)
CANDIDATE_IDS = ("snr5_t9_cap09", "snr8_t9_cap07")
RUN_ID_BASE = 7_520_100
MAX_CONCURRENT = 6


def _definition(*, candidate_ids: tuple[str, ...], trainer_seed: int, run_id_base: int) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/table9_seed{TABLE9_DATA_SEED}_t{trainer_seed}",
        nominal_candidate_ids=candidate_ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in candidate_ids},
        expected_run_count=len(candidate_ids),
        run_id_base=run_id_base,
        common_data_seed=TABLE9_DATA_SEED,
        trainer_seed=trainer_seed,
        run_name_prefix=f"snrsub_table9_seed{TABLE9_DATA_SEED}_t{trainer_seed}",
        table9_run_name_prefix=f"t9snr{trainer_seed}",
        panel_source="table9_snr_subset",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_table9_snr_subset",
        provenance_panel="delphi_3e18_one_phase_table9_snr_subset",
        wandb_tags=(
            "delphi-3e18",
            "one-phase",
            "table9-snr-subset",
            "seed-matched",
            "table9",
            f"trainer-seed-{trainer_seed}",
        ),
    )


VALIDATION_GROUPS = tuple(
    (_definition(candidate_ids=CANDIDATE_IDS, trainer_seed=seed, run_id_base=RUN_ID_BASE + 10 * seed), CANDIDATE_IDS)
    for seed in TRAINER_SEEDS
)
ALL_DEFINITION = _definition(candidate_ids=CANDIDATE_IDS, trainer_seed=0, run_id_base=0)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the frozen candidate table and return the rows grouped by trainer seed."""
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
    """Bind both proposals to the OlmoBaseEval Easy data seed and each trainer seed."""
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
        raise ValueError(f"Expected {MAX_CONCURRENT} rows, found {len(all_specs)}")
    if len({run_spec.run_id for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Run IDs are not unique")
    if len({run_spec.run_name for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Run names are not unique")
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
        sweep.save_sweep_manifest(
            sweep.SaveSweepManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / definition.experiment_name.rsplit("/", 1)[-1]),
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
        raise ValueError(f"Release all {MAX_CONCURRENT} rows with --max-concurrent={MAX_CONCURRENT}")
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The validation must use the frozen candidate table")
    if args.expected_candidate_sha256 != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The frozen candidate hash cannot be overridden")

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
        logger.info("Wrote %d rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
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
        logger.info("Built %d trainings and evaluations; skipping launch in CI", MAX_CONCURRENT)
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 OlmoBaseEval Easy SNR-subset optima (SNR >= 5 and >= 8) at data seed 662009, trainer seeds "
            "0, 1 and 2, on v6e-8"
        ),
    )


if __name__ == "__main__":
    main()
