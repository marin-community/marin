# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replicate the five best under-replicated Table-9 floor coordinates of the frozen bank at 3e18 FLOPs.

The rows come from `design_delphi_floor_replicates_20260907.py`: the five best-measured optima-stratum
coordinates with at most two existing runs, each trained at the Table-9 validation data seed (662009) with
trainer seeds 0 and 1, so that every candidate floor coordinate reaches three runs. The two tables hold the
same mixtures; the sweep loader aliases identical mixtures within a table, so each trainer seed reads its own
copy. Every run gets the inline Uncheatable evaluation and the native Table-9 evaluation of the epoch-cap sweeps.
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

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_floor_replicates_3e18_20260907"
DEFAULT_CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory/two_phase_many/reference_outputs/delphi_floor_replicates_design_20260907"
)
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights_t0.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "f4913b5f093f083115b08449289631c368291fd87d35601be8af6e9b4929e23b"
# Same mixtures at trainer seed 1; the sweep loader aliases identical rows within one table, so the second
# trainer seed reads its own copy of the table.
REPLICATE_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights_t1.csv"
EXPECTED_REPLICATE_WEIGHTS_SHA256 = "f4913b5f093f083115b08449289631c368291fd87d35601be8af6e9b4929e23b"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
TABLE9_DATA_SEED = 662_009
MAX_CONCURRENT = 10

CANDIDATE_IDS = (
    "floor_1_e8df5e6e_cap17",
    "floor_2_d3ffb3bd_cap17",
    "floor_3_a446352f_cap17",
    "floor_4_5969f449_cap17",
    "floor_5_c8a7abc2_cap17",
)


def _definition(
    *,
    group: str,
    candidate_ids: tuple[str, ...],
    trainer_seed: int,
    run_id_base: int,
) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/{group}_seed{TABLE9_DATA_SEED}_t{trainer_seed}",
        nominal_candidate_ids=candidate_ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in candidate_ids},
        expected_run_count=len(candidate_ids),
        run_id_base=run_id_base,
        common_data_seed=TABLE9_DATA_SEED,
        trainer_seed=trainer_seed,
        run_name_prefix=f"flr_{group}_seed{TABLE9_DATA_SEED}_t{trainer_seed}",
        table9_run_name_prefix=f"t9r{trainer_seed}",
        panel_source="floor_replicates",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_floor_replicates",
        provenance_panel="delphi_3e18_one_phase_floor_replicates",
        wandb_tags=("delphi-3e18", "one-phase", "floor-replicates", "table9", group),
    )


SEED0_DEFINITION = _definition(group="floor", candidate_ids=CANDIDATE_IDS, trainer_seed=0, run_id_base=7_395_000)
SEED1_DEFINITION = _definition(group="floor", candidate_ids=CANDIDATE_IDS, trainer_seed=1, run_id_base=7_395_100)
VALIDATION_GROUPS = (
    (SEED0_DEFINITION, DEFAULT_CANDIDATE_WEIGHTS, EXPECTED_CANDIDATE_WEIGHTS_SHA256),
    (SEED1_DEFINITION, REPLICATE_CANDIDATE_WEIGHTS, EXPECTED_REPLICATE_WEIGHTS_SHA256),
)
ALL_CANDIDATE_IDS = CANDIDATE_IDS


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the trainer-seed-0 and trainer-seed-1 tables, each against its own frozen hash and definition."""
    groups = []
    for definition, path, digest in VALIDATION_GROUPS:
        candidates, _ = sweep.load_candidate_mixtures(path, digest, definition=definition)
        if [candidate.candidate_id for candidate in candidates] != list(definition.nominal_candidate_ids):
            raise ValueError(f"{path} does not hold exactly {definition.experiment_name}'s rows")
        groups.append((definition, candidates))
    return groups


def build_validation_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture], list[base.DelphiSwarmRunSpec]]]:
    """Bind the five floor coordinates to the Table-9 data seed at trainer seeds 0 and 1."""
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
        raise ValueError(f"Expected {MAX_CONCURRENT} replicate rows, found {len(all_specs)}")
    if len({run_spec.run_id for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Replicate run IDs are not unique")
    if len({run_spec.run_name for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Replicate run names are not unique")
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
    analysis_output_path: str,
) -> None:
    tables = {definition.experiment_name: (path, digest) for definition, path, digest in VALIDATION_GROUPS}
    for definition, candidates, run_specs in groups:
        path, digest = tables[definition.experiment_name]
        sweep.save_sweep_manifest(
            sweep.SaveSweepManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR / definition.experiment_name.rsplit("/", 1)[-1]),
                candidate_weights_path=str(path),
                candidate_weights_sha256=digest,
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
        raise ValueError(f"Release all {MAX_CONCURRENT} replicate rows with --max-concurrent={MAX_CONCURRENT}")
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The replicates must use the frozen design table")
    if args.expected_candidate_sha256 != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The frozen design hash cannot be overridden")

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
        _save_local_manifests(groups, analysis_output_path=args.analysis_output_path)
        logger.info("Wrote %d replicate rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    tables = {definition.experiment_name: (path, digest) for definition, path, digest in VALIDATION_GROUPS}
    steps = []
    with executor_context():
        for definition, candidates, run_specs in groups:
            path, digest = tables[definition.experiment_name]
            artifacts = sweep.build_launch_artifacts(
                run_specs=run_specs,
                candidates=candidates,
                candidate_weights_path=path,
                candidate_weights_sha256=digest,
                analysis_output_path=args.analysis_output_path,
                validation_configs=validation_configs,
                definition=definition,
            )
            steps.extend(artifacts.steps)

    if os.getenv("CI") is not None:
        logger.info("Built ten floor-replicate trainings and evaluations; skipping launch in CI")
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 floor replicates: the five best under-replicated Table-9 bank coordinates at data seed "
            "662009, trainer seeds 0 and 1"
        ),
    )


if __name__ == "__main__":
    main()
