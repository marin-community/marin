# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Confirm selected Delphi 3e18 WSPU policies at their OLMix comparator seeds."""

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
from experiments.domain_phase_mix import launch_delphi_one_phase_weibull_softplus_epoch_cap_sweep_3e18 as wspu
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

ROOT_EXPERIMENT_NAME = (
    "pinlin_calvin_xu/data_mixture/delphi_one_phase_weibull_softplus_olmix_seed_confirmation_3e18_20260903"
)
LOCAL_ARTIFACT_DIR = wspu.DEFAULT_CANDIDATE_DIR / "olmix_seed_confirmation_launch_dry_run"
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
TRAINER_SEED = 0
MAX_CONCURRENT = 4

UNCHEATABLE_CANDIDATE_IDS = ("wspu_uncheatable_cap06",)
TABLE9_CANDIDATE_IDS = tuple(f"wspu_table9_cap{cap:02d}" for cap in (6, 7, 8))


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
        run_name_prefix=f"wspu_{target}_seed{data_seed}",
        table9_run_name_prefix=f"t9w{target[0]}",
        panel_source="weibull_softplus_unscaled_olmix_seed_confirmation",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_wspu_olmix_seed_confirmation",
        provenance_panel="delphi_3e18_one_phase_wspu_olmix_seed_confirmation",
        wandb_tags=(
            "delphi-3e18",
            "one-phase",
            "weibull-softplus-unscaled",
            "olmix-seed-confirmation",
            target,
        ),
    )


UNCHEATABLE_DEFINITION = _definition(
    target="uncheatable",
    candidate_ids=UNCHEATABLE_CANDIDATE_IDS,
    data_seed=UNCHEATABLE_DATA_SEED,
    run_id_base=7_360_000,
)
TABLE9_DEFINITION = _definition(
    target="table9",
    candidate_ids=TABLE9_CANDIDATE_IDS,
    data_seed=TABLE9_DATA_SEED,
    run_id_base=7_360_100,
)
CONFIRMATION_GROUPS = (
    (UNCHEATABLE_DEFINITION, UNCHEATABLE_CANDIDATE_IDS),
    (TABLE9_DEFINITION, TABLE9_CANDIDATE_IDS),
)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the frozen WSPU table and return only the requested confirmation rows."""
    candidates, _ = sweep.load_candidate_mixtures(
        wspu.DEFAULT_CANDIDATE_WEIGHTS,
        wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=wspu.SWEEP_DEFINITION,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    groups = []
    for definition, candidate_ids in CONFIRMATION_GROUPS:
        selected = [by_id[candidate_id] for candidate_id in candidate_ids]
        if len(selected) != definition.expected_run_count:
            raise ValueError(f"Expected {definition.expected_run_count} {definition.experiment_name} candidates")
        groups.append((definition, selected))
    return groups


def build_confirmation_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture], list[base.DelphiSwarmRunSpec]]]:
    """Bind each target's frozen policies to its comparator's data seed."""
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
        raise ValueError(f"Expected {MAX_CONCURRENT} confirmation rows, found {len(all_specs)}")
    if len({run_spec.run_id for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Confirmation run IDs are not unique")
    if len({run_spec.run_name for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Confirmation run names are not unique")
    return groups


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    return sweep.parse_sweep_args(
        default_candidate_weights=wspu.DEFAULT_CANDIDATE_WEIGHTS,
        expected_candidate_sha256=wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
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
                candidate_weights_sha256=wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
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
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    if args.candidate_weights != wspu.DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The confirmation must use the frozen WSPU candidate table")
    if args.expected_candidate_sha256 != wspu.EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The frozen WSPU candidate hash cannot be overridden")

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
    groups = build_confirmation_run_specs(
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
        logger.info("Wrote %d confirmation rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
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
        logger.info("Built four WSPU confirmation trainings and evaluations; skipping launch in CI")
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 WSPU confirmation: Uncheatable cap 6 and Table-9 caps 6-8 "
            "at their presented OLMix comparator data seeds"
        ),
    )


if __name__ == "__main__":
    main()
