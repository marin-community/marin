# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate four frozen official-RegMix endpoints and their MARINER blends.

Each objective has one newly proposed endpoint and one 50% blend with its
previously trained MARINER endpoint. Both use trainer seed zero and the
objective's existing data seed. The candidates preserve the offline replay's
runtime counts. The cap64 metadata is inactive; training has no epoch cap.
Inline Uncheatable and dependent native OlmoBaseEval Easy evaluate each run.
Stable executor paths resume checkpoints and reuse completed outputs.
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

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_regmix_reference_3e18_20260913"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
DEFAULT_CANDIDATE_DIR = REFERENCE_OUTPUTS / "regmix_official_rerun_20260913"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "7c4b8b005b5a8d7eaaa19278d236591a00873f7e3ff3da5ff5031270f91c3899"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
TRAINER_SEED = 0
UNCHEATABLE_RUN_ID_BASE = 7_470_000
TABLE9_RUN_ID_BASE = 7_470_100
MAX_CONCURRENT = 4

UNCHEATABLE_CANDIDATE_IDS = ("rgref_u_endpoint_cap64", "rgref_u_midpoint_cap64")
TABLE9_CANDIDATE_IDS = ("rgref_t9_endpoint_cap64", "rgref_t9_midpoint_cap64")
ALL_CANDIDATE_IDS = (*UNCHEATABLE_CANDIDATE_IDS, *TABLE9_CANDIDATE_IDS)


def _definition(
    *,
    target: str,
    candidate_ids: tuple[str, ...],
    data_seed: int,
    run_id_base: int,
) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/{target}_seed{data_seed}_t{TRAINER_SEED}",
        nominal_candidate_ids=candidate_ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in candidate_ids},
        expected_run_count=len(candidate_ids),
        run_id_base=run_id_base,
        common_data_seed=data_seed,
        trainer_seed=TRAINER_SEED,
        run_name_prefix=f"rgref_{target}_seed{data_seed}_t{TRAINER_SEED}",
        table9_run_name_prefix=f"t9rgref_{target}",
        panel_source="regmix_reference_validation",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_regmix_reference",
        provenance_panel="delphi_3e18_regmix_reference",
        wandb_tags=("delphi-3e18", "one-phase", "regmix-reference", "seed-matched", target, "trainer-seed-0"),
    )


VALIDATION_DEFINITIONS = (
    _definition(
        target="uncheatable",
        candidate_ids=UNCHEATABLE_CANDIDATE_IDS,
        data_seed=UNCHEATABLE_DATA_SEED,
        run_id_base=UNCHEATABLE_RUN_ID_BASE,
    ),
    _definition(
        target="table9",
        candidate_ids=TABLE9_CANDIDATE_IDS,
        data_seed=TABLE9_DATA_SEED,
        run_id_base=TABLE9_RUN_ID_BASE,
    ),
)
ALL_DEFINITION = _definition(target="all", candidate_ids=ALL_CANDIDATE_IDS, data_seed=0, run_id_base=0)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Verify the frozen endpoints and blends and group them by objective's matched data seed."""
    candidates, _ = sweep.load_candidate_mixtures(
        DEFAULT_CANDIDATE_WEIGHTS,
        EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=ALL_DEFINITION,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        (definition, [by_id[candidate_id] for candidate_id in definition.nominal_candidate_ids])
        for definition in VALIDATION_DEFINITIONS
    ]


def build_validation_run_specs(
    *,
    template: base.DelphiSwarmRunSpec,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture], list[base.DelphiSwarmRunSpec]]]:
    """Keep the endpoint training recipe while replacing only mixture and identity."""
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
    all_specs = [spec for _, _, specs in groups for spec in specs]
    if len(all_specs) != MAX_CONCURRENT:
        raise ValueError(f"Expected {MAX_CONCURRENT} validation runs, found {len(all_specs)}")
    if len({spec.run_id for spec in all_specs}) != len(all_specs):
        raise ValueError("Run IDs are not unique")
    if len({spec.run_name for spec in all_specs}) != len(all_specs):
        raise ValueError("Run names are not unique")
    return groups


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", default=EXPECTED_CANDIDATE_WEIGHTS_SHA256)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-type", default=sweep.TPU_TYPE)
    parser.add_argument("--tpu-region", default=sweep.TPU_REGION)
    parser.add_argument("--tpu-zone", default=sweep.TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.argv = [sys.argv[0], *remaining]

    hardware = (args.tpu_type, args.tpu_region, args.tpu_zone)
    expected_hardware = (sweep.TPU_TYPE, sweep.TPU_REGION, sweep.TPU_ZONE)
    if hardware != expected_hardware:
        raise ValueError(f"This launcher is pinned to {expected_hardware}, got {hardware}")
    if args.max_concurrent != MAX_CONCURRENT:
        raise ValueError(
            f"Release all {MAX_CONCURRENT} reference validation runs with --max-concurrent={MAX_CONCURRENT}"
        )
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("Validation must use the frozen RegMix candidate table")
    if args.expected_candidate_sha256 != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The frozen RegMix candidate hash cannot be overridden")
    if args.analysis_output_path != base.DEFAULT_ANALYSIS_OUTPUT_PATH:
        raise ValueError("Validation must use the endpoint architecture and optimizer calibration")
    if sweep.MIXTURE_BLOCK_SIZE != 2048 or base.MIXTURE_BLOCK_SIZE != 2048:
        raise ValueError("Validation must retain the endpoint 1/2048 mixture sampler")

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
        for definition, candidates, run_specs in groups:
            sweep.save_sweep_manifest(
                sweep.SaveSweepManifestConfig(
                    output_path=str(LOCAL_ARTIFACT_DIR / definition.experiment_name.rsplit("/", 1)[-1]),
                    candidate_weights_path=str(args.candidate_weights),
                    candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                    source_panel=base.DEFAULT_SOURCE_PANEL,
                    source_panel_sha256=base.SOURCE_PANEL_SHA256,
                    analysis_output_path=args.analysis_output_path,
                    candidates_json=json.dumps([asdict(candidate) for candidate in candidates], sort_keys=True),
                    run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
                    sweep_definition_json=json.dumps(asdict(definition), sort_keys=True),
                )
            )
        logger.info("Wrote %d RegMix validation rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
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
        logger.info("Built %d RegMix train/eval pairs; skipping launch in CI", MAX_CONCURRENT)
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Four official-RegMix validations at 3e18: one endpoint and one MARINER blend per objective, "
            "trainer seed zero, Uncheatable data seed 666200 and suite data seed 662009; v6e-8 in us-east5-b"
        ),
    )


if __name__ == "__main__":
    main()
