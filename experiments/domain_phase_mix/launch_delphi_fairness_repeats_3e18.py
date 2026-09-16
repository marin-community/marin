# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed-matched repeats of the frozen procedure's proposals and Olmix's best 3e18 policies.

Twelve runs at 3e18 FLOPs on the same hardware (v6e-8) with matched (data seed, trainer seed) pairs, so the two
methods can be compared as means over three seeds each. Uncheatable at data seed 666200 and Table 9 at 662009 are
the seeds of the frozen-procedure validation; trainer seed 0 adds the Olmix policies at those seeds (their original
runs used their own run-id data seeds on v5p-8), and trainer seeds 1 and 2 repeat every policy:

- Olmix: KL 0.1 for Uncheatable and KL 0.005 for Table 9, the best measured settings of the July sweep, trained on
  the exact runtime mixtures of those runs (Levanter's block quantization of their weights, see
  `olmix_quantization.md` beside the candidate table; the Table-9 mixture reaches 4.10 epochs on its largest
  bucket under that rule, hence its `cap05` suffix).
- Ours: the frozen Uncheatable proposal (cap 6, interior) and the Table-9 proposals at caps 6 and 8, copied from
  the frozen-procedure validation table.

The KL-0 controls at trainer seed 0 are the frozen-procedure validation runs
(`launch_delphi_frozen_procedure_validation_3e18.py`).
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

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_fairness_repeats_3e18_20260908"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
DEFAULT_CANDIDATE_DIR = REFERENCE_OUTPUTS / "delphi_fairness_repeats_3e18_20260908"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "a6c7264085b994cea63d48a6c9de739461732cf65a0491b43f185b2318e33b02"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
REPEAT_TRAINER_SEEDS = (1, 2)
MAX_CONCURRENT = 12

OLMIX_UNCHEATABLE = "olmix_u_kl0p1_cap04"
OLMIX_TABLE9 = "olmix_t9_kl0p005_cap05"
OURS_UNCHEATABLE = "lwspu_u_snc_cap06"
OURS_TABLE9 = ("lwspu_t9_snc_cap06", "lwspu_t9_snc_cap08")
ALL_CANDIDATE_IDS = (OLMIX_UNCHEATABLE, OLMIX_TABLE9, OURS_UNCHEATABLE, *OURS_TABLE9)


def _definition(
    *,
    target: str,
    candidate_ids: tuple[str, ...],
    data_seed: int,
    trainer_seed: int,
    run_id_base: int,
) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/{target}_seed{data_seed}_t{trainer_seed}",
        nominal_candidate_ids=candidate_ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in candidate_ids},
        expected_run_count=len(candidate_ids),
        run_id_base=run_id_base,
        common_data_seed=data_seed,
        trainer_seed=trainer_seed,
        run_name_prefix=f"lwspufr_{target}_seed{data_seed}_t{trainer_seed}",
        table9_run_name_prefix=f"t9r{trainer_seed}",
        panel_source="fairness_repeats",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_fairness_repeats",
        provenance_panel="delphi_3e18_one_phase_fairness_repeats",
        wandb_tags=(
            "delphi-3e18",
            "one-phase",
            "fairness-repeats",
            "seed-matched",
            target,
            f"trainer-seed-{trainer_seed}",
        ),
    )


def _groups() -> tuple[tuple[sweep.SweepDefinition, tuple[str, ...]], ...]:
    groups = []
    for target, data_seed, olmix, ours, run_id_base in (
        ("uncheatable", UNCHEATABLE_DATA_SEED, OLMIX_UNCHEATABLE, (OURS_UNCHEATABLE,), 7_403_000),
        ("table9", TABLE9_DATA_SEED, OLMIX_TABLE9, OURS_TABLE9, 7_403_100),
    ):
        # Trainer seed 0 already holds our KL-0 controls; only the Olmix policy is added there.
        for trainer_seed in (0, *REPEAT_TRAINER_SEEDS):
            candidate_ids = (olmix,) if trainer_seed == 0 else (olmix, *ours)
            definition = _definition(
                target=target,
                candidate_ids=candidate_ids,
                data_seed=data_seed,
                trainer_seed=trainer_seed,
                run_id_base=run_id_base + 10 * trainer_seed,
            )
            groups.append((definition, candidate_ids))
    return tuple(groups)


VALIDATION_GROUPS = _groups()
ALL_DEFINITION = _definition(target="all", candidate_ids=ALL_CANDIDATE_IDS, data_seed=0, trainer_seed=0, run_id_base=0)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the frozen candidate table and return the twelve rows grouped by (target, trainer seed)."""
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
    """Bind every policy to its target's data seed and the group's trainer seed."""
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
        raise ValueError(f"Expected {MAX_CONCURRENT} repeat rows, found {len(all_specs)}")
    if len({run_spec.run_id for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Repeat run IDs are not unique")
    if len({run_spec.run_name for run_spec in all_specs}) != len(all_specs):
        raise ValueError("Repeat run names are not unique")
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
        raise ValueError(f"Release all {MAX_CONCURRENT} repeat rows with --max-concurrent={MAX_CONCURRENT}")
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The repeats must use the frozen candidate table")
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
        logger.info("Wrote %d repeat rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
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
        logger.info("Built twelve seed-matched repeat trainings and evaluations; skipping launch in CI")
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 seed-matched repeats: Olmix best policies (KL 0.1 Uncheatable, KL 0.005 Table 9) and the "
            "frozen procedure's proposals at trainer seeds 0/1/2 on v6e-8"
        ),
    )


if __name__ == "__main__":
    main()
