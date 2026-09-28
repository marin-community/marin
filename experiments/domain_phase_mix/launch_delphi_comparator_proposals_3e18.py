# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comparator proposals at 3e18 FLOPs: mixtures optimized under the complexity ladder's alternative surrogates.

The offline ladder (Appendix "Conventional nonlinear comparators") scores prediction and selection among measured
runs; this validation measures optimization quality. Every comparator is fitted on the frozen Qwen3 3e18 swarm and
optimized without a cap or penalty (`materialize_delphi_comparator_proposals_20260909.py`): the additive quadratic
in log-epochs under MARINER's floor and log link (the prespecified primary comparator, three trainer seeds per
objective), the natural cubic spline under the same link, RegMix's gradient-boosted trees with RegMix's top-k
sampling proposal, Hellinger kernel ridge, and MARINER with the benefit power fixed at one (single runs each; the
power-one Uncheatable proposal coincides with the frozen procedure's mixture and is not retrained). Thirteen runs.
Uncheatable proposals train at data seed 666200 and suite proposals at data seed 662009, the seed design of the
seed-matched repeats, so every run is paired with the frozen procedure's own validation and repeats.
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

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_comparator_proposals_3e18_20260909"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
DEFAULT_CANDIDATE_DIR = REFERENCE_OUTPUTS / "delphi_comparator_proposals_3e18_20260909"
DEFAULT_CANDIDATE_WEIGHTS = DEFAULT_CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "6bfa57119f3bd334ff6db9912ec397ddbda1d0c612b36cc09f4bc06f854660fc"
LOCAL_ARTIFACT_DIR = DEFAULT_CANDIDATE_DIR / "launch_dry_run"
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
TRAINER_SEEDS = (0, 1, 2)

UNCHEATABLE_QUADRATIC = "cmp_u_quad_cap08"
UNCHEATABLE_SPLINE = "cmp_u_spline_cap08"
UNCHEATABLE_LIGHTGBM = "cmp_u_lgbm_cap08"
UNCHEATABLE_KRR = "cmp_u_krr_cap12"
UNCHEATABLE_MARINER_POWER1 = "cmp_u_mk1_cap06"
TABLE9_QUADRATIC = "cmp_t9_quad_cap12"
TABLE9_SPLINE = "cmp_t9_spline_cap12"
TABLE9_LIGHTGBM = "cmp_t9_lgbm_cap12"
TABLE9_KRR = "cmp_t9_krr_cap16"
TABLE9_MARINER_POWER1 = "cmp_t9_mk1_cap08"
# The candidate table's frozen order (the loader checks it).
ALL_CANDIDATE_IDS = (
    UNCHEATABLE_QUADRATIC,
    UNCHEATABLE_SPLINE,
    UNCHEATABLE_LIGHTGBM,
    UNCHEATABLE_KRR,
    UNCHEATABLE_MARINER_POWER1,
    TABLE9_QUADRATIC,
    TABLE9_SPLINE,
    TABLE9_LIGHTGBM,
    TABLE9_KRR,
    TABLE9_MARINER_POWER1,
)
# The primary comparator gets three trainer seeds; the other comparators get one run each. The Uncheatable
# power-one proposal is the frozen procedure's own mixture (every Uncheatable task already selects power one), so
# it stays in the table for the record and is not trained again.
UNCHEATABLE_GROUPS = {
    0: (UNCHEATABLE_QUADRATIC, UNCHEATABLE_SPLINE, UNCHEATABLE_LIGHTGBM, UNCHEATABLE_KRR),
    1: (UNCHEATABLE_QUADRATIC,),
    2: (UNCHEATABLE_QUADRATIC,),
}
TABLE9_GROUPS = {
    0: (TABLE9_QUADRATIC, TABLE9_SPLINE, TABLE9_LIGHTGBM, TABLE9_KRR, TABLE9_MARINER_POWER1),
    1: (TABLE9_QUADRATIC,),
    2: (TABLE9_QUADRATIC,),
}
MAX_CONCURRENT = 13
UNCHEATABLE_RUN_ID_BASE = 7_440_000
TABLE9_RUN_ID_BASE = 7_440_100


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
        run_name_prefix=f"cmp_{target}_seed{data_seed}_t{trainer_seed}",
        table9_run_name_prefix=f"t9c{trainer_seed}",
        panel_source="comparator_proposals",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_one_phase_comparator_proposals",
        provenance_panel="delphi_3e18_one_phase_comparator_proposals",
        wandb_tags=(
            "delphi-3e18",
            "one-phase",
            "comparator-proposals",
            "seed-matched",
            target,
            f"trainer-seed-{trainer_seed}",
        ),
    )


def _groups() -> tuple[tuple[sweep.SweepDefinition, tuple[str, ...]], ...]:
    groups = []
    for target, data_seed, plan, run_id_base in (
        ("uncheatable", UNCHEATABLE_DATA_SEED, UNCHEATABLE_GROUPS, UNCHEATABLE_RUN_ID_BASE),
        ("table9", TABLE9_DATA_SEED, TABLE9_GROUPS, TABLE9_RUN_ID_BASE),
    ):
        for trainer_seed in TRAINER_SEEDS:
            candidate_ids = plan[trainer_seed]
            groups.append(
                (
                    _definition(
                        target=target,
                        candidate_ids=candidate_ids,
                        data_seed=data_seed,
                        trainer_seed=trainer_seed,
                        run_id_base=run_id_base + 10 * trainer_seed,
                    ),
                    candidate_ids,
                )
            )
    return tuple(groups)


VALIDATION_GROUPS = _groups()
ALL_DEFINITION = _definition(target="all", candidate_ids=ALL_CANDIDATE_IDS, data_seed=0, trainer_seed=0, run_id_base=0)


def selected_candidates() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    """Load the frozen candidate table and return the rows grouped by objective and trainer seed."""
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
    """Bind each proposal to its objective's matched data seed and trainer seeds."""
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
        raise ValueError(f"Expected {MAX_CONCURRENT} validation rows, found {len(all_specs)}")
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
        raise ValueError(f"Release all {MAX_CONCURRENT} validation rows with --max-concurrent={MAX_CONCURRENT}")
    if args.candidate_weights != DEFAULT_CANDIDATE_WEIGHTS:
        raise ValueError("The validation must use the frozen comparator candidate table")
    if args.expected_candidate_sha256 != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError("The frozen comparator candidate hash cannot be overridden")

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
        logger.info("Wrote %d validation rows under %s", MAX_CONCURRENT, LOCAL_ARTIFACT_DIR)
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
        logger.info("Built %d comparator-proposal trainings and evaluations; skipping launch in CI", MAX_CONCURRENT)
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=(
            "Delphi 3e18 comparator proposals: quadratic (floor link, three trainer seeds), spline, RegMix trees, "
            "Hellinger kernel ridge and power-one MARINER optima; Uncheatable at data seed 666200, suite at 662009, "
            "on v6e-8"
        ),
    )


if __name__ == "__main__":
    main()
