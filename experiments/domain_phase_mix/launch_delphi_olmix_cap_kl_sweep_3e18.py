# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched-Olmix epoch-cap x KL sweep at 3e18 FLOPs (Qwen3 360M/1.6B) on v5p-8 in us-east5-a.

The matched Olmix policy (per-task log-linear laws fitted on the frozen Qwen swarm, exact KL-regularized proposer)
crossed with epoch caps {1, 4, 8, 12, uncapped} and the KL axis of the paper's MARINER KL table, deduplicated to 39
distinct mixtures (`materialize_delphi_olmix_cap_kl_sweep_20260926.py`; `grid_map.csv` maps all 90 cells to the run
that measures them). One run per mixture, no repeats: Uncheatable proposals at data seed 666200 and suite proposals
at data seed 662009 (the seed-matched validation design), trainer seed 0. Training runs on v5p-8 in us-east5-a, the
swarm's own hardware; the OlmoBaseEval Easy evaluations use the swarm's v6e-8 evaluation resources.

``--canary CANDIDATE_ID`` builds the full sweep and launches only that candidate's training and evaluation, with the
same run id and output path as in the full launch, so the full launch reuses it.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

ROOT_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_olmix_cap_kl_sweep_3e18_20260926"
CANDIDATE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory/two_phase_many/reference_outputs/delphi_olmix_cap_kl_sweep_3e18_20260926"
)
CANDIDATE_WEIGHTS = CANDIDATE_DIR / "candidate_weights.csv"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "c138bfe222970d4b2cc3780d576f3667d0bcedf972c0aefd7b9bb3a0d4911e8c"
LOCAL_ARTIFACT_DIR = CANDIDATE_DIR / "launch_dry_run"
HARDWARE = (base.TARGET_TPU_TYPE, base.DEFAULT_TPU_REGION, base.DEFAULT_TPU_ZONE)
UNCHEATABLE_DATA_SEED = 666_200
TABLE9_DATA_SEED = 662_009
TRAINER_SEED = 0
UNCHEATABLE_RUN_ID_BASE = 7_500_000
TABLE9_RUN_ID_BASE = 7_500_100
TOTAL_RUNS = 39


def candidate_ids() -> tuple[str, ...]:
    summary = json.loads((CANDIDATE_DIR / "summary.json").read_text())
    return tuple(summary["candidate_ids"])


def _definition(target: str, ids: tuple[str, ...], data_seed: int, run_id_base: int) -> sweep.SweepDefinition:
    return sweep.SweepDefinition(
        experiment_name=f"{ROOT_EXPERIMENT_NAME}/{target}_seed{data_seed}_t{TRAINER_SEED}",
        nominal_candidate_ids=ids,
        expected_alias_map={candidate_id: candidate_id for candidate_id in ids},
        expected_run_count=len(ids),
        run_id_base=run_id_base,
        common_data_seed=data_seed,
        trainer_seed=TRAINER_SEED,
        run_name_prefix=f"olmixsw_{target}_seed{data_seed}_t{TRAINER_SEED}",
        table9_run_name_prefix="t9ok",
        panel_source="olmix_cap_kl_sweep",
        table9_wandb_group="olmo_base_eval_table9_delphi_3e18_olmix_cap_kl_sweep",
        provenance_panel="delphi_3e18_olmix_cap_kl_sweep",
        wandb_tags=("delphi-3e18", "one-phase", "olmix-cap-kl-sweep", "seed-matched", target, "v5p-8"),
    )


def groups() -> list[tuple[sweep.SweepDefinition, list[sweep.CandidateMixture]]]:
    ids = candidate_ids()
    everything = _definition("all", ids, 0, 0)
    candidates, _ = sweep.load_candidate_mixtures(
        CANDIDATE_WEIGHTS, EXPECTED_CANDIDATE_WEIGHTS_SHA256, definition=everything
    )
    by_id = {c.candidate_id: c for c in candidates}
    result = []
    for target, prefix, seed, run_id_base in (
        ("uncheatable", "olmixq_u_", UNCHEATABLE_DATA_SEED, UNCHEATABLE_RUN_ID_BASE),
        ("table9", "olmixq_t9_", TABLE9_DATA_SEED, TABLE9_RUN_ID_BASE),
    ):
        selected = tuple(i for i in ids if i.startswith(prefix))
        result.append((_definition(target, selected, seed, run_id_base), [by_id[i] for i in selected]))
    if sum(len(c) for _, c in result) != TOTAL_RUNS:
        raise ValueError(f"Expected {TOTAL_RUNS} runs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", help="launch only this candidate")
    parser.add_argument("--max-concurrent", type=int, default=TOTAL_RUNS)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args, remaining = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    sys.argv = [sys.argv[0], *remaining]
    tpu_type, tpu_region, tpu_zone = HARDWARE
    expected_prefix = marin_prefix_for_region(tpu_region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    base.completed_adamh_heuristic = sweep.current_completed_adamh_heuristic
    template = base.load_source_panel(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=args.analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )[0]
    plan = []
    for definition, candidates in groups():
        specs = sweep.build_run_specs(
            template=template,
            candidates=candidates,
            tpu_type=tpu_type,
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            definition=definition,
        )
        if args.canary:
            keep = [i for i, c in enumerate(candidates) if c.candidate_id == args.canary]
            candidates, specs = [candidates[i] for i in keep], [specs[i] for i in keep]
        if candidates:
            plan.append((definition, candidates, specs))
    all_specs = [s for _, _, specs in plan for s in specs]
    if args.canary and len(all_specs) != 1:
        raise ValueError(f"Unknown canary candidate: {args.canary}")
    if len({s.run_id for s in all_specs}) != len(all_specs) or len({s.run_name for s in all_specs}) != len(all_specs):
        raise ValueError("Run ids or names are not unique")
    if args.dry_run:
        for definition, candidates, specs in plan:
            sweep.save_sweep_manifest(
                sweep.SaveSweepManifestConfig(
                    output_path=str(LOCAL_ARTIFACT_DIR / definition.experiment_name.rsplit("/", 1)[-1]),
                    candidate_weights_path=str(CANDIDATE_WEIGHTS),
                    candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                    source_panel=base.DEFAULT_SOURCE_PANEL,
                    source_panel_sha256=base.SOURCE_PANEL_SHA256,
                    analysis_output_path=args.analysis_output_path,
                    candidates_json=json.dumps([asdict(c) for c in candidates], sort_keys=True),
                    run_specs_json=json.dumps([asdict(s) for s in specs], sort_keys=True),
                    sweep_definition_json=json.dumps(asdict(definition), sort_keys=True),
                )
            )
        logger.info("Dry run: %d runs %s", len(all_specs), [(s.run_id, s.run_name, s.tpu_type) for s in all_specs][:3])
        return

    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False)
        for name, step in base._default_validation_sets(tokenizer=llama3_tokenizer).items()
    }
    steps = []
    with executor_context():
        for definition, candidates, specs in plan:
            if args.canary:
                # Only the count checks change; training and evaluation steps keep their full-launch identities.
                definition = replace(
                    definition,
                    nominal_candidate_ids=(args.canary,),
                    expected_alias_map={args.canary: args.canary},
                    expected_run_count=1,
                )
            steps.extend(
                sweep.build_launch_artifacts(
                    run_specs=specs,
                    candidates=candidates,
                    candidate_weights_path=CANDIDATE_WEIGHTS,
                    candidate_weights_sha256=EXPECTED_CANDIDATE_WEIGHTS_SHA256,
                    analysis_output_path=args.analysis_output_path,
                    validation_configs=validation_configs,
                    definition=definition,
                ).steps
            )
    if os.getenv("CI") is not None:
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"Delphi 3e18 matched-Olmix cap x KL sweep on {tpu_type} ({len(all_specs)} runs)",
    )


if __name__ == "__main__":
    main()
