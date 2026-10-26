# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the pretraining arms of the SFT midtraining probe at Delphi 3e18 (2026-10-14).

Twenty-one runs on v6e-8 in us-east5-b at MARINER's Table-9 data seed 662009 and trainer seeds 0-2, so every arm
pairs with the existing ``lwspu_t9_snc_cap08`` runs (arm A): three two-phase candidates for arm B (B1-B3), their
shared single-phase average C, and the Dolmino-late schedule at A's aggregate (BP), and (added 2026-10-14 10:40 PT)
the code-late and math-late recipes at TV 0.20 at the anchor (CL, ML). The frozen panel is
``reference_outputs/sft_midtraining_probe_20261014/arms_source_panel.csv`` from
``materialize_sft_probe_arms_20261014.py``; every run gets the native Table-9 evaluation.

Release policy: submit the canary first (``--run-only sftp_c_t0``), then the rest (``--run-only '^(?!.*sftp_c_t0$)'``
is not needed: completed steps are skipped, so the second parent may simply name every other run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_sft_probe_arms_3e18_20261014"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "sft_midtraining_probe_20261014"
DEFAULT_SOURCE_PANEL = PANEL_DIR / "arms_source_panel.csv"
SOURCE_PANEL_SHA256 = "752b62d1f517f466f5b724c88fb2218691b9584abb255fe89b8bfdfaf050faf5"
LOCAL_ARTIFACT_DIR = PANEL_DIR / "launch_dry_run"
EXPECTED_RUNS = 21
EXPECTED_ARM_COUNTS = {"B1": 3, "B2": 3, "B3": 3, "C": 3, "BP": 3, "CL": 3, "ML": 3}
DATA_SEED = 662_009
TPU_TYPE = "v6e-8"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-b"
# Fallback pools in the checkpoints' region when the v6e-8 pool churns (2026-10-14 14:05 PT: B' runs restarted 10-14
# times by failing v6e-8 workers). Hardware is recorded in each run spec; the step identity (name + versioned fields)
# does not include it, so a resubmission on another pool resumes the same output path from its checkpoints.
TPU_POOLS = {"v6e-8": "us-east5-b", "v6e-4": "us-east5-b", "v5p-8": "us-east5-a"}
TABLE9_WANDB_GROUP = "olmo_base_eval_table9_delphi_3e18_sft_probe_arms_20261014"
TRAINING_WANDB_GROUP = "delphi_3e18_sft_probe_arms_20261014"


def load_panel(path: Path) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_PANEL_SHA256:
        raise ValueError(f"Arm panel SHA-256 changed: {digest} != {SOURCE_PANEL_SHA256}")
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} rows, found {len(frame)}")
    if frame["arm"].value_counts().to_dict() != EXPECTED_ARM_COUNTS:
        raise ValueError(f"Arm counts changed: {frame['arm'].value_counts().to_dict()}")
    if frame["candidate_id"].duplicated().any() or frame["run_id"].duplicated().any():
        raise ValueError("Duplicate run identities in the arm panel")
    if (frame["data_seed"] != DATA_SEED).any():
        raise ValueError("Every arm must use MARINER's Table-9 data seed")
    return frame


def build_run_specs(
    frame: pd.DataFrame, *, template: base.DelphiSwarmRunSpec, tpu_type: str, tpu_region: str, tpu_zone: str
) -> list[base.DelphiSwarmRunSpec]:
    tensor_parallel_size = base._tensor_parallel_size(template.model_hidden_dim, tpu_type)
    run_specs = []
    for run_order, row in enumerate(frame.to_dict(orient="records")):
        row_strings = {key: str(value) for key, value in row.items()}
        phase_weights = base._phase_weights_from_row(row_strings, source_run_name=row["candidate_id"])
        max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
        run_specs.append(
            replace(
                template,
                run_order=run_order,
                run_id=int(row["run_id"]),
                run_name=str(row["candidate_id"]),
                source_run_name=str(row["source_candidate"]),
                source_experiment=EXPERIMENT_NAME,
                panel_source=f"sft_probe_arm_{row['arm']}",
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
            )
        )
    return run_specs


def build_launch_artifacts(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    frame: pd.DataFrame,
    analysis_output_path: str,
    validation_configs,
    max_retries_failure: int,
    max_task_failures: int,
    table9_eval_resources: ResourceConfig = base.TABLE9_EVAL_RESOURCES,
) -> base.LaunchArtifacts:
    arms = dict(zip(frame["candidate_id"], frame["arm"], strict=True))
    base_artifacts = base.build_launch_artifacts(
        run_specs=run_specs,
        analysis_output_path=analysis_output_path,
        source_panel=str(DEFAULT_SOURCE_PANEL),
        validation_configs=validation_configs,
        experiment_name=EXPERIMENT_NAME,
        wandb_tags=("delphi-3e18", "sft-midtraining-probe", "arms", "seed-matched"),
        training_wandb_group=TRAINING_WANDB_GROUP,
        table9_wandb_group=TABLE9_WANDB_GROUP,
        provenance_panel="delphi_3e18_sft_probe_arms_20261014",
        provenance_scale="3e18",
        steps_per_eval=1000,
        permanent_checkpoint_interval=5000,
        source_panel_sha256=SOURCE_PANEL_SHA256,
    )
    training_steps = []
    eval_steps = []
    for original_step, run_spec in zip(base_artifacts.training_steps, run_specs, strict=True):
        if original_step.resources is None:
            raise ValueError(f"Training step {original_step.name} has no resource configuration")
        training_step = replace(
            original_step,
            fn=remote(
                sweep.run_one_phase_training,  # installs the current AdamH adapter, as arm A's launcher did
                resources=original_step.resources,
                env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                max_retries_failure=max_retries_failure,
                max_task_failures=max_task_failures,
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                resource_config=table9_eval_resources,
                wandb_group=TABLE9_WANDB_GROUP,
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_sft_probe_arms_20261014",
                    "scale": "3e18",
                    "arm": arms[run_spec.run_name],
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )
    return base.LaunchArtifacts(base_artifacts.manifest_step, training_steps, eval_steps)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-type", default=TPU_TYPE, choices=sorted(TPU_POOLS))
    parser.add_argument("--tpu-region", default=TPU_REGION)
    parser.add_argument("--tpu-zone", default=None, help="Defaults to the pool's zone")
    parser.add_argument(
        "--eval-tpu-type", default=TPU_TYPE, choices=sorted(TPU_POOLS), help="Pool for the Table-9 evals"
    )
    parser.add_argument("--max-concurrent", type=int, default=EXPECTED_RUNS)
    parser.add_argument("--max-retries-failure", type=int, default=4)
    parser.add_argument("--max-task-failures", type=int, default=4)
    parser.add_argument("--run-only", action="append", default=None, help="Regex on step names; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_zone is None:
        args.tpu_zone = TPU_POOLS[args.tpu_type]
    if args.tpu_region != TPU_REGION or args.tpu_zone != TPU_POOLS[args.tpu_type]:
        raise ValueError(
            f"{args.tpu_type} runs in {TPU_REGION}/{TPU_POOLS[args.tpu_type]}, got {(args.tpu_region, args.tpu_zone)}"
        )
    eval_resources = ResourceConfig.with_tpu(
        args.eval_tpu_type, regions=[TPU_REGION], zone=TPU_POOLS[args.eval_tpu_type], disk="80g"
    )
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    frame = load_panel(args.source_panel)
    base.completed_adamh_heuristic = sweep.current_completed_adamh_heuristic
    source_specs = base.load_source_panel(
        source_panel=base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    run_specs = build_run_specs(
        frame, template=source_specs[0], tpu_type=args.tpu_type, tpu_region=args.tpu_region, tpu_zone=args.tpu_zone
    )
    if args.dry_run:
        LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_ARTIFACT_DIR / "run_specs.json").write_text(
            json.dumps([asdict(spec) for spec in run_specs], indent=2, sort_keys=True) + "\n"
        )
        logger.info("Wrote %d run specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            frame=frame,
            analysis_output_path=args.analysis_output_path,
            validation_configs=validation_configs,
            max_retries_failure=args.max_retries_failure,
            max_task_failures=args.max_task_failures,
            table9_eval_resources=eval_resources,
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d arm trainings; skipping launch in CI", len(artifacts.training_steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent, run_only=args.run_only),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: SFT midtraining probe arms (B1-B3, C, B') at Delphi 3e18 with Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
