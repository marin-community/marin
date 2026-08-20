# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the canonical Delphi 3e18 fit panel through the phase-0 boundary.

The replay is rooted at the exact clean commit used by the July 2026 full
swarm. It preserves every state-affecting input through update 2400, including
the original 3007-step optimizer schedule, while Levanter's forced final hooks
retain and evaluate the boundary state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

ORIGINAL_CODE_COMMIT = "a12bd2d96e648dfc75be2347d45aa0fcb41968b9"
ORIGINAL_IRIS_JOB = "/calvinxu/dm-delphi-3e18-augmented-swarm-20260714-retry1"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_replay_20260820"
LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_3e18_phase0_prefix_replay_20260820"
    / "launch_dry_run"
)

MIXTURE_BLOCK_SIZE = 2048
DEFAULT_MAX_CONCURRENT = 56
EXPECTED_FULL_TRAIN_STEPS = 3007
EXPECTED_PREFIX_TRAIN_STEPS = 2400
EXPECTED_PREFIX_HF_STEP = 2399
EXPECTED_FULL_TRAIN_TOKENS = 1_576_534_016
EXPECTED_PREFIX_TRAIN_TOKENS = 1_258_291_200
EXPECTED_SOURCE_COORDINATE_HASH = "4db8e7f70dda72f2bc8a04fa4b8271f1a5959aa27e7119ad4f62f951cd1b2864"


@dataclass(frozen=True)
class PrefixTrainingConfig:
    """Runtime configuration for one exact phase-0 replay."""

    analysis_output_path: str
    output_path: str
    run_spec: base.DelphiSwarmRunSpec
    validation_configs: dict[str, DatasetComponent] | None
    prefix_train_steps: int
    optimizer_schedule_num_train_steps: int
    replay_code_commit: str


@dataclass(frozen=True)
class SavePrefixManifestConfig:
    """Configuration for the immutable phase-0 replay manifest."""

    output_path: str
    source_panel: str
    analysis_output_path: str
    run_specs_json: str
    launch_audit_json: str
    replay_code_commit: str


@dataclass(frozen=True)
class PrefixLaunchArtifacts:
    """Resolved manifest, prefix-training, and boundary-evaluation graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def phase_0_boundary(full_train_steps: int, batch_size: int) -> tuple[int, int]:
    """Return completed phase-0 updates and the corresponding exported step."""
    prefix_train_steps = base.PHASE_SCHEDULE.phases[1].get_start_step_aligned(
        full_train_steps,
        batch_size,
        MIXTURE_BLOCK_SIZE,
    )
    if prefix_train_steps <= 0 or prefix_train_steps >= full_train_steps:
        raise ValueError(f"Invalid phase-0 boundary {prefix_train_steps} for {full_train_steps} steps")
    return prefix_train_steps, prefix_train_steps - 1


def _source_coordinate_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = [
        {
            "source_run_name": spec.source_run_name,
            "panel_source": spec.panel_source,
            "phase_weights": spec.phase_weights,
        }
        for spec in run_specs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_replay_specs(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[base.DelphiSwarmRunSpec], dict[str, Any]]:
    """Load the canonical panel and freeze exact boundary accounting."""
    run_specs = base.load_source_panel(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    if run_specs[0].run_order != 0 or run_specs[0].source_run_name != "baseline_proportional":
        raise ValueError(f"Canonical canary row changed: {run_specs[0].source_run_name}")
    coordinate_hash = _source_coordinate_hash(run_specs)
    if coordinate_hash != EXPECTED_SOURCE_COORDINATE_HASH:
        raise ValueError(f"Unexpected source coordinate hash: {coordinate_hash}")

    full_train_steps = {spec.train_steps for spec in run_specs}
    full_train_tokens = {spec.realized_train_tokens for spec in run_specs}
    if full_train_steps != {EXPECTED_FULL_TRAIN_STEPS}:
        raise ValueError(f"Unexpected full training horizons: {sorted(full_train_steps)}")
    if full_train_tokens != {EXPECTED_FULL_TRAIN_TOKENS}:
        raise ValueError(f"Unexpected full token budgets: {sorted(full_train_tokens)}")

    boundaries = {phase_0_boundary(spec.train_steps, spec.batch_size) for spec in run_specs}
    if boundaries != {(EXPECTED_PREFIX_TRAIN_STEPS, EXPECTED_PREFIX_HF_STEP)}:
        raise ValueError(f"Unexpected phase-0 boundaries: {sorted(boundaries)}")

    prefix_train_tokens = EXPECTED_PREFIX_TRAIN_STEPS * base.TARGET_BATCH_SIZE * base.SEQ_LEN_DELPHI
    if prefix_train_tokens != EXPECTED_PREFIX_TRAIN_TOKENS:
        raise ValueError(f"Unexpected phase-0 token budget: {prefix_train_tokens}")

    audit: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "original_code_commit": ORIGINAL_CODE_COMMIT,
        "original_iris_job": ORIGINAL_IRIS_JOB,
        "source_panel": source_panel,
        "source_panel_sha256": base.SOURCE_PANEL_SHA256,
        "source_coordinate_hash": coordinate_hash,
        "run_count": len(run_specs),
        "full_train_steps": EXPECTED_FULL_TRAIN_STEPS,
        "full_expected_hf_step": EXPECTED_FULL_TRAIN_STEPS - 1,
        "full_train_tokens": EXPECTED_FULL_TRAIN_TOKENS,
        "prefix_train_steps": EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_expected_hf_step": EXPECTED_PREFIX_HF_STEP,
        "prefix_trainer_state_step": EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_train_tokens": EXPECTED_PREFIX_TRAIN_TOKENS,
        "realized_prefix_fraction": EXPECTED_PREFIX_TRAIN_STEPS / EXPECTED_FULL_TRAIN_STEPS,
        "optimizer_schedule_num_train_steps": EXPECTED_FULL_TRAIN_STEPS,
        "steps_per_eval": 1000,
        "checkpoint_keep_policy": [{"every": 5000}],
        "smooth_boundary_eval_scheduled": True,
        "native_table9_boundary_eval_scheduled": True,
        "full_trainer_state_retained": True,
        "identity_contract": {
            "preserved": [
                "source panel bytes and all 280 coordinates",
                "phase-0 weights and full-horizon data schedule",
                "model architecture and initialization seed",
                "data seed and data-loader implementation",
                "optimizer configuration and 3007-step schedule horizon",
                "batch size, sequence length, precision, mesh, TPU type, region, and zone",
                "all training/eval/checkpoint hooks before update 2400",
            ],
            "allowed_differences": [
                "execution stops after update 2400",
                "output paths and W&B tags identify the replay",
                "forced final hooks retain and evaluate the boundary state after update 2400",
            ],
        },
    }
    return run_specs, audit


def _prefix_train_config(
    *,
    run_spec: base.DelphiSwarmRunSpec,
    candidate,
    validation_configs: dict[str, DatasetComponent] | None,
    replay_code_commit: str,
) -> train_lm.TrainLmConfig:
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    tensor_parallel_size = base._tensor_parallel_size(candidate.model_config.hidden_dim, run_spec.tpu_type)

    data = base._build_mixture_data(run_spec)
    data = base._add_validation_components(data, validation_configs)
    config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    "issue-6611",
                    "delphi-3e18-augmented-swarm",
                    "phase0-prefix-replay",
                    f"replay_code_commit={replay_code_commit}",
                    "fit-panel",
                    "completed-adamh",
                    f"panel_source={run_spec.panel_source}",
                    f"source_run={run_spec.source_run_name}",
                    f"FLOPs={run_spec.target_flops:.1e}",
                    f"D={run_spec.realized_train_tokens:.1e}",
                    f"D/N={run_spec.realized_train_tokens / params:.3f}",
                    f"label={base.LABEL}",
                    f"N={params:.1e}",
                    f"data_seed={run_spec.data_seed}",
                    f"trainer_seed={run_spec.trainer_seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=run_spec.batch_size,
            per_device_parallelism=-1,
            num_train_steps=EXPECTED_PREFIX_TRAIN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tensor_parallel_size},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=run_spec.trainer_seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=base.SEQ_LEN_DELPHI,
        model=candidate.model_config,
        optimizer=candidate.optimizer_config,
        optimizer_schedule_num_train_steps=EXPECTED_FULL_TRAIN_STEPS,
        data_seed=run_spec.data_seed,
    )
    if config.trainer.profiler.is_enabled:
        raise ValueError("Profiler must remain disabled for bitwise phase-0 replay")
    return config


def run_phase_0_prefix(config: PrefixTrainingConfig) -> None:
    """Train one canonical row to the exact phase-0 boundary."""
    run_spec = config.run_spec
    expected_prefix_steps, _ = phase_0_boundary(run_spec.train_steps, run_spec.batch_size)
    if config.prefix_train_steps != expected_prefix_steps:
        raise ValueError(f"Prefix horizon changed: {config.prefix_train_steps} != {expected_prefix_steps}")
    if config.optimizer_schedule_num_train_steps != run_spec.train_steps:
        raise ValueError(
            "Optimizer horizon must remain the original full horizon: "
            f"{config.optimizer_schedule_num_train_steps} != {run_spec.train_steps}"
        )

    scaling_fits = base._read_scaling_fits(config.analysis_output_path)
    candidate = base._candidate_for_budget(scaling_fits=scaling_fits)
    if candidate.train_steps != run_spec.train_steps:
        raise ValueError(f"Resolved train steps changed: {candidate.train_steps} != {run_spec.train_steps}")
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    if int(params) != run_spec.total_trainable_params:
        raise ValueError(f"Resolved parameter count changed: {int(params)} != {run_spec.total_trainable_params}")

    inner_config = _prefix_train_config(
        run_spec=run_spec,
        candidate=candidate,
        validation_configs=config.validation_configs,
        replay_code_commit=config.replay_code_commit,
    )
    if inner_config.trainer.num_train_steps != config.prefix_train_steps:
        raise ValueError("Prefix trainer horizon changed after config construction")
    if inner_config.optimizer_schedule_num_train_steps != config.optimizer_schedule_num_train_steps:
        raise ValueError("Optimizer schedule horizon changed after config construction")

    resources = ResourceConfig.with_tpu(
        run_spec.tpu_type,
        regions=[run_spec.tpu_region],
        zone=run_spec.tpu_zone,
    )
    run_levanter_train_lm(
        TrainLmOnPodConfig(
            train_config=inner_config,
            resources=resources,
            output_path=config.output_path,
            env_vars={
                "GIT_COMMIT": config.replay_code_commit,
                "MARIN_PREFIX": marin_prefix_for_region(run_spec.tpu_region),
                SKIP_EVAL_HARNESS_ENV_VAR: "1",
            },
        )
    )


def save_prefix_manifest(config: SavePrefixManifestConfig) -> None:
    """Persist source identities and exact phase-boundary semantics."""
    run_specs = [base.DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    launch_audit = json.loads(config.launch_audit_json)
    if launch_audit.get("replay_code_commit") != config.replay_code_commit:
        raise ValueError("Replay code commit differs between manifest config and launch audit")
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    fields = [
        "run_order",
        "run_name",
        "source_run_name",
        "source_experiment",
        "panel_source",
        "data_seed",
        "trainer_seed",
        "full_train_steps",
        "full_train_tokens",
        "prefix_train_steps",
        "prefix_train_tokens",
        "prefix_hf_step",
        "prefix_trainer_state_step",
        "optimizer_schedule_num_train_steps",
        "replay_code_commit",
        "tpu_type",
        "tpu_region",
        "tpu_zone",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for spec in run_specs:
        writer.writerow(
            {
                "run_order": spec.run_order,
                "run_name": spec.run_name,
                "source_run_name": spec.source_run_name,
                "source_experiment": spec.source_experiment,
                "panel_source": spec.panel_source,
                "data_seed": spec.data_seed,
                "trainer_seed": spec.trainer_seed,
                "full_train_steps": spec.train_steps,
                "full_train_tokens": spec.realized_train_tokens,
                "prefix_train_steps": EXPECTED_PREFIX_TRAIN_STEPS,
                "prefix_train_tokens": EXPECTED_PREFIX_TRAIN_TOKENS,
                "prefix_hf_step": EXPECTED_PREFIX_HF_STEP,
                "prefix_trainer_state_step": EXPECTED_PREFIX_TRAIN_STEPS,
                "optimizer_schedule_num_train_steps": spec.train_steps,
                "replay_code_commit": config.replay_code_commit,
                "tpu_type": spec.tpu_type,
                "tpu_region": spec.tpu_region,
                "tpu_zone": spec.tpu_zone,
            }
        )
    with fs.open(os.path.join(config.output_path, "prefix_training_manifest.csv"), "w") as handle:
        handle.write(buffer.getvalue())
    with fs.open(os.path.join(config.output_path, "source_run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)
    with fs.open(os.path.join(config.output_path, "launch_audit.json"), "w") as handle:
        json.dump(launch_audit, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[base.DelphiSwarmRunSpec],
    analysis_output_path: str,
    source_panel: str,
    validation_configs: dict[str, DatasetComponent],
    launch_audit: dict[str, Any],
) -> PrefixLaunchArtifacts:
    """Build the idempotent 280-prefix plus 280-Table-9 graph."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
            fn=remote(
                run_phase_0_prefix,
                resources=resources,
                env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=PrefixTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=run_spec,
                validation_configs=validation_configs,
                prefix_train_steps=EXPECTED_PREFIX_TRAIN_STEPS,
                optimizer_schedule_num_train_steps=EXPECTED_FULL_TRAIN_STEPS,
                replay_code_commit=launch_audit["replay_code_commit"],
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_boundary_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{EXPECTED_PREFIX_HF_STEP}",
                request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                resource_config=base.TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_3e18_phase0_prefix_replay",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_280row_phase0_prefix_replay",
                    "scale": "3e18",
                    "temporal_position": "phase_0_boundary",
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_prefix_manifest,
        config=SavePrefixManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            launch_audit_json=json.dumps(launch_audit, sort_keys=True),
            replay_code_commit=launch_audit["replay_code_commit"],
        ),
    )
    return PrefixLaunchArtifacts(manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps)


def _write_local_dry_run(
    *,
    source_panel: str,
    analysis_output_path: str,
    run_specs: list[base.DelphiSwarmRunSpec],
    launch_audit: dict[str, Any],
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_prefix_manifest(
        SavePrefixManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            source_panel=source_panel,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            launch_audit_json=json.dumps(launch_audit, sort_keys=True),
            replay_code_commit=launch_audit["replay_code_commit"],
        )
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=base.DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--run-order", type=int, default=None, help="Launch one canonical row as an idempotent canary")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {base.DEFAULT_TPU_REGION}/{base.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")

    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs, launch_audit = load_replay_specs(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    replay_code_commit = get_git_commit()
    if replay_code_commit is None or len(replay_code_commit) != 40:
        raise ValueError(f"Could not resolve a full replay code commit: {replay_code_commit!r}")
    launch_audit["replay_code_commit"] = replay_code_commit
    launch_audit["profiler_enabled"] = False
    if args.run_order is not None:
        if not 0 <= args.run_order < len(run_specs):
            raise ValueError(f"--run-order must be in [0, {len(run_specs) - 1}]")
        run_specs = [run_specs[args.run_order]]
        launch_audit["selected_run_orders"] = [args.run_order]
    else:
        launch_audit["selected_run_orders"] = list(range(len(run_specs)))
    launch_audit["selected_run_count"] = len(run_specs)
    launch_audit["max_concurrent"] = args.max_concurrent
    if args.dry_run:
        _write_local_dry_run(
            source_panel=args.source_panel,
            analysis_output_path=args.analysis_output_path,
            run_specs=run_specs,
            launch_audit=launch_audit,
        )
        logger.info("Wrote %d phase-0 replay specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=args.analysis_output_path,
            source_panel=str(args.source_panel),
            validation_configs=validation_configs,
            launch_audit=launch_audit,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built phase-0 replay graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: exact phase-0 replay of the canonical 280-row Delphi 3e18 panel "
            "with retained boundary state and boundary Uncheatable/Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
