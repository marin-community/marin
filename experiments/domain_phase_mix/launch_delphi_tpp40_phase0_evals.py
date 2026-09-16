# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill exact TPP40 boundary validation and Table 9 without restarting training.

Use the frozen regional assignment and persisted training configs, not a rebuilt
training graph. With --watch, a CPU parent picks up newly saved checkpoints;
no accelerator is allocated until a permanent boundary checkpoint is ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, replace

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.main import eval_lm, export_lm_to_hf
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig, initialize
from levanter.utils.mesh import MeshConfig
from marin.evaluation.olmo_base_eval.run import RESULTS_FILENAME, olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_tpp40 as tpp40
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge
from experiments.domain_phase_mix.delphi_tpp40_evaluation_identity import table9_request_set_identity
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_tpp40_phase0_evals_20260907"
ASSIGNMENT_SHA256 = "8074b0d3a92e5e002336389849f33bbd630d9be2ea1580ccf436dfb2b40ea836"
ASSIGNMENT_RELATIVE_PATH = "experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v2.json"
CHECKPOINT_STEP = tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP
RESULT_FILE = "phase0_validation_result.json"


@dataclass(frozen=True)
class BoundaryCheckpoint:
    """Persisted training identity of one ready, assigned boundary checkpoint."""

    training_output_path: str
    analysis_output_path: str
    run_spec: base.DelphiSwarmRunSpec
    metadata_sha256: str

    @property
    def checkpoint_path(self) -> str:
        return f"{self.training_output_path}/checkpoints/step-{CHECKPOINT_STEP}"


@dataclass(frozen=True)
class BoundaryEvalConfig:
    checkpoint: BoundaryCheckpoint
    validation_configs: dict[str, DatasetComponent]
    output_path: str


def ready_checkpoints(root: str, assigned_orders: tuple[int, ...]) -> list[BoundaryCheckpoint]:
    """Discover ready checkpoints, rejecting ambiguous or inconsistent identities."""
    fs, path = fsspec.core.url_to_fs(root)
    found: dict[int, BoundaryCheckpoint] = {}
    for run_path in sorted(fs.ls(path, detail=False)):
        if not run_path.rsplit("/", 1)[-1].startswith("fit_"):
            continue
        training_path = fs.unstrip_protocol(run_path)
        order = int(training_path.rsplit("/fit_", 1)[1].split("_", 1)[0])
        if order not in assigned_orders:
            continue
        metadata = bridge._checkpoint_metadata(
            f"{training_path}/checkpoints/step-{CHECKPOINT_STEP}", expected_step=CHECKPOINT_STEP
        )
        if metadata is None:
            continue
        info = bridge._read_json(f"{training_path}/.executor_info")
        config = info["config"]
        if not isinstance(config, dict):
            raise ValueError(f"Training config is not an object: {training_path}")
        spec = base.DelphiSwarmRunSpec(**config["run_spec"])
        if config["output_path"] != training_path or spec.run_order != order:
            raise ValueError(f"Training identity disagrees with output path: {training_path}")
        if spec.train_steps != tpp40.EXPECTED_FINAL_CHECKPOINT_STEP + 1:
            raise ValueError(f"Unexpected TPP40 training horizon: {training_path}")
        if tpp40._phase_0_checkpoint_step(spec.train_steps)[0] != CHECKPOINT_STEP:
            raise ValueError(f"Unexpected TPP40 phase boundary: {training_path}")
        if order in found:
            raise ValueError(f"Multiple boundary checkpoints for assigned row {order}")
        found[order] = BoundaryCheckpoint(training_path, config["analysis_output_path"], spec, metadata[1])
    return [found[order] for order in sorted(found)]


def validation_result(raw: dict, config: BoundaryEvalConfig) -> dict:
    """Require every requested component before publishing a boundary result."""
    for name in config.validation_configs:
        key = f"eval/{name}/bpb"
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Missing or invalid validation metric: {key}")
    return {
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_path": config.checkpoint.checkpoint_path,
        "checkpoint_metadata_sha256": config.checkpoint.metadata_sha256,
        "training_output_path": config.checkpoint.training_output_path,
        "run_order": config.checkpoint.run_spec.run_order,
        "data_seed": config.checkpoint.run_spec.data_seed,
        "trainer_seed": config.checkpoint.run_spec.trainer_seed,
        "eval_batch_size": bridge.EVAL_BATCH_SIZE,
        "validation_caches": {name: component.cache_dir for name, component in config.validation_configs.items()},
        "metrics": raw,
    }


def completed_validation_result(config: BoundaryEvalConfig) -> dict | None:
    """Reuse complete metrics only for the identical checkpoint and evaluation."""
    path = f"{config.output_path}/{RESULT_FILE}"
    fs, fs_path = fsspec.core.url_to_fs(path)
    if not fs.exists(fs_path):
        return None
    result = bridge._read_json(path)
    if result != validation_result(result["metrics"], config):
        raise ValueError(f"Saved boundary validation identity does not match: {path}")
    return result


def run_boundary_validation(config: BoundaryEvalConfig) -> None:
    """Evaluate and export the exact saved model; never enter a training loop."""
    checkpoint = config.checkpoint
    metadata = bridge._checkpoint_metadata(checkpoint.checkpoint_path, expected_step=CHECKPOINT_STEP)
    if metadata is None or metadata[1] != checkpoint.metadata_sha256:
        raise ValueError(f"Boundary checkpoint changed or disappeared: {checkpoint.checkpoint_path}")
    candidate = base._candidate_for_run_spec(
        scaling_fits=base._read_scaling_fits(checkpoint.analysis_output_path), run_spec=checkpoint.run_spec
    )
    model = candidate.model_config
    if (
        model.total_trainable_params(base.completed_adamh_heuristic.vocab_size)
        != checkpoint.run_spec.total_trainable_params
    ):
        raise ValueError("Resolved architecture does not match the persisted training config")
    trainer = TrainerConfig(
        tracker=JsonFileTrackerConfig(output_path=config.output_path),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=bridge.EVAL_BATCH_SIZE,
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}),
        seed=checkpoint.run_spec.trainer_seed,
        allow_nondivisible_batch_size=True,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    if completed_validation_result(config) is None:
        eval_lm.main(
            eval_lm.EvalLmConfig(
                checkpoint_path=checkpoint.checkpoint_path,
                trainer=trainer,
                data=LmDataConfig(
                    tokenizer=llama3_tokenizer,
                    cache_dir=None,
                    auto_build_caches=False,
                    components=config.validation_configs,
                    train_weights={name: 0.0 for name in config.validation_configs},
                ),
                max_eval_length=base.SEQ_LEN_DELPHI,
                model=model,
            )
        )
        raw = bridge._read_json(f"{config.output_path}/{bridge.RAW_RESULT_FILE}")
        bridge._write_json(f"{config.output_path}/{RESULT_FILE}", validation_result(raw, config))
    else:
        logger.info("Reusing completed boundary validation at %s", config.output_path)
        # Export still needs distributed initialization, but must not overwrite saved metrics on exit.
        trainer = replace(trainer, tracker=NoopConfig())
        initialize(trainer)
    export_lm_to_hf.main(
        export_lm_to_hf.ConvertLmConfig(
            trainer=trainer,
            checkpoint_path=checkpoint.checkpoint_path,
            output_dir=f"{config.output_path}/hf/step-{CHECKPOINT_STEP}",
            model=model,
            tokenizer=llama3_tokenizer,
        )
    )


def build_steps(
    checkpoints: list[BoundaryCheckpoint], validation_configs: dict[str, DatasetComponent], *, side: str
) -> list[ExecutorStep]:
    deployment = bridge.BRIDGE_SIDES[side]
    prefix = marin_prefix_for_region(deployment.region)
    resources = ResourceConfig.with_tpu(
        bridge.EVALUATOR_TPU_TYPE, regions=[deployment.region], zone=deployment.evaluator_zone, disk="80g"
    )
    steps = []
    with executor_context():
        for checkpoint in checkpoints:
            name = f"tpp40_phase0_{side}_{checkpoint.run_spec.run_name}"
            validation = ExecutorStep(
                name=f"{EXPERIMENT_NAME}/{side}/{checkpoint.run_spec.run_name}",
                fn=remote(
                    run_boundary_validation,
                    resources=resources,
                    env_vars={"MARIN_PREFIX": prefix, base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                ),
                resources=resources,
                config=BoundaryEvalConfig(checkpoint, validation_configs, this_output_path()),
            )
            steps.append(
                olmo_base_eval_step(
                    name=name,
                    checkpoint=validation / f"hf/step-{CHECKPOINT_STEP}",
                    request_set_dir=base.TABLE9_REQUEST_SET_DIR,
                    resource_config=resources,
                    wandb_group="delphi_tpp40_phase0_table9_20260907",
                    provenance={
                        "panel": "delphi_tpp40_augmented_fit_swarm",
                        "checkpoint_step": str(CHECKPOINT_STEP),
                        "source_checkpoint": checkpoint.checkpoint_path,
                        "source_run_name": checkpoint.run_spec.source_run_name,
                        "swarm_run_name": checkpoint.run_spec.run_name,
                        "region": deployment.region,
                    },
                )
            )
    return steps


def result_paths(steps: list[ExecutorStep], prefix: str) -> dict[int, str]:
    resolver = Executor(prefix=prefix, executor_info_base_path=f"{prefix}/experiments")
    with executor_context():
        for step in steps:
            resolver.compute_version(step, is_pseudo_dep=False)
    return {
        int(step.config.eval_config.provenance["swarm_run_name"].split("_")[1]): resolver.output_paths[step]
        for step in steps
    }


def require_complete_table9(paths: dict[int, str]) -> None:
    for order, path in paths.items():
        result = bridge._read_json(f"{path}/{RESULTS_FILENAME}")
        macro = result.get("table9_macro_bpb")
        components = result.get("table9_components", {})
        if (
            not isinstance(macro, int | float)
            or not math.isfinite(macro)
            or len(components) != 51
            or any(not math.isfinite(value) for value in components.values())
        ):
            raise ValueError(f"Row {order} lacks a complete Table-9 macro: {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=tuple(bridge.BRIDGE_SIDES), required=True)
    parser.add_argument("--run-orders", default="all")
    parser.add_argument("--canary-order", type=int, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-path")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("Polling interval must be positive")
    deployment = bridge.BRIDGE_SIDES[args.side]
    prefix = bridge._set_region_prefix(deployment)
    assigned, audit = tpp40._assignment_orders(
        f"{prefix}/{ASSIGNMENT_RELATIVE_PATH}",
        args.side,
        tpu_region=deployment.region,
        experiment_name=tpp40.EXPERIMENT_NAME,
        expected_assignment_sha256=ASSIGNMENT_SHA256,
    )
    requested = assigned if args.run_orders == "all" else tpp40._parse_run_orders(args.run_orders)
    if not set(requested) <= set(assigned) or args.canary_order not in requested:
        raise ValueError("Requested rows and canary must belong to the frozen regional assignment")
    with executor_context():
        validation_configs, _ = bridge._validation_configs()
    for component in validation_configs.values():
        if not component.cache_dir.startswith(prefix + "/"):
            raise ValueError(f"Validation cache is not region-local: {component.cache_dir}")
        if not fsspec.open(f"{component.cache_dir}/validation/.stats.json").fs.exists(
            f"{component.cache_dir}/validation/.stats.json"
        ):
            raise FileNotFoundError(f"Validation cache is not ready: {component.cache_dir}")
    request_identity = table9_request_set_identity(f"{prefix}/raw/eval-datasets/olmo_base_eval_table9/v2")
    root = f"{prefix}/{tpp40.EXPERIMENT_NAME}"
    canary_passed = False
    completed: set[int] = set()
    while True:
        checkpoints = ready_checkpoints(root, requested)
        for checkpoint in checkpoints:
            tpp40._require_regional_input_path(
                checkpoint.analysis_output_path, region=deployment.region, label="analysis"
            )
            if checkpoint.run_spec.tpu_region != deployment.region:
                raise ValueError(f"Saved training region disagrees with assignment: {checkpoint.training_output_path}")
        ready = {checkpoint.run_spec.run_order for checkpoint in checkpoints}
        selected = (
            checkpoints
            if canary_passed or args.dry_run
            else [checkpoint for checkpoint in checkpoints if checkpoint.run_spec.run_order == args.canary_order]
        )
        steps = build_steps(selected, validation_configs, side=args.side)
        paths = result_paths(steps, prefix)
        if not canary_passed and not steps and not args.watch and not args.dry_run:
            raise FileNotFoundError(f"Canary row {args.canary_order} has no ready boundary checkpoint")
        manifest = {
            **audit,
            "checkpoint_step": CHECKPOINT_STEP,
            "requested_orders": list(requested),
            "ready_orders": sorted(ready),
            "selected_orders": sorted(paths),
            "completed_orders": sorted(completed),
            "table9_output_paths": paths,
            "table9_request_identity": request_identity,
            "canary_order": args.canary_order,
            "canary_passed": canary_passed,
            "max_concurrent": 2 * len(steps),
        }
        if args.manifest_path:
            bridge._write_json(args.manifest_path, manifest)
        logger.info(
            "Boundary evaluations: %s",
            json.dumps(
                {
                    "ready": len(ready),
                    "requested": len(requested),
                    "completed": len(completed),
                    "released": len(steps),
                    "canary_passed": canary_passed,
                }
            ),
        )
        if args.dry_run:
            return
        if steps:
            executor_main(
                ExecutorMainConfig(prefix=prefix, max_concurrent=2 * len(steps)),
                steps=steps,
                description=f"TPP40 {args.side}: exact step-{CHECKPOINT_STEP} validation and Table 9",
            )
            require_complete_table9(paths)
            completed.update(paths)
            manifest["completed_orders"] = sorted(completed)
            manifest["canary_passed"] = True
            if args.manifest_path:
                bridge._write_json(args.manifest_path, manifest)
            if not canary_passed:
                canary_passed = True
                continue
        if completed == set(requested) or not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
