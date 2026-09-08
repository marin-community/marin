# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the frozen TPP40 bridge checkpoints on English Uncheatable.

This sidecar reconstructs the original content-addressed training outputs for
the frozen bridge row, then evaluates the exact phase-boundary and final
Orbax checkpoints. East5 and Europe both use one v6e-8 evaluator so the paired
comparison isolates the training deployment rather than evaluation hardware.

The launcher is ready-only and idempotent: each invocation schedules only
materialized checkpoints whose result step has not already succeeded. A strict
invocation after training completes can require both row/checkpoint cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.main import eval_lm
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.context import executor_context
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_tpp40 as tpp40
from experiments.domain_phase_mix.delphi_tpp40_evaluation_identity import validation_payload_identity
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import executor_status_succeeded
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = (
    SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "delphi_tpp40_europe_readiness_20260830"
)
ACCEPTANCE_CONTRACT_PATH = REFERENCE_DIR / "bridge_acceptance_contract_v4.json"
EVALUATOR_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_tpp40_bridge_uncheatable_v1_20260830"
BRIDGE_RUN_ORDERS = (2,)
CHECKPOINT_STEPS = (tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP, tpp40.EXPECTED_FINAL_CHECKPOINT_STEP)
EXPECTED_CONTRACT_SHA256 = "f0441b8927e3e7d32bbdbe781ed3008dbb46a1cd98ff540661423e850ee936df"
EXPECTED_UNCHEATABLE_NAMES = tuple(f"uncheatable_eval/{name}" for name in UNCHEATABLE_SUBSETS)
EVALUATOR_TPU_TYPE = "v6e-8"
EVAL_BATCH_SIZE = 128
RESULT_FILE = "bridge_result.json"
RAW_RESULT_FILE = "eval_results.json"


@dataclass(frozen=True)
class BridgeSide:
    """Frozen training and evaluation deployment for one bridge side."""

    name: str
    training_tpu_type: str
    region: str
    training_zone: str
    evaluator_zone: str
    training_experiment_name: str
    training_wandb_group: str
    table9_wandb_group: str


BRIDGE_SIDES = {
    "east5": BridgeSide(
        name="east5",
        training_tpu_type="v5p-8",
        region="us-east5",
        training_zone="us-east5-a",
        evaluator_zone="us-east5-b",
        training_experiment_name=tpp40.EXPERIMENT_NAME,
        training_wandb_group=tpp40.DEFAULT_TRAINING_WANDB_GROUP,
        table9_wandb_group=tpp40.DEFAULT_TABLE9_WANDB_GROUP,
    ),
    "europe": BridgeSide(
        name="europe",
        training_tpu_type="v6e-8",
        region="europe-west4",
        training_zone="europe-west4-a",
        evaluator_zone="europe-west4-a",
        training_experiment_name=(
            "pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_europe_v6e_bridge_v2_20260830"
        ),
        training_wandb_group="delphi_tpp40_europe_v6e_bridge_v2_20260830",
        table9_wandb_group="olmo_base_eval_table9_delphi_tpp40_europe_v6e_bridge_v2_20260830",
    ),
}


@dataclass(frozen=True)
class BridgeUncheatableEvalConfig:
    """Runtime inputs for one exact-checkpoint evaluation."""

    side: str
    analysis_output_path: str
    run_spec: base.DelphiSwarmRunSpec
    checkpoint_step: int
    checkpoint_path: str
    validation_configs: dict[str, DatasetComponent]
    validation_payload_sha256: str
    output_path: str


@dataclass(frozen=True)
class EvalRecord:
    """Resolved identity and readiness for one bridge evaluation cell."""

    side: str
    run_order: int
    run_name: str
    checkpoint_step: int
    training_output_path: str
    checkpoint_path: str
    checkpoint_ready: bool
    eval_output_path: str
    eval_already_succeeded: bool
    validation_payload_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_acceptance_contract() -> dict[str, Any]:
    encoded = ACCEPTANCE_CONTRACT_PATH.read_bytes()
    observed_sha256 = _sha256_bytes(encoded)
    if observed_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"Bridge acceptance contract changed: {observed_sha256} != {EXPECTED_CONTRACT_SHA256}")
    contract = json.loads(encoded)
    bridge = contract["bridge"]
    if tuple(bridge["run_orders"]) != BRIDGE_RUN_ORDERS:
        raise ValueError(f"Frozen bridge rows changed: {bridge['run_orders']}")
    run_orders_sha256 = _sha256_bytes(
        json.dumps(list(BRIDGE_RUN_ORDERS), sort_keys=True, separators=(",", ":")).encode()
    )
    if bridge.get("run_orders_sha256") != run_orders_sha256:
        raise ValueError("Frozen bridge run-order digest changed")
    if (bridge["phase_0_checkpoint_step"], bridge["endpoint_checkpoint_step"]) != CHECKPOINT_STEPS:
        raise ValueError("Frozen bridge checkpoint steps changed")
    return contract


def _set_region_prefix(side: BridgeSide) -> str:
    prefix = marin_prefix_for_region(side.region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {prefix!r}")
    os.environ["MARIN_PREFIX"] = prefix
    return prefix


def _regional_inputs(side: BridgeSide) -> tuple[str, str]:
    source_panel = tpp40._regional_input_path(base.DEFAULT_SOURCE_PANEL, region=side.region)
    analysis_output_path = tpp40._regional_input_path(base.DEFAULT_ANALYSIS_OUTPUT_PATH, region=side.region)
    return source_panel, analysis_output_path


def _run_specs(side: BridgeSide) -> tuple[list[base.DelphiSwarmRunSpec], str, str]:
    source_panel, analysis_output_path = _regional_inputs(side)
    all_specs, audit = tpp40.build_run_specs(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_type=side.training_tpu_type,
        tpu_region=side.region,
        tpu_zone=side.training_zone,
    )
    contract_bridge = _load_acceptance_contract()["bridge"]
    for field in (
        "source_panel_sha256",
        "source_coordinate_hash",
        "fixed_identity_hash",
        "scientific_identity_hash",
    ):
        if audit.get(field) != contract_bridge.get(field):
            raise ValueError(f"Bridge {field} changed: {audit.get(field)} != {contract_bridge.get(field)}")
    train_steps = {spec.train_steps for spec in all_specs}
    if len(train_steps) != 1:
        raise ValueError(f"Bridge rows have inconsistent training horizons: {sorted(train_steps)}")
    train_step_count = train_steps.pop()
    phase_0_checkpoint_step, _ = tpp40._phase_0_checkpoint_step(train_step_count)
    if phase_0_checkpoint_step != tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP:
        raise ValueError(
            f"Phase-0 checkpoint changed: {phase_0_checkpoint_step} != {tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP}"
        )
    final_checkpoint_steps = {spec.expected_checkpoint_step for spec in all_specs}
    if final_checkpoint_steps != {tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}:
        raise ValueError(
            "Final checkpoint changed: " f"{sorted(final_checkpoint_steps)} != {[tpp40.EXPECTED_FINAL_CHECKPOINT_STEP]}"
        )
    selected = [all_specs[run_order] for run_order in BRIDGE_RUN_ORDERS]
    if tuple(spec.run_order for spec in selected) != BRIDGE_RUN_ORDERS:
        raise ValueError("Bridge run-order indexing changed")
    return selected, source_panel, analysis_output_path


def _validation_configs() -> tuple[dict[str, DatasetComponent], dict[str, DatasetComponent]]:
    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    full: dict[str, DatasetComponent] = {}
    for name, step in validation_steps.items():
        component = step_to_lm_mixture_component(step, include_raw_paths=False)
        if not isinstance(component, DatasetComponent):
            raise TypeError(f"Validation component {name!r} has unsupported type {type(component).__name__}")
        full[name] = component
    uncheatable = {name: full[name] for name in EXPECTED_UNCHEATABLE_NAMES}
    if tuple(uncheatable) != EXPECTED_UNCHEATABLE_NAMES:
        raise ValueError("English Uncheatable component order changed")
    return full, uncheatable


def _require_uncheatable_caches(
    validation_configs: dict[str, DatasetComponent],
    *,
    side: BridgeSide,
) -> tuple[str, ...]:
    required_prefix = marin_prefix_for_region(side.region).rstrip("/") + "/"
    fs = fsspec.filesystem("gcs")
    paths: list[str] = []
    for name in EXPECTED_UNCHEATABLE_NAMES:
        component = validation_configs[name]
        cache_dir = component.cache_dir
        if not cache_dir.startswith(required_prefix):
            raise ValueError(f"Uncheatable cache {name!r} is not region-local: {cache_dir}")
        status_path = f"{cache_dir.rstrip('/')}/.executor_status"
        try:
            with fs.open(status_path, "rt") as handle:
                status = handle.read()
        except FileNotFoundError as error:
            raise ValueError(f"Uncheatable cache {name!r} lacks executor status: {cache_dir}") from error
        if not executor_status_succeeded(status):
            raise ValueError(f"Uncheatable cache {name!r} is incomplete: {cache_dir}")
        if not fs.exists(f"{cache_dir.rstrip('/')}/validation/.stats.json"):
            raise ValueError(f"Uncheatable cache {name!r} lacks validation/.stats.json: {cache_dir}")
        paths.append(cache_dir)
    if len(paths) != len(EXPECTED_UNCHEATABLE_NAMES) or len(set(paths)) != len(paths):
        raise ValueError("Expected seven distinct English Uncheatable caches")
    return tuple(paths)


def _original_training_paths(
    *,
    side: BridgeSide,
    run_specs: list[base.DelphiSwarmRunSpec],
    source_panel: str,
    analysis_output_path: str,
    full_validation_configs: dict[str, DatasetComponent],
    prefix: str,
) -> list[str]:
    table9_resources = ResourceConfig.with_tpu(
        EVALUATOR_TPU_TYPE,
        regions=[side.region],
        zone=side.evaluator_zone,
        disk="80g",
    )
    with executor_context():
        artifacts = base.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=analysis_output_path,
            source_panel=source_panel,
            validation_configs=full_validation_configs,
            experiment_name=side.training_experiment_name,
            architecture_target_flops=base.TARGET_FLOPS,
            wandb_tags=(
                "delphi-tpp40-augmented-swarm",
                "architecture=3e18-selected",
                "total-tpp=40",
                "fit-panel",
                "two-phase",
                f"deployment={side.region}-{side.training_tpu_type}",
            ),
            training_wandb_group=side.training_wandb_group,
            table9_wandb_group=side.table9_wandb_group,
            provenance_panel="delphi_tpp40_augmented_fit_swarm",
            provenance_scale="fixed_n_total_tpp40",
            steps_per_eval=tpp40.STEPS_PER_EVAL,
            permanent_checkpoint_interval=tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP,
            table9_eval_resources=table9_resources,
        )
    resolver = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description=f"Resolve frozen {side.name} TPP40 bridge training outputs.",
    )
    with executor_context():
        for training_step in artifacts.training_steps:
            resolver.compute_version(training_step, is_pseudo_dep=False)
    return [resolver.output_paths[step] for step in artifacts.training_steps]


def _checkpoint_metadata(checkpoint_path: str, *, expected_step: int) -> tuple[dict[str, object], str] | None:
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    fs, _, _ = fsspec.get_fs_token_paths(metadata_path)
    if not fs.exists(metadata_path):
        return None
    with fs.open(metadata_path, "rb") as handle:
        encoded = handle.read()
    metadata = json.loads(encoded)
    if int(metadata.get("step", -1)) != expected_step:
        raise ValueError(f"Checkpoint metadata step changed at {checkpoint_path}: {metadata.get('step')}")
    if metadata.get("is_temporary", False):
        raise ValueError(f"Frozen bridge checkpoint is marked temporary: {checkpoint_path}")
    has_manifest = fs.exists(os.path.join(checkpoint_path, "manifest.ocdbt"))
    has_data_dir = fs.exists(os.path.join(checkpoint_path, "d"))
    if not has_manifest and not has_data_dir:
        raise ValueError(f"Checkpoint metadata exists without tensor payload: {checkpoint_path}")
    return metadata, _sha256_bytes(encoded)


def _uncheatable_metrics(raw_metrics: dict[str, object]) -> tuple[dict[str, float], float]:
    component_metrics: dict[str, float] = {}
    for name in EXPECTED_UNCHEATABLE_NAMES:
        key = f"eval/{name}/bpb"
        if key not in raw_metrics:
            raise ValueError(f"Missing Uncheatable metric {key!r}")
        raw_value = raw_metrics[key]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"Non-numeric Uncheatable metric {key!r}: {raw_value!r}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite Uncheatable metric {key!r}: {value}")
        component_metrics[name] = value
    # Levanter accumulates and averages the tag values in float32. Recreate that
    # arithmetic exactly before comparing the JSON-widened reported scalar.
    component_values = np.asarray(tuple(component_metrics.values()), dtype=np.float32)
    component_mask = np.ones(component_values.shape, dtype=bool)
    macro_bpb = float(np.mean(component_values, where=component_mask))
    reported_key = "eval/uncheatable_eval/macro_bpb"
    if reported_key not in raw_metrics:
        raise ValueError(f"Missing reported Uncheatable macro {reported_key!r}")
    raw_reported_macro = raw_metrics[reported_key]
    if isinstance(raw_reported_macro, bool) or not isinstance(raw_reported_macro, int | float):
        raise ValueError(f"Non-numeric reported Uncheatable macro {reported_key!r}: {raw_reported_macro!r}")
    reported_macro = float(raw_reported_macro)
    if not math.isfinite(reported_macro):
        raise ValueError(f"Non-finite reported Uncheatable macro {reported_key!r}: {reported_macro}")
    if macro_bpb != reported_macro:
        raise ValueError(f"Uncheatable macro mismatch: computed={macro_bpb}, reported={reported_macro}")
    return component_metrics, reported_macro


def _read_json(path: str) -> dict[str, object]:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    with fs.open(path, "rt") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, object]) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with fs.open(path, "wt") as handle:
        handle.write(encoded)


def run_bridge_uncheatable_eval(config: BridgeUncheatableEvalConfig) -> None:
    """Restore one exact Orbax checkpoint and materialize its seven-component result."""
    metadata_result = _checkpoint_metadata(config.checkpoint_path, expected_step=config.checkpoint_step)
    if metadata_result is None:
        raise FileNotFoundError(f"Exact checkpoint is not ready: {config.checkpoint_path}")
    checkpoint_metadata, checkpoint_metadata_sha256 = metadata_result

    scaling_fits = base._read_scaling_fits(config.analysis_output_path)
    candidate = base._candidate_for_run_spec(scaling_fits=scaling_fits, run_spec=config.run_spec)
    total_trainable_params = int(
        candidate.model_config.total_trainable_params(base.completed_adamh_heuristic.vocab_size)
    )
    if total_trainable_params != config.run_spec.total_trainable_params:
        raise ValueError(
            f"Resolved parameter count changed: {total_trainable_params} != " f"{config.run_spec.total_trainable_params}"
        )
    model_config_payload = asdict(candidate.model_config)
    model_config_sha256 = _sha256_bytes(
        json.dumps(model_config_payload, default=str, separators=(",", ":"), sort_keys=True).encode()
    )
    tensor_parallel_size = tpp40._tensor_parallel_size(candidate.model_config.hidden_dim, EVALUATOR_TPU_TYPE)
    data = LmDataConfig(
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        auto_build_caches=False,
        components=config.validation_configs,
        train_weights={name: 0.0 for name in config.validation_configs},
    )
    trainer = TrainerConfig(
        tracker=JsonFileTrackerConfig(output_path=config.output_path),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=EVAL_BATCH_SIZE,
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": tensor_parallel_size},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
        seed=config.run_spec.trainer_seed,
        allow_nondivisible_batch_size=True,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    eval_lm.main(
        eval_lm.EvalLmConfig(
            checkpoint_path=config.checkpoint_path,
            trainer=trainer,
            data=data,
            max_eval_length=base.SEQ_LEN_DELPHI,
            model=candidate.model_config,
        )
    )

    raw_result_path = os.path.join(config.output_path, RAW_RESULT_FILE)
    raw_metrics = _read_json(raw_result_path)
    component_metrics, macro_bpb = _uncheatable_metrics(raw_metrics)
    _write_json(
        os.path.join(config.output_path, RESULT_FILE),
        {
            "schema_version": 1,
            "acceptance_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "evaluator_tpu_type": EVALUATOR_TPU_TYPE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "side": config.side,
            "run_order": config.run_spec.run_order,
            "run_name": config.run_spec.run_name,
            "source_run_name": config.run_spec.source_run_name,
            "data_seed": config.run_spec.data_seed,
            "trainer_seed": config.run_spec.trainer_seed,
            "total_trainable_params": total_trainable_params,
            "model_config_sha256": model_config_sha256,
            "checkpoint_step": config.checkpoint_step,
            "checkpoint_path": config.checkpoint_path,
            "checkpoint_metadata": checkpoint_metadata,
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
            "validation_payload_sha256": config.validation_payload_sha256,
            "component_bpb": component_metrics,
            "macro_bpb": macro_bpb,
            "raw_result_path": raw_result_path,
        },
    )


def _eval_steps(
    *,
    side: BridgeSide,
    run_specs: list[base.DelphiSwarmRunSpec],
    training_output_paths: list[str],
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    validation_payload_sha256: str,
) -> list[ExecutorStep]:
    resources = ResourceConfig.with_tpu(
        EVALUATOR_TPU_TYPE,
        regions=[side.region],
        zone=side.evaluator_zone,
        disk="80g",
    )
    steps: list[ExecutorStep] = []
    with executor_context():
        for run_spec, training_output_path in zip(run_specs, training_output_paths, strict=True):
            for checkpoint_step in CHECKPOINT_STEPS:
                steps.append(
                    ExecutorStep(
                        name=(f"{EVALUATOR_EXPERIMENT_NAME}/{side.name}/{run_spec.run_name}/step-{checkpoint_step}"),
                        fn=remote(
                            run_bridge_uncheatable_eval,
                            resources=resources,
                            env_vars={
                                "MARIN_PREFIX": marin_prefix_for_region(side.region),
                                base.HF_HUB_DISABLE_XET_ENV_VAR: "1",
                            },
                        ),
                        resources=resources,
                        config=BridgeUncheatableEvalConfig(
                            side=side.name,
                            analysis_output_path=analysis_output_path,
                            run_spec=run_spec,
                            checkpoint_step=checkpoint_step,
                            checkpoint_path=os.path.join(training_output_path, f"checkpoints/step-{checkpoint_step}"),
                            validation_configs=validation_configs,
                            validation_payload_sha256=validation_payload_sha256,
                            output_path=this_output_path(),
                        ),
                    )
                )
    return steps


def _resolve_eval_paths(steps: list[ExecutorStep], *, prefix: str, side: BridgeSide) -> list[str]:
    resolver = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description=f"Resolve {side.name} TPP40 bridge Uncheatable outputs.",
    )
    with executor_context():
        for step in steps:
            resolver.compute_version(step, is_pseudo_dep=False)
    return [resolver.output_paths[step] for step in steps]


def _validate_completed_result(record: EvalRecord) -> None:
    payload = _read_json(os.path.join(record.eval_output_path, RESULT_FILE))
    if payload.get("acceptance_contract_sha256") != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"Completed result has wrong acceptance contract: {record.eval_output_path}")
    expected_identity = (record.side, record.run_order, record.checkpoint_step, record.checkpoint_path)
    observed_identity = (
        payload.get("side"),
        payload.get("run_order"),
        payload.get("checkpoint_step"),
        payload.get("checkpoint_path"),
    )
    if observed_identity != expected_identity:
        raise ValueError(f"Completed result identity mismatch: {observed_identity} != {expected_identity}")
    metadata_result = _checkpoint_metadata(record.checkpoint_path, expected_step=record.checkpoint_step)
    if metadata_result is None:
        raise ValueError(f"Completed result checkpoint disappeared: {record.checkpoint_path}")
    _, current_metadata_sha256 = metadata_result
    if payload.get("checkpoint_metadata_sha256") != current_metadata_sha256:
        raise ValueError(f"Completed result checkpoint metadata changed: {record.eval_output_path}")
    if payload.get("validation_payload_sha256") != record.validation_payload_sha256:
        raise ValueError(f"Completed result validation payload changed: {record.eval_output_path}")
    total_trainable_params = payload.get("total_trainable_params")
    if isinstance(total_trainable_params, bool) or not isinstance(total_trainable_params, int):
        raise ValueError(f"Completed result lacks parameter-count identity: {record.eval_output_path}")
    model_config_sha256 = payload.get("model_config_sha256")
    if not isinstance(model_config_sha256, str) or len(model_config_sha256) != 64:
        raise ValueError(f"Completed result lacks model-config identity: {record.eval_output_path}")
    component_metrics = payload.get("component_bpb")
    if not isinstance(component_metrics, dict):
        raise ValueError(f"Completed result lacks component metrics: {record.eval_output_path}")
    if "macro_bpb" not in payload:
        raise ValueError(f"Completed result lacks macro_bpb: {record.eval_output_path}")
    _uncheatable_metrics(
        {
            **{f"eval/{name}/bpb": value for name, value in component_metrics.items()},
            "eval/uncheatable_eval/macro_bpb": payload["macro_bpb"],
        }
    )


def _records(
    *,
    side: BridgeSide,
    run_specs: list[base.DelphiSwarmRunSpec],
    training_output_paths: list[str],
    eval_steps: list[ExecutorStep],
    eval_output_paths: list[str],
    validation_payload_sha256: str,
    inspect_readiness: bool,
) -> list[EvalRecord]:
    step_cells = [
        (run_spec, training_output_path, checkpoint_step)
        for run_spec, training_output_path in zip(run_specs, training_output_paths, strict=True)
        for checkpoint_step in CHECKPOINT_STEPS
    ]
    if len(step_cells) != len(eval_steps) or len(eval_steps) != len(eval_output_paths):
        raise ValueError("Bridge evaluation step accounting changed")

    records: list[EvalRecord] = []
    for (run_spec, training_output_path, checkpoint_step), eval_output_path in zip(
        step_cells, eval_output_paths, strict=True
    ):
        checkpoint_path = os.path.join(training_output_path, f"checkpoints/step-{checkpoint_step}")
        checkpoint_ready = False
        eval_already_succeeded = False
        if inspect_readiness:
            checkpoint_ready = _checkpoint_metadata(checkpoint_path, expected_step=checkpoint_step) is not None
            eval_already_succeeded = (
                StatusFile(eval_output_path, worker_id="tpp40-bridge-uncheatable-audit").status == STATUS_SUCCESS
            )
        record = EvalRecord(
            side=side.name,
            run_order=run_spec.run_order,
            run_name=run_spec.run_name,
            checkpoint_step=checkpoint_step,
            training_output_path=training_output_path,
            checkpoint_path=checkpoint_path,
            checkpoint_ready=checkpoint_ready,
            eval_output_path=eval_output_path,
            eval_already_succeeded=eval_already_succeeded,
            validation_payload_sha256=validation_payload_sha256,
        )
        if eval_already_succeeded:
            _validate_completed_result(record)
        records.append(record)
    return records


def _write_ready_manifest(
    *,
    side: BridgeSide,
    prefix: str,
    records: list[EvalRecord],
    cache_paths: tuple[str, ...],
    validation_payload_sha256: str,
    persist: bool,
    dry_run_output: Path,
) -> tuple[str, str]:
    payload = {
        "schema_version": 1,
        "acceptance_contract_path": str(ACCEPTANCE_CONTRACT_PATH),
        "acceptance_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "side": asdict(side),
        "evaluator_tpu_type": EVALUATOR_TPU_TYPE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "run_orders": list(BRIDGE_RUN_ORDERS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "uncheatable_components": list(EXPECTED_UNCHEATABLE_NAMES),
        "validation_cache_paths": list(cache_paths),
        "validation_cache_paths_sha256": (
            hashlib.sha256(json.dumps(cache_paths, separators=(",", ":")).encode()).hexdigest()
        ),
        "validation_payload_sha256": validation_payload_sha256,
        "ready_count": sum(record.checkpoint_ready for record in records),
        "completed_eval_count": sum(record.eval_already_succeeded for record in records),
        "records": [asdict(record) for record in records],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = _sha256_bytes(encoded.encode())
    if persist:
        output_path = os.path.join(
            prefix,
            EVALUATOR_EXPERIMENT_NAME,
            "ready_manifests",
            f"{side.name}-{digest[:16]}.json",
        )
        fs, _, _ = fsspec.get_fs_token_paths(output_path)
        if fs.exists(output_path):
            with fs.open(output_path, "rt") as handle:
                if handle.read() != encoded:
                    raise RuntimeError(f"Ready manifest collision at {output_path}")
        else:
            fs.makedirs(os.path.dirname(output_path), exist_ok=True)
            with fs.open(output_path, "wt") as handle:
                handle.write(encoded)
    else:
        dry_run_output.mkdir(parents=True, exist_ok=True)
        output_path = str(dry_run_output / f"{side.name}_bridge_uncheatable_ready_manifest.json")
        Path(output_path).write_text(encoded)
    return output_path, digest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=tuple(BRIDGE_SIDES), required=True)
    parser.add_argument("--tpu-type", required=True)
    parser.add_argument("--tpu-region", required=True)
    parser.add_argument("--tpu-zone", required=True)
    parser.add_argument("--max-concurrent", type=int, default=len(BRIDGE_RUN_ORDERS) * len(CHECKPOINT_STEPS))
    parser.add_argument("--minimum-ready-checkpoints", type=int, default=1)
    parser.add_argument("--require-all-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output", type=Path, default=REFERENCE_DIR / "bridge_uncheatable_dry_run")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    cell_count = len(BRIDGE_RUN_ORDERS) * len(CHECKPOINT_STEPS)
    if not 1 <= args.max_concurrent <= cell_count:
        raise ValueError(f"--max-concurrent must be in [1, {cell_count}]")
    if not 0 <= args.minimum_ready_checkpoints <= cell_count:
        raise ValueError(f"--minimum-ready-checkpoints must be in [0, {cell_count}]")

    _load_acceptance_contract()
    side = BRIDGE_SIDES[args.side]
    requested_placement = (args.tpu_type, args.tpu_region, args.tpu_zone)
    frozen_placement = (EVALUATOR_TPU_TYPE, side.region, side.evaluator_zone)
    if requested_placement != frozen_placement:
        raise ValueError(f"Evaluator placement changed: {requested_placement} != {frozen_placement}")
    prefix = _set_region_prefix(side)
    run_specs, source_panel, analysis_output_path = _run_specs(side)
    full_validation_configs, uncheatable_validation_configs = _validation_configs()
    cache_paths = _require_uncheatable_caches(uncheatable_validation_configs, side=side)
    validation_identity = validation_payload_identity(dict(zip(EXPECTED_UNCHEATABLE_NAMES, cache_paths, strict=True)))
    validation_payload_sha256 = validation_identity["payload_sha256"]
    training_output_paths = _original_training_paths(
        side=side,
        run_specs=run_specs,
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        full_validation_configs=full_validation_configs,
        prefix=prefix,
    )
    eval_steps = _eval_steps(
        side=side,
        run_specs=run_specs,
        training_output_paths=training_output_paths,
        analysis_output_path=analysis_output_path,
        validation_configs=uncheatable_validation_configs,
        validation_payload_sha256=validation_payload_sha256,
    )
    eval_output_paths = _resolve_eval_paths(eval_steps, prefix=prefix, side=side)
    records = _records(
        side=side,
        run_specs=run_specs,
        training_output_paths=training_output_paths,
        eval_steps=eval_steps,
        eval_output_paths=eval_output_paths,
        validation_payload_sha256=validation_payload_sha256,
        inspect_readiness=not args.dry_run,
    )
    ready_count = sum(record.checkpoint_ready for record in records)
    pending_steps = [
        step
        for step, record in zip(eval_steps, records, strict=True)
        if record.checkpoint_ready and not record.eval_already_succeeded
    ]
    persist_manifest = not args.dry_run and os.getenv("CI") is None
    manifest_path, manifest_sha256 = _write_ready_manifest(
        side=side,
        prefix=prefix,
        records=records,
        cache_paths=cache_paths,
        validation_payload_sha256=validation_payload_sha256,
        persist=persist_manifest,
        dry_run_output=args.dry_run_output,
    )
    logger.info(
        "%s bridge Uncheatable snapshot: ready=%d/%d completed=%d pending=%d manifest=%s sha256=%s",
        side.name,
        ready_count,
        len(records),
        sum(record.eval_already_succeeded for record in records),
        len(pending_steps),
        manifest_path,
        manifest_sha256,
    )

    if args.dry_run or os.getenv("CI") is not None:
        return
    if args.require_all_checkpoints and ready_count != len(records):
        raise RuntimeError(f"Found {ready_count}/{len(records)} ready {side.name} bridge checkpoints; all are required")
    if ready_count < args.minimum_ready_checkpoints:
        raise RuntimeError(
            f"Found {ready_count}/{len(records)} ready {side.name} bridge checkpoints; "
            f"minimum is {args.minimum_ready_checkpoints}"
        )
    if not pending_steps:
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=pending_steps,
        description=(
            f"Frozen TPP40 {side.name} bridge: exact-checkpoint seven-component English Uncheatable evaluation; "
            f"ready manifest {manifest_path}"
        ),
    )


if __name__ == "__main__":
    main()
