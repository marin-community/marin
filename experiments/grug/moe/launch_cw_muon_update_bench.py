# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave H100 launcher for the standalone Grug MoE Muon update benchmark."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fsspec
import jax
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from levanter.distributed import DistributedConfig
from levanter.optim.grugmuon import DEFAULT_MAX_GROUPED_STACK_SIZE, STACK_BATCH_SHARDED
from levanter.utils.jax_utils import sync_global_devices
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.cw_storage import set_default_cw_grug_moe_prefix
from experiments.grug.moe.muon_update_bench import (
    BENCH_KINDS,
    MUONH_UPDATE_BENCH,
    BenchConfig,
    create_mesh,
    dtype_from_name,
    dtype_name,
    emit_jsonl,
    estimate_grouping,
    run_config,
    summary_row,
    synthetic_shapes,
    variant_label,
)

GPUS_PER_NODE = 8
DEFAULT_OUTPUT_SUBDIR = "experiments/grug-moe-cw/muon-update-bench"
WANDB_LOG_TIMEOUT = 120
WANDB_PAYLOAD_ENV = "MUON_BENCH_WANDB_PAYLOAD"

set_default_cw_grug_moe_prefix()


@dataclass(frozen=True)
class MuonUpdateBenchLaunchConfig:
    output_path: str
    run_id: str
    resources: ResourceConfig
    layers: int = 2
    ns4d_group_size: int | None = None
    ns4d_group_axis: str = "data"
    hidden_dim: int = 2560
    intermediate_dim: int = 1280
    num_experts: int = 256
    dtype: str = "bf16"
    ns_compute_dtype: str = "input"
    sweep_backend_steps: tuple[int, ...] = (1, 5)
    sweep_max_grouped_stack_sizes: tuple[int, ...] = (256, 512)
    orthogonalization_layout: str = STACK_BATCH_SHARDED
    replica_axis: int = 1
    data_axis: int = 1
    expert_axis: int = 8
    model_axis: int = 1
    learning_rate: float = 0.02
    nesterov: bool = True
    expert_grouped_muonh_packed_entry: bool = False
    warmup: int = 1
    iters: int = 3
    grouped_expert_consumer_tokens_per_expert: int = 1
    grouped_expert_consumer_chunk_tokens: int = 0
    grouped_expert_consumer_chunk_tokens_per_expert: int = 0
    bench_kinds: tuple[str, ...] = (MUONH_UPDATE_BENCH,)
    mode: str = "both"
    compile_only: bool = False
    disable_abstract_mesh: bool = False
    allow_boundary_collectives: bool = False
    require_no_boundary_collectives: bool = False
    write_compiled_hlo: bool = False
    profile: bool = False
    wandb: bool = False
    wandb_project: str = "marin_moe"
    wandb_group: str = "grug-moe-cw-muon-update-bench"


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key, "")
    return int(raw) if raw else None


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def env_int_csv(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{key}={raw!r} must contain at least one integer")
    return values


def env_str_csv(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{key}={raw!r} must contain at least one value")
    invalid_values = sorted(set(values) - set(BENCH_KINDS))
    if invalid_values:
        raise ValueError(f"{key} has unknown benchmark kinds {invalid_values}; expected one of {BENCH_KINDS}.")
    return values


def _bench_configs(config: MuonUpdateBenchLaunchConfig) -> list[BenchConfig]:
    dtype = dtype_name(dtype_from_name(config.dtype))
    return [
        BenchConfig(
            layers=config.layers,
            ns4d_group_size=config.ns4d_group_size,
            ns4d_group_axis=config.ns4d_group_axis,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            num_experts=config.num_experts,
            dtype=dtype,
            backend_steps=backend_steps,
            orthogonalization_layout=config.orthogonalization_layout,
            max_grouped_stack_size=max_grouped_stack_size,
            replica_axis=config.replica_axis,
            data_axis=config.data_axis,
            expert_axis=config.expert_axis,
            model_axis=config.model_axis,
            learning_rate=config.learning_rate,
            nesterov=config.nesterov,
            ns_compute_dtype=config.ns_compute_dtype,
            expert_grouped_muonh_packed_entry=config.expert_grouped_muonh_packed_entry,
            grouped_expert_consumer_tokens_per_expert=config.grouped_expert_consumer_tokens_per_expert,
            grouped_expert_consumer_chunk_tokens=config.grouped_expert_consumer_chunk_tokens,
            grouped_expert_consumer_chunk_tokens_per_expert=config.grouped_expert_consumer_chunk_tokens_per_expert,
        )
        for backend_steps in config.sweep_backend_steps
        for max_grouped_stack_size in config.sweep_max_grouped_stack_sizes
    ]


def _run_args(config: MuonUpdateBenchLaunchConfig, profile_dir: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        mode=config.mode,
        warmup=config.warmup,
        iters=config.iters,
        compile_only=config.compile_only,
        hlo_output=None,
        compiled_hlo_output=f"{config.output_path}/compiled_hlo.txt" if config.write_compiled_hlo else None,
        output=None,
        disable_abstract_mesh=config.disable_abstract_mesh,
        allow_boundary_collectives=config.allow_boundary_collectives,
        require_no_boundary_collectives=config.require_no_boundary_collectives,
        profile_dir=profile_dir,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(child) for child in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with fsspec.open(path, "w") as fp:
        fp.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sync_global_devices_if_multihost(name: str) -> None:
    if jax.process_count() <= 1:
        return
    sync_global_devices(f"muon_update_bench_{name}")


def _upload_directory(local_dir: Path, remote_dir: str) -> list[str]:
    if not local_dir.exists():
        return []

    uploaded = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(local_dir).as_posix()
        remote_path = f"{remote_dir.rstrip('/')}/{relative_path}"
        with path.open("rb") as src, fsspec.open(remote_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        uploaded.append(remote_path)
    return uploaded


def _wandb_metric_row(row: dict[str, Any], row_index: int) -> dict[str, int | float | str | bool]:
    metrics: dict[str, int | float | str | bool] = {"bench/row_index": row_index}
    for key, value in row.items():
        if isinstance(value, bool | int | float | str):
            metrics[f"bench/{key}"] = value
    return metrics


def _log_summary_to_wandb_process(
    run_id: str,
    wandb_project: str,
    wandb_group: str,
    launch_config: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        emit_jsonl({"event": "wandb_skipped", "reason": "WANDB_API_KEY is not set"})
        return

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "marin-community"),
        project=wandb_project,
        name=run_id,
        group=wandb_group or None,
        tags=["grug", "moe", "muonh", "cw", "h100", "benchmark"],
        config=launch_config,
        settings=wandb.Settings(
            console="off",
            init_timeout=30,
            login_timeout=30,
            quiet=True,
            silent=True,
            summary_timeout=30,
            x_disable_meta=True,
            x_disable_stats=True,
            x_file_stream_timeout_seconds=30,
            x_file_transfer_timeout_seconds=30,
            x_graphql_timeout_seconds=30,
            x_service_wait=30,
        ),
    )

    try:
        run.summary["output_path"] = payload["output_path"]
        run.summary["profile_output_path"] = payload.get("profile_output_path")
        summary_rows = payload["summary_table"]
        best_row = max(
            summary_rows,
            key=lambda row: row.get("median_h100_bf16_peak_pct") or float("-inf"),
            default=None,
        )
        if best_row is not None:
            for key, value in _wandb_metric_row(best_row, 0).items():
                if key.startswith("bench/"):
                    run.summary[f"topline/{key.removeprefix('bench/')}"] = value
        for row_index, row in enumerate(summary_rows):
            run.log(_wandb_metric_row(row, row_index), step=row_index)
        emit_jsonl({"event": "wandb_logged", "entity": run.entity, "project": run.project, "name": run.name})
    finally:
        run.finish(quiet=True)


def _log_summary_to_wandb(config: MuonUpdateBenchLaunchConfig, payload: dict[str, Any]) -> None:
    if not config.wandb or jax.process_index() != 0:
        return

    payload_file = None
    try:
        with tempfile.NamedTemporaryFile("w", prefix="muon-update-bench-wandb-", suffix=".json", delete=False) as fp:
            payload_file = Path(fp.name)
            json.dump(
                {
                    "run_id": config.run_id,
                    "wandb_project": config.wandb_project,
                    "wandb_group": config.wandb_group,
                    "launch_config": _json_safe(asdict(config)),
                    "payload": payload,
                },
                fp,
            )
            fp.write("\n")

        env = os.environ.copy()
        env[WANDB_PAYLOAD_ENV] = str(payload_file)
        result = subprocess.run(
            [sys.executable, "-m", "experiments.grug.moe.launch_cw_muon_update_bench"],
            env=env,
            timeout=WANDB_LOG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        emit_jsonl({"event": "wandb_timeout", "timeout_seconds": WANDB_LOG_TIMEOUT})
        return
    finally:
        if payload_file is not None:
            payload_file.unlink(missing_ok=True)

    if result.returncode != 0:
        emit_jsonl({"event": "wandb_failed", "exit_code": result.returncode})


def _log_summary_to_wandb_from_payload_file(path: str) -> None:
    with Path(path).open() as fp:
        payload = json.load(fp)
    _log_summary_to_wandb_process(
        payload["run_id"],
        payload["wandb_project"],
        payload["wandb_group"],
        payload["launch_config"],
        payload["payload"],
    )


def _run_muon_update_bench_local(config: MuonUpdateBenchLaunchConfig) -> None:
    DistributedConfig().initialize()
    configs = _bench_configs(config)
    first_config = configs[0]
    mesh = create_mesh(
        first_config.replica_axis,
        first_config.data_axis,
        first_config.expert_axis,
        first_config.model_axis,
    )
    emit_jsonl(
        {
            "event": "launch_metadata",
            "run_id": config.run_id,
            "output_path": config.output_path,
            "launch_config": _json_safe(asdict(config)),
            "devices": int(np.asarray(jax.devices()).size),
            "local_devices": int(np.asarray(jax.local_devices()).size),
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "device_kinds": sorted({getattr(device, "device_kind", "") for device in jax.devices()}),
        }
    )

    profile_dir = None
    profile_output_path = None
    if config.profile:
        profile_dir = Path("/tmp/muon_update_bench_profiles") / config.run_id / f"process_{jax.process_index()}"
        profile_output_path = f"{config.output_path}/profiler/process_{jax.process_index()}"

    args = _run_args(config, profile_dir)
    total_variants = len(configs) * len(config.bench_kinds)
    results = [
        run_config(args, mesh, bench_config, bench_kind, total_variants)
        for bench_config in configs
        for bench_kind in config.bench_kinds
    ]
    summary_rows = [summary_row(result) for result in results]
    uploaded_profile_files = (
        _upload_directory(profile_dir, profile_output_path) if profile_dir and profile_output_path else []
    )
    payload = {
        "run_id": config.run_id,
        "output_path": config.output_path,
        "profile_output_path": profile_output_path,
        "uploaded_profile_files": uploaded_profile_files,
        "launch_config": _json_safe(asdict(config)),
        "synthetic_shapes": synthetic_shapes(first_config),
        "group_estimates": {
            variant_label(bench_config, bench_kind): [asdict(estimate) for estimate in estimate_grouping(bench_config)]
            for bench_config in configs
            for bench_kind in config.bench_kinds
        },
        "results": results,
        "summary_table": summary_rows,
    }
    emit_jsonl({"event": "summary_table", "rows": summary_rows})
    if uploaded_profile_files:
        emit_jsonl(
            {
                "event": "profile_uploaded",
                "profile_output_path": profile_output_path,
                "files": uploaded_profile_files,
            }
        )
    _write_json(f"{config.output_path}/summary.json", payload)
    _sync_global_devices_if_multihost("before_wandb")
    _log_summary_to_wandb(config, payload)
    _sync_global_devices_if_multihost("after_wandb")


def run_muon_update_bench(config: MuonUpdateBenchLaunchConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_muon_update_bench_local,
        resources=config.resources,
        max_retries_failure=0,
    )


def build_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "muon-update-bench-%Y%m%d-%H%M%S"
    )
    resources = ResourceConfig.with_gpu(
        "H100",
        count=GPUS_PER_NODE,
        cpu=env_int("MUON_BENCH_WORKER_CPU", 8),
        ram=os.environ.get("MUON_BENCH_WORKER_RAM", "256g"),
        disk=os.environ.get("MUON_BENCH_WORKER_DISK", "256g"),
        replicas=env_int("MUON_BENCH_GPU_REPLICAS", 1),
        image=os.environ.get("MUON_BENCH_TASK_IMAGE") or None,
    )
    config = MuonUpdateBenchLaunchConfig(
        output_path=this_output_path(),
        run_id=run_id,
        resources=resources,
        layers=env_int("MUON_BENCH_LAYERS", 2),
        ns4d_group_size=env_optional_int("MUON_BENCH_NS4D_GROUP_SIZE"),
        ns4d_group_axis=os.environ.get("MUON_BENCH_NS4D_GROUP_AXIS", "data"),
        hidden_dim=env_int("MUON_BENCH_HIDDEN_DIM", 2560),
        intermediate_dim=env_int("MUON_BENCH_INTERMEDIATE_DIM", 1280),
        num_experts=env_int("MUON_BENCH_NUM_EXPERTS", 256),
        dtype=os.environ.get("MUON_BENCH_DTYPE", "bf16"),
        ns_compute_dtype=os.environ.get("MUON_BENCH_NS_COMPUTE_DTYPE", "input"),
        sweep_backend_steps=env_int_csv("MUON_BENCH_SWEEP_BACKEND_STEPS", (1, 5)),
        sweep_max_grouped_stack_sizes=env_int_csv(
            "MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES", (DEFAULT_MAX_GROUPED_STACK_SIZE, 512)
        ),
        orthogonalization_layout=os.environ.get("MUON_BENCH_ORTHOGONALIZATION_LAYOUT", STACK_BATCH_SHARDED),
        replica_axis=env_int("MUON_BENCH_REPLICA_AXIS", 1),
        data_axis=env_int("MUON_BENCH_DATA_AXIS", 1),
        expert_axis=env_int("MUON_BENCH_EXPERT_AXIS", 8),
        model_axis=env_int("MUON_BENCH_MODEL_AXIS", 1),
        learning_rate=env_float("MUON_BENCH_LEARNING_RATE", 0.02),
        nesterov=env_bool("MUON_BENCH_NESTEROV", True),
        expert_grouped_muonh_packed_entry=env_bool("MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY", False),
        warmup=env_int("MUON_BENCH_WARMUP", 1),
        iters=env_int("MUON_BENCH_ITERS", 3),
        grouped_expert_consumer_tokens_per_expert=env_int("MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT", 1),
        grouped_expert_consumer_chunk_tokens=env_int("MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS", 0),
        grouped_expert_consumer_chunk_tokens_per_expert=env_int(
            "MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT",
            0,
        ),
        bench_kinds=env_str_csv("MUON_BENCH_KINDS", (MUONH_UPDATE_BENCH,)),
        mode=os.environ.get("MUON_BENCH_MODE", "both"),
        compile_only=env_bool("MUON_BENCH_COMPILE_ONLY", False),
        disable_abstract_mesh=env_bool("MUON_BENCH_DISABLE_ABSTRACT_MESH", False),
        allow_boundary_collectives=env_bool("MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES", False),
        require_no_boundary_collectives=env_bool("MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES", False),
        write_compiled_hlo=env_bool("MUON_BENCH_WRITE_COMPILED_HLO", False),
        profile=env_bool("MUON_BENCH_ENABLE_JAX_PROFILE", False),
        wandb=env_bool("MUON_BENCH_WANDB", False) or os.environ.get("MUON_BENCH_TRACKER", "").lower() == "wandb",
        wandb_project=os.environ.get("MUON_BENCH_WANDB_PROJECT") or os.environ.get("WANDB_PROJECT", "marin_moe"),
        wandb_group=os.environ.get("MUON_BENCH_WANDB_GROUP", "grug-moe-cw-muon-update-bench"),
    )
    return ExecutorStep(
        name=f"{DEFAULT_OUTPUT_SUBDIR}/{run_id}",
        fn=run_muon_update_bench,
        config=config,
    )


def main() -> None:
    if payload_path := os.environ.get(WANDB_PAYLOAD_ENV):
        _log_summary_to_wandb_from_payload_file(payload_path)
        return

    executor_main(steps=[build_step()])


if __name__ == "__main__":
    main()
