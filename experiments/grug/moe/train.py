# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.data import AsyncDataset, DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text import GrugLmExample, LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.models.lm_model import LmExample
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.model import GrugModelConfig, Transformer

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3
CANARY_RUNTIME_CONFIG_MARKER = "CANARY_RUNTIME_CONFIG_JSON"
CANARY_COMPILE_MEMORY_MARKER = "CANARY_COMPILE_MEMORY_JSON"
CANARY_GPU_MEMORY_MARKER = "CANARY_GPU_MEMORY_JSON"
COMPILE_MEMORY_BYTE_FIELDS: tuple[tuple[str, str], ...] = (
    ("argument", "argument_size_in_bytes"),
    ("output", "output_size_in_bytes"),
    ("alias", "alias_size_in_bytes"),
    ("temp", "temp_size_in_bytes"),
    ("code", "generated_code_size_in_bytes"),
    ("peak", "peak_memory_in_bytes"),
    ("host_argument", "host_argument_size_in_bytes"),
    ("host_output", "host_output_size_in_bytes"),
    ("host_alias", "host_alias_size_in_bytes"),
    ("host_temp", "host_temp_size_in_bytes"),
    ("host_code", "host_generated_code_size_in_bytes"),
)


@dataclass(frozen=True)
class CanaryDiagnosticsConfig:
    """Low-overhead diagnostics emitted only for explicitly enabled canary runs."""

    enabled: bool = False
    compile_memory_enabled: bool = True
    gpu_memory_snapshot_interval: int = 10
    compile_temp_baseline_bytes: int | None = None
    nvidia_smi_enabled: bool = True


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    train_batch_pspec: P = field(default_factory=lambda: P(("data", "expert")))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 0.0  # Weight on logsumexp (z-loss) stabilization term.


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    eval_batch_pspec: P = field(default_factory=lambda: P(("data", "expert")))
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
    batch_pspec: P = P(("data", "expert")),
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    axis_resource = batch_pspec[0]
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": axis_resource},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    batch_axis_resource = eval_cfg.eval_batch_pspec[0]
    eval_axis_mapping = {"batch": batch_axis_resource}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / BYTES_PER_GIB


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default

    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False

    raise ValueError(f"Unknown {key}={raw!r}, expected a boolean value")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def _env_float_or_none(key: str) -> float | None:
    raw = os.environ.get(key, "")
    return float(raw) if raw else None


def _canary_diagnostics_from_env() -> CanaryDiagnosticsConfig:
    enabled = _env_bool("CANARY_MEMORY_DIAGNOSTICS", False)
    compile_temp_baseline_gib = _env_float_or_none("CANARY_COMPILE_TEMP_BASELINE_GIB")
    return CanaryDiagnosticsConfig(
        enabled=enabled,
        compile_memory_enabled=_env_bool("CANARY_COMPILE_MEMORY_ENABLED", enabled),
        gpu_memory_snapshot_interval=_env_int("CANARY_GPU_MEMORY_INTERVAL", 10),
        compile_temp_baseline_bytes=(
            int(compile_temp_baseline_gib * BYTES_PER_GIB) if compile_temp_baseline_gib is not None else None
        ),
        nvidia_smi_enabled=_env_bool("CANARY_NVIDIA_SMI_ENABLED", True),
    )


def _jsonable_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _exception_payload(exc: BaseException) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "error": str(exc)}


def _canary_base_payload(*, phase: str | None = None, step: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
    }
    if phase is not None:
        payload["phase"] = phase
    if step is not None:
        payload["step"] = step
    return payload


def _log_canary_marker(marker: str, payload: dict[str, Any]) -> None:
    logger.info("%s %s", marker, json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _device_identity(device) -> dict[str, Any]:
    return {
        "id": _jsonable_scalar(getattr(device, "id", None)),
        "platform": str(getattr(device, "platform", "unknown")),
        "kind": str(getattr(device, "device_kind", "unknown")),
        "process_index": _jsonable_scalar(getattr(device, "process_index", None)),
    }


def _compile_memory_payload(memory_analysis) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, attr in COMPILE_MEMORY_BYTE_FIELDS:
        try:
            value = getattr(memory_analysis, attr)
        except AttributeError:
            continue

        if value is None:
            continue

        try:
            bytes_value = int(value)
        except (TypeError, ValueError):
            continue
        payload[f"{name}_bytes"] = bytes_value
        payload[f"{name}_gib"] = _bytes_to_gib(bytes_value)

    return payload


def _log_canary_compile_memory(
    diagnostics: CanaryDiagnosticsConfig,
    train_step,
    state: GrugTrainState,
    batch,
    *,
    compute_watch: bool,
    step: int,
) -> None:
    if not diagnostics.enabled or not diagnostics.compile_memory_enabled:
        return

    payload = _canary_base_payload(phase="compile_train_step", step=step)
    payload["compute_watch"] = compute_watch
    try:
        compiled = train_step.lower(state, batch, compute_watch=compute_watch).compile()
        memory_analysis = compiled.memory_analysis()
    except AttributeError as exc:
        payload.update({"status": "unsupported", **_exception_payload(exc)})
        _log_canary_marker(CANARY_COMPILE_MEMORY_MARKER, payload)
        return
    except Exception as exc:
        payload.update({"status": "error", **_exception_payload(exc)})
        _log_canary_marker(CANARY_COMPILE_MEMORY_MARKER, payload)
        return

    if memory_analysis is None:
        payload.update({"status": "unsupported", "reason": "compiled executable returned no memory analysis"})
        _log_canary_marker(CANARY_COMPILE_MEMORY_MARKER, payload)
        return

    payload["status"] = "ok"
    payload.update(_compile_memory_payload(memory_analysis))

    temp_bytes = payload.get("temp_bytes")
    summary: dict[str, float | int] = {}
    if isinstance(temp_bytes, int):
        summary["canary/compile/temp_bytes"] = temp_bytes
        summary["canary/compile/temp_gib"] = _bytes_to_gib(temp_bytes)
        if diagnostics.compile_temp_baseline_bytes is not None and diagnostics.compile_temp_baseline_bytes > 0:
            ratio = temp_bytes / diagnostics.compile_temp_baseline_bytes
            payload["temp_baseline_ratio"] = ratio
            summary["canary/compile/temp_baseline_ratio"] = ratio

    peak_bytes = payload.get("peak_bytes")
    if isinstance(peak_bytes, int):
        summary["canary/compile/peak_bytes"] = peak_bytes
        summary["canary/compile/peak_gib"] = _bytes_to_gib(peak_bytes)

    _log_canary_marker(CANARY_COMPILE_MEMORY_MARKER, payload)
    if summary:
        levanter.tracker.log_summary(summary)


def _memory_stat_int(stats: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in stats or stats[key] is None:
            continue
        try:
            return int(stats[key])
        except (TypeError, ValueError):
            continue
    return None


def _jax_device_memory_snapshot() -> tuple[dict[str, Any], int | None]:
    devices = []
    max_bytes = None
    any_supported = False
    for device in jax.local_devices():
        device_payload = _device_identity(device)
        try:
            stats = device.memory_stats()
        except AttributeError as exc:
            devices.append({**device_payload, "status": "unsupported", **_exception_payload(exc)})
            continue
        except Exception as exc:
            devices.append({**device_payload, "status": "error", **_exception_payload(exc)})
            continue

        if stats is None:
            devices.append({**device_payload, "status": "unsupported", "reason": "memory_stats returned None"})
            continue

        try:
            stats_payload = {str(key): _jsonable_scalar(value) for key, value in stats.items()}
            used_bytes = _memory_stat_int(stats, "bytes_in_use", "bytes_used", "used_bytes")
            peak_bytes = _memory_stat_int(stats, "peak_bytes_in_use", "peak_bytes_used", "peak_used_bytes")
            limit_bytes = _memory_stat_int(stats, "bytes_limit", "memory_limit", "total_bytes")
        except Exception as exc:
            devices.append({**device_payload, "status": "error", **_exception_payload(exc)})
            continue

        any_supported = True
        device_payload.update(
            {
                "status": "ok",
                "stats": stats_payload,
                "used_bytes": used_bytes,
                "peak_bytes": peak_bytes,
                "limit_bytes": limit_bytes,
            }
        )
        for key in ("used_bytes", "peak_bytes", "limit_bytes"):
            bytes_value = device_payload.get(key)
            if isinstance(bytes_value, int):
                device_payload[f"{key[:-6]}_gib"] = _bytes_to_gib(bytes_value)
                max_bytes = bytes_value if max_bytes is None else max(max_bytes, bytes_value)
        devices.append(device_payload)

    status = "ok" if any_supported else "unsupported"
    return {"status": status, "devices": devices}, max_bytes


def _nvidia_smi_memory_snapshot() -> tuple[dict[str, Any], int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError as exc:
        return {"status": "unsupported", **_exception_payload(exc)}, None
    except subprocess.TimeoutExpired as exc:
        return {"status": "error", **_exception_payload(exc)}, None

    if result.returncode != 0:
        return {"status": "error", "returncode": result.returncode, "stderr": result.stderr.strip()}, None

    devices = []
    max_used_bytes = None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, used_mib, total_mib = parts
        try:
            device_index = int(index)
            used_mib_value = float(used_mib)
            total_mib_value = float(total_mib)
        except ValueError:
            continue

        used_bytes = int(used_mib_value * 1024**2)
        total_bytes = int(total_mib_value * 1024**2)
        max_used_bytes = used_bytes if max_used_bytes is None else max(max_used_bytes, used_bytes)
        devices.append(
            {
                "index": device_index,
                "used_mib": used_mib_value,
                "total_mib": total_mib_value,
                "used_bytes": used_bytes,
                "total_bytes": total_bytes,
                "used_gib": _bytes_to_gib(used_bytes),
                "total_gib": _bytes_to_gib(total_bytes),
            }
        )

    if not devices:
        return {"status": "unsupported", "reason": "nvidia-smi returned no GPU rows"}, None

    return {"status": "ok", "devices": devices}, max_used_bytes


def _log_canary_gpu_memory(
    diagnostics: CanaryDiagnosticsConfig,
    *,
    phase: str,
    step: int | None,
    peak_used_bytes: int | None,
) -> int | None:
    if not diagnostics.enabled:
        return peak_used_bytes

    try:
        jax_snapshot, jax_max_bytes = _jax_device_memory_snapshot()
    except Exception as exc:
        jax_snapshot, jax_max_bytes = {"status": "error", **_exception_payload(exc)}, None

    if diagnostics.nvidia_smi_enabled:
        try:
            nvidia_snapshot, nvidia_max_bytes = _nvidia_smi_memory_snapshot()
        except Exception as exc:
            nvidia_snapshot, nvidia_max_bytes = {"status": "error", **_exception_payload(exc)}, None
    else:
        nvidia_snapshot, nvidia_max_bytes = {"status": "disabled"}, None

    snapshot_max_bytes = max(
        (value for value in (jax_max_bytes, nvidia_max_bytes) if value is not None),
        default=None,
    )
    if snapshot_max_bytes is not None:
        peak_used_bytes = snapshot_max_bytes if peak_used_bytes is None else max(peak_used_bytes, snapshot_max_bytes)

    payload = _canary_base_payload(phase=phase, step=step)
    payload.update(
        {
            "status": "ok" if snapshot_max_bytes is not None else "unsupported",
            "jax_device_memory": jax_snapshot,
            "nvidia_smi_memory": nvidia_snapshot,
            "snapshot_max_used_bytes": snapshot_max_bytes,
            "peak_used_bytes": peak_used_bytes,
        }
    )
    if snapshot_max_bytes is not None:
        payload["snapshot_max_used_gib"] = _bytes_to_gib(snapshot_max_bytes)
    if peak_used_bytes is not None:
        payload["peak_used_gib"] = _bytes_to_gib(peak_used_bytes)

    _log_canary_marker(CANARY_GPU_MEMORY_MARKER, payload)
    if peak_used_bytes is not None:
        levanter.tracker.log_summary(
            {
                "canary/gpu/peak_used_bytes": peak_used_bytes,
                "canary/gpu/peak_used_gib": _bytes_to_gib(peak_used_bytes),
            }
        )
        if step is not None and snapshot_max_bytes is not None:
            levanter.tracker.log({"canary/gpu/max_used_gib": _bytes_to_gib(snapshot_max_bytes)}, step=step)

    return peak_used_bytes


def _log_canary_runtime_config(
    *,
    diagnostics: CanaryDiagnosticsConfig,
    config: GrugRunConfig,
    batch_schedule: BatchSchedule,
    profiler_enabled: bool,
    profiler_num_steps: int,
) -> None:
    if not diagnostics.enabled:
        return

    devices = jax.devices()
    local_devices = jax.local_devices()
    watch_config = config.trainer.trainer.watch
    payload = _canary_base_payload(phase="after_backend_init")
    payload.update(
        {
            "status": "ok",
            "run_id": config.trainer.trainer.id,
            "env_run_id": os.environ.get("RUN_ID"),
            "config_id": os.environ.get("CANARY_CONFIG_ID"),
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "device_kinds": sorted({str(getattr(device, "device_kind", "unknown")) for device in devices}),
            "device_platforms": sorted({str(getattr(device, "platform", "unknown")) for device in devices}),
            "local_device_kinds": sorted({str(getattr(device, "device_kind", "unknown")) for device in local_devices}),
            "train_batch_size": batch_schedule.batch_size_at_step(0),
            "max_seq_len": config.model.max_seq_len,
            "train_steps": config.trainer.trainer.num_train_steps,
            "watch_enabled": watch_config.is_enabled,
            "watch_interval": watch_config.interval,
            "profiler_enabled": profiler_enabled,
            "profiler_start_step": config.trainer.trainer.profiler.start_step,
            "profiler_num_steps": profiler_num_steps,
            "canary_compile_memory_enabled": diagnostics.compile_memory_enabled,
            "canary_gpu_memory_snapshot_interval": diagnostics.gpu_memory_snapshot_interval,
        }
    )
    _log_canary_marker(CANARY_RUNTIME_CONFIG_MARKER, payload)
    levanter.tracker.log_summary(
        {
            "canary/runtime/device_count": jax.device_count(),
            "canary/runtime/local_device_count": jax.local_device_count(),
            "canary/runtime/train_batch_size": batch_schedule.batch_size_at_step(0),
            "canary/runtime/max_seq_len": config.model.max_seq_len,
            "canary/runtime/train_steps": config.trainer.trainer.num_train_steps,
            "canary/runtime/profiler_enabled": profiler_enabled,
            "canary/runtime/watch_enabled": watch_config.is_enabled,
        }
    )


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_blocks = list(model.blocks)
    moe_idx = 0
    for i, block in enumerate(model.blocks):
        if block.mlp is None:
            continue
        new_bias = -qb_betas[moe_idx]
        new_bias = new_bias - jnp.mean(new_bias)
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_bias)
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
        moe_idx += 1
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        metrics = {"train/loss": loss, **summarized_metrics}
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        params = optax.apply_updates(qb_params, updates)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
                params,
            )

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    canary_diagnostics = _canary_diagnostics_from_env()
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Build data/model state under the trainer mesh so all arrays are sharded consistently.
    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_schedule = trainer.batch_schedule
        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=trainer.num_train_steps)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0
        _log_canary_runtime_config(
            diagnostics=canary_diagnostics,
            config=config,
            batch_schedule=batch_schedule,
            profiler_enabled=profiler_enabled,
            profiler_num_steps=profiler_num_steps,
        )
        canary_peak_used_bytes = _log_canary_gpu_memory(
            canary_diagnostics,
            phase="after_backend_init",
            step=None,
            peak_used_bytes=None,
        )

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
            batch_pspec=config.trainer.train_batch_pspec,
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        canary_peak_used_bytes = _log_canary_gpu_memory(
            canary_diagnostics,
            phase="after_trainer_state_init",
            step=int(state.step),
            peak_used_bytes=canary_peak_used_bytes,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
            )

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                callbacks.profile(
                    str(trainer.log_dir / run_id / "profiler"),
                    profiler_cfg.start_step,
                    profiler_num_steps,
                    profiler_cfg.perfetto_link,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0
        canary_compile_memory_logged = False

        # Main optimization loop.
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                if not canary_compile_memory_logged:
                    _log_canary_compile_memory(
                        canary_diagnostics,
                        train_step,
                        state,
                        batch,
                        compute_watch=compute_watch,
                        step=current_step,
                    )
                    canary_compile_memory_logged = True
                    canary_peak_used_bytes = _log_canary_gpu_memory(
                        canary_diagnostics,
                        phase="after_compile",
                        step=current_step,
                        peak_used_bytes=canary_peak_used_bytes,
                    )

                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])
                if canary_diagnostics.enabled:
                    if step == 0:
                        canary_peak_used_bytes = _log_canary_gpu_memory(
                            canary_diagnostics,
                            phase="after_step_0",
                            step=step,
                            peak_used_bytes=canary_peak_used_bytes,
                        )
                    elif (
                        canary_diagnostics.gpu_memory_snapshot_interval > 0
                        and step % canary_diagnostics.gpu_memory_snapshot_interval == 0
                    ):
                        canary_peak_used_bytes = _log_canary_gpu_memory(
                            canary_diagnostics,
                            phase="after_step_interval",
                            step=step,
                            peak_used_bytes=canary_peak_used_bytes,
                        )

                if jnp.isnan(metrics["train/loss"]):
                    logger.error(f"NaN loss at step {int(state.step)}. Stopping training.")
                    break
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=int(state.step))
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
    )


__all__ = [
    "CanaryDiagnosticsConfig",
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
]
