# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import base64
import gc
import json
import logging
import math
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, cast

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
import levanter.tracker
from levanter import callbacks
from levanter.callbacks import StepInfo
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.mixture import MixtureDataset
from levanter.data.dataset import AsyncDataset
from levanter.data.text import (
    DatasetComponent,
    DpoExample,
    LmDataConfig,
    PreferenceChatLmDatasetFormat,
)
from haliax.jax_utils import is_jax_array_like
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request, _infer_max_pages_from_hbm
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.metrics import Metric, ReductionType
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import (
    barrier_sync_with_tag,
    estimated_free_device_memory,
    parameter_count,
    replicate_model_to_local_mesh,
)
from levanter.utils.mesh import create_local_mesh
from levanter.utils.tree_utils import inference_mode

logger = logging.getLogger(__name__)


def simpo_loss_from_logps(
    avg_logp_chosen: hax.NamedArray | jnp.ndarray,
    avg_logp_rejected: hax.NamedArray | jnp.ndarray,
    *,
    beta: float,
    gamma_beta_ratio: float,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    if isinstance(avg_logp_chosen, hax.NamedArray) or isinstance(avg_logp_rejected, hax.NamedArray):
        if not isinstance(avg_logp_chosen, hax.NamedArray) or not isinstance(avg_logp_rejected, hax.NamedArray):
            raise TypeError(
                "avg_logp_chosen and avg_logp_rejected must both be NamedArray when using named computations."
            )
        logits = (avg_logp_chosen - avg_logp_rejected) - gamma_beta_ratio
        loss = hax.mean(hax.nn.softplus(-beta * logits)).scalar()
        metrics = {
            "simpo_loss": Metric.from_value(loss, ReductionType.MEAN),
            "simpo_chosen_logp": Metric.from_value(hax.mean(avg_logp_chosen).scalar(), ReductionType.MEAN),
            "simpo_rejected_logp": Metric.from_value(hax.mean(avg_logp_rejected).scalar(), ReductionType.MEAN),
            "simpo_margin": Metric.from_value(
                hax.mean(avg_logp_chosen - avg_logp_rejected).scalar(), ReductionType.MEAN
            ),
            "simpo_accuracy": Metric.from_value(hax.mean(logits > 0).scalar(), ReductionType.MEAN),
        }
        return loss, metrics

    logits = (avg_logp_chosen - avg_logp_rejected) - gamma_beta_ratio
    loss = jnp.mean(hax.nn.softplus(-beta * logits))
    metrics = {
        "simpo_loss": Metric.from_value(loss, ReductionType.MEAN),
        "simpo_chosen_logp": Metric.from_value(jnp.mean(avg_logp_chosen), ReductionType.MEAN),
        "simpo_rejected_logp": Metric.from_value(jnp.mean(avg_logp_rejected), ReductionType.MEAN),
        "simpo_margin": Metric.from_value(jnp.mean(avg_logp_chosen - avg_logp_rejected), ReductionType.MEAN),
        "simpo_accuracy": Metric.from_value(jnp.mean(logits > 0), ReductionType.MEAN),
    }
    return loss, metrics


def _average_logp(model: LmHeadModel, example: LmExample, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=None, reduction_axis=(), key=key)
    Pos = example.tokens.resolve_axis("position")
    logp_sum = -hax.sum(nll, axis=Pos)
    denom = hax.sum(example.loss_weight, axis=Pos)
    zeros = hax.zeros_like(logp_sum)
    return hax.where(denom != 0, logp_sum / denom, zeros)


def _validate_preference_chat_formats(config: LmDataConfig) -> None:
    formats: dict[str, PreferenceChatLmDatasetFormat] = {}
    for name, component in config.components.items():
        if not isinstance(component, DatasetComponent):
            raise ValueError(f"SimPO training requires DatasetComponent, got {type(component)} for {name}")
        fmt = component.format
        if not isinstance(fmt, PreferenceChatLmDatasetFormat):
            raise ValueError(
                f"SimPO training requires preference_chat datasets. Component '{name}' has format {type(fmt).__name__}"
            )
        formats[name] = fmt

    packed = {name: fmt for name, fmt in formats.items() if fmt.pack}
    if packed:
        bad = ", ".join(sorted(packed.keys()))
        raise ValueError(f"Packed preference_chat datasets are not supported yet. Packed datasets: {bad}")

    non_raise = {name: fmt for name, fmt in formats.items() if fmt.slice_strategy != "raise"}
    if non_raise:
        bad = ", ".join(sorted(non_raise.keys()))
        raise ValueError(f"preference_chat slice_strategy must be 'raise' for now. Invalid datasets: {bad}")


def _num_validation_sequences(total_sequences: int, fraction: float) -> int:
    if total_sequences <= 1:
        return 0
    if fraction <= 0:
        return 0
    num_val = int(total_sequences * fraction)
    if num_val <= 0:
        num_val = 1
    if num_val >= total_sequences:
        num_val = total_sequences - 1
    return num_val


def _build_validation_split(
    config: LmDataConfig,
    Pos: Axis,
    *,
    batch_schedule: BatchSchedule,
    key: jrandom.PRNGKey,
    fraction: float,
) -> tuple[AsyncDataset[DpoExample], dict[str, AsyncDataset[DpoExample]]]:
    """Build train/validation split from LmDataConfig by holding out a fraction of each component."""
    train_caches = config.build_caches("train")
    token_datasets = config.build_token_datasets(train_caches, Pos, split="train")

    num_validation_sequences: dict[str, int] = {}
    for name, dataset in token_datasets.items():
        total_len = len(dataset.as_sync_dataset())
        num_val = _num_validation_sequences(total_len, fraction)
        if num_val > 0:
            num_validation_sequences[name] = num_val

    if not num_validation_sequences:
        train_dataset = cast(AsyncDataset[DpoExample], config.train_set(Pos, batch_schedule, key=key))
        return train_dataset, {}

    config_with_val = dataclasses.replace(config, num_validation_sequences=num_validation_sequences)
    train_dataset = cast(AsyncDataset[DpoExample], config_with_val.train_set(Pos, batch_schedule, key=key))
    validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config_with_val.validation_sets(Pos))
    return train_dataset, validation_sets


InferenceMode = Literal["global_mesh", "host_data_parallel"]


def _block_until_ready_tree(tree) -> None:
    def _block(x):
        if is_jax_array_like(x):
            jax.block_until_ready(x)
        return x

    jax.tree_util.tree_map(_block, tree)


def _normalize_eval_at_steps(eval_at_steps: list[int] | None) -> set[int] | None:
    if eval_at_steps is None:
        return None

    normalized: set[int] = set()
    for step in eval_at_steps:
        step_int = int(step)
        if step_int <= 0:
            raise ValueError(f"inference_eval.eval_at_steps must be positive, got {step_int}")
        normalized.add(step_int)
    return normalized


def _should_run_inference_eval_step(
    *,
    step: int,
    eval_every: int,
    eval_at_steps: set[int] | None,
) -> bool:
    if step <= 0:
        return False
    if eval_at_steps is not None:
        return step in eval_at_steps
    if eval_every <= 0:
        raise ValueError(f"inference_eval.eval_every must be positive, got {eval_every}")
    return step % eval_every == 0


def _resolve_inference_prompts(inference_config: "InferenceEvalConfig") -> list[str]:
    if inference_config.prompts_path is not None:
        prompts_path = Path(inference_config.prompts_path)
        if not prompts_path.exists():
            raise ValueError(f"inference_eval.prompts_path does not exist: {prompts_path}")
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"inference_eval.prompts_path contains no non-empty prompts: {prompts_path}")
        return prompts

    if inference_config.synthetic_prompt_count is not None:
        count = int(inference_config.synthetic_prompt_count)
        if count <= 0:
            raise ValueError(f"inference_eval.synthetic_prompt_count must be positive, got {count}")
        template = inference_config.synthetic_prompt_template
        return [template.format(index=i) for i in range(count)]

    if not inference_config.prompts:
        raise ValueError(
            "inference_eval.prompts must be non-empty when prompts_path and synthetic_prompt_count are unset."
        )
    return list(inference_config.prompts)


def _prompt_shard_bounds(num_prompts: int, process_index: int, process_count: int) -> tuple[int, int]:
    if process_count <= 0:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if process_index < 0 or process_index >= process_count:
        raise ValueError(f"process_index={process_index} out of range for process_count={process_count}")
    start = (process_index * num_prompts) // process_count
    end = ((process_index + 1) * num_prompts) // process_count
    return start, end


def _shard_prompts_for_host(
    prompts: list[str],
    *,
    process_index: int,
    process_count: int,
) -> tuple[list[str], int, int]:
    start, end = _prompt_shard_bounds(len(prompts), process_index, process_count)
    return prompts[start:end], start, end


def _encode_host_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(zlib.compress(serialized)).decode("ascii")


def _decode_host_payload(payload: str) -> dict[str, object]:
    decoded = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    parsed = json.loads(decoded.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected host payload dict, got {type(parsed).__name__}.")
    return parsed


def _gather_host_payload_to_leader(
    *,
    local_payload: dict[str, object],
    process_index: int,
    process_count: int,
    tag: str,
    timeout_sec: float = 600.0,
) -> list[dict[str, object]] | None:
    if process_count == 1:
        return [local_payload]

    import jax._src.distributed as distributed

    client = distributed.global_state.client
    if client is None:
        raise RuntimeError("Host payload gather requires jax distributed client to be initialized.")

    payload_key = f"{tag}_payload_{process_index:04d}"
    payload_value = _encode_host_payload(local_payload)
    timeout_ms = int(timeout_sec * 1000.0)
    client.key_value_set(payload_key, payload_value)
    client.wait_at_barrier(f"{tag}_after_set", timeout_in_ms=timeout_ms)

    gathered_payloads: list[dict[str, object]] | None = None
    if process_index == 0:
        gathered_payloads = []
        for host_index in range(process_count):
            host_key = f"{tag}_payload_{host_index:04d}"
            host_payload = client.blocking_key_value_get(host_key, timeout_in_ms=timeout_ms)
            gathered_payloads.append(_decode_host_payload(host_payload))

    client.wait_at_barrier(f"{tag}_after_get", timeout_in_ms=timeout_ms)
    return gathered_payloads


def _inference_eval_host_output_path(
    *,
    output_dir: str,
    step: int,
    process_index: int,
    process_count: int,
) -> Path:
    return Path(output_dir) / f"step_{step:06d}" / f"host_{process_index:04d}_of_{process_count:04d}.jsonl"


def _write_inference_rows_jsonl(output_path: Path, rows: list[dict[str, object]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _build_host_local_engine_config(
    *,
    inference_config: "InferenceEvalConfig",
    max_seq_len: int,
    prompt_tokens: list[list[int]],
    devices,
    shard_locally: bool,
) -> InferenceEngineConfig:
    prompt_count = len(prompt_tokens)
    local_prompt_count = max(1, prompt_count)

    base_max_seqs = int(inference_config.max_seqs) if inference_config.max_seqs is not None else local_prompt_count
    if base_max_seqs <= 0:
        raise ValueError(f"inference_eval.max_seqs must be positive when set, got {base_max_seqs}")
    max_seqs = min(base_max_seqs, local_prompt_count) if shard_locally else base_max_seqs
    if prompt_count > max_seqs:
        raise ValueError(
            f"inference_eval engine max_seqs={max_seqs} cannot cover prompt_count={prompt_count} "
            f"(shard_locally={shard_locally})."
        )

    base_max_seqs_in_prefill = (
        int(inference_config.max_seqs_in_prefill) if inference_config.max_seqs_in_prefill is not None else max_seqs
    )
    if base_max_seqs_in_prefill <= 0:
        raise ValueError(
            f"inference_eval.max_seqs_in_prefill must be positive when set, got {base_max_seqs_in_prefill}"
        )
    max_seqs_in_prefill = (
        min(base_max_seqs_in_prefill, local_prompt_count) if shard_locally else base_max_seqs_in_prefill
    )
    if prompt_count > max_seqs_in_prefill:
        raise ValueError(
            f"inference_eval engine max_seqs_in_prefill={max_seqs_in_prefill} cannot cover prompt_count={prompt_count} "
            f"(shard_locally={shard_locally})."
        )

    max_tokens_per_round = (
        int(inference_config.max_tokens_per_round) if inference_config.max_tokens_per_round is not None else None
    )
    if max_tokens_per_round is not None and max_tokens_per_round <= 0:
        raise ValueError(f"inference_eval.max_tokens_per_round must be positive when set, got {max_tokens_per_round}")
    effective_tpr = max_tokens_per_round if max_tokens_per_round is not None else max_seqs
    max_queued_tokens = max(int(inference_config.max_queued_tokens), max_seqs, max_seqs_in_prefill, effective_tpr)

    max_prefill_size = (
        int(inference_config.max_prefill_size) if inference_config.max_prefill_size is not None else max_seq_len
    )
    if max_prefill_size <= 0:
        raise ValueError(f"inference_eval.max_prefill_size must be positive when set, got {max_prefill_size}")

    return InferenceEngineConfig(
        max_seq_len=max_seq_len,
        hbm_utilization=inference_config.hbm_utilization,
        page_size=inference_config.page_size,
        max_rounds=inference_config.max_rounds,
        reset_mode=inference_config.reset_mode,
        cleanup_mode=inference_config.cleanup_mode,
        max_stop_seqs=inference_config.max_stop_seqs,
        max_stop_tokens=inference_config.max_stop_tokens,
        seed=inference_config.seed,
        max_seqs=max_seqs,
        max_pages=inference_config.max_pages,
        max_queued_tokens=max_queued_tokens,
        max_seqs_in_prefill=max_seqs_in_prefill,
        max_prefill_size=max_prefill_size,
        max_tokens_per_round=max_tokens_per_round,
        devices=devices,
    )


def _create_inference_eval_callback(
    inference_config: "InferenceEvalConfig",
    tokenizer,
    model_max_seq_len: int,
    compute_axis_mapping,
):
    prompts = _resolve_inference_prompts(inference_config)
    eval_at_steps = _normalize_eval_at_steps(inference_config.eval_at_steps)
    max_seq_len = inference_config.max_seq_len or model_max_seq_len
    if max_seq_len <= 0:
        raise ValueError(f"inference_eval.max_seq_len must be positive, got {max_seq_len}")
    if max_seq_len > model_max_seq_len:
        raise ValueError(
            "inference_eval.max_seq_len cannot exceed model max sequence length: "
            f"{max_seq_len} > {model_max_seq_len}"
        )
    if inference_config.inference_mode not in ("global_mesh", "host_data_parallel"):
        raise ValueError(
            f"Unknown inference_eval.inference_mode={inference_config.inference_mode!r}. "
            "Expected one of: 'global_mesh', 'host_data_parallel'."
        )
    if inference_config.max_logged_samples < 0:
        raise ValueError(
            "inference_eval.max_logged_samples must be non-negative, " f"got {inference_config.max_logged_samples}"
        )

    def _run_with_inference_context(*, is_multihost: bool, local_mesh, fn):
        if is_multihost:
            with hax.partitioning.set_mesh(local_mesh):
                with hax.axis_mapping({}):
                    return fn()
        with hax.axis_mapping(compute_axis_mapping):
            return fn()

    def inference_eval_callback(step: StepInfo):
        step_number = int(jax.device_get(step.state.step))
        if not _should_run_inference_eval_step(
            step=step_number,
            eval_every=inference_config.eval_every,
            eval_at_steps=eval_at_steps,
        ):
            return

        is_multihost = jax.process_count() > 1
        process_index = int(jax.process_index())
        process_count = int(jax.process_count())
        is_leader = process_index == 0

        if is_multihost and not inference_config.allow_multihost:
            if is_leader:
                logger.info(
                    "[Step %d] Skipping inference evaluation on multi-host (allow_multihost=False).",
                    step_number,
                )
            return

        _block_until_ready_tree(step.state)
        if is_multihost:
            barrier_sync_with_tag(f"levanter_inference_pre_{step_number}", timeout=120.0)

        total_start_time = time.perf_counter()
        local_devices = list(jax.local_devices())
        memory_device = local_devices[0] if local_devices else None
        hbm_free_before = estimated_free_device_memory(memory_device) if memory_device is not None else None

        model = inference_mode(step.eval_model, True)
        local_model = None
        local_mesh = None
        engine = None

        try:
            local_mesh_devices = None
            inference_model = model
            if is_multihost:
                local_mesh = create_local_mesh(devices=local_devices)
                local_mesh_devices = list(local_mesh.devices.flat)
                replication_start = time.perf_counter()
                local_model = replicate_model_to_local_mesh(model, local_mesh=local_mesh)
                _block_until_ready_tree(local_model)
                replication_time = time.perf_counter() - replication_start
                if is_leader:
                    logger.info("[Step %d] Inference model replication time: %.2fs", step_number, replication_time)
                inference_model = local_model

            if inference_config.inference_mode == "host_data_parallel":
                local_prompts, shard_start, shard_end = _shard_prompts_for_host(
                    prompts,
                    process_index=process_index,
                    process_count=process_count,
                )
            else:
                local_prompts = list(prompts)
                shard_start, shard_end = 0, len(prompts)

            prompt_tokens = [tokenizer.encode(prompt) for prompt in local_prompts]
            engine_config = _build_host_local_engine_config(
                inference_config=inference_config,
                max_seq_len=max_seq_len,
                prompt_tokens=prompt_tokens,
                devices=local_mesh_devices,
                shard_locally=inference_config.inference_mode == "host_data_parallel" and is_multihost,
            )

            if engine_config.max_pages is None:
                inferred_pages = _run_with_inference_context(
                    is_multihost=is_multihost,
                    local_mesh=local_mesh,
                    fn=lambda: _infer_max_pages_from_hbm(inference_model, engine_config),
                )
                engine_config = dataclasses.replace(engine_config, max_pages=int(inferred_pages))

            engine_start = time.perf_counter()
            engine = _run_with_inference_context(
                is_multihost=is_multihost,
                local_mesh=local_mesh,
                fn=lambda: InferenceEngine.from_model_with_config(
                    model=inference_model,
                    tokenizer=tokenizer,
                    config=engine_config,
                ),
            )
            engine_creation_time = time.perf_counter() - engine_start

            hbm_free_after_engine = estimated_free_device_memory(memory_device) if memory_device is not None else None

            request_meta: list[tuple[int, str]] = []
            requests: list[Request] = []
            base_key = jrandom.PRNGKey(step_number + int(inference_config.seed))
            for local_idx, tokens in enumerate(prompt_tokens):
                global_prompt_index = shard_start + local_idx
                request_key = jrandom.fold_in(base_key, global_prompt_index)
                requests.append(
                    Request(
                        prompt_tokens=tokens,
                        request_id=global_prompt_index,
                        decode_params=SeqDecodingParams(
                            max_num_tokens=jnp.array(len(tokens) + inference_config.max_new_tokens, dtype=jnp.int32),
                            temperature=jnp.array(inference_config.temperature, dtype=jnp.float32),
                            stop_tokens=None,
                            key=request_key,
                        ),
                        n_generations=1,
                    )
                )
                request_meta.append((global_prompt_index, local_prompts[local_idx]))

            if requests:
                generation_start = time.perf_counter()
                result = _run_with_inference_context(
                    is_multihost=is_multihost,
                    local_mesh=local_mesh,
                    fn=lambda: engine.generate(requests),
                )
                generation_time = time.perf_counter() - generation_start
                for tokens in result.tokens:
                    jax.block_until_ready(tokens)
                _block_until_ready_tree(engine.gen_state)
                total_generated = int(result.total_generated)
                result_tokens = result.tokens
            else:
                generation_time = 0.0
                total_generated = 0
                result_tokens = []

            hbm_free_after_generation = (
                estimated_free_device_memory(memory_device) if memory_device is not None else None
            )
            tokens_per_second = total_generated / generation_time if generation_time > 0 else 0.0

            local_rows: list[dict[str, object]] = []
            for (global_prompt_index, prompt), tokens in zip(request_meta, result_tokens):
                local_rows.append(
                    {
                        "step": step_number,
                        "process_index": process_index,
                        "process_count": process_count,
                        "global_prompt_index": global_prompt_index,
                        "prompt": prompt,
                        "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "generated_token_count": len(tokens),
                    }
                )

            if inference_config.host_data_parallel_output_dir is not None:
                output_path = _inference_eval_host_output_path(
                    output_dir=inference_config.host_data_parallel_output_dir,
                    step=step_number,
                    process_index=process_index,
                    process_count=process_count,
                )
                rows_written = _write_inference_rows_jsonl(output_path, local_rows)
                logger.info(
                    "[Step %d] Inference host rows written: process=%d/%d rows=%d path=%s",
                    step_number,
                    process_index,
                    process_count,
                    rows_written,
                    output_path,
                )

            samples_per_host = (
                0
                if inference_config.max_logged_samples == 0
                else max(1, math.ceil(inference_config.max_logged_samples / max(process_count, 1)))
            )
            local_payload: dict[str, object] = {
                "process_index": process_index,
                "process_count": process_count,
                "local_num_prompts": len(local_prompts),
                "local_total_generated": total_generated,
                "local_generation_time_sec": generation_time,
                "local_engine_creation_time_sec": engine_creation_time,
                "local_tokens_per_second": tokens_per_second,
                "shard_start": shard_start,
                "shard_end": shard_end,
                "sample_rows": local_rows[:samples_per_host],
            }

            if inference_config.inference_mode == "host_data_parallel":
                gathered_payloads = _gather_host_payload_to_leader(
                    local_payload=local_payload,
                    process_index=process_index,
                    process_count=process_count,
                    tag=f"train_simpo_inference_eval_step_{step_number}",
                )
            else:
                gathered_payloads = [local_payload] if is_leader else None

            if is_leader:
                if gathered_payloads is None:
                    raise RuntimeError("Leader expected gathered host payloads, got None.")

                total_tokens = int(sum(int(p["local_total_generated"]) for p in gathered_payloads))
                total_prompts = int(sum(int(p["local_num_prompts"]) for p in gathered_payloads))
                max_generation_time = float(max(float(p["local_generation_time_sec"]) for p in gathered_payloads))
                max_engine_creation_time = float(
                    max(float(p["local_engine_creation_time_sec"]) for p in gathered_payloads)
                )
                total_time = time.perf_counter() - total_start_time
                round_tokens_per_second = total_tokens / max_generation_time if max_generation_time > 0 else 0.0

                gathered_samples: list[dict[str, object]] = []
                for payload in gathered_payloads:
                    payload_samples = payload.get("sample_rows", [])
                    if not isinstance(payload_samples, list):
                        continue
                    for row in payload_samples:
                        if isinstance(row, dict):
                            gathered_samples.append(row)
                gathered_samples.sort(key=lambda row: (int(row["global_prompt_index"]), int(row["process_index"])))
                gathered_samples = gathered_samples[: inference_config.max_logged_samples]

                if gathered_samples:
                    try:
                        import wandb

                        samples_table = wandb.Table(
                            columns=["step", "prompt_id", "process", "prompt", "generated_text", "num_tokens"]
                        )
                        for sample in gathered_samples:
                            samples_table.add_data(
                                int(sample["step"]),
                                int(sample["global_prompt_index"]),
                                int(sample["process_index"]),
                                str(sample["prompt"]),
                                str(sample["generated_text"]),
                                int(sample["generated_token_count"]),
                            )
                        levanter.tracker.log({"inference_eval/samples": samples_table}, step=step_number)
                    except ImportError:
                        pass

                metrics: dict[str, float | int] = {
                    "inference_eval/total_tokens": total_tokens,
                    "inference_eval/num_prompts": total_prompts,
                    "inference_eval/avg_tokens_per_prompt": (
                        (total_tokens / total_prompts) if total_prompts > 0 else 0.0
                    ),
                    "inference_eval/engine_creation_time_sec": max_engine_creation_time,
                    "inference_eval/generation_time_sec": max_generation_time,
                    "inference_eval/tokens_per_second": round_tokens_per_second,
                    "inference_eval/total_callback_time_sec": total_time,
                    "inference_eval/num_hosts": process_count,
                    "inference_eval/max_seq_len": max_seq_len,
                }
                if hbm_free_before is not None:
                    metrics["inference_eval/hbm_free_before_gib"] = hbm_free_before
                if hbm_free_after_engine is not None:
                    metrics["inference_eval/hbm_free_after_engine_gib"] = hbm_free_after_engine
                if hbm_free_after_generation is not None:
                    metrics["inference_eval/hbm_free_after_gen_gib"] = hbm_free_after_generation
                if hbm_free_before is not None and hbm_free_after_engine is not None:
                    metrics["inference_eval/hbm_used_by_engine_gib"] = hbm_free_before - hbm_free_after_engine

                levanter.tracker.log(metrics, step=step_number)

                logger.info(
                    "[Step %d] Inference eval complete: mode=%s prompts=%d tokens=%d generation_time=%.2fs",
                    step_number,
                    inference_config.inference_mode,
                    total_prompts,
                    total_tokens,
                    max_generation_time,
                )
        finally:
            if engine is not None:
                del engine
            if local_model is not None:
                del local_model
            gc.collect()
            if is_multihost:
                barrier_sync_with_tag(f"levanter_inference_post_{step_number}", timeout=120.0)

    return inference_eval_callback


@dataclass
class InferenceEvalConfig:
    """Configuration for running inference evaluation during training."""

    enabled: bool = False
    """Whether to run inference evaluation."""
    eval_every: int = 10
    """Run inference every N steps."""
    eval_at_steps: list[int] | None = None
    """Run inference at these exact step numbers. When set, this overrides eval_every."""
    inference_mode: InferenceMode = "global_mesh"
    """Inference execution mode: global_mesh or host_data_parallel."""
    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
        ]
    )
    """Prompts to generate from during evaluation."""
    prompts_path: str | None = None
    """Optional local text file with one prompt per line."""
    synthetic_prompt_count: int | None = None
    """Generate this many synthetic prompts using synthetic_prompt_template."""
    synthetic_prompt_template: str = "Synthetic prompt {index}: explain one key concept in machine learning."
    """Template used when synthetic_prompt_count is set. Must accept {index}."""
    max_new_tokens: int = 128
    """Maximum number of new tokens to generate per prompt."""
    temperature: float = 0.7
    """Sampling temperature for generation."""
    max_seq_len: int | None = 512
    """Maximum sequence length for inference. Kept small to avoid OOM during training."""

    hbm_utilization: float = 0.2
    """Fraction of HBM to use for inference KV cache (keep low to avoid OOM during training)."""
    page_size: int = 128
    """KV page size used by inference engine."""
    max_pages: int | None = 64
    """Maximum number of KV cache pages. If None, infer from hbm_utilization."""
    max_rounds: int = 32
    """Maximum decode rounds for each inference eval call."""
    max_tokens_per_round: int | None = None
    """Maximum tokens packed per decode round."""
    max_queued_tokens: int = 512
    """Queue capacity for prefill/decode token handoff."""
    max_seqs: int | None = None
    """Maximum sequences in local eval engine. Defaults to prompt count."""
    max_seqs_in_prefill: int | None = None
    """Maximum sequences to prefill at once. Defaults to max_seqs."""
    max_prefill_size: int | None = None
    """Prefill token budget. Defaults to max_seq_len."""
    reset_mode: Literal["logical", "physical"] = "physical"
    """Inference reset mode."""
    cleanup_mode: Literal["none", "end", "incremental"] = "end"
    """Inference cleanup mode."""
    max_stop_seqs: int = 4
    """Max stop sequences per request for eval engine sizing."""
    max_stop_tokens: int = 16
    """Max stop tokens per stop sequence for eval engine sizing."""
    seed: int = 0
    """Base seed offset for deterministic eval request keys."""

    allow_multihost: bool = True
    """If True, run inference on all hosts in multi-host training."""
    host_data_parallel_output_dir: str | None = "train_simpo_inference_eval_outputs"
    """Optional output dir for host-local inference JSONL rows."""
    max_logged_samples: int = 16
    """Maximum number of sample rows to emit to tracker per eval step."""


@dataclass
class TrainSimpoConfig:
    data: LmDataConfig = field(default_factory=LmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    validation_split_fraction: float | None = 0.1

    initialize_from_hf: bool | str = False
    use_hf_model_config: bool = False

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None

    inference_eval: InferenceEvalConfig = field(default_factory=InferenceEvalConfig)
    """Configuration for running inference evaluation during training."""


def main(config: TrainSimpoConfig):
    _validate_preference_chat_formats(config.data)

    tokenizer = config.data.the_tokenizer

    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def loss_function(model: LmHeadModel, example: DpoExample, *, key=None):
        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        avg_logp_chosen = _average_logp(model, example.chosen, key=key_chosen)
        avg_logp_rejected = _average_logp(model, example.rejected, key=key_rejected)

        return simpo_loss_from_logps(
            avg_logp_chosen,
            avg_logp_rejected,
            beta=config.beta,
            gamma_beta_ratio=config.gamma_beta_ratio,
        )

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
        del loader_key

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        parameter_axis_mapping = trainer.parameter_axis_mapping

        model_max_seq_len = config.model.max_seq_len
        train_length = config.train_seq_len if config.train_seq_len is not None else model_max_seq_len

        if train_length <= 0:
            raise ValueError(f"train_length must be positive, got {train_length}")

        if train_length > model_max_seq_len:
            raise ValueError(f"train_length ({train_length}) cannot exceed model max_seq_len ({model_max_seq_len}).")

        if train_length != model_max_seq_len:
            logger.info(f"Training with sequence length {train_length} (model supports {model_max_seq_len}).")

        Pos = config.model.max_Pos.resize(train_length)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        validation_sets: dict[str, AsyncDataset[DpoExample]] = {}
        if config.validation_split_fraction is not None:
            fraction = config.validation_split_fraction
            if fraction < 0 or fraction >= 1:
                raise ValueError(f"validation_split_fraction must be in [0, 1), got {fraction}")
            train_dataset, validation_sets = _build_validation_split(
                config.data,
                Pos,
                batch_schedule=config.trainer.batch_schedule,
                key=data_key,
                fraction=fraction,
            )
        else:
            train_dataset = cast(
                AsyncDataset[DpoExample],
                config.data.train_set(Pos, config.trainer.batch_schedule, key=data_key),
            )
            validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config.data.validation_sets(Pos))

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0:
            if config.initialize_from_hf:
                assert converter is not None
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model if not config.use_hf_model_config else None,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            elif config.initialize_from_checkpoint_path is not None:
                state = load_checkpoint(state, config.initialize_from_checkpoint_path)
                state = dataclasses.replace(state, step=jnp.array(0))
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        flops_per_token = config.model.flops_per_token(vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )

        if isinstance(train_dataset, MixtureDataset):
            last_stage = -1

            def log_mixture_weights(step_info):
                nonlocal last_stage
                seq_index = trainer.config.batch_schedule.global_data_offset_by_step(step_info.step)
                block_id = seq_index // train_dataset.block_size
                stage = train_dataset._get_stage_for_block(block_id)
                weights = train_dataset.weight_stages[stage][1]
                if stage != last_stage:
                    metrics = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
                    metrics["mixture/stage"] = stage
                    levanter.tracker.log(metrics, step=step_info.step)
                    last_stage = stage

            trainer.add_hook(log_mixture_weights, every=1)

        if validation_sets:
            for name, dataset in validation_sets.items():
                trainer.add_eval_hook(dataset, name=name or None)
        else:
            logger.warning("No validation datasets provided.")

        if config.inference_eval.enabled:
            resolved_prompts = _resolve_inference_prompts(config.inference_eval)
            normalized_eval_steps = _normalize_eval_at_steps(config.inference_eval.eval_at_steps)
            if normalized_eval_steps is None:
                schedule_desc = f"every {config.inference_eval.eval_every} steps"
            else:
                schedule_desc = f"at steps {sorted(normalized_eval_steps)}"
            logger.info(
                "Inference evaluation enabled: mode=%s schedule=%s prompts=%d",
                config.inference_eval.inference_mode,
                schedule_desc,
                len(resolved_prompts),
            )
            inference_callback = _create_inference_eval_callback(
                config.inference_eval,
                tokenizer,
                model_max_seq_len=model_max_seq_len,
                compute_axis_mapping=trainer.compute_axis_mapping,
            )
            hook_every = 1 if normalized_eval_steps is not None else config.inference_eval.eval_every
            if hook_every <= 0:
                raise ValueError(
                    f"inference_eval.eval_every must be positive when eval_at_steps is unset, got {hook_every}"
                )
            trainer.add_hook(inference_callback, every=hook_every)

        if config.hf_save_path is not None and config.hf_save_steps is not None:
            assert converter is not None, "converter must be set when saving HF checkpoints"
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            def save_policy_hf_checkpoint(step):
                if step.step == 0:
                    return
                upload_to_hf = config.hf_upload or False
                hf_upload_kwargs = {}
                if upload_to_hf is not None:
                    hf_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"

                converter.save_pretrained(
                    step.eval_model,
                    os.path.join(full_save_path, f"step-{step.step}"),
                    upload_to_hf=upload_to_hf,
                    dtype=save_dtype,
                    **hf_upload_kwargs,
                )

            trainer.add_hook(save_policy_hf_checkpoint, every=config.hf_save_steps)

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        last_info = trainer.train(state, train_loader)

        if trainer.config.checkpointer is not None:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
