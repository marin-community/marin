# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import base64
import gc
import json
import logging
import math
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional, TypedDict

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.utils import INVALID
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import (
    barrier_sync_with_tag,
    estimated_free_device_memory,
    multihost_broadcast_sync,
    replicate_model_to_local_mesh,
    use_cpu_device,
)
from levanter.utils.mesh import create_local_mesh

logger = logging.getLogger(__name__)

InferenceMode = Literal["global_mesh", "host_data_parallel"]


def _create_engine(
    *,
    model: LmHeadModel,
    tokenizer,
    engine_config: InferenceEngineConfig,
    memory_device,
    is_leader: bool,
) -> tuple[InferenceEngine, float, float | None]:
    engine_start_time = time.perf_counter()
    if is_leader:
        logger.info("Engine creation starting")
    try:
        engine = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=engine_config)
    except Exception:
        logger.exception("Engine creation failed")
        raise
    if is_leader:
        logger.info("Engine creation completed")
    engine_creation_time = time.perf_counter() - engine_start_time
    hbm_free_after_engine = estimated_free_device_memory(memory_device)
    return engine, engine_creation_time, hbm_free_after_engine


def _build_requests_for_rounds(
    *,
    prompt_ids: list[list[int]],
    stop_tokens: hax.NamedArray | None,
    config: "SampleLmMultihostConfig",
    base_key,
    rounds: int,
    round_offset: int = 0,
    prompt_id_offset: int = 0,
    total_num_prompts: int | None = None,
) -> tuple[list[Request], list[tuple[int, int]]]:
    requests: list[Request] = []
    request_meta: list[tuple[int, int]] = []
    num_prompts = len(prompt_ids)
    prompts_per_round = total_num_prompts if total_num_prompts is not None else num_prompts
    for round_delta in range(rounds):
        round_index = round_offset + round_delta
        round_key = jrandom.fold_in(base_key, round_index)
        for prompt_index, tokens in enumerate(prompt_ids):
            global_prompt_index = prompt_id_offset + prompt_index
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(tokens) + config.max_new_tokens, dtype=jnp.int32),
                stop_tokens=stop_tokens,
                temperature=jnp.array(config.temperature, dtype=jnp.float32),
                key=jrandom.fold_in(round_key, prompt_index),
            )
            requests.append(
                Request(
                    prompt_tokens=list(map(int, tokens)),
                    request_id=round_index * prompts_per_round + global_prompt_index,
                    decode_params=seq_params,
                    n_generations=int(config.n_generations),
                )
            )
            request_meta.append((round_index, global_prompt_index))
    return requests, request_meta


@dataclass
class SampleLmMultihostConfig:
    """Configuration for multi-host text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    engine: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(max_seq_len=1024))

    prompts: list[str] | str | tuple[str, ...] = (
        "What is the square root of 17?",
        "Four score and seven years ago, our",
        "On the first day of Christmas, my true love gave to me",
        "In a hole in the ground there lived a hobbit, not a nasty, dirty, wet hole",
    )
    stop_sequence: str | None = "."
    max_new_tokens: int = 192
    temperature: float = 0.7
    seed: int = 2

    n_generations: int = 1
    n_rounds: int = 1
    inference_mode: InferenceMode = "global_mesh"
    """Execution mode for multi-host inference.

    - ``global_mesh``: existing path with one global mesh across all hosts.
    - ``host_data_parallel``: planned M10 path for host-sharded prompt execution.
    """
    host_data_parallel_output_dir: str | None = "host_data_parallel_outputs"
    """Directory for per-host JSONL generation dumps in host-data-parallel mode.

    Set to ``None`` to disable local dump files.
    """
    enable_host_rows_gather: bool = True
    """Whether to gather host rows to leader and emit merged all-host JSONL output (M10.6)."""
    skip_round_barrier: bool = False
    skip_samples_table: bool = False
    force_post_generate_block_until_ready: bool = False
    force_post_generate_barrier: bool = False
    barrier_before_leader_postprocess: bool = False
    skip_leader_postprocess: bool = False
    skip_metrics_log: bool = False
    force_samples_rows_log: bool = False
    defer_tracker_logs_until_end: bool = True
    emit_minimal_tracker_probe: bool = False
    emit_minimal_tracker_probe_all_hosts: bool = False
    leader_postprocess_sleep_sec: float = 0.0

    log_kernel_jaxprs_path: Optional[str] = None


def _require(condition: bool, message: str) -> None:
    """Raise a ValueError when a precondition is not met."""

    if not condition:
        raise ValueError(message)


def _load_model(config: SampleLmMultihostConfig, vocab_axis: Axis, tokenizer, *, key) -> LmHeadModel:
    """Load a model either from a checkpoint or HF repo."""

    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    mp = config.trainer.mp

    if config.checkpoint_path is not None:
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, vocab_axis, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = mp.cast_to_compute(model)
        return model

    assert hasattr(config.model, "hf_checkpoint_converter"), "model config lacks HF loader"
    converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)

    # Get config from HF checkpoint, but override with user-specified model settings
    # that materially affect inference kernel behavior.
    hf_config = converter.hf_config_from_hf_checkpoint(config.hf_checkpoint)
    overrides = {
        "use_tpu_ragged_paged_attention": config.model.use_tpu_ragged_paged_attention,
        "ragged_paged_q_block_size": config.model.ragged_paged_q_block_size,
        "ragged_paged_kv_block_pages": config.model.ragged_paged_kv_block_pages,
    }
    merged_config = converter.config_from_hf_config(hf_config, overrides=overrides)
    logger.info(
        "HF merged model config: ragged_enabled=%s q_block=%s kv_block_pages=%s",
        getattr(merged_config, "use_tpu_ragged_paged_attention", "n/a"),
        getattr(merged_config, "ragged_paged_q_block_size", "n/a"),
        getattr(merged_config, "ragged_paged_kv_block_pages", "n/a"),
    )

    model = converter.load_pretrained(
        config.model.model_type, ref=config.hf_checkpoint, config=merged_config, dtype=config.trainer.mp.compute_dtype
    )
    return model  # type: ignore[return-value]


def _normalize_prompts(prompts: list[str] | str | tuple[str, ...]) -> list[str]:
    """Normalize prompt input into a list of strings."""

    if isinstance(prompts, str):
        return [prompts]
    return list(prompts)


def _validate_engine_config(
    config: SampleLmMultihostConfig,
    engine_config: InferenceEngineConfig,
    prompt_ids: list[list[int]],
    stop_ids: list[int] | None,
    *,
    is_multihost: bool,
) -> None:
    """Validate engine sizing constraints for deterministic multi-host inference.

    Since we use per-round mode (each round is a separate generate() call with reset between),
    we only need capacity for ONE round at a time.
    """

    if is_multihost:
        _require(engine_config.max_pages is not None, "engine.max_pages must be explicit for multi-host inference")
        _require(
            engine_config.max_prefill_size is not None,
            "engine.max_prefill_size must be explicit for multi-host inference",
        )

    _require(engine_config.max_rounds <= engine_config.max_seq_len, "engine.max_rounds must be <= engine.max_seq_len")
    _require(config.n_generations > 0, "n_generations must be >= 1")

    num_prompts = len(prompt_ids)
    # Per-round mode: only need capacity for one round at a time
    sequences_per_round = num_prompts * int(config.n_generations)
    prompt_tokens_per_round = sum(len(tokens) for tokens in prompt_ids)
    max_prompt_len = max((len(tokens) for tokens in prompt_ids), default=0)

    _require(
        engine_config.max_seqs_in_prefill >= num_prompts,
        "engine.max_seqs_in_prefill must cover all prompts per round",
    )
    _require(engine_config.max_seqs >= sequences_per_round, "engine.max_seqs must cover all generations per round")
    _require(engine_config.max_seq_len >= max_prompt_len + config.max_new_tokens, "engine.max_seq_len is too small")

    if engine_config.max_prefill_size is not None:
        _require(
            engine_config.max_prefill_size >= prompt_tokens_per_round,
            "engine.max_prefill_size must cover the sum of prompt lengths per round",
        )

    if stop_ids is not None:
        _require(engine_config.max_stop_seqs >= 1, "engine.max_stop_seqs must be >= 1 when using stop_sequence")
        _require(
            engine_config.max_stop_tokens >= len(stop_ids),
            "engine.max_stop_tokens must be >= stop_sequence token length",
        )

    if engine_config.max_pages is not None:
        pages_per_seq = math.ceil((max_prompt_len + config.max_new_tokens) / engine_config.page_size)
        required_pages = pages_per_seq * sequences_per_round
        _require(
            engine_config.max_pages >= required_pages,
            "engine.max_pages="
            f"{engine_config.max_pages} is too small for {sequences_per_round} sequences per round "
            f"(est. {required_pages} pages needed at page_size={engine_config.page_size}). "
            "Increase engine.max_pages or reduce num_prompts/n_generations/max_new_tokens.",
        )


def _validate_tracker_logging_safety(config: SampleLmMultihostConfig, *, is_multihost: bool) -> None:
    """Fail fast for known-unsafe in-loop tracker emission on multi-host multi-round sampling."""

    if not is_multihost:
        return
    if int(config.n_rounds) <= 1:
        return
    if config.defer_tracker_logs_until_end:
        return
    if config.skip_leader_postprocess:
        return

    has_leader_in_loop_tracker_emission = (
        not config.skip_metrics_log or not config.skip_samples_table or config.emit_minimal_tracker_probe
    )
    if not has_leader_in_loop_tracker_emission:
        return

    raise ValueError(
        "Unsafe tracker logging configuration for multi-host multi-round sampling: "
        "in-loop tracker emission with defer_tracker_logs_until_end=false is known to trigger "
        "round-boundary launch-group failures. Use defer_tracker_logs_until_end=true or disable "
        "leader in-loop tracker paths (set skip_metrics_log=true, skip_samples_table=true, "
        "emit_minimal_tracker_probe=false)."
    )


def _validate_inference_mode_safety(config: SampleLmMultihostConfig, *, is_multihost: bool) -> None:
    """Validate selected inference-mode constraints and fail fast for unsupported modes."""

    if config.inference_mode == "global_mesh":
        return

    if config.inference_mode != "host_data_parallel":
        raise ValueError(
            f"Unknown inference_mode={config.inference_mode!r}. "
            "Expected one of: 'global_mesh', 'host_data_parallel'."
        )

    _require(is_multihost, "inference_mode=host_data_parallel requires multi-host execution.")
    _require(int(config.n_rounds) == 1, "inference_mode=host_data_parallel currently requires n_rounds=1 (M10.1).")
    _require(
        int(config.n_generations) == 1,
        "inference_mode=host_data_parallel currently requires n_generations=1 (M10.1).",
    )
    _require(
        config.defer_tracker_logs_until_end,
        "inference_mode=host_data_parallel currently requires defer_tracker_logs_until_end=true (M10.1).",
    )


def _prompt_shard_bounds(num_prompts: int, process_index: int, process_count: int) -> tuple[int, int]:
    """Return deterministic contiguous prompt-shard bounds for one host."""

    if process_count <= 0:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if process_index < 0 or process_index >= process_count:
        raise ValueError(f"process_index={process_index} out of range for process_count={process_count}")

    start = (process_index * num_prompts) // process_count
    end = ((process_index + 1) * num_prompts) // process_count
    return start, end


def _shard_prompts_for_host(
    prompts: list[str],
    prompt_ids: list[list[int]],
    *,
    process_index: int,
    process_count: int,
) -> tuple[list[str], list[list[int]], int, int]:
    """Shard prompts deterministically for one host, preserving global prompt ordering."""

    if len(prompts) != len(prompt_ids):
        raise ValueError(
            f"prompts/prompt_ids length mismatch: len(prompts)={len(prompts)} len(prompt_ids)={len(prompt_ids)}"
        )

    start, end = _prompt_shard_bounds(len(prompts), process_index, process_count)
    return prompts[start:end], prompt_ids[start:end], start, end


def _derive_host_local_engine_config(
    *,
    config: SampleLmMultihostConfig,
    local_prompt_ids: list[list[int]],
    devices,
) -> InferenceEngineConfig:
    """Derive host-local engine sizing from the local prompt shard.

    This prevents M10 host-data-parallel runs from keeping global-shape limits
    (for example `max_seqs=128`) on every host when each host only serves a
    subset of prompts.
    """

    base = config.engine
    local_num_prompts = len(local_prompt_ids)
    local_sequences = max(1, local_num_prompts * int(config.n_generations))
    local_prefill_seqs = max(1, local_num_prompts)
    local_prompt_tokens = max(1, sum(len(tokens) for tokens in local_prompt_ids))

    local_max_seqs = min(base.max_seqs, local_sequences)
    local_max_seqs_in_prefill = min(base.max_seqs_in_prefill, local_prefill_seqs)

    if base.max_tokens_per_round is None:
        local_max_tokens_per_round = None
        effective_tokens_per_round = local_max_seqs
    else:
        # Preserve configured scheduler budget so M10.5 can tune host-DP the same
        # way M9 tuned the global path.
        local_max_tokens_per_round = base.max_tokens_per_round
        effective_tokens_per_round = local_max_tokens_per_round

    # Keep configured queue budget while enforcing minimal local-shard validity.
    local_max_queued_tokens = max(
        base.max_queued_tokens, local_max_seqs, local_max_seqs_in_prefill, effective_tokens_per_round
    )

    if base.max_prefill_size is None:
        local_max_prefill_size = None
    else:
        local_max_prefill_size = min(base.max_prefill_size, local_prompt_tokens)

    # Preserve configured page budget for host-DP tuning sweeps.
    local_max_pages = base.max_pages

    return replace(
        base,
        max_seqs=local_max_seqs,
        max_seqs_in_prefill=local_max_seqs_in_prefill,
        max_tokens_per_round=local_max_tokens_per_round,
        max_queued_tokens=local_max_queued_tokens,
        max_prefill_size=local_max_prefill_size,
        max_pages=local_max_pages,
        devices=devices,
    )


def _host_output_path(output_dir: str, *, process_index: int, process_count: int) -> Path:
    """Return a deterministic per-host JSONL output path."""

    return Path(output_dir) / f"host_{process_index:04d}_of_{process_count:04d}.jsonl"


def _merged_host_output_path(output_dir: str, *, process_count: int) -> Path:
    """Return a deterministic leader-written merged JSONL path for all hosts."""

    return Path(output_dir) / f"all_hosts_merged_of_{process_count:04d}.jsonl"


def _filter_generated_tokens(sequence_tokens, *, pad_token_id: int | None) -> list[int]:
    """Drop padding/sentinel tokens and normalize generated tokens to Python ints."""

    filtered_tokens: list[int] = []
    for token in sequence_tokens:
        token_int = int(token)
        if token_int == INVALID:
            continue
        if pad_token_id is not None and token_int == int(pad_token_id):
            continue
        filtered_tokens.append(token_int)
    return filtered_tokens


def _build_host_generation_rows(
    *,
    requests: list[Request],
    request_meta: list[tuple[int, int]],
    result_tokens: list[list[int]],
    local_prompts: list[str],
    shard_start: int,
    shard_end: int,
    process_index: int,
    process_count: int,
    n_generations: int,
    tokenizer,
) -> list[dict[str, object]]:
    """Build one structured row per generated sequence for one host shard."""

    if len(request_meta) != len(requests):
        raise RuntimeError(
            f"request metadata mismatch: len(request_meta)={len(request_meta)} len(requests)={len(requests)}"
        )
    expected_sequences = len(requests) * n_generations
    if len(result_tokens) != expected_sequences:
        raise RuntimeError(
            f"Expected {expected_sequences} generated sequences for host {process_index}, "
            f"but got {len(result_tokens)}."
        )

    rows: list[dict[str, object]] = []
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    for sequence_index, sequence_tokens in enumerate(result_tokens):
        request_index = sequence_index // n_generations
        generation_index = sequence_index % n_generations
        if request_index >= len(requests):
            raise RuntimeError(
                f"Generated sequence index {sequence_index} maps to request {request_index}, "
                f"but only {len(requests)} requests exist."
            )

        request = requests[request_index]
        round_index, global_prompt_index = request_meta[request_index]
        local_prompt_index = global_prompt_index - shard_start
        if local_prompt_index < 0 or local_prompt_index >= len(local_prompts):
            raise RuntimeError(
                f"Global prompt index {global_prompt_index} is outside local shard "
                f"[{shard_start}:{shard_end}) for process {process_index}."
            )

        filtered_tokens = _filter_generated_tokens(sequence_tokens, pad_token_id=pad_token_id)
        generated_text = tokenizer.decode(filtered_tokens, skip_special_tokens=True)

        rows.append(
            {
                "process_index": process_index,
                "process_count": process_count,
                "shard_start": shard_start,
                "shard_end": shard_end,
                "round_index": round_index,
                "global_prompt_index": global_prompt_index,
                "local_prompt_index": local_prompt_index,
                "request_id": int(request.request_id),
                "generation_index": generation_index,
                "prompt": local_prompts[local_prompt_index],
                "prompt_token_count": len(request.prompt_tokens),
                "generated_token_count": len(filtered_tokens),
                "generated_tokens": filtered_tokens,
                "generated_text": generated_text,
            }
        )
    return rows


def _write_rows_jsonl(*, output_path: Path, rows: list[dict[str, object]]) -> int:
    """Write JSONL rows to disk and return the number of written rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _write_host_generations_jsonl(
    *,
    output_path: Path,
    requests: list[Request],
    request_meta: list[tuple[int, int]],
    result_tokens: list[list[int]],
    local_prompts: list[str],
    shard_start: int,
    shard_end: int,
    process_index: int,
    process_count: int,
    n_generations: int,
    tokenizer,
) -> int:
    """Write one JSONL line per generated sequence for one host shard."""

    rows = _build_host_generation_rows(
        requests=requests,
        request_meta=request_meta,
        result_tokens=result_tokens,
        local_prompts=local_prompts,
        shard_start=shard_start,
        shard_end=shard_end,
        process_index=process_index,
        process_count=process_count,
        n_generations=n_generations,
        tokenizer=tokenizer,
    )
    return _write_rows_jsonl(output_path=output_path, rows=rows)


def _encode_host_rows_payload(rows: list[dict[str, object]]) -> str:
    """Serialize + compress host rows for JAX distributed key-value transport."""

    serialized = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(zlib.compress(serialized)).decode("ascii")


def _decode_host_rows_payload(payload: str) -> list[dict[str, object]]:
    """Decode rows from the compressed transport payload."""

    decoded = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    rows = json.loads(decoded.decode("utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected gathered rows payload list, got {type(rows).__name__}.")
    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected gathered row object, got {type(row).__name__}.")
        parsed_rows.append(row)
    return parsed_rows


def _gather_host_rows_to_leader(
    *,
    local_rows: list[dict[str, object]],
    process_index: int,
    process_count: int,
    timeout_sec: float = 600.0,
) -> list[list[dict[str, object]]] | None:
    """Gather host-local rows to process 0 in deterministic host-index order."""

    if process_count == 1:
        return [local_rows]

    import jax._src.distributed as distributed

    client = distributed.global_state.client
    if client is None:
        raise RuntimeError("Host-row gather requires jax distributed client to be initialized.")

    kv_prefix = "sample_lm_multihost_m10_6_host_rows"
    payload_key = f"{kv_prefix}_payload_{process_index:04d}"
    payload = _encode_host_rows_payload(local_rows)
    client.key_value_set(payload_key, payload)
    timeout_ms = int(timeout_sec * 1000.0)
    client.wait_at_barrier(f"{kv_prefix}_after_set", timeout_in_ms=timeout_ms)

    gathered: list[list[dict[str, object]]] | None = None
    if process_index == 0:
        gathered = []
        for host_index in range(process_count):
            host_payload_key = f"{kv_prefix}_payload_{host_index:04d}"
            host_payload = client.blocking_key_value_get(host_payload_key, timeout_in_ms=timeout_ms)
            gathered.append(_decode_host_rows_payload(host_payload))

    client.wait_at_barrier(f"{kv_prefix}_after_get", timeout_in_ms=timeout_ms)
    return gathered


def _row_sort_key(row: dict[str, object]) -> tuple[int, int, int, int]:
    """Deterministic global ordering for merged host rows."""

    return (
        int(row["round_index"]),
        int(row["global_prompt_index"]),
        int(row["generation_index"]),
        int(row["process_index"]),
    )


def _merge_gathered_host_rows(
    gathered_rows_by_host: list[list[dict[str, object]]], *, process_count: int
) -> list[dict[str, object]]:
    """Merge gathered rows with source validation and deterministic ordering."""

    if len(gathered_rows_by_host) != process_count:
        raise RuntimeError(f"Expected gathered rows from {process_count} hosts, got {len(gathered_rows_by_host)}.")

    merged_rows: list[dict[str, object]] = []
    for host_index, host_rows in enumerate(gathered_rows_by_host):
        for row in host_rows:
            row_process_index = int(row.get("process_index", -1))
            if row_process_index != host_index:
                raise RuntimeError(
                    "Gathered row source mismatch: " f"host_index={host_index} row.process_index={row_process_index}"
                )
            merged_rows.append(row)

    merged_rows.sort(key=_row_sort_key)
    return merged_rows


class _PromptPayload(TypedDict, total=False):
    ok: bool
    prompt_ids: list[list[int]]
    stop_ids: list[int] | None
    base_seed: int
    error: str


def _broadcast_prompt_payload(
    tokenizer,
    prompts: list[str],
    stop_sequence: str | None,
    base_seed: int,
    *,
    is_leader: bool,
) -> tuple[list[list[int]], list[int] | None, int]:
    """Tokenize prompts on the leader and broadcast token IDs to all hosts."""

    if is_leader:
        try:
            prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            if stop_sequence is None:
                stop_ids = None
            else:
                stop_ids = tokenizer(stop_sequence, add_special_tokens=False)["input_ids"]
            payload: _PromptPayload = {
                "ok": True,
                "prompt_ids": prompt_ids,
                "stop_ids": stop_ids,
                "base_seed": base_seed,
            }
        except Exception as exc:
            payload = {"ok": False, "error": repr(exc)}
    else:
        payload = {"ok": False}

    payload = multihost_broadcast_sync(payload, is_source=is_leader)
    if not payload.get("ok"):
        error_message = payload.get("error", "unknown error")
        raise RuntimeError(f"Leader tokenization failed: {error_message}")

    prompt_ids = payload.get("prompt_ids")
    base_seed_value = payload.get("base_seed")
    if prompt_ids is None or base_seed_value is None:
        raise RuntimeError("Leader broadcast missing required prompt payload")

    stop_ids = payload.get("stop_ids")
    return prompt_ids, stop_ids, int(base_seed_value)


def _log_decode_stats(engine: InferenceEngine, stage: str, *, is_leader: bool) -> None:
    """Log decode-state stats from the leader for round boundary diagnostics."""

    stats = jax.device_get(engine.gen_state.decode_state.stats())
    if not is_leader:
        return
    logger.info(
        "RoundStats[%s]: active=%d pages_in_use=%d free=%d max_refcount=%d",
        stage,
        int(stats.active_seqs),
        int(stats.pages_in_use),
        int(stats.free_pages),
        int(stats.max_refcount),
    )


def _log_host_dp_samples_to_tracker(
    *,
    merged_rows: list[dict[str, object]],
    generation_time: float,
    total_generated: int,
    process_count: int,
) -> None:
    """Log merged host-DP samples to the active tracker (wandb or other) on leader only.

    Called after gather + merge completes, so all generation is done and this is safe
    for multi-host execution (no in-loop tracker emission).
    """
    samples_rows = []
    for row in merged_rows:
        samples_rows.append(
            (
                row.get("round_index", 0),
                row.get("global_prompt_index", 0),
                row.get("generation_index", 0),
                row.get("prompt", ""),
                row.get("generated_text", ""),
                row.get("generated_token_count", 0),
            )
        )

    samples_payload: dict[str, object] = {}
    try:
        import wandb

        samples_table = wandb.Table(
            columns=["round", "prompt_id", "generation_id", "prompt", "generated_text", "num_tokens"]
        )
        for r in samples_rows:
            samples_table.add_data(*r)
        samples_payload["sample/samples"] = samples_table
    except ImportError:
        samples_payload["sample/samples_rows"] = samples_rows

    metrics_payload: dict[str, object] = {
        "sample/total_prompts": len(merged_rows),
        "sample/total_generated_tokens": sum(int(r.get("generated_token_count", 0)) for r in merged_rows),
        "sample/num_hosts": process_count,
        "sample/inference_mode": "host_data_parallel",
    }

    levanter.tracker.log({**metrics_payload, **samples_payload}, step=0)
    logger.info(
        "Host data-parallel wandb samples logged: rows=%d total_tokens=%d",
        len(merged_rows),
        metrics_payload["sample/total_generated_tokens"],
    )


def _run_host_data_parallel_inference(
    *,
    config: SampleLmMultihostConfig,
    tokenizer,
    prompts: list[str],
    prompt_ids: list[list[int]],
    stop_ids: list[int] | None,
    base_seed: int,
    is_multihost: bool,
    is_leader: bool,
) -> None:
    """Run host-local generation and write per-host JSONL output files (M10.3)."""
    try:
        process_index = int(jax.process_index())
        process_count = int(jax.process_count())
        local_prompts, local_prompt_ids, shard_start, shard_end = _shard_prompts_for_host(
            prompts,
            prompt_ids,
            process_index=process_index,
            process_count=process_count,
        )
        logger.info(
            "Host data-parallel prompt shard: process=%d/%d range=[%d:%d) local_prompts=%d global_prompts=%d",
            process_index,
            process_count,
            shard_start,
            shard_end,
            len(local_prompts),
            len(prompts),
        )
        if len(local_prompts) != len(local_prompt_ids):
            raise RuntimeError(
                f"Local prompt shard mismatch: len(local_prompts)={len(local_prompts)} "
                f"len(local_prompt_ids)={len(local_prompt_ids)}"
            )

        local_devices = list(jax.local_devices())
        local_mesh = create_local_mesh(devices=local_devices)
        local_mesh_devices = list(local_mesh.devices.flat)
        local_engine_config = _derive_host_local_engine_config(
            config=config,
            local_prompt_ids=local_prompt_ids,
            devices=local_mesh_devices,
        )
        logger.info(
            "Host data-parallel local engine sizing: process=%d/%d max_seqs=%d max_seqs_in_prefill=%d "
            "max_prefill_size=%s max_pages=%s max_tokens_per_round=%s max_queued_tokens=%d",
            process_index,
            process_count,
            int(local_engine_config.max_seqs),
            int(local_engine_config.max_seqs_in_prefill),
            str(local_engine_config.max_prefill_size),
            str(local_engine_config.max_pages),
            str(local_engine_config.max_tokens_per_round),
            int(local_engine_config.max_queued_tokens),
        )
        _validate_engine_config(
            config,
            local_engine_config,
            local_prompt_ids,
            stop_ids,
            is_multihost=is_multihost,
        )

        if stop_ids is None:
            stop_tokens = None
        else:
            stop_tokens = hax.named(jnp.asarray(stop_ids, dtype=jnp.int32), axis="position").broadcast_axis(
                {"stop_seq": 1}
            )

        key = jrandom.PRNGKey(config.seed)
        with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
            vocab_size = len(tokenizer)
            vocab_axis = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
            if is_leader:
                logger.info("Model load starting (processes=%d)", process_count)
            model = _load_model(config, vocab_axis, tokenizer, key=key)
            if is_leader:
                logger.info("Model load completed")

            memory_device = local_mesh_devices[0] if local_mesh_devices else None
            replication_start = time.perf_counter()
            local_model = replicate_model_to_local_mesh(model, local_mesh=local_mesh)
            for leaf in jax.tree.leaves(local_model):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()
            replication_time = time.perf_counter() - replication_start
            logger.info(
                "Host data-parallel model replication finished: process=%d/%d time_sec=%.2f",
                process_index,
                process_count,
                replication_time,
            )

            base_key = jrandom.PRNGKey(base_seed)
            requests, request_meta = _build_requests_for_rounds(
                prompt_ids=local_prompt_ids,
                stop_tokens=stop_tokens,
                config=config,
                base_key=base_key,
                rounds=1,
                round_offset=0,
                prompt_id_offset=shard_start,
                total_num_prompts=len(prompts),
            )

            hbm_free_before_engine = estimated_free_device_memory(memory_device)
            with hax.partitioning.set_mesh(local_mesh), hax.axis_mapping({}):
                engine, engine_creation_time, hbm_free_after_engine = _create_engine(
                    model=local_model,
                    tokenizer=tokenizer,
                    engine_config=local_engine_config,
                    memory_device=memory_device,
                    is_leader=is_leader,
                )
                logger.info(
                    "Host data-parallel local engine ready: process=%d/%d engine_creation_time_sec=%.2f",
                    process_index,
                    process_count,
                    engine_creation_time,
                )

                if config.log_kernel_jaxprs_path:
                    jaxprs_path = f"{config.log_kernel_jaxprs_path}.host{process_index}"
                    engine.write_kernel_jaxprs(jaxprs_path)

                result_tokens: list[list[int]]
                total_generated = 0
                if requests:
                    generate_start = time.perf_counter()
                    result = engine.generate(requests)
                    generation_time = time.perf_counter() - generate_start
                    result_tokens = result.tokens
                    total_generated = int(result.total_generated)
                    jax.block_until_ready(jax.tree.leaves(engine.gen_state))
                    _log_decode_stats(engine, "round0_after_generate_host_local", is_leader=is_leader)
                else:
                    generation_time = 0.0
                    result_tokens = []
                    logger.info(
                        "Host data-parallel shard has no local requests: process=%d/%d", process_index, process_count
                    )

                local_rows = _build_host_generation_rows(
                    requests=requests,
                    request_meta=request_meta,
                    result_tokens=result_tokens,
                    local_prompts=local_prompts,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    process_index=process_index,
                    process_count=process_count,
                    n_generations=int(config.n_generations),
                    tokenizer=tokenizer,
                )

                rows_written = 0
                if config.host_data_parallel_output_dir is not None:
                    output_path = _host_output_path(
                        config.host_data_parallel_output_dir,
                        process_index=process_index,
                        process_count=process_count,
                    )
                    rows_written = _write_rows_jsonl(output_path=output_path, rows=local_rows)
                    logger.info(
                        "Host data-parallel wrote local generations: process=%d/%d rows=%d path=%s",
                        process_index,
                        process_count,
                        rows_written,
                        output_path,
                    )
                else:
                    logger.info(
                        "Host data-parallel local generation dump disabled: process=%d/%d",
                        process_index,
                        process_count,
                    )

                if config.enable_host_rows_gather:
                    gathered_rows = _gather_host_rows_to_leader(
                        local_rows=local_rows,
                        process_index=process_index,
                        process_count=process_count,
                    )
                    if is_leader:
                        assert gathered_rows is not None
                        merged_rows = _merge_gathered_host_rows(gathered_rows, process_count=process_count)
                        if config.host_data_parallel_output_dir is not None:
                            merged_output_path = _merged_host_output_path(
                                config.host_data_parallel_output_dir, process_count=process_count
                            )
                            merged_rows_written = _write_rows_jsonl(output_path=merged_output_path, rows=merged_rows)
                            logger.info(
                                "Host data-parallel gathered + merged rows: hosts=%d rows=%d path=%s",
                                process_count,
                                merged_rows_written,
                                merged_output_path,
                            )
                        else:
                            logger.info(
                                "Host data-parallel gathered + merged rows: hosts=%d rows=%d (output disabled)",
                                process_count,
                                len(merged_rows),
                            )
                elif is_leader:
                    logger.info("Host data-parallel gather disabled (M10.4 baseline emulation).")
                    if not config.skip_samples_table:
                        logger.warning(
                            "skip_samples_table=false but enable_host_rows_gather=false: "
                            "cannot log samples to wandb without gathered rows. "
                            "Set enable_host_rows_gather=true to enable wandb samples logging."
                        )

                # WandB logging: log samples table and metrics on leader after gather
                if is_leader and config.enable_host_rows_gather and not config.skip_samples_table:
                    assert merged_rows is not None  # noqa: S101
                    _log_host_dp_samples_to_tracker(
                        merged_rows=merged_rows,
                        generation_time=generation_time,
                        total_generated=total_generated,
                        process_count=process_count,
                    )

                total_sequences = len(result_tokens)
                tokens_per_second = total_generated / generation_time if generation_time > 0 else 0.0
                logger.info(
                    "Host data-parallel round complete: process=%d/%d requests=%d sequences=%d generated_tokens=%d "
                    "generation_time_sec=%.2f tokens_per_sec=%.1f",
                    process_index,
                    process_count,
                    len(requests),
                    total_sequences,
                    total_generated,
                    generation_time,
                    tokens_per_second,
                )

                if is_leader:
                    logger.info(
                        "Host data-parallel M10.6 completed local execution, host-row gather, and leader merge."
                    )

                if hbm_free_before_engine is not None and hbm_free_after_engine is not None:
                    logger.info(
                        "Host data-parallel HBM stats: process=%d/%d free_before=%.2fGiB free_after_engine=%.2fGiB used=%.2fGiB",
                        process_index,
                        process_count,
                        hbm_free_before_engine,
                        hbm_free_after_engine,
                        hbm_free_before_engine - hbm_free_after_engine,
                    )

            del engine
            del local_model
            del model
            gc.collect()
    finally:
        if is_multihost:
            barrier_sync_with_tag("sample_lm_multihost_done")


def main(config: SampleLmMultihostConfig):
    """Run multi-host sampling with a globally sharded model."""

    levanter.initialize(config)

    is_multihost = jax.process_count() > 1
    is_leader = jax.process_index() == 0
    if is_leader:
        logger.info("Inference mode: %s", config.inference_mode)

    if is_multihost:
        barrier_sync_with_tag("sample_lm_multihost_start")

    tokenizer_name = config.tokenizer
    if tokenizer_name is None and config.hf_checkpoint is not None:
        tokenizer_name = config.hf_checkpoint.model_name_or_path
    _require(tokenizer_name is not None, "Must specify tokenizer or hf_checkpoint with tokenizer")
    tokenizer = load_tokenizer(tokenizer_name)

    prompts = _normalize_prompts(config.prompts)
    _require(prompts, "prompts must be non-empty")

    if is_multihost:
        prompt_ids, stop_ids, base_seed = _broadcast_prompt_payload(
            tokenizer,
            prompts,
            config.stop_sequence,
            int(config.engine.seed),
            is_leader=is_leader,
        )
    else:
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        if config.stop_sequence is None:
            stop_ids = None
        else:
            stop_ids = tokenizer(config.stop_sequence, add_special_tokens=False)["input_ids"]
        base_seed = int(config.engine.seed)

    if stop_ids is not None and len(stop_ids) == 0:
        raise ValueError("stop_sequence must be non-empty if provided")

    try:
        _validate_tracker_logging_safety(config, is_multihost=is_multihost)
        _validate_inference_mode_safety(config, is_multihost=is_multihost)
    except Exception:
        logger.exception("Config validation failed")
        raise

    if config.inference_mode == "host_data_parallel":
        _run_host_data_parallel_inference(
            config=config,
            tokenizer=tokenizer,
            prompts=prompts,
            prompt_ids=prompt_ids,
            stop_ids=stop_ids,
            base_seed=base_seed,
            is_multihost=is_multihost,
            is_leader=is_leader,
        )
        levanter.current_tracker().finish()
        return

    try:
        _validate_engine_config(
            config,
            config.engine,
            prompt_ids,
            stop_ids,
            is_multihost=is_multihost,
        )
    except Exception:
        logger.exception("Config validation failed")
        raise

    if stop_ids is None:
        stop_tokens = None
    else:
        stop_tokens = hax.named(jnp.asarray(stop_ids, dtype=jnp.int32), axis="position").broadcast_axis(
            {"stop_seq": 1}
        )

    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        vocab_axis = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        if is_leader:
            logger.info("Model load starting (processes=%d)", jax.process_count())
        try:
            model = _load_model(config, vocab_axis, tokenizer, key=key)
        except Exception:
            logger.exception("Model load failed")
            raise
        if is_leader:
            logger.info("Model load completed")

        memory_device = jax.local_devices()[0] if jax.local_devices() else None
        hbm_free_before_engine = estimated_free_device_memory(memory_device)
        engine, engine_creation_time, hbm_free_after_engine = _create_engine(
            model=model,
            tokenizer=tokenizer,
            engine_config=config.engine,
            memory_device=memory_device,
            is_leader=is_leader,
        )

        if config.log_kernel_jaxprs_path:
            jaxprs_path = config.log_kernel_jaxprs_path
            if is_multihost:
                jaxprs_path = f"{jaxprs_path}.host{jax.process_index()}"
            engine.write_kernel_jaxprs(jaxprs_path)

        base_key = jrandom.PRNGKey(base_seed)
        deferred_tracker_logs: list[tuple[int, dict[str, object]]] = []
        try:
            # Per-round generation: each round is a separate generate() call with reset between them.
            # This is safe now that the reset path has been fixed (see CODEX_FIX_M5.2.md).
            total_expected = len(prompt_ids) * int(config.n_generations)
            for round_index in range(int(config.n_rounds)):
                requests, _ = _build_requests_for_rounds(
                    prompt_ids=prompt_ids,
                    stop_tokens=stop_tokens,
                    config=config,
                    base_key=base_key,
                    rounds=1,
                    round_offset=round_index,
                )
                if is_leader:
                    logger.info(
                        "Round %d starting: %d prompts x %d generations",
                        round_index,
                        len(prompts),
                        int(config.n_generations),
                    )
                if config.engine.debug_stats_every_n:
                    logger.warning("Round %d: generate start", round_index)
                start_time = time.time()
                result = engine.generate(requests)
                duration = time.time() - start_time
                if config.engine.debug_stats_every_n:
                    logger.warning("Round %d: generate done", round_index)

                if config.force_post_generate_block_until_ready:
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: post-generate block_until_ready start", round_index)
                    jax.block_until_ready(jax.tree.leaves(engine.gen_state))
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: post-generate block_until_ready done", round_index)

                if is_multihost and config.force_post_generate_barrier:
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: post-generate barrier start", round_index)
                    barrier_sync_with_tag(f"sample_lm_multihost_round_{round_index}_post_generate_ready")
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: post-generate barrier done", round_index)

                if len(result.tokens) != total_expected:
                    raise RuntimeError(
                        f"Expected {total_expected} sequences but got {len(result.tokens)}. "
                        "Check engine.max_seqs/max_seqs_in_prefill/max_prefill_size."
                    )
                if config.engine.debug_stats_every_n:
                    logger.warning("Round %d: decode stats start", round_index)
                _log_decode_stats(engine, f"round{round_index}_after_generate", is_leader=is_leader)
                if config.engine.debug_stats_every_n:
                    logger.warning("Round %d: decode stats done", round_index)

                if is_multihost and config.barrier_before_leader_postprocess and not config.skip_round_barrier:
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: pre-postprocess barrier start", round_index)
                    barrier_sync_with_tag(f"sample_lm_multihost_round_{round_index}")
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: pre-postprocess barrier done", round_index)

                hbm_free_after_gen = estimated_free_device_memory(memory_device)
                tokens_per_second = result.total_generated / duration if duration > 0 else 0.0

                if is_leader and not config.skip_leader_postprocess:
                    logger.info(
                        "Round %d finished in %.2fs: %d tokens across %d sequences",
                        round_index,
                        duration,
                        result.total_generated,
                        len(result.tokens),
                    )

                    metrics: dict[str, float | int] = {
                        "sample/total_tokens": result.total_generated,
                        "sample/num_sequences": len(result.tokens),
                        "sample/round_time_sec": duration,
                        "sample/tokens_per_second": tokens_per_second,
                        "sample/num_prompts": len(prompts),
                        "sample/n_generations": int(config.n_generations),
                    }
                    if round_index == 0:
                        metrics["sample/engine_creation_time_sec"] = engine_creation_time
                        if hbm_free_before_engine is not None:
                            metrics["sample/hbm_free_before_engine_gib"] = hbm_free_before_engine
                        if hbm_free_after_engine is not None:
                            metrics["sample/hbm_free_after_engine_gib"] = hbm_free_after_engine
                        if hbm_free_before_engine is not None and hbm_free_after_engine is not None:
                            metrics["sample/hbm_used_by_engine_gib"] = hbm_free_before_engine - hbm_free_after_engine
                    if hbm_free_after_gen is not None:
                        metrics["sample/hbm_free_after_gen_gib"] = hbm_free_after_gen
                    if not config.skip_metrics_log:
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: metrics log start", round_index)
                        if config.defer_tracker_logs_until_end:
                            deferred_tracker_logs.append((round_index, metrics))
                        else:
                            levanter.tracker.log(metrics, step=round_index)
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: metrics log done", round_index)
                    elif config.engine.debug_stats_every_n:
                        logger.warning("Round %d: metrics log skipped by config", round_index)

                    samples_rows: list[tuple[int, int, int, str, str, int]] = []
                    for sequence_index, sequence_tokens in enumerate(result.tokens):
                        filtered_tokens = [
                            token for token in sequence_tokens if token != tokenizer.pad_token_id and token != INVALID
                        ]
                        generated_text = tokenizer.decode(filtered_tokens, skip_special_tokens=True)
                        logger.info("Sequence %d: %s", sequence_index, generated_text)
                        prompt_index = sequence_index // int(config.n_generations)
                        generation_index = sequence_index % int(config.n_generations)
                        prompt_text = prompts[prompt_index] if prompt_index < len(prompts) else ""
                        samples_rows.append(
                            (
                                round_index,
                                prompt_index,
                                generation_index,
                                prompt_text,
                                generated_text,
                                len(filtered_tokens),
                            )
                        )
                    if not config.skip_samples_table:
                        if config.force_samples_rows_log:
                            if config.engine.debug_stats_every_n:
                                logger.warning("Round %d: samples rows log start (forced)", round_index)
                            samples_payload: dict[str, object] = {"sample/samples": samples_rows}
                            if config.defer_tracker_logs_until_end:
                                deferred_tracker_logs.append((round_index, samples_payload))
                            else:
                                levanter.tracker.log(samples_payload, step=round_index)
                            if config.engine.debug_stats_every_n:
                                logger.warning("Round %d: samples rows log done (forced)", round_index)
                        else:
                            try:
                                import wandb

                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples table start", round_index)
                                samples_table = wandb.Table(
                                    columns=[
                                        "round",
                                        "prompt_id",
                                        "generation_id",
                                        "prompt",
                                        "generated_text",
                                        "num_tokens",
                                    ]
                                )
                                for row in samples_rows:
                                    samples_table.add_data(*row)
                                samples_payload = {"sample/samples": samples_table}
                                if config.defer_tracker_logs_until_end:
                                    deferred_tracker_logs.append((round_index, samples_payload))
                                else:
                                    levanter.tracker.log(samples_payload, step=round_index)
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples table done", round_index)
                            except ImportError:
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples rows log start", round_index)
                                samples_payload = {"sample/samples": samples_rows}
                                if config.defer_tracker_logs_until_end:
                                    deferred_tracker_logs.append((round_index, samples_payload))
                                else:
                                    levanter.tracker.log(samples_payload, step=round_index)
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples rows log done", round_index)
                    if config.emit_minimal_tracker_probe:
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: minimal tracker probe start", round_index)
                        probe_payload: dict[str, object] = {"sample/minimal_probe": float(round_index)}
                        if config.defer_tracker_logs_until_end:
                            deferred_tracker_logs.append((round_index, probe_payload))
                        else:
                            levanter.tracker.log(probe_payload, step=round_index)
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: minimal tracker probe done", round_index)
                    if config.leader_postprocess_sleep_sec > 0:
                        if config.engine.debug_stats_every_n:
                            logger.warning(
                                "Round %d: leader postprocess sleep start (%.3fs)",
                                round_index,
                                config.leader_postprocess_sleep_sec,
                            )
                        time.sleep(config.leader_postprocess_sleep_sec)
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: leader postprocess sleep done", round_index)
                if is_leader and config.skip_leader_postprocess and config.engine.debug_stats_every_n:
                    logger.warning("Round %d: leader postprocess skipped by config", round_index)
                if config.emit_minimal_tracker_probe_all_hosts:
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: all-host minimal tracker probe start", round_index)
                    all_host_probe_payload: dict[str, object] = {
                        "sample/minimal_probe_all_hosts": float(round_index),
                        "sample/minimal_probe_all_hosts_process_index": int(jax.process_index()),
                    }
                    if config.defer_tracker_logs_until_end:
                        deferred_tracker_logs.append((round_index, all_host_probe_payload))
                    else:
                        levanter.tracker.log(all_host_probe_payload, step=round_index)
                    if config.engine.debug_stats_every_n:
                        logger.warning("Round %d: all-host minimal tracker probe done", round_index)
                if is_multihost and not config.skip_round_barrier and not config.barrier_before_leader_postprocess:
                    barrier_sync_with_tag(f"sample_lm_multihost_round_{round_index}")
        finally:
            if is_multihost:
                barrier_sync_with_tag("sample_lm_multihost_done")

        if is_leader and config.defer_tracker_logs_until_end:
            if config.engine.debug_stats_every_n:
                logger.warning("Deferred tracker log flush start (%d entries)", len(deferred_tracker_logs))
            for step, payload in deferred_tracker_logs:
                levanter.tracker.log(payload, step=step)
            if config.engine.debug_stats_every_n:
                logger.warning("Deferred tracker log flush done")

        del engine
        gc.collect()
        hbm_free_after_cleanup = estimated_free_device_memory(memory_device)
        if is_leader and hbm_free_after_cleanup is not None:
            levanter.tracker.log({"sample/hbm_free_after_cleanup_gib": hbm_free_after_cleanup}, step=config.n_rounds)

    levanter.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
