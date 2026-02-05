# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import gc
import math
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

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
    use_cpu_device,
)

logger = logging.getLogger(__name__)


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
) -> tuple[list[Request], list[tuple[int, int]]]:
    requests: list[Request] = []
    request_meta: list[tuple[int, int]] = []
    num_prompts = len(prompt_ids)
    for round_delta in range(rounds):
        round_index = round_offset + round_delta
        round_key = jrandom.fold_in(base_key, round_index)
        for prompt_index, tokens in enumerate(prompt_ids):
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(tokens) + config.max_new_tokens, dtype=jnp.int32),
                stop_tokens=stop_tokens,
                temperature=jnp.array(config.temperature, dtype=jnp.float32),
                key=jrandom.fold_in(round_key, prompt_index),
            )
            requests.append(
                Request(
                    prompt_tokens=list(map(int, tokens)),
                    request_id=round_index * num_prompts + prompt_index,
                    decode_params=seq_params,
                    n_generations=int(config.n_generations),
                )
            )
            request_meta.append((round_index, prompt_index))
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
    skip_round_barrier: bool = False
    skip_samples_table: bool = False

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
    hf_config = converter.hf_config_from_hf_checkpoint(config.hf_checkpoint)
    overrides = {"use_tpu_ragged_paged_attention": config.model.use_tpu_ragged_paged_attention}
    merged_config = converter.config_from_hf_config(hf_config, overrides=overrides)

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
    """Validate engine sizing constraints for deterministic multi-host inference."""

    if is_multihost:
        _require(engine_config.max_pages is not None, "engine.max_pages must be explicit for multi-host inference")
        _require(
            engine_config.max_prefill_size is not None,
            "engine.max_prefill_size must be explicit for multi-host inference",
        )

    _require(engine_config.max_rounds <= engine_config.max_seq_len, "engine.max_rounds must be <= engine.max_seq_len")
    _require(config.n_generations > 0, "n_generations must be >= 1")

    num_prompts = len(prompt_ids)
    rounds = max(1, int(config.n_rounds))
    total_sequences = num_prompts * int(config.n_generations) * rounds
    total_prompt_tokens = sum(len(tokens) for tokens in prompt_ids) * rounds
    max_prompt_len = max((len(tokens) for tokens in prompt_ids), default=0)
    total_prefill_seqs = num_prompts * rounds

    _require(
        engine_config.max_seqs_in_prefill >= total_prefill_seqs,
        "engine.max_seqs_in_prefill must cover all prompts (including batched rounds)",
    )
    _require(engine_config.max_seqs >= total_sequences, "engine.max_seqs must cover all generations")
    _require(engine_config.max_seq_len >= max_prompt_len + config.max_new_tokens, "engine.max_seq_len is too small")

    if engine_config.max_prefill_size is not None:
        _require(
            engine_config.max_prefill_size >= total_prompt_tokens,
            "engine.max_prefill_size must cover the sum of prompt lengths (including batched rounds)",
        )

    if stop_ids is not None:
        _require(engine_config.max_stop_seqs >= 1, "engine.max_stop_seqs must be >= 1 when using stop_sequence")
        _require(
            engine_config.max_stop_tokens >= len(stop_ids),
            "engine.max_stop_tokens must be >= stop_sequence token length",
        )

    if engine_config.max_pages is not None:
        pages_per_seq = math.ceil((max_prompt_len + config.max_new_tokens) / engine_config.page_size)
        required_pages = pages_per_seq * total_sequences
        _require(
            engine_config.max_pages >= required_pages,
            "engine.max_pages="
            f"{engine_config.max_pages} is too small for {total_sequences} sequences "
            f"(est. {required_pages} pages needed at page_size={engine_config.page_size}). "
            "Increase engine.max_pages or reduce n_rounds/num_prompts/max_new_tokens.",
        )


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


def main(config: SampleLmMultihostConfig):
    """Run multi-host sampling with a globally sharded model."""

    levanter.initialize(config)

    is_multihost = jax.process_count() > 1
    is_leader = jax.process_index() == 0

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
        _validate_engine_config(
            config,
            config.engine,
            prompt_ids,
            stop_ids,
            is_multihost=is_multihost,
        )
    except Exception:
        logger.exception("Engine config validation failed")
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
        try:
            if int(config.n_rounds) > 1:
                if is_leader:
                    logger.warning(
                        "n_rounds=%d detected; using single-batch mode to avoid multi-round TPU instability.",
                        int(config.n_rounds),
                    )
                requests, request_meta = _build_requests_for_rounds(
                    prompt_ids=prompt_ids,
                    stop_tokens=stop_tokens,
                    config=config,
                    base_key=base_key,
                    rounds=int(config.n_rounds),
                )
                total_expected = len(requests) * int(config.n_generations)
                if is_leader:
                    logger.info(
                        "Round batch starting: %d rounds x %d prompts x %d generations",
                        int(config.n_rounds),
                        len(prompts),
                        int(config.n_generations),
                    )
                if config.engine.debug_stats_every_n:
                    logger.warning("RoundBatch: generate start")
                start_time = time.time()
                result = engine.generate(requests)
                duration = time.time() - start_time
                if config.engine.debug_stats_every_n:
                    logger.warning("RoundBatch: generate done")

                if len(result.tokens) != total_expected:
                    raise RuntimeError(
                        f"Expected {total_expected} sequences but got {len(result.tokens)}. "
                        "Check engine.max_seqs/max_seqs_in_prefill/max_prefill_size."
                    )
                if config.engine.debug_stats_every_n:
                    logger.warning("RoundBatch: decode stats start")
                _log_decode_stats(engine, "round_batch_after_generate", is_leader=is_leader)
                if config.engine.debug_stats_every_n:
                    logger.warning("RoundBatch: decode stats done")

                hbm_free_after_gen = estimated_free_device_memory(memory_device)
                tokens_per_second = result.total_generated / duration if duration > 0 else 0.0

                if is_leader:
                    logger.info(
                        "Round batch finished in %.2fs: %d tokens across %d sequences",
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
                        "sample/num_rounds": int(config.n_rounds),
                        "sample/rounds_batched": 1,
                    }
                    metrics["sample/engine_creation_time_sec"] = engine_creation_time
                    if hbm_free_before_engine is not None:
                        metrics["sample/hbm_free_before_engine_gib"] = hbm_free_before_engine
                    if hbm_free_after_engine is not None:
                        metrics["sample/hbm_free_after_engine_gib"] = hbm_free_after_engine
                    if hbm_free_before_engine is not None and hbm_free_after_engine is not None:
                        metrics["sample/hbm_used_by_engine_gib"] = hbm_free_before_engine - hbm_free_after_engine
                    if hbm_free_after_gen is not None:
                        metrics["sample/hbm_free_after_gen_gib"] = hbm_free_after_gen
                    if config.engine.debug_stats_every_n:
                        logger.warning("RoundBatch: metrics log start")
                    levanter.tracker.log(metrics, step=0)
                    if config.engine.debug_stats_every_n:
                        logger.warning("RoundBatch: metrics log done")

                    samples_rows: list[tuple[int, int, int, str, str, int]] = []
                    for request_index, (round_index, prompt_index) in enumerate(request_meta):
                        for generation_index in range(int(config.n_generations)):
                            sequence_index = request_index * int(config.n_generations) + generation_index
                            sequence_tokens = result.tokens[sequence_index]
                            filtered_tokens = [
                                token
                                for token in sequence_tokens
                                if token != tokenizer.pad_token_id and token != INVALID
                            ]
                            generated_text = tokenizer.decode(filtered_tokens, skip_special_tokens=True)
                            logger.info("Sequence %d: %s", sequence_index, generated_text)
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
                        try:
                            import wandb

                            if config.engine.debug_stats_every_n:
                                logger.warning("RoundBatch: samples table start")
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
                            levanter.tracker.log({"sample/samples": samples_table}, step=0)
                            if config.engine.debug_stats_every_n:
                                logger.warning("RoundBatch: samples table done")
                        except ImportError:
                            if config.engine.debug_stats_every_n:
                                logger.warning("RoundBatch: samples rows log start")
                            levanter.tracker.log({"sample/samples": samples_rows}, step=0)
                            if config.engine.debug_stats_every_n:
                                logger.warning("RoundBatch: samples rows log done")
            else:
                total_expected = len(prompt_ids) * int(config.n_generations)
                for round_index in range(int(config.n_rounds)):
                    if round_index > 0:
                        if is_leader:
                            logger.info(
                                "Recreating engine for round %d to avoid multi-round reset instability", round_index
                            )
                        del engine
                        gc.collect()
                        engine, _, _ = _create_engine(
                            model=model,
                            tokenizer=tokenizer,
                            engine_config=config.engine,
                            memory_device=memory_device,
                            is_leader=is_leader,
                        )
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

                    hbm_free_after_gen = estimated_free_device_memory(memory_device)
                    tokens_per_second = result.total_generated / duration if duration > 0 else 0.0

                    if is_leader:
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
                                metrics["sample/hbm_used_by_engine_gib"] = (
                                    hbm_free_before_engine - hbm_free_after_engine
                                )
                        if hbm_free_after_gen is not None:
                            metrics["sample/hbm_free_after_gen_gib"] = hbm_free_after_gen
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: metrics log start", round_index)
                        levanter.tracker.log(metrics, step=round_index)
                        if config.engine.debug_stats_every_n:
                            logger.warning("Round %d: metrics log done", round_index)

                        samples_rows: list[tuple[int, int, int, str, str, int]] = []
                        for sequence_index, sequence_tokens in enumerate(result.tokens):
                            filtered_tokens = [
                                token
                                for token in sequence_tokens
                                if token != tokenizer.pad_token_id and token != INVALID
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
                                levanter.tracker.log({"sample/samples": samples_table}, step=round_index)
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples table done", round_index)
                            except ImportError:
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples rows log start", round_index)
                                levanter.tracker.log({"sample/samples": samples_rows}, step=round_index)
                                if config.engine.debug_stats_every_n:
                                    logger.warning("Round %d: samples rows log done", round_index)
                    if is_multihost and not config.skip_round_barrier:
                        barrier_sync_with_tag(f"sample_lm_multihost_round_{round_index}")
        finally:
            if is_multihost:
                barrier_sync_with_tag("sample_lm_multihost_done")

        del engine
        gc.collect()
        hbm_free_after_cleanup = estimated_free_device_memory(memory_device)
        if is_leader and hbm_free_after_cleanup is not None:
            levanter.tracker.log({"sample/hbm_free_after_cleanup_gib": hbm_free_after_cleanup}, step=config.n_rounds)

    levanter.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
