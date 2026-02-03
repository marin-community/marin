# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import gc
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
    model = converter.load_pretrained(
        config.model.model_type, ref=config.hf_checkpoint, dtype=config.trainer.mp.compute_dtype
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
    total_sequences = num_prompts * int(config.n_generations)
    total_prompt_tokens = sum(len(tokens) for tokens in prompt_ids)
    max_prompt_len = max((len(tokens) for tokens in prompt_ids), default=0)

    _require(engine_config.max_seqs_in_prefill >= num_prompts, "engine.max_seqs_in_prefill must cover all prompts")
    _require(engine_config.max_seqs >= total_sequences, "engine.max_seqs must cover all generations")
    _require(engine_config.max_seq_len >= max_prompt_len + config.max_new_tokens, "engine.max_seq_len is too small")

    if engine_config.max_prefill_size is not None:
        _require(
            engine_config.max_prefill_size >= total_prompt_tokens,
            "engine.max_prefill_size must cover the sum of prompt lengths",
        )

    if stop_ids is not None:
        _require(engine_config.max_stop_seqs >= 1, "engine.max_stop_seqs must be >= 1 when using stop_sequence")
        _require(
            engine_config.max_stop_tokens >= len(stop_ids),
            "engine.max_stop_tokens must be >= stop_sequence token length",
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

    _validate_engine_config(
        config,
        config.engine,
        prompt_ids,
        stop_ids,
        is_multihost=is_multihost,
    )

    if stop_ids is None:
        stop_tokens = None
    else:
        stop_tokens = hax.named(jnp.asarray(stop_ids, dtype=jnp.int32), axis="position").broadcast_axis({"stop_seq": 1})

    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        vocab_axis = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, vocab_axis, tokenizer, key=key)

        memory_device = jax.local_devices()[0] if jax.local_devices() else None
        hbm_free_before_engine = estimated_free_device_memory(memory_device)
        engine_start_time = time.perf_counter()
        engine = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.engine)
        engine_creation_time = time.perf_counter() - engine_start_time
        hbm_free_after_engine = estimated_free_device_memory(memory_device)

        if config.log_kernel_jaxprs_path:
            jaxprs_path = config.log_kernel_jaxprs_path
            if is_multihost:
                jaxprs_path = f"{jaxprs_path}.host{jax.process_index()}"
            engine.write_kernel_jaxprs(jaxprs_path)

        base_key = jrandom.PRNGKey(base_seed)
        requests: list[Request] = []
        for request_index, tokens in enumerate(prompt_ids):
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(tokens) + config.max_new_tokens, dtype=jnp.int32),
                stop_tokens=stop_tokens,
                temperature=jnp.array(config.temperature, dtype=jnp.float32),
                key=jrandom.fold_in(base_key, request_index),
            )
            requests.append(
                Request(
                    prompt_tokens=list(map(int, tokens)),
                    request_id=request_index,
                    decode_params=seq_params,
                    n_generations=int(config.n_generations),
                )
            )

        total_expected = len(prompt_ids) * int(config.n_generations)
        try:
            for round_index in range(int(config.n_rounds)):
                start_time = time.time()
                result = engine.generate(requests)
                duration = time.time() - start_time

                if len(result.tokens) != total_expected:
                    raise RuntimeError(
                        f"Expected {total_expected} sequences but got {len(result.tokens)}. "
                        "Check engine.max_seqs/max_seqs_in_prefill/max_prefill_size."
                    )

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
                    levanter.tracker.log(metrics, step=round_index)

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
                    try:
                        import wandb

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
                    except ImportError:
                        levanter.tracker.log({"sample/samples": samples_rows}, step=round_index)
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
