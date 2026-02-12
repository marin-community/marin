# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Test per-round reset with the actual sample_lm_multihost logic.

This script uses the same logic as sample_lm_multihost.py but forces the per-round
code path to verify that reset between generate() calls works correctly.
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

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
    multihost_broadcast_sync,
    use_cpu_device,
)

logger = logging.getLogger(__name__)


@dataclass
class TestPerRoundConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    tokenizer: str | None = None
    engine: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(max_seq_len=1024))

    prompts: list[str] | str | tuple[str, ...] = ("Hello, world!",)
    max_new_tokens: int = 64
    temperature: float = 0.7
    n_rounds: int = 2  # Number of rounds to run (each is a separate generate() call)
    seed: int = 42


def _load_model(config: TestPerRoundConfig, vocab_axis: Axis, tokenizer, *, key) -> LmHeadModel:
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
    hf_config = converter.hf_config_from_hf_checkpoint(config.hf_checkpoint)
    overrides = {"use_tpu_ragged_paged_attention": config.model.use_tpu_ragged_paged_attention}
    merged_config = converter.config_from_hf_config(hf_config, overrides=overrides)
    model = converter.load_pretrained(
        config.model.model_type, ref=config.hf_checkpoint, config=merged_config, dtype=config.trainer.mp.compute_dtype
    )
    return model


def _log_decode_stats(engine: InferenceEngine, stage: str, *, is_leader: bool) -> None:
    stats = jax.device_get(engine.gen_state.decode_state.stats())
    if not is_leader:
        return
    logger.info(
        "DecodeStats[%s]: active=%d pages_in_use=%d free=%d max_refcount=%d",
        stage,
        int(stats.active_seqs),
        int(stats.pages_in_use),
        int(stats.free_pages),
        int(stats.max_refcount),
    )


def main(config: TestPerRoundConfig):
    """Test per-round reset with multiple generate() calls."""

    levanter.initialize(config)

    is_multihost = jax.process_count() > 1
    is_leader = jax.process_index() == 0

    if is_leader:
        logger.info("Starting per-round reset test (n_rounds=%d, processes=%d)", config.n_rounds, jax.process_count())

    if is_multihost:
        barrier_sync_with_tag("test_perround_start")

    tokenizer_name = config.tokenizer
    if tokenizer_name is None and config.hf_checkpoint is not None:
        tokenizer_name = config.hf_checkpoint.model_name_or_path
    assert tokenizer_name is not None
    tokenizer = load_tokenizer(tokenizer_name)

    prompts = list(config.prompts) if isinstance(config.prompts, (list, tuple)) else [config.prompts]

    if is_multihost:
        payload = {"prompt_ids": tokenizer(prompts, add_special_tokens=False)["input_ids"], "seed": config.seed}
        payload = multihost_broadcast_sync(payload, is_source=is_leader)
        prompt_ids = payload["prompt_ids"]
        base_seed = payload["seed"]
    else:
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        base_seed = config.seed

    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        vocab_axis = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

        if is_leader:
            logger.info("Loading model...")
        model = _load_model(config, vocab_axis, tokenizer, key=key)
        if is_leader:
            logger.info("Model loaded")

        _ = jax.local_devices()[0] if jax.local_devices() else None

        if is_leader:
            logger.info("Creating engine...")
        engine = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.engine)
        if is_leader:
            logger.info("Engine created")

        base_key = jrandom.PRNGKey(base_seed)
        num_prompts = len(prompt_ids)

        # ====================================================================
        # PER-ROUND LOOP: This is the pattern that was failing before M5.2 fix
        # ====================================================================
        for round_index in range(config.n_rounds):
            if is_leader:
                logger.info("=" * 60)
                logger.info("ROUND %d of %d", round_index, config.n_rounds)
                logger.info("=" * 60)

            # Build requests for this round
            round_key = jrandom.fold_in(base_key, round_index)
            requests = []
            for prompt_index, tokens in enumerate(prompt_ids):
                seq_params = SeqDecodingParams(
                    max_num_tokens=jnp.array(len(tokens) + config.max_new_tokens, dtype=jnp.int32),
                    stop_tokens=None,
                    temperature=jnp.array(config.temperature, dtype=jnp.float32),
                    key=jrandom.fold_in(round_key, prompt_index),
                )
                requests.append(
                    Request(
                        prompt_tokens=list(map(int, tokens)),
                        request_id=round_index * num_prompts + prompt_index,
                        decode_params=seq_params,
                        n_generations=1,
                    )
                )

            if is_leader:
                logger.info("Round %d: generate starting (%d requests)", round_index, len(requests))

            start_time = time.time()
            result = engine.generate(requests)
            duration = time.time() - start_time

            if is_leader:
                logger.info(
                    "Round %d: generate done (%.2fs, %d tokens)", round_index, duration, result.total_generated
                )

            _log_decode_stats(engine, f"round{round_index}_after_generate", is_leader=is_leader)

            # Log generated text
            if is_leader:
                for seq_idx, seq_tokens in enumerate(result.tokens):
                    filtered = [t for t in seq_tokens if t != tokenizer.pad_token_id and t != INVALID]
                    text = tokenizer.decode(filtered, skip_special_tokens=True)
                    logger.info(
                        "Round %d Seq %d: %s", round_index, seq_idx, text[:100] + "..." if len(text) > 100 else text
                    )

            if is_multihost:
                barrier_sync_with_tag(f"test_perround_round_{round_index}")

        # ====================================================================
        # SUCCESS
        # ====================================================================
        if is_leader:
            logger.info("=" * 60)
            logger.info("SUCCESS: All %d rounds completed with per-round reset!", config.n_rounds)
            logger.info("=" * 60)

        del engine
        gc.collect()

    if is_multihost:
        barrier_sync_with_tag("test_perround_done")

    levanter.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
