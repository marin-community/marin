# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Minimal reproducer for the per-round reset TPU failure (M5.2).

This script explicitly calls engine.generate() twice to trigger the reset path.
The second generate() call calls engine.reset() at its start, which is where
the TPU slice failure occurs on multi-host.

Usage:
    python lib/levanter/infra/launch.py --foreground --zone us-central1-a \
        --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand -- \
        -e TPU_STDERR_LOG_LEVEL 0 -e TPU_MIN_LOG_LEVEL 0 -e JAX_ASYNC_ERROR_CHECKING 1 -- \
        uv run lib/levanter/scripts/test_reset_multihost.py \
        --config_path config/sampler/sample_llama8b_multihost_real_1prompt_256.yaml \
        2>&1 | tee /tmp/levanter_run_m52_reset_test.log
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
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import (
    barrier_sync_with_tag,
    estimated_free_device_memory,
    use_cpu_device,
)

logger = logging.getLogger(__name__)


@dataclass
class TestResetConfig:
    """Configuration for reset test."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    engine: InferenceEngineConfig = field(default_factory=lambda: InferenceEngineConfig(max_seq_len=1024))

    prompt: str = "Hello, world!"
    max_new_tokens: int = 32
    temperature: float = 0.7
    seed: int = 42


def _log_all_hosts(msg: str, *args, is_leader: bool = False) -> None:
    """Log on all hosts but prefix with host index."""
    prefix = f"[host={jax.process_index()}]"
    if is_leader:
        prefix = f"[LEADER host={jax.process_index()}]"
    logger.info(f"{prefix} {msg}", *args)


def _log_decode_stats(engine: InferenceEngine, stage: str) -> None:
    """Log decode-state stats from ALL hosts (required for multi-host safety)."""
    # CRITICAL: All hosts must call device_get on sharded arrays
    stats = jax.device_get(engine.gen_state.decode_state.stats())
    is_leader = jax.process_index() == 0
    if is_leader:
        logger.info(
            "DecodeStats[%s]: active=%d pages_in_use=%d free=%d max_refcount=%d",
            stage,
            int(stats.active_seqs),
            int(stats.pages_in_use),
            int(stats.free_pages),
            int(stats.max_refcount),
        )


def _load_model(config: TestResetConfig, vocab_axis: Axis, tokenizer, *, key) -> LmHeadModel:
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


def main(config: TestResetConfig):
    """Test reset by calling generate() twice."""

    levanter.initialize(config)

    is_multihost = jax.process_count() > 1
    is_leader = jax.process_index() == 0

    _log_all_hosts("Starting reset test (processes=%d)", jax.process_count(), is_leader=is_leader)

    if is_multihost:
        barrier_sync_with_tag("test_reset_start")

    tokenizer_name = config.tokenizer
    if tokenizer_name is None and config.hf_checkpoint is not None:
        tokenizer_name = config.hf_checkpoint.model_name_or_path
    assert tokenizer_name is not None, "Must specify tokenizer or hf_checkpoint with tokenizer"
    tokenizer = load_tokenizer(tokenizer_name)

    # Tokenize prompt
    prompt_ids = tokenizer(config.prompt, add_special_tokens=False)["input_ids"]
    _log_all_hosts("Prompt tokenized: %d tokens", len(prompt_ids), is_leader=is_leader)

    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        vocab_axis = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

        _log_all_hosts("Loading model...", is_leader=is_leader)
        model = _load_model(config, vocab_axis, tokenizer, key=key)
        _log_all_hosts("Model loaded", is_leader=is_leader)

        memory_device = jax.local_devices()[0] if jax.local_devices() else None
        hbm_free_before = estimated_free_device_memory(memory_device)
        _log_all_hosts("HBM free before engine: %s GiB", hbm_free_before, is_leader=is_leader)

        _log_all_hosts("Creating engine...", is_leader=is_leader)
        engine = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.engine)
        _log_all_hosts("Engine created", is_leader=is_leader)

        hbm_free_after_engine = estimated_free_device_memory(memory_device)
        _log_all_hosts("HBM free after engine: %s GiB", hbm_free_after_engine, is_leader=is_leader)

        # Build a simple request
        def build_request(request_id: int, seed_offset: int = 0) -> Request:
            req_key = jrandom.fold_in(jrandom.PRNGKey(config.seed), seed_offset)
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(prompt_ids) + config.max_new_tokens, dtype=jnp.int32),
                stop_tokens=None,
                temperature=jnp.array(config.temperature, dtype=jnp.float32),
                key=req_key,
            )
            return Request(
                prompt_tokens=list(map(int, prompt_ids)),
                request_id=request_id,
                decode_params=seq_params,
                n_generations=1,
            )

        # ============================================================
        # FIRST GENERATE CALL
        # ============================================================
        _log_all_hosts("=" * 60, is_leader=is_leader)
        _log_all_hosts("FIRST GENERATE CALL", is_leader=is_leader)
        _log_all_hosts("=" * 60, is_leader=is_leader)

        request1 = build_request(request_id=0, seed_offset=0)

        _log_all_hosts("First generate: start", is_leader=is_leader)
        start_time = time.time()
        result1 = engine.generate([request1])
        duration1 = time.time() - start_time
        _log_all_hosts(
            "First generate: done (%.2fs, %d tokens)", duration1, result1.total_generated, is_leader=is_leader
        )

        _log_decode_stats(engine, "after_first_generate")

        if is_leader:
            generated_text1 = tokenizer.decode(result1.tokens[0], skip_special_tokens=True)
            logger.info(
                "First generation: %s",
                generated_text1[:100] + "..." if len(generated_text1) > 100 else generated_text1,
            )

        if is_multihost:
            barrier_sync_with_tag("after_first_generate")

        # ============================================================
        # EXPLICIT BLOCK_UNTIL_READY BEFORE SECOND CALL
        # ============================================================
        _log_all_hosts("Calling jax.block_until_ready on gen_state...", is_leader=is_leader)

        # Block until all pending computations are done
        jax.block_until_ready(jax.tree.leaves(engine.gen_state))

        _log_all_hosts("block_until_ready completed", is_leader=is_leader)

        if is_multihost:
            barrier_sync_with_tag("after_block_until_ready")

        # ============================================================
        # SECOND GENERATE CALL (triggers reset at start)
        # ============================================================
        _log_all_hosts("=" * 60, is_leader=is_leader)
        _log_all_hosts("SECOND GENERATE CALL (will trigger reset)", is_leader=is_leader)
        _log_all_hosts("=" * 60, is_leader=is_leader)

        request2 = build_request(request_id=1, seed_offset=1)

        _log_all_hosts("Second generate: about to call (this triggers reset internally)", is_leader=is_leader)

        # This is where the TPU slice failure occurs on multi-host!
        # Inside generate(), the first thing that happens is engine.reset()
        start_time = time.time()
        result2 = engine.generate([request2])
        duration2 = time.time() - start_time

        _log_all_hosts(
            "Second generate: done (%.2fs, %d tokens)", duration2, result2.total_generated, is_leader=is_leader
        )

        _log_decode_stats(engine, "after_second_generate")

        if is_leader:
            generated_text2 = tokenizer.decode(result2.tokens[0], skip_special_tokens=True)
            logger.info(
                "Second generation: %s",
                generated_text2[:100] + "..." if len(generated_text2) > 100 else generated_text2,
            )

        if is_multihost:
            barrier_sync_with_tag("after_second_generate")

        # ============================================================
        # SUCCESS
        # ============================================================
        _log_all_hosts("=" * 60, is_leader=is_leader)
        _log_all_hosts("SUCCESS: Both generate() calls completed!", is_leader=is_leader)
        _log_all_hosts("=" * 60, is_leader=is_leader)

        del engine
        gc.collect()

    if is_multihost:
        barrier_sync_with_tag("test_reset_done")

    levanter.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
