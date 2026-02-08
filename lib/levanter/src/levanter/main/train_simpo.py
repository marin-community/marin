# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, cast

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


def _create_inference_eval_callback(
    inference_config: "InferenceEvalConfig",
    tokenizer,
    model_max_seq_len: int,
    compute_axis_mapping,
):
    """Create a callback that runs inference evaluation during training.

    This callback pauses training, creates an InferenceEngine, generates samples
    from the prompts, logs the results to WandB (including a samples table),
    and then discards the engine to free memory.

    For multi-host training (e.g., v5p-64 with 8 hosts):
    - We create a per-host replicated model copy for inference
    - Each host runs inference independently on its local devices
    - This avoids XLA launch ID mismatch issues with globally-sharded inference
    - Only process 0 logs results
    """
    prompts = inference_config.prompts
    max_new_tokens = inference_config.max_new_tokens
    temperature = inference_config.temperature
    max_seq_len = inference_config.max_seq_len or min(model_max_seq_len, 512)
    hbm_utilization = inference_config.hbm_utilization
    max_pages = inference_config.max_pages
    allow_multihost = inference_config.allow_multihost

    def inference_eval_callback(step: StepInfo):
        import sys
        import time
        import traceback

        def debug_print(msg):
            """Print with flush for immediate output."""
            print(f"[INFERENCE DEBUG][Process {jax.process_index()}][Step {step.step}] {msg}", flush=True)
            sys.stdout.flush()

        def block_until_ready_tree(tree):
            def _block(x):
                if is_jax_array_like(x):
                    jax.block_until_ready(x)
                return x

            jax.tree_util.tree_map(_block, tree)

        debug_print(">>> CALLBACK ENTERED")

        if step.step == 0:
            debug_print("Skipping step 0")
            return

        is_multihost = jax.process_count() > 1
        is_leader = jax.process_index() == 0

        debug_print(f"is_multihost={is_multihost}, is_leader={is_leader}, process_count={jax.process_count()}")

        debug_print("Blocking until training step is fully complete...")
        block_until_ready_tree(step.state)
        debug_print("Training state ready")

        # CRITICAL: Sync all hosts BEFORE starting inference
        # This ensures all hosts enter the inference code path together
        if is_multihost:
            debug_print("Barrier BEFORE inference (ensuring all hosts enter together)...")
            barrier_sync_with_tag(f"levanter_inference_pre_{step.step}", timeout=120.0)
            debug_print("Pre-inference barrier passed")

        if is_multihost and not allow_multihost:
            if is_leader:
                logger.info(f"[Step {step.step}] Skipping inference evaluation on multi-host (allow_multihost=False)")
            debug_print("allow_multihost=False, returning")
            return

        prompt_texts = prompts

        debug_print("Starting inference...")
        if is_leader:
            logger.info(f"[Step {step.step}] Running inference evaluation with {len(prompts)} prompts...")

        # Start total timing
        total_start_time = time.perf_counter()

        # Get local devices for this host
        local_devices = jax.local_devices()
        memory_device = local_devices[0]

        # Log memory before inference
        debug_print("Getting HBM memory stats...")
        hbm_free_before = estimated_free_device_memory(memory_device)
        if hbm_free_before is not None and is_leader:
            logger.info(f"[Step {step.step}] HBM free before inference: {hbm_free_before:.2f} GiB")
        debug_print(f"HBM free before: {hbm_free_before}")

        debug_print("Getting eval_model...")
        model = step.eval_model
        debug_print(f"Model type: {type(model)}")
        debug_print("Calling inference_mode...")
        model = inference_mode(model, True)
        debug_print("inference_mode done")

        # For multi-host: replicate model to local devices and run inference in local mesh
        # For single-host: run inference with global mesh as before
        local_model = None
        local_mesh = None
        local_mesh_devices = list(local_devices)

        try:
            max_prefill_size = max_seq_len
            resolved_max_pages = max_pages

            # Prepare prompt tokens and parameters
            prompt_tokens = [tokenizer.encode(prompt) for prompt in prompt_texts]
            base_seed = int(jax.device_get(step.state.step))

            if is_multihost:
                # Create a local mesh for this host's devices
                debug_print(f"Creating local mesh with {len(local_devices)} devices...")
                local_mesh = create_local_mesh(devices=local_devices)
                debug_print(f"Local mesh created: {local_mesh}")
                local_mesh_devices = list(local_mesh.devices.flat)

                # Replicate the globally-sharded model to local devices
                # This triggers an all-gather to collect all shards
                debug_print("Replicating model to local devices (all-gather)...")
                replication_start = time.perf_counter()
                local_model = replicate_model_to_local_mesh(model, local_mesh=local_mesh)
                block_until_ready_tree(local_model)
                replication_time = time.perf_counter() - replication_start
                debug_print(f"Model replicated in {replication_time:.2f}s")
                if is_leader:
                    logger.info(f"[Step {step.step}] Model replication time: {replication_time:.2f}s")

                # Log memory after replication
                hbm_free_after_replication = estimated_free_device_memory(memory_device)
                if hbm_free_after_replication is not None and is_leader:
                    logger.info(
                        f"[Step {step.step}] HBM free after model replication: {hbm_free_after_replication:.2f} GiB"
                    )

                # Infer max_pages using the local model and local devices
                if resolved_max_pages is None:
                    debug_print("Inferring max_pages for inference KV cache...")
                    budget_config = InferenceEngineConfig(
                        max_seq_len=max_seq_len,
                        max_seqs=len(prompt_texts),
                        hbm_utilization=hbm_utilization,
                        page_size=128,
                        max_pages=None,
                        max_rounds=32,
                        max_prefill_size=max_prefill_size,
                        devices=local_mesh_devices,
                    )
                    # Use the local mesh context for budget inference
                    with hax.partitioning.set_mesh(local_mesh):
                        with hax.axis_mapping({}):  # No sharding for local inference
                            resolved_max_pages = _infer_max_pages_from_hbm(local_model, budget_config)

                inference_model = local_model
            else:
                # Single-host: use original model with global mesh
                if resolved_max_pages is None:
                    debug_print("Inferring max_pages for inference KV cache...")
                    budget_config = InferenceEngineConfig(
                        max_seq_len=max_seq_len,
                        max_seqs=len(prompt_texts),
                        hbm_utilization=hbm_utilization,
                        page_size=128,
                        max_pages=None,
                        max_rounds=32,
                        max_prefill_size=max_prefill_size,
                        devices=None,
                    )
                    with hax.axis_mapping(compute_axis_mapping):
                        resolved_max_pages = _infer_max_pages_from_hbm(model, budget_config)
                inference_model = model

            if len(prompt_tokens) != len(prompt_texts):
                raise RuntimeError(
                    f"Prompt tokenization mismatch: {len(prompt_tokens)} tokens for {len(prompt_texts)} prompts."
                )
            if resolved_max_pages is None:
                raise RuntimeError("resolved_max_pages must be set before creating the inference engine.")

            debug_print(
                f"Creating engine config with max_seq_len={max_seq_len}, "
                f"max_seqs={len(prompt_texts)}, max_pages={resolved_max_pages}"
            )
            engine_config = InferenceEngineConfig(
                max_seq_len=max_seq_len,
                max_seqs=len(prompt_texts),
                hbm_utilization=hbm_utilization,
                page_size=128,
                max_pages=resolved_max_pages,
                max_rounds=32,
                max_prefill_size=max_prefill_size,
                devices=local_mesh_devices if is_multihost else None,
            )
            debug_print("Engine config created")

            # Time engine creation
            debug_print(">>> Creating InferenceEngine...")
            engine_start_time = time.perf_counter()

            if is_multihost:
                # Run engine creation and inference within local mesh context
                with hax.partitioning.set_mesh(local_mesh):
                    with hax.axis_mapping({}):  # No sharding for local inference
                        engine = InferenceEngine.from_model_with_config(
                            model=inference_model,
                            tokenizer=tokenizer,
                            config=engine_config,
                        )
            else:
                with hax.axis_mapping(compute_axis_mapping):
                    engine = InferenceEngine.from_model_with_config(
                        model=inference_model,
                        tokenizer=tokenizer,
                        config=engine_config,
                    )

            engine_creation_time = time.perf_counter() - engine_start_time
            debug_print(f"<<< Engine created in {engine_creation_time:.2f}s")
            if is_leader:
                logger.info(f"[Step {step.step}] Engine creation time: {engine_creation_time:.2f}s")

            # Log memory after engine creation (KV cache allocated)
            hbm_free_after_engine = estimated_free_device_memory(memory_device)
            if hbm_free_after_engine is not None and is_leader:
                logger.info(f"[Step {step.step}] HBM free after engine creation: {hbm_free_after_engine:.2f} GiB")

            debug_print("Creating requests...")
            requests = []
            base_key = jax.random.PRNGKey(base_seed)
            for i, tokens in enumerate(prompt_tokens):
                key = jax.random.fold_in(base_key, i)
                requests.append(
                    Request(
                        prompt_tokens=tokens,
                        request_id=i,
                        decode_params=SeqDecodingParams(
                            max_num_tokens=jnp.array(len(tokens) + max_new_tokens, dtype=jnp.int32),
                            temperature=jnp.array(temperature, dtype=jnp.float32),
                            stop_tokens=None,
                            key=key,
                        ),
                        n_generations=1,
                    )
                )
            debug_print(f"Created {len(requests)} requests")

            # Time generation - each host runs independently with local model
            debug_print(f">>> Starting generation with {len(requests)} requests...")
            generation_start_time = time.perf_counter()

            if is_multihost:
                with hax.partitioning.set_mesh(local_mesh):
                    with hax.axis_mapping({}):  # No sharding for local inference
                        result = engine.generate(requests)
            else:
                with hax.axis_mapping(compute_axis_mapping):
                    result = engine.generate(requests)

            # Block until results are ready
            debug_print("Blocking until inference results ready...")
            for tokens in result.tokens:
                jax.block_until_ready(tokens)
            block_until_ready_tree(engine.gen_state)
            debug_print("All inference results materialized")

            generation_time = time.perf_counter() - generation_start_time
            debug_print(f"<<< Generation complete in {generation_time:.2f}s, {result.total_generated} tokens")
            if is_leader:
                logger.info(f"[Step {step.step}] Generation time: {generation_time:.2f}s")

            # Calculate tokens per second
            tokens_per_second = result.total_generated / generation_time if generation_time > 0 else 0
            if is_leader:
                logger.info(
                    f"[Step {step.step}] Generated {result.total_generated} tokens at {tokens_per_second:.1f} tokens/sec"
                )

            # Log memory after generation
            hbm_free_after_gen = estimated_free_device_memory(memory_device)
            if hbm_free_after_gen is not None and is_leader:
                logger.info(f"[Step {step.step}] HBM free after generation: {hbm_free_after_gen:.2f} GiB")

            # Decode results
            debug_print("Decoding results...")
            decoded_texts = []
            for i, (prompt, tokens) in enumerate(zip(prompts, result.tokens)):
                generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append((prompt, generated_text, len(tokens)))
            debug_print(f"Decoded {len(decoded_texts)} results")

            # Only leader logs to wandb/logger
            if is_leader:
                try:
                    import wandb

                    samples_table = wandb.Table(
                        columns=["step", "prompt_id", "prompt", "generated_text", "num_tokens"]
                    )

                    for i, (prompt, generated_text, num_tokens) in enumerate(decoded_texts):
                        logger.info(f"\n[Step {step.step}] Prompt {i}: {prompt[:50]}...")
                        logger.info(f"[Step {step.step}] Generated: {generated_text[:200]}...")
                        samples_table.add_data(step.step, i, prompt, generated_text, num_tokens)

                    levanter.tracker.log(
                        {"inference_eval/samples": samples_table},
                        step=step.step,
                    )

                except ImportError:
                    for i, (prompt, generated_text, num_tokens) in enumerate(decoded_texts):
                        logger.info(f"\n[Step {step.step}] Prompt {i}: {prompt[:50]}...")
                        logger.info(f"[Step {step.step}] Generated: {generated_text[:200]}...")

                metrics: dict = {
                    "inference_eval/total_tokens": result.total_generated,
                    "inference_eval/num_prompts": len(prompts),
                    "inference_eval/avg_tokens_per_prompt": result.total_generated / len(prompts),
                    "inference_eval/engine_creation_time_sec": engine_creation_time,
                    "inference_eval/generation_time_sec": generation_time,
                    "inference_eval/tokens_per_second": tokens_per_second,
                }

                if hbm_free_before is not None:
                    metrics["inference_eval/hbm_free_before_gib"] = hbm_free_before
                if hbm_free_after_engine is not None:
                    metrics["inference_eval/hbm_free_after_engine_gib"] = hbm_free_after_engine
                if hbm_free_after_gen is not None:
                    metrics["inference_eval/hbm_free_after_gen_gib"] = hbm_free_after_gen
                if hbm_free_before is not None and hbm_free_after_engine is not None:
                    metrics["inference_eval/hbm_used_by_engine_gib"] = hbm_free_before - hbm_free_after_engine

                levanter.tracker.log(metrics, step=step.step)

            # Cleanup
            del engine
            if local_model is not None:
                del local_model
            gc.collect()

            # Log memory after cleanup
            hbm_free_after_cleanup = estimated_free_device_memory(memory_device)
            if hbm_free_after_cleanup is not None and is_leader:
                logger.info(f"[Step {step.step}] HBM free after cleanup: {hbm_free_after_cleanup:.2f} GiB")

            total_time = time.perf_counter() - total_start_time
            debug_print(f"Inference evaluation complete in {total_time:.2f}s")
            if is_leader:
                logger.info(f"[Step {step.step}] Inference evaluation complete in {total_time:.2f}s")

        except Exception as e:
            debug_print(f"!!! EXCEPTION: {e}")
            logger.error(f"[Step {step.step}] Inference evaluation failed: {e}")
            traceback.print_exc()
            # Cleanup on error
            if local_model is not None:
                del local_model
            gc.collect()

        # Sync all processes before returning to training
        # This ensures all processes complete inference before any resume training
        if is_multihost:
            debug_print("Syncing host processes (barrier_sync_with_tag)...")
            barrier_sync_with_tag(f"levanter_inference_post_{step.step}", timeout=120.0)
            debug_print("Barrier passed, resuming training")

        debug_print("<<< CALLBACK EXITING")

    return inference_eval_callback


@dataclass
class InferenceEvalConfig:
    """Configuration for running inference evaluation during training."""

    enabled: bool = False
    """Whether to run inference evaluation."""
    eval_every: int = 10
    """Run inference every N steps."""
    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
        ]
    )
    """Prompts to generate from during evaluation."""
    max_new_tokens: int = 128
    """Maximum number of new tokens to generate per prompt."""
    temperature: float = 0.7
    """Sampling temperature for generation."""
    max_seq_len: int | None = 512
    """Maximum sequence length for inference. Kept small to avoid OOM during training."""
    hbm_utilization: float = 0.2
    """Fraction of HBM to use for inference KV cache (keep low to avoid OOM during training)."""
    max_pages: int = 64
    """Maximum number of KV cache pages. Kept small to avoid VMEM exhaustion in paged attention."""
    allow_multihost: bool = True
    """If True, run inference on all hosts during multi-host training.
    If False, skip inference entirely on multi-host setups."""


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
            logger.info(
                f"Inference evaluation enabled: running every {config.inference_eval.eval_every} steps "
                f"with {len(config.inference_eval.prompts)} prompts"
            )
            inference_callback = _create_inference_eval_callback(
                config.inference_eval,
                tokenizer,
                model_max_seq_len=model_max_seq_len,
                compute_axis_mapping=trainer.compute_axis_mapping,
            )
            trainer.add_hook(inference_callback, every=config.inference_eval.eval_every)

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
