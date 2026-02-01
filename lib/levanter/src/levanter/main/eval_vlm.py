# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM Benchmark Evaluation Script.

This module provides a command-line interface for evaluating Vision-Language Models
on standard benchmarks using the lm-eval-harness framework.

Supported Benchmarks:
- MMMU, ChartQA (already in lm-eval-harness)
- MME, GQA, RealWorldQA, SEED, MMStar, AI2D, OCRBench (direct evaluation)

Default Model: LLaVA-OneVision architecture
- Vision Encoder: SigLIP (384x384, patch16)
- Language Model: Qwen3-1.7B
- Projector: 2-layer MLP

Usage:
    python -m levanter.main.eval_vlm \
        --checkpoint_path /path/to/checkpoint \
        --eval_harness.task_spec='["mme", "gqa"]'
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_processor, load_tokenizer
from levanter.data.image import BatchImageProcessor, ImageMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode
from levanter.vlm_eval_harness import VLMEvalHarnessConfig, run_vlm_eval_harness, run_vlm_benchmark_direct
from haliax.partitioning import ResourceAxis


# Benchmarks that are in lm-eval-harness
LM_EVAL_HARNESS_TASKS = {"mmmu", "chartqa"}

# Benchmarks that need direct evaluation
DIRECT_EVAL_TASKS = {"mme", "gqa", "realworldqa", "seed", "mmstar", "ai2d", "ocrbench"}


logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT MODEL CONFIGURATION (matches demo_vlm_train.py)
# ============================================================================

# Flash attention block size (matches training config for memory efficiency)
FLASH_ATTENTION_BLOCK_SIZE = 1024

# Vision encoder: SigLIP (matches google/siglip2-so400m-patch16-384)
DEFAULT_VISION_CONFIG = SiglipVisionConfig(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    image_size=384,
    patch_size=16,
    gradient_checkpointing=False,  # Not needed for inference
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,  # Critical for memory efficiency
)

# Language model: Qwen3-1.7B
DEFAULT_TEXT_CONFIG = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    gradient_checkpointing=False,  # Not needed for inference
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,  # Critical for memory efficiency
)

# Qwen3-1.7B <|image_pad|> token ID
DEFAULT_IMAGE_TOKEN_INDEX = 151655

# Combined VLM config
DEFAULT_VLM_CONFIG = LlavaOnevisionConfig(
    vision_config=DEFAULT_VISION_CONFIG,
    text_config=DEFAULT_TEXT_CONFIG,
    vision_encoder_type="siglip",
    vision_feature_select_strategy="full",
    vision_aspect_ratio="single",
    disable_anyres=True,
    image_token_index=DEFAULT_IMAGE_TOKEN_INDEX,
    gradient_checkpointing=False,
)

# Vision feature height: 384 // 16 = 24
DEFAULT_VISION_FEATURE_HEIGHT = DEFAULT_VISION_CONFIG.image_size // DEFAULT_VISION_CONFIG.patch_size

# Default paths
DEFAULT_PROCESSOR_PATH = "gs://marin-vlm/processors/llava-onevision-qwen2-0.5b-ov-hf"
DEFAULT_TOKENIZER_PATH = "gs://marin-vlm/tokenizers/Qwen3-1.7B"

# Default mesh config with vision_batch sharding (critical for VLM inference)
# Without this, the vision encoder's attention runs OOM because vision_batch
# axis is not sharded across devices. This matches the training config.
DEFAULT_MESH_CONFIG = MeshConfig(
    axes={"data": -1, "replica": 1, "model": 1},
    compute_mapping={
        # vision_batch is created by flattening (batch, num_patches) in get_image_features
        # Must be sharded to avoid OOM in vision encoder's splash attention
        "vision_batch": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    },
    param_mapping={"embed": "data"},
)


def _default_trainer_config():
    """Create a TrainerConfig with VLM-specific mesh settings."""
    return TrainerConfig(mesh=DEFAULT_MESH_CONFIG)


@dataclass
class EvalVLMConfig:
    """Configuration for VLM benchmark evaluation.

    Default model configuration matches demo_vlm_train.py:
    - Vision: SigLIP (384x384, patch16, 1152 hidden)
    - LLM: Qwen3-1.7B (2048 hidden, 28 layers)
    - Single resolution mode (disable_anyres=True)
    """

    # Checkpoint loading options (mutually exclusive)
    checkpoint_path: Optional[str] = None
    """Path to a Levanter checkpoint directory."""
    hf_checkpoint: Optional[RepoRef] = None
    """HuggingFace checkpoint reference (e.g., 'lmms-lab/llava-onevision-qwen2-7b-ov')."""

    # Model config - defaults to LLaVA-OneVision (SigLIP + Qwen3-1.7B)
    # Uses VLM-specific mesh config with vision_batch sharding to avoid OOM
    trainer: TrainerConfig = field(default_factory=_default_trainer_config)
    model: LlavaOnevisionConfig = field(default_factory=lambda: DEFAULT_VLM_CONFIG)

    # VLM Evaluation config
    eval_harness: VLMEvalHarnessConfig = field(default_factory=lambda: VLMEvalHarnessConfig(
        task_spec=["mme"],  # Default task
        max_examples=None,
        generation_kwargs={"max_gen_toks": 64, "temperature": 0.0, "n": 1},
    ))

    # Processor/tokenizer paths - defaults to GCS paths for Qwen3
    processor_path: Optional[str] = DEFAULT_PROCESSOR_PATH
    """HuggingFace processor path."""
    tokenizer_path: Optional[str] = DEFAULT_TOKENIZER_PATH
    """HuggingFace tokenizer path."""

    # Evaluation length - matches model's max_seq_len
    max_eval_length: int = 4096
    """Maximum sequence length for evaluation."""

    # Image processing options - matches demo_vlm_train.py
    image_size: int = 384
    """Image size for processing (384 for SigLIP)."""
    patch_size: int = 16
    """Patch size for vision encoder (16 for SigLIP patch16)."""
    vision_feature_height: int = DEFAULT_VISION_FEATURE_HEIGHT  # 24 for patch_size=16
    """Vision feature height (num_image_tokens = height^2). Default: 24 = 384//16."""
    disable_anyres: bool = True
    """If True, disable anyres (use single image resolution). Default: True to match training."""

    # Output options
    output_dir: Optional[str] = None
    """Directory to save evaluation results locally. If None, saves to ./vlm_eval_results/."""
    save_samples: bool = True
    """If True, save individual sample outputs to local files."""


def _load_vlm_model(
    config: EvalVLMConfig,
    Vocab: Axis,
    mp: jmp.Policy,
    parameter_axis_mapping,
    tokenizer,
    key,
) -> LlavaOnevisionModel:
    """Load VLM model from checkpoint or HuggingFace."""
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Cannot specify both checkpoint_path and hf_checkpoint")
    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")

    if config.checkpoint_path is not None:
        # Load from Levanter checkpoint
        logger.info(f"Loading model from checkpoint: {config.checkpoint_path}")
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

    elif config.hf_checkpoint is not None:
        # Load from HuggingFace
        logger.info(f"Loading model from HuggingFace: {config.hf_checkpoint}")
        model_config = config.model

        # Create converter directly with our already-loaded tokenizer to avoid
        # re-loading from GCS path (which causes path handling issues)
        from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig
        converter = HFCheckpointConverter(
            LevConfigClass=model_config.__class__,
            reference_checkpoint=str(config.hf_checkpoint),
            HfConfigClass=HfLlavaOnevisionConfig,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )
        model = converter.load_pretrained(
            model_config.model_type,
            ref=config.hf_checkpoint,
            config=model_config,  # Use our config with flash_attention_block_size!
            dtype=mp.compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

    return model


def main(config: EvalVLMConfig):
    """Main function for VLM benchmark evaluation."""
    levanter.initialize(config)

    # Determine processor/tokenizer paths
    processor_path = config.processor_path
    if processor_path is None:
        if config.hf_checkpoint is not None:
            processor_path = str(config.hf_checkpoint)
        else:
            processor_path = config.model.default_hf_checkpoint_path

    tokenizer_path = config.tokenizer_path or processor_path

    logger.info(f"Loading processor from: {processor_path}")
    logger.info(f"Loading tokenizer from: {tokenizer_path}")

    # Load processor and tokenizer
    processor = load_processor(processor_path, trust_remote_code=True)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup axes
    Batch = config.trainer.EvalBatch
    Pos = config.model.max_Pos.resize(config.max_eval_length)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    # Determine vocab size
    vocab_size = len(tokenizer)

    mp: jmp.Policy = config.trainer.mp
    key = jax.random.PRNGKey(0)

    with config.trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        # Round vocab size for partitioning (must be inside mesh context)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")
        # Load model
        model = _load_vlm_model(
            config=config,
            Vocab=Vocab,
            mp=mp,
            parameter_axis_mapping=parameter_axis_mapping,
            tokenizer=tokenizer,
            key=key,
        )

        # Put model in inference mode
        model = inference_mode(model, True)
        model = mp.cast_to_compute(model)

        # Create BatchImageProcessor for image processing
        # Determine grid pinpoints for anyres
        if config.disable_anyres:
            # Single resolution mode - use image_size as single pinpoint
            grid_pinpoints = [[config.image_size, config.image_size]]  # [[384, 384]]
            max_num_patches = 0  # Only base patch
        elif hasattr(config.model, "image_grid_pinpoints") and config.model.image_grid_pinpoints:
            grid_pinpoints = config.model.image_grid_pinpoints
            max_resolution = max(max(h, w) for h, w in grid_pinpoints)
            max_patches_per_dim = max_resolution // config.patch_size  # Use patch_size, not image_size!
            max_num_patches = max_patches_per_dim * max_patches_per_dim
        else:
            grid_pinpoints = None
            max_num_patches = 9  # Default for anyres_max_9

        image_processor = BatchImageProcessor(
            processor=processor,
            tokenizer=tokenizer,
            max_length=config.max_eval_length,
            padding=False,  # Don't pad to max_length - use actual sequence length for inference
            disable_anyres=config.disable_anyres,
            grid_pinpoints=grid_pinpoints,
            patch_size=config.patch_size,  # Use patch_size (16), not image_size (384)!
            vision_feature_height=config.vision_feature_height,
            max_num_patches=max_num_patches,
        )

        # Run evaluation
        logger.info("Starting VLM benchmark evaluation...")
        logger.info(f"Tasks: {config.eval_harness.task_spec}")

        # Separate tasks into lm-eval-harness and direct evaluation
        task_spec = config.eval_harness.task_spec
        lm_eval_tasks = []
        direct_tasks = []

        for task in task_spec:
            task_name = task.lower() if isinstance(task, str) else task.task.lower()
            if task_name in LM_EVAL_HARNESS_TASKS:
                lm_eval_tasks.append(task)
            elif task_name in DIRECT_EVAL_TASKS:
                direct_tasks.append(task_name)
            else:
                logger.warning(f"Unknown task: {task_name}, trying lm-eval-harness")
                lm_eval_tasks.append(task)

        all_results = {}

        # Run lm-eval-harness tasks (MMMU, ChartQA)
        if lm_eval_tasks:
            logger.info(f"Running lm-eval-harness tasks: {lm_eval_tasks}")
            harness_config = VLMEvalHarnessConfig(
                task_spec=lm_eval_tasks,
                max_examples=config.eval_harness.max_examples,
                max_images=config.eval_harness.max_images,
                generation_kwargs=config.eval_harness.generation_kwargs,
                custom_task_path=config.eval_harness.custom_task_path,
                output_dir=config.eval_harness.output_dir or config.output_dir,
                vlm_batch_size=config.eval_harness.vlm_batch_size,
                checkpoint_interval=config.eval_harness.checkpoint_interval,
                checkpoint_dir=config.eval_harness.checkpoint_dir,
                resume_from_checkpoint=config.eval_harness.resume_from_checkpoint,
                auto_resume=config.eval_harness.auto_resume,
            )
            harness_results = run_vlm_eval_harness(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                config=harness_config,
                EvalBatch=Batch,
                EvalPos=Pos,
                axis_resources=compute_axis_mapping,
                mp=mp,
            )
            if harness_results and "results" in harness_results:
                all_results.update(harness_results["results"])

        # Run direct evaluation tasks (MME, GQA, etc.)
        if direct_tasks:
            logger.info(f"Running direct evaluation tasks: {direct_tasks}")
            for task_name in direct_tasks:
                logger.info(f"Evaluating {task_name}...")
                try:
                    task_results = run_vlm_benchmark_direct(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        benchmark_name=task_name,
                        EvalBatch=Batch,
                        EvalPos=Pos,
                        axis_resources=compute_axis_mapping,
                        mp=mp,
                        max_examples=config.eval_harness.max_examples,
                        generation_kwargs=config.eval_harness.generation_kwargs,
                        checkpoint_interval=config.eval_harness.checkpoint_interval,
                        checkpoint_dir=config.eval_harness.checkpoint_dir,
                        resume_from_checkpoint=config.eval_harness.resume_from_checkpoint,
                        auto_resume=config.eval_harness.auto_resume,
                        output_dir=config.eval_harness.output_dir or config.output_dir,
                    )
                    all_results[task_name] = {
                        "accuracy": task_results.get("accuracy", 0.0),
                        "correct": task_results.get("correct", 0),
                        "total": task_results.get("total", 0),
                    }
                    # Log to tracker
                    levanter.tracker.log({
                        f"vlm_eval/{task_name}/accuracy": task_results.get("accuracy", 0.0)
                    }, step=0)
                except Exception as e:
                    logger.error(f"Failed to evaluate {task_name}: {e}")
                    all_results[task_name] = {"error": str(e)}

        # Collect sample outputs from harness results if available
        sample_outputs = {}
        if lm_eval_tasks and harness_results and "sample_outputs" in harness_results:
            sample_outputs.update(harness_results["sample_outputs"])

        # Print summary
        print("\n" + "=" * 60)
        print("VLM Benchmark Evaluation Results")
        print("=" * 60)

        for task_name, metrics in all_results.items():
            print(f"\n{task_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
                elif metric_name == "error":
                    print(f"  ERROR: {value}")

        print("=" * 60)

        # Save results locally
        output_dir = config.output_dir or "./vlm_eval_results"
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_names = "_".join([t.lower() for t in task_spec[:3]])  # Use first 3 task names
        if len(task_spec) > 3:
            task_names += f"_and_{len(task_spec) - 3}_more"

        # Save summary results
        results_file = os.path.join(output_dir, f"results_{task_names}_{timestamp}.json")
        results_data = {
            "timestamp": timestamp,
            "checkpoint": config.checkpoint_path or str(config.hf_checkpoint),
            "tasks": [str(t) for t in task_spec],
            "results": all_results,
            # Evaluation metadata for reproducibility
            "metadata": {
                # Model configuration
                "model": {
                    "vision_encoder": config.model.vision_encoder_type,
                    "vision_hidden_size": config.model.vision_config.hidden_size,
                    "vision_image_size": config.model.vision_config.image_size,
                    "vision_patch_size": config.model.vision_config.patch_size,
                    "text_hidden_size": config.model.text_config.hidden_dim,
                    "text_num_layers": config.model.text_config.num_layers,
                    "text_max_seq_len": config.model.text_config.max_seq_len,
                },
                # Evaluation parameters
                "eval_config": {
                    "max_eval_length": config.max_eval_length,
                    "vlm_batch_size": config.eval_harness.vlm_batch_size,
                    "max_examples": config.eval_harness.max_examples,
                    "generation_kwargs": config.eval_harness.generation_kwargs,
                },
                # Image processing settings
                "image_processing": {
                    "image_size": config.image_size,
                    "patch_size": config.patch_size,
                    "vision_feature_height": config.vision_feature_height,
                    "disable_anyres": config.disable_anyres,
                },
                # Paths
                "paths": {
                    "processor_path": config.processor_path,
                    "tokenizer_path": config.tokenizer_path,
                },
                # Hardware info
                "hardware": {
                    "num_devices": jax.device_count(),
                    "device_type": str(jax.devices()[0].platform) if jax.devices() else "unknown",
                },
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_file}")
        print(f"\nResults saved to: {results_file}")

        # Save sample outputs if enabled
        if config.save_samples and sample_outputs:
            samples_file = os.path.join(output_dir, f"samples_{task_names}_{timestamp}.json")
            with open(samples_file, "w") as f:
                json.dump(sample_outputs, f, indent=2, default=str)
            logger.info(f"Sample outputs saved to: {samples_file}")
            print(f"Sample outputs saved to: {samples_file}")

            # Also print a few sample outputs to console
            print("\n" + "-" * 60)
            print("Sample Outputs (first 3 per task):")
            print("-" * 60)
            for task_name, samples in sample_outputs.items():
                print(f"\n[{task_name}]")
                for i, sample in enumerate(samples[:3]):
                    prompt = sample.get("prompt", "")
                    if len(prompt) > 200:
                        prompt = prompt[:200] + "..."
                    print(f"  [{i+1}] Prompt: {prompt}")
                    print(f"      Response: {sample.get('generation', '')}")
                    print()

        return {"results": all_results, "sample_outputs": sample_outputs}


if __name__ == "__main__":
    levanter.config.main(main)()
