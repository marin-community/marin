# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLM Benchmark Evaluation Script.

This module provides a command-line interface for evaluating Vision-Language Models
on standard benchmarks using the lm-eval-harness framework.

Supported Benchmarks:
- MMMU, ChartQA (already in lm-eval-harness)
- MME, GQA, RealWorldQA, SEED, MMStar, AI2D, OCRBench (custom tasks)

Usage:
    python -m levanter.main.eval_vlm \
        --config eval_vlm_config.yaml \
        --eval_harness.task_spec='["mmmu", "mme"]' \
        --checkpoint_path /path/to/checkpoint
"""

import logging
import os
from dataclasses import dataclass, field
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
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode
from levanter.vlm_eval_harness import VLMEvalHarnessConfig, run_vlm_eval_harness, run_vlm_benchmark_direct


# Benchmarks that are in lm-eval-harness
LM_EVAL_HARNESS_TASKS = {"mmmu", "chartqa"}

# Benchmarks that need direct evaluation
DIRECT_EVAL_TASKS = {"mme", "gqa", "realworldqa", "seed", "mmstar", "ai2d", "ocrbench"}


logger = logging.getLogger(__name__)


@dataclass
class EvalVLMConfig:
    """Configuration for VLM benchmark evaluation."""

    # Checkpoint loading options (mutually exclusive)
    checkpoint_path: Optional[str] = None
    """Path to a Levanter checkpoint directory."""
    hf_checkpoint: Optional[RepoRef] = None
    """HuggingFace checkpoint reference (e.g., 'lmms-lab/llava-onevision-qwen2-7b-ov')."""

    # Model and training config
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LlavaOnevisionConfig = field(default_factory=LlavaOnevisionConfig)
    data: ImageMixtureDatasetConfig = field(default_factory=ImageMixtureDatasetConfig)

    # VLM Evaluation config
    eval_harness: VLMEvalHarnessConfig = field(default_factory=VLMEvalHarnessConfig)

    # Processor/tokenizer paths (can differ from checkpoint)
    processor_path: Optional[str] = None
    """HuggingFace processor path. If not specified, uses hf_checkpoint or model default."""
    tokenizer_path: Optional[str] = None
    """HuggingFace tokenizer path. If not specified, uses processor's tokenizer."""

    # Evaluation length
    max_eval_length: int = 2048
    """Maximum sequence length for evaluation."""

    # Image processing options
    image_size: int = 384
    """Image size for processing."""
    vision_feature_height: int = 27
    """Vision feature height (num_image_tokens = height^2)."""
    disable_anyres: bool = False
    """If True, disable anyres (use single image resolution)."""


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
        if not hasattr(model_config, "hf_checkpoint_converter"):
            raise ValueError("Model config does not have an HF checkpoint converter.")

        converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
        model = converter.load_pretrained(
            model_config.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype
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
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
    if vocab_size != Vocab.size:
        logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

    mp: jmp.Policy = config.trainer.mp
    key = jax.random.PRNGKey(0)

    with config.trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
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
        grid_pinpoints = None
        if not config.disable_anyres and hasattr(config.model, "image_grid_pinpoints"):
            grid_pinpoints = config.model.image_grid_pinpoints

        # Calculate max_num_patches
        if config.disable_anyres:
            max_num_patches = 0  # Only base patch
        elif grid_pinpoints:
            max_resolution = max(max(h, w) for h, w in grid_pinpoints)
            patch_size = config.image_size
            max_patches_per_dim = max_resolution // patch_size
            max_num_patches = max_patches_per_dim * max_patches_per_dim
        else:
            max_num_patches = 9  # Default for anyres_max_9

        image_processor = BatchImageProcessor(
            processor=processor,
            tokenizer=tokenizer,
            max_length=config.max_eval_length,
            padding=True,
            truncation=True,
            disable_anyres=config.disable_anyres,
            grid_pinpoints=grid_pinpoints,
            patch_size=config.image_size,
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
                    )
                    all_results[task_name] = {
                        "accuracy": task_results.get("accuracy", 0.0),
                        "correct": task_results.get("correct", 0),
                        "total": task_results.get("total", 0),
                    }
                    # Log to tracker
                    levanter.tracker.log({
                        f"vlm_eval/{task_name}/accuracy": task_results.get("accuracy", 0.0)
                    })
                except Exception as e:
                    logger.error(f"Failed to evaluate {task_name}: {e}")
                    all_results[task_name] = {"error": str(e)}

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

        return {"results": all_results}


if __name__ == "__main__":
    levanter.config.main(main)()
