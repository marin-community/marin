#!/usr/bin/env python3
# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Launch script for VLM (Vision-Language Model) training with LLaVA OneVision.

This script provides a complete training pipeline for LLaVA OneVision models
using real parquet data, with performance optimizations for TPU/GPU training.

Usage:
    # Train from scratch with small model config
    python launch_vlm_training.py

    # Train with HuggingFace pretrained weights
    python launch_vlm_training.py --initialize_from_hf

    # Train with a single parquet file
    python launch_vlm_training.py --train_data /path/to/train.parquet --val_data /path/to/val.parquet

    # Train with a folder containing multiple parquet files
    python launch_vlm_training.py --train_data /path/to/train_folder/ --val_data /path/to/val_folder/

    # Train with glob pattern
    python launch_vlm_training.py --train_data "/path/to/data/*.parquet"

    # Full training run with optimizations
    python launch_vlm_training.py --initialize_from_hf --num_train_steps 10000 --train_batch_size 32

    # High-performance training with all optimizations enabled
    python launch_vlm_training.py --initialize_from_hf --mp bfloat16 \\
        --freeze_vision_encoder --per_device_parallelism 8

Performance Optimization Flags:
    --mp bfloat16           : Use mixed precision (bfloat16) for faster training
    --no_flash_attention    : Disable flash attention (enabled by default)
    --freeze_vision_encoder : Freeze vision encoder (only train projector + LLM)
    --per_device_parallelism: Number of examples per device (for gradient accumulation)
    --fsdp_axis             : FSDP sharding axis (default: embed)
"""

import argparse
import asyncio
import dataclasses
import logging

import jmp  # For mixed precision policy

import levanter.main.train_vlm as train_vlm
from levanter.data.image import ConversationDatasetSourceConfig, ImageMixtureDatasetConfig
from levanter.distributed import DistributedConfig, RayConfig
from levanter.models.llava_onevision import LlavaOnevisionConfig
from levanter.models.siglip import SiglipVisionConfig
from levanter.models.qwen import Qwen3Config, QwenConfig
from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.layers.attention import AttentionBackend
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.checkpoint import CheckpointerConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch VLM training with LLaVA OneVision")

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="./output",
        help="Path to training data. Can be: a single parquet file, a directory containing parquet files, "
        "or a glob pattern (e.g., '/path/to/*.parquet')",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data. Same format as --train_data (defaults to train_data)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/vlm_cache",
        help="Directory for data caching",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching and use streaming mode (processes images on-the-fly, saves disk space)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="HuggingFace model name for processor and optional weight initialization",
    )
    parser.add_argument(
        "--initialize_from_hf",
        action="store_true",  # Default is False; we use custom weight loading for SigLIP + Qwen3
        help="Initialize model weights from HuggingFace checkpoint (for unified llava-onevision models)",
    )
    parser.add_argument(
        "--use_hf_model_config",
        action="store_true",  # Default is False; use custom SigLIP + Qwen3 config
        help="Use model config from HuggingFace checkpoint (set to True to load full llava-onevision model)",
    )
    parser.add_argument(
        "--use_small_model",
        action="store_true",
        help="Use small model config for testing (overrides --use_hf_model_config)",
    )

    # Training arguments
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=20000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of epochs to train (default: 1). If 0, train indefinitely until num_train_steps is reached. "
        "If > 0, dataset will cycle through the data for the specified number of epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )

    # === Performance Optimization Arguments ===
    parser.add_argument(
        "--mp",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", None],
        help="Mixed precision mode: bfloat16 (recommended for TPU), float16 (GPU), or float32 (full precision)",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention (enabled by default for memory-efficient attention computation)",
    )
    parser.add_argument(
        "--flash_attention_block_size",
        type=int,
        default=64,
        help="Block size for flash attention (default: 512, use smaller values if OOM)",
    )
    parser.add_argument(
        "--per_device_parallelism",
        type=int,
        default=-1,
        help="Number of examples to process per device. -1 means train_batch_size/num_devices. "
        "Set lower for gradient accumulation to save memory.",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action="store_true",
        help="Freeze vision encoder weights (only train projector and LLM). "
        "Reduces compute by ~30% and often improves fine-tuning results.",
    )
    parser.add_argument(
        "--freeze_llm",
        action="store_true",
        help="Freeze LLM weights (only train projector and vision encoder). "
        "Useful for vision encoder fine-tuning or projector-only training.",
    )
    parser.add_argument(
        "--fsdp_axis",
        type=str,
        default="embed",
        help="Axis to use for FSDP sharding. Options: embed, mlp, or comma-separated list",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default to reduce memory usage)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/vlm_output",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--hf_save_path",
        type=str,
        default=None,
        help="Path to save HuggingFace format checkpoints",
    )
    parser.add_argument(
        "--hf_save_steps",
        type=int,
        default=1000,
        help="Save HF checkpoint every N steps",
    )
    parser.add_argument(
        "--checkpointer_path",
        type=str,
        default=None,
        help="Path for Levanter checkpoints (defaults to output_dir/checkpoints)",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="marin-vlm",
        help="Weights & Biases project name (None to disable)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    # Distributed arguments
    parser.add_argument(
        "--no_distributed",
        action="store_true",
        help="Disable JAX distributed initialization",
    )

    # Evaluation arguments
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=10,
        help="Maximum number of evaluation batches",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=500,  # Default to less frequent eval to reduce memory pressure from dual JIT
        help="How often to run evaluation (in steps). Higher values reduce JIT compilation memory overhead.",
    )
    parser.add_argument(
        "--per_device_eval_parallelism",
        type=int,
        default=-1,  # Same as training to potentially reuse XLA compilation cache
        help="Number of examples to process per device during evaluation. "
        "Default: -1 (same as training batch size).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Disable evaluation completely to save memory",
    )

    return parser.parse_args()


def get_model_config(args) -> LlavaOnevisionConfig:
    """Get model configuration based on arguments with performance optimizations."""

    # Determine gradient checkpointing setting
    use_gradient_checkpointing = not args.no_gradient_checkpointing

    # Determine attention backend (flash attention enabled by default)
    use_flash = not args.no_flash_attention
    if use_flash:
        attn_backend = AttentionBackend.DEFAULT
        flash_block_size = args.flash_attention_block_size
    else:
        attn_backend = AttentionBackend.VANILLA
        flash_block_size = None

    if args.use_small_model:
        # Small model config for testing
        logger.info("Using small model config for testing")
        vision_config = SiglipVisionConfig(
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=384,
            gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention=use_flash,
            attn_backend=attn_backend,
            flash_attention_block_size=flash_block_size,
        )
        text_config = QwenConfig(
            hidden_dim=128,
            intermediate_dim=512,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            gradient_checkpointing=use_gradient_checkpointing,
            attn_backend=attn_backend,
            flash_attention_block_size=flash_block_size,
        )
    else:
        # Custom config: SigLIP2 (from google/siglip2-so400m-patch16-384) + Qwen3-1.7B
        # Vision: SigLIP2 so400m-patch16-384 config (using SigLIP architecture)
        # LLM: Qwen3-1.7B config (not Qwen2)
        logger.info("Using custom config: SigLIP2-so400m-patch16 + Qwen3-1.7B")

        # SigLIP2 so400m-patch16-384 config (from HuggingFace)
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=384,
            patch_size=16,
            gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention=use_flash,
            attn_backend=attn_backend,
            flash_attention_block_size=flash_block_size,
        )

        # Qwen3-1.7B config (from HuggingFace Qwen/Qwen3-1.7B)
        text_config = Qwen3Config(
            hidden_dim=2048,
            intermediate_dim=6144,
            num_layers=28,
            num_heads=16,
            num_kv_heads=8,
            max_seq_len=40960,
            gradient_checkpointing=use_gradient_checkpointing,
            attn_backend=attn_backend,
            flash_attention_block_size=flash_block_size,
            rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),
            use_bias=False,
            tie_word_embeddings=True,
        )

    config = LlavaOnevisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        gradient_checkpointing=use_gradient_checkpointing,
    )

    # Log optimization settings
    logger.info(f"  Gradient checkpointing: {use_gradient_checkpointing}")
    logger.info(f"  Flash attention: {use_flash}")
    if use_flash:
        logger.info(f"  Flash attention block size: {flash_block_size}")

    return config


def main():
    args = parse_args()

    # Set validation data to train data if not specified
    if args.val_data is None:
        args.val_data = args.train_data

    logger.info("=" * 60)
    logger.info("VLM Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Initialize from HF: {args.initialize_from_hf}")
    logger.info(f"Num train steps: {args.num_train_steps}")
    logger.info(f"Batch size: {args.train_batch_size}")

    # Log performance optimization settings
    logger.info("-" * 60)
    logger.info("Performance Optimizations:")
    logger.info(f"  Mixed precision: {args.mp or 'disabled (float32)'}")
    logger.info(f"  Flash attention: {not args.no_flash_attention}")
    logger.info(f"  Freeze vision encoder: {args.freeze_vision_encoder}")
    logger.info(f"  Per-device parallelism: {args.per_device_parallelism}")
    logger.info(f"  FSDP axis: {args.fsdp_axis}")
    logger.info(f"  Gradient checkpointing: {not args.no_gradient_checkpointing}")
    logger.info("-" * 60)

    # Create data config
    data_config = ImageMixtureDatasetConfig(
        cache_dir=args.cache_dir,
        configs={
            "train": ConversationDatasetSourceConfig(
                train_urls=[f"file://{args.train_data}"],
                validation_urls=[f"file://{args.val_data}"],
                cache_dir=f"{args.cache_dir}/train",
            ),
        },
        train_weights={"train": 1.0},
        processor=args.model_name,
        max_length=args.max_length,
        use_cache=not args.no_cache,  # Use streaming mode if --no_cache is set
    )

    if args.no_cache:
        logger.info("Using streaming mode (no caching) - images will be processed on-the-fly")

    # Log dataset file count
    logger.info("-" * 60)
    logger.info("Dataset Files:")
    for name, source_config in data_config.configs.items():
        train_urls = source_config.urls_for_split("train")
        val_urls = source_config.urls_for_split("validation")
        logger.info(f"  {name}: {len(train_urls)} train file(s), {len(val_urls)} validation file(s)")
    logger.info("-" * 60)

    # Calculate num_train_steps based on epoch if specified
    num_train_steps = args.num_train_steps
    if args.epoch > 0:
        # Build training datasets to get the actual dataset size
        logger.info("Building training datasets to calculate epoch-based steps...")
        train_datasets = data_config.training_sets()

        # Calculate total dataset size from all training datasets
        total_dataset_size = 0
        for name, ds in train_datasets.items():
            try:
                ds_len = asyncio.run(ds.async_len())
                total_dataset_size += ds_len
                logger.info(f"  Dataset '{name}': {ds_len:,} samples")
            except Exception as e:
                logger.warning(f"Could not get length of dataset '{name}': {e}")

        if total_dataset_size > 0:
            # Calculate steps needed for the specified number of epochs
            steps_per_epoch = total_dataset_size // args.train_batch_size
            epoch_based_steps = steps_per_epoch * args.epoch
            num_train_steps = epoch_based_steps
            logger.info(
                f"Epoch-based training: {args.epoch} epoch(s) = {num_train_steps:,} steps "
                f"({total_dataset_size:,} samples / {args.train_batch_size} batch_size * {args.epoch} epochs)"
            )
        else:
            logger.warning("Could not determine dataset size, using --num_train_steps instead")

    # Create model config with optimizations
    model_config = get_model_config(args)

    # Create optimizer config
    warmup_steps = int(num_train_steps * args.warmup_ratio)
    optimizer_config = AdamConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup=warmup_steps,
    )

    # Create tracker config
    if args.wandb_project:
        tracker_config = WandbConfig(
            project=args.wandb_project,
            name=args.wandb_run_name,
        )
    else:
        tracker_config = NoopConfig()

    # Create distributed config
    distributed_config = DistributedConfig(initialize_jax_distributed=not args.no_distributed)

    # Set checkpoint path
    checkpointer_path = args.checkpointer_path or f"{args.output_dir}/checkpoints"
    checkpointer_config = CheckpointerConfig(base_path=checkpointer_path)

    # Parse FSDP axis (can be comma-separated for multi-axis)
    fsdp_axis = args.fsdp_axis
    if "," in fsdp_axis:
        fsdp_axis = [ax.strip() for ax in fsdp_axis.split(",")]

    # Convert mixed precision string to jmp.Policy
    # jmp.get_policy accepts strings like "f32", "bf16", "bfloat16", or
    # "compute=bfloat16,params=float32,output=float32"
    if args.mp:
        mp_policy = jmp.get_policy(args.mp)
    else:
        mp_policy = jmp.get_policy("f32")  # Default to full precision

    # Create trainer config with performance optimizations
    trainer_config = train_vlm.TrainerConfig(
        num_train_steps=num_train_steps,
        train_batch_size=args.train_batch_size,
        per_device_parallelism=args.per_device_parallelism,
        per_device_eval_parallelism=args.per_device_eval_parallelism,  # Smaller eval batch to save memory
        max_eval_batches=args.max_eval_batches,
        steps_per_eval=args.steps_per_eval,
        tracker=tracker_config,
        checkpointer=checkpointer_config,
        distributed=distributed_config,
        ray=RayConfig(auto_start_cluster=False),
        # # FSDP configuration
        # fsdp_axis=fsdp_axis,
        # Mixed precision configuration
        mp=mp_policy,
    )

    # Create main training config
    # Note: When using custom config (SigLIP + Qwen3), we disable use_hf_model_config
    # and initialize_from_hf since we'll load weights separately
    use_custom_config = not args.use_small_model and not args.use_hf_model_config
    config = train_vlm.TrainVLMConfig(
        data=data_config,
        model=model_config,
        trainer=trainer_config,
        optimizer=optimizer_config,
        # Disable HF loading when using custom config - we'll load weights separately
        initialize_from_hf=(
            False
            if use_custom_config
            else (
                args.initialize_from_hf
                if args.initialize_from_hf
                else args.model_name if args.use_hf_model_config else False
            )
        ),
        use_hf_model_config=args.use_hf_model_config and not args.use_small_model,
        hf_save_path=args.hf_save_path,
        hf_save_steps=args.hf_save_steps,
        # Custom weight loading paths for hybrid model
        # Though it's SigLIP2, the architecture is the same as SigLIP, so we use the siglip config.
        vision_checkpoint="google/siglip2-so400m-patch16-384" if use_custom_config else None,
        llm_checkpoint="Qwen/Qwen3-1.7B" if use_custom_config else None,
        # Evaluation control
        no_eval=args.no_eval,
        # Epoch control
        epoch=args.epoch,
    )

    # Handle freezing if requested
    if args.freeze_vision_encoder:
        config = dataclasses.replace(config, freeze_vision_encoder=True)
    if args.freeze_llm:
        config = dataclasses.replace(config, freeze_llm=True)

    logger.info("=" * 60)
    logger.info("Starting VLM training...")
    logger.info(f"Checkpoints will be saved to: {checkpointer_path}")
    if args.hf_save_path:
        logger.info(f"HF checkpoints will be saved to: {args.hf_save_path}")
    if args.epoch > 0:
        logger.info(f"Training for {args.epoch} epoch(s) ({num_train_steps:,} steps)")
    else:
        logger.info(f"Training for {num_train_steps:,} steps (no epoch limit)")

    # Run training
    train_vlm.main(config)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
