# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Training script for Vision-Language Models (VLM) like LLaVA OneVision.

This module provides training functionality for multimodal models that combine
vision encoders with language models.
"""

import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, cast

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from haliax.state_dict import from_torch_compatible_state_dict

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data.image import (
    ImageDataLoader,
    ImageIODatasetConfig,
    ImageMixtureDatasetConfig,
    ImageTextDataset,
    StreamingImageDataset,
)
from levanter.data.mixture import MixtureDataset
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)

# Constants for VLM training configuration
DEFAULT_NUM_PATCHES = 3 * 3 + 1  # 3x3 grid + base image for default anyres_max_9 config


def _load_vision_weights(model, checkpoint_path, axis_mapping, mp, tokenizer):
    """Load vision encoder weights from a separate HuggingFace checkpoint.

    Args:
        model: The LlavaOnevisionModel to load weights into
        checkpoint_path: HuggingFace checkpoint path (e.g., 'google/siglip-so400m-patch14-384')
        axis_mapping: Axis mapping for sharding
        mp: Mixed precision policy
        tokenizer: Already-loaded tokenizer to pass to HFCheckpointConverter

    Returns:
        Model with vision weights loaded
    """
    from transformers import SiglipConfig as HfSiglipConfig

    # Create converter to load state dict from HF checkpoint
    # We pass the already-loaded tokenizer from main() to avoid network calls
    # (Ray cluster workers have separate filesystems, so _hf_load_with_rank_sync
    # doesn't work for cross-node caching)
    vision_config = model.config.vision_config
    converter = HFCheckpointConverter(
        vision_config.__class__,
        reference_checkpoint=checkpoint_path,
        trust_remote_code=True,
        tokenizer=tokenizer,
        HfConfigClass=HfSiglipConfig,
    )

    # Load state dict from HF checkpoint
    state_dict = converter.load_state_dict()

    # Detect checkpoint type by checking key prefixes:
    # - Standalone vision model (e.g., HuggingFace SigLIP): keys start with "vision_model."
    # - Full VLM checkpoint (our saved checkpoints): keys start with "vision_tower.vision_model."
    sample_key = next(iter(state_dict.keys()), "")
    if sample_key.startswith("vision_tower."):
        prefix = "vision_tower"
        logger.info(f"Detected full VLM checkpoint format, using prefix='vision_tower'")
    else:
        prefix = None
        logger.info(f"Detected standalone vision model checkpoint format, using prefix=None")

    vision_tower = model.vision_tower
    vision_tower = from_torch_compatible_state_dict(vision_tower, state_dict, prefix=prefix)

    # Replace vision tower in the model
    model = dataclasses.replace(model, vision_tower=vision_tower)
    logger.info(f"Loaded vision weights from {checkpoint_path}")
    return model


def _load_llm_weights(model, checkpoint_path, axis_mapping, mp, tokenizer):
    """Load language model weights from a separate HuggingFace checkpoint.

    Args:
        model: The LlavaOnevisionModel to load weights into
        checkpoint_path: HuggingFace checkpoint path (e.g., 'Qwen/Qwen3-1.7B')
        axis_mapping: Axis mapping for sharding
        mp: Mixed precision policy
        tokenizer: Already-loaded tokenizer to pass to HFCheckpointConverter
                   (avoids network calls on non-leader workers in distributed mode)

    Returns:
        Model with LLM weights loaded
    """
    from transformers import Qwen3Config as HfQwen3Config

    # Create converter to load state dict from HF checkpoint
    # We pass the already-loaded tokenizer from main() to avoid network calls
    # on non-leader workers, which would fail with local_files_only=True
    text_config = model.config.text_config
    converter = HFCheckpointConverter(
        text_config.__class__,
        reference_checkpoint=checkpoint_path,
        trust_remote_code=True,
        tokenizer=tokenizer,
        HfConfigClass=HfQwen3Config,
    )

    # Load state dict from HF checkpoint
    state_dict = converter.load_state_dict()

    # Detect checkpoint type by checking key prefixes:
    # - Standalone LLM (e.g., HuggingFace Qwen3): keys start with "model." or "lm_head."
    # - Full VLM checkpoint (our saved checkpoints): keys start with "language_model."
    sample_key = next(iter(state_dict.keys()), "")
    if sample_key.startswith("language_model."):
        prefix = "language_model"
        logger.info(f"Detected full VLM checkpoint format, using prefix='language_model'")
    else:
        prefix = None
        logger.info(f"Detected standalone LLM checkpoint format, using prefix=None")

    language_model = model.language_model
    language_model = from_torch_compatible_state_dict(language_model, state_dict, prefix=prefix)

    # Replace language model in the model
    model = dataclasses.replace(model, language_model=language_model)
    logger.info(f"Loaded LLM weights from {checkpoint_path}")
    return model


def _compute_max_num_patches(config, first_ex=None):
    """Compute maximum number of grid patches for anyres image processing.

    This returns the max number of GRID patches (excluding the base patch).
    The total patches = max_num_patches + 1 (for base) is computed in _pad_pixel_values().

    For packed datasets (e.g., PackedVLMDataset), first_ex["pixel_values"] has shape
    (max_patches, C, H, W) where max_patches is the packing config's max_patches.
    In this case, we use the actual shape from first_ex to determine NumPatches,
    which may be larger than 1 even when disable_anyres is True.

    Args:
        config: VLM training config with model.image_grid_pinpoints and vision_config
        first_ex: Optional first example from dataset for fallback

    Returns:
        Maximum number of grid patches (excluding base)
    """
    # If we have a first example, prefer its actual shape
    # This is critical for packed mode where max_patches may be > 1 even with disable_anyres
    if first_ex is not None:
        pv = first_ex["pixel_values"]
        if pv.ndim == 4:  # (num_patches, C, H, W)
            # first_ex has total patches, subtract 1 for grid patches only
            # (but for packed mode, this is the packing's max_patches, so return total - 1)
            return pv.shape[0] - 1

    # Handle disable_anyres case: vision_aspect_ratio="single" means no grid patches
    vision_aspect_ratio = getattr(config.model, "vision_aspect_ratio", "anyres_max_9")
    if vision_aspect_ratio == "single" or (
        isinstance(vision_aspect_ratio, str) and not vision_aspect_ratio.startswith("anyres")
    ):
        return 0  # No grid patches, only base patch

    grid_pinpoints = config.model.image_grid_pinpoints
    patch_size = config.model.vision_config.image_size

    # Handle empty grid_pinpoints case (shouldn't happen with proper config)
    if grid_pinpoints is not None and len(grid_pinpoints) == 0:
        return 0  # No grid patches, only base patch

    if grid_pinpoints:
        max_resolution = max(max(h, w) for h, w in grid_pinpoints)
        max_patches_per_dim = max_resolution // patch_size
        # Return grid patches only; +1 for base is added in _pad_pixel_values()
        return max_patches_per_dim * max_patches_per_dim
    else:
        return DEFAULT_NUM_PATCHES


def _get_vocab_size_from_hf_config(hf_config):
    """Extract vocab_size from HuggingFace config, handling nested text_config."""
    vocab_size = getattr(hf_config, "vocab_size", None)
    if vocab_size is None and hasattr(hf_config, "text_config"):
        vocab_size = hf_config.text_config.vocab_size
    return vocab_size


def _get_image_shape_from_config(config) -> tuple:
    """Compute image shape (Channels, Height, Width) from model config.

    This provides a fallback when first example extraction fails (e.g., due to
    "Already borrowed" errors from HuggingFace tokenizers in distributed settings).

    For VLM models, pixel_values shape is (num_patches, C, H, W) where:
    - C = 3 (RGB channels)
    - H = W = image_size from vision config (typically 384)

    Args:
        config: VLM training config with model.vision_config

    Returns:
        Tuple of (channels, height, width)
    """
    channels = 3  # RGB
    # Get image_size from vision config (this determines H and W of each patch)
    image_size = getattr(config.model.vision_config, "image_size", 384)
    return (channels, image_size, image_size)


def _get_first_example(dataset, max_retries: int = 3):
    """Extract the first example from a dataset (cached or streaming).

    This is used to determine image axes (Channels, Height, Width) from actual data.
    For streaming datasets, this uses get_batch which processes data on-the-fly
    without affecting subsequent iteration.

    Includes retry logic for "Already borrowed" errors from HuggingFace tokenizers,
    which can occur in distributed settings when the tokenizer's Rust backend
    state is not properly reset after unpickling.

    Args:
        dataset: An AsyncDataset or MixtureDataset
        max_retries: Maximum number of retries for "Already borrowed" errors

    Returns:
        The first example dict, or None if extraction failed
    """
    import asyncio

    def _force_recreate_processor(ds):
        """Force recreation of BatchImageProcessor's internal processor."""
        if hasattr(ds, "_batch_processor") and ds._batch_processor is not None:
            bp = ds._batch_processor
            if hasattr(bp, "_processor"):
                bp._processor = None
            if hasattr(bp, "_cached_token_ids"):
                bp._cached_token_ids = None
        if hasattr(ds, "datasets"):
            for underlying_ds in ds.datasets.values():
                _force_recreate_processor(underlying_ds)
        if hasattr(ds, "dataset"):
            _force_recreate_processor(ds.dataset)

    for attempt in range(max_retries):
        try:
            # MixtureDataset case - get from first underlying dataset
            if hasattr(dataset, "datasets"):
                first_ds = next(iter(dataset.datasets.values()))
                return _get_first_example(first_ds, max_retries=1)

            # EpochDataset case - unwrap and get from underlying dataset
            if hasattr(dataset, "dataset") and hasattr(dataset, "max_epochs"):
                return _get_first_example(dataset.dataset, max_retries=1)

            # ProcessedImageCache case - use cache directly
            if hasattr(dataset, "cache"):
                return dataset.cache.get_batch_sync([0])[0]

            # StreamingImageDataset or other AsyncDataset - use get_batch
            if hasattr(dataset, "get_batch"):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(dataset.get_batch([0]))
                    return result[0]
                finally:
                    loop.close()

            return None
        except Exception as e:
            error_msg = str(e)
            if "Already borrowed" in error_msg and attempt < max_retries - 1:
                logger.warning(
                    f"Got 'Already borrowed' error on attempt {attempt + 1}/{max_retries}, "
                    f"forcing processor recreation and retrying..."
                )
                _force_recreate_processor(dataset)
                continue
            logger.warning(f"Failed to extract first example: {e}")
            return None

    return None


def _determine_vocab_size(config, converter, tokenizer):
    """Determine the vocab size to use for model initialization.

    Prioritizes HF checkpoint vocab size over tokenizer vocab size when loading
    from checkpoints, as HF models may pad vocab for efficiency.

    Returns:
        tuple: (vocab_size, source_description) for logging
    """
    tokenizer_vocab_size = len(tokenizer)

    if config.initialize_from_hf and converter is not None:
        hf_vocab_size = _get_vocab_size_from_hf_config(converter.default_hf_config)
        if hf_vocab_size is not None and hf_vocab_size > tokenizer_vocab_size:
            return hf_vocab_size, f"HF checkpoint vocab size {hf_vocab_size} (tokenizer has {tokenizer_vocab_size})"

    # Check vlm_checkpoint first (complete VLM checkpoint)
    elif config.vlm_checkpoint:
        # Handle both local paths and GCS paths (gs://...)
        if config.vlm_checkpoint.startswith("gs://"):
            # Use fsspec to read config.json from GCS
            import json
            import fsspec
            fs, _ = fsspec.core.url_to_fs(config.vlm_checkpoint)
            config_path = os.path.join(config.vlm_checkpoint, "config.json")
            with fs.open(config_path, "r") as f:
                config_dict = json.load(f)
            # Get vocab_size from text_config (for VLM models like LLaVA)
            if "text_config" in config_dict and "vocab_size" in config_dict["text_config"]:
                hf_vocab_size = config_dict["text_config"]["vocab_size"]
            elif "vocab_size" in config_dict:
                hf_vocab_size = config_dict["vocab_size"]
            else:
                hf_vocab_size = None
        else:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(config.vlm_checkpoint, trust_remote_code=True)
            hf_vocab_size = _get_vocab_size_from_hf_config(hf_config)

        if hf_vocab_size is not None:
            return hf_vocab_size, f"VLM checkpoint vocab size {hf_vocab_size} (tokenizer has {tokenizer_vocab_size})"

    elif config.llm_checkpoint:
        from transformers import AutoConfig

        llm_hf_config = AutoConfig.from_pretrained(config.llm_checkpoint, trust_remote_code=True)
        hf_vocab_size = _get_vocab_size_from_hf_config(llm_hf_config)
        if hf_vocab_size is not None and hf_vocab_size > tokenizer_vocab_size:
            return hf_vocab_size, f"LLM checkpoint vocab size {hf_vocab_size} (tokenizer has {tokenizer_vocab_size})"

    return tokenizer_vocab_size, None


def compute_vlm_loss(
    model: LlavaOnevisionModel,
    example,
    *,
    key=None,
    reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
    reduction_axis: Optional[hax.AxisSelection] = None,
    block_size: Optional[int] = 4096,
) -> jax.numpy.ndarray | hax.NamedArray:
    """Compute the loss for a VLM example using blockwise cross-entropy.

    This computes masked cross-entropy loss consistent with HuggingFace's implementation:
    loss = -sum(log_probs * mask) / sum(mask)

    Uses blockwise cross-entropy to avoid materializing the full logits tensor
    (batch * seq * vocab), which can cause OOM for large vocab sizes.

    Only tokens where loss_mask > 0 contribute to the loss. This is important for
    VLM training where we typically mask out image tokens and user prompts.

    Args:
        model: The LlavaOnevisionModel to compute loss for.
        example: A batch containing input_ids, pixel_values, and optionally loss_mask.
        key: Random key for any stochastic operations.
        reduction: Reduction function to apply to the loss (default: hax.mean).
            Note: When loss_mask is present, we use HF-compatible masked mean
            (sum of masked losses / count of valid tokens) instead of simple mean.
        reduction_axis: Axis to reduce over.
        block_size: Block size for blockwise cross-entropy computation.
            Set to None to use full logits (may cause OOM for large vocab).

    Returns:
        The computed loss value.
    """
    from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

    # Forward pass through the model
    # Get grid_mask and unpad_indices from example (for fixed-shape processing)
    grid_mask = getattr(example, "grid_mask", None)
    unpad_indices = getattr(example, "unpad_indices", None)

    # Get precomputed combined_mask and position_ids (for CPU precomputation optimization)
    combined_mask = getattr(example, "combined_mask", None)
    position_ids = getattr(example, "position_ids", None)

    # Use forward_with_activations for blockwise computation
    activations, lm_head = model.forward_with_activations(
        example.input_ids,
        pixel_values=example.pixel_values,
        grid_mask=grid_mask,
        unpad_indices=unpad_indices,
        combined_mask=combined_mask,
        position_ids=position_ids,
        key=key,
    )

    # Get axes for cross-entropy computation
    Pos = example.input_ids.resolve_axis("position")
    Embed = model.config.TextEmbed
    Vocab = model.Vocab

    # Get targets (shifted by 1 for next-token prediction)
    targets = hax.roll(example.input_ids, -1, Pos)

    # Compute loss weight from loss_mask
    if example.loss_mask is not None:
        # Shift loss mask to align with targets
        loss_weight = hax.roll(example.loss_mask, -1, Pos)
        # Zero out last position (roll wraps first element to last, but there's no valid target there)
        not_last_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
        loss_weight = loss_weight * not_last_mask.astype(loss_weight.dtype)
    else:
        # Create a mask that excludes the last token
        not_last_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))
        loss_weight = not_last_mask.astype(jnp.float32)

    # Use fused_cross_entropy_loss for blockwise computation
    # This avoids materializing the full (batch, seq, vocab) logits tensor
    per_token_loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings=activations,
        pred_lm_head=lm_head,
        Contract=Embed,
        Label=Vocab,
        target_y=targets,
        reduction=None,  # We'll handle reduction ourselves for masked loss
        weight=None,  # We'll apply mask after
        logsumexp_weight=0.0,
        block_size=block_size,
    )

    # Apply loss mask if available (HuggingFace-consistent masked mean)
    if example.loss_mask is not None:
        masked_loss = per_token_loss * loss_weight

        # Compute token-weighted loss across entire batch (more stable than per-example mean)
        # This avoids division by zero for samples with no valid tokens
        # loss = sum(masked_loss) / sum(mask) across all tokens in batch
        total_masked_loss = hax.sum(masked_loss, axis=None)  # Sum all axes
        total_mask = hax.sum(loss_weight, axis=None)  # Sum all axes

        # Add small epsilon to avoid division by zero
        loss = total_masked_loss / (total_mask + 1e-8)
    else:
        # No mask - use standard reduction
        if reduction is not None:
            loss = reduction(per_token_loss, axis=reduction_axis)
        else:
            loss = per_token_loss

    return loss


@dataclass
class TrainVLMConfig:
    """Configuration for training Vision-Language Models."""

    data: Union[ImageIODatasetConfig, ImageMixtureDatasetConfig] = field(default_factory=ImageMixtureDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LlavaOnevisionConfig = field(default_factory=LlavaOnevisionConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint
    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    # Performance optimization options
    freeze_vision_encoder: bool = False
    """If True, freeze vision encoder weights during training (only train projector + LLM)."""
    freeze_llm: bool = False
    """If True, freeze LLM weights during training (only train projector + vision encoder)."""

    # Custom weight loading for hybrid models (e.g., SigLIP + Qwen3)
    vlm_checkpoint: Optional[str] = None
    """Complete VLM HuggingFace checkpoint path (loads vision encoder + projector + LLM).
    Use this for loading from a previously trained VLM, e.g., for stage 2 training."""
    vision_checkpoint: Optional[str] = None
    """HuggingFace checkpoint for vision encoder only (e.g., 'google/siglip-so400m-patch14-384').
    Use this with llm_checkpoint for loading separate vision and LLM weights."""
    llm_checkpoint: Optional[str] = None
    """HuggingFace checkpoint for language model (e.g., 'Qwen/Qwen3-1.7B')"""

    # Evaluation control
    no_eval: bool = False
    """If True, disable evaluation completely to save memory."""

    # Epoch control
    epoch: int = 0
    """Number of epochs to train. If 0, train indefinitely until num_train_steps is reached."""

    # Levanter checkpoint initialization
    initialize_from_checkpoint_path: Optional[str] = None
    """If provided, will initialize model weights from this Levanter checkpoint path, resetting step to 0.
    Use this for loading from a previous training run's Levanter checkpoint while starting a new training stage."""

    # Streaming mode performance tuning
    streaming_max_buffered_batches: int = 4
    """Maximum buffered batches in streaming mode (default: 4). Increase for better throughput."""
    streaming_prefetch_size: int = 2
    """Prefetch size in streaming mode (default: 2). Increase for better throughput."""


def main(config: TrainVLMConfig):
    """Main training function for VLM."""
    # Pass image_grid_pinpoints from model config to data config
    # This must happen BEFORE the_processor is accessed (which is cached)
    if config.model.image_grid_pinpoints is not None:
        config.data.image_grid_pinpoints = config.model.image_grid_pinpoints

    tokenizer = config.data.the_tokenizer

    # Calculate num_train_steps based on epoch if specified
    if config.epoch > 0:
        logger.info("Building training datasets to calculate epoch-based steps...")
        # Build training datasets to get the actual dataset size
        train_datasets = config.data.training_sets()

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
            train_batch_size = config.trainer.train_batch_size
            steps_per_epoch = total_dataset_size // train_batch_size
            epoch_based_steps = steps_per_epoch * config.epoch
            logger.info(
                f"Epoch-based training: {config.epoch} epoch(s) = {epoch_based_steps:,} steps "
                f"({total_dataset_size:,} samples / {train_batch_size} batch_size * {config.epoch} epochs)"
            )
            # Update trainer config with calculated num_train_steps
            config = dataclasses.replace(
                config,
                trainer=dataclasses.replace(config.trainer, num_train_steps=epoch_based_steps),
            )
        else:
            logger.warning("Could not determine dataset size, using num_train_steps from config instead")

    # Handle HuggingFace checkpoint initialization
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        if isinstance(config.initialize_from_hf, str):
            converter = config.model.hf_checkpoint_converter(ref_checkpoint=config.initialize_from_hf)
            converter = converter.replaced(tokenizer=tokenizer)
        else:
            converter = config.model.hf_checkpoint_converter(
                ref_checkpoint=config.data.processor  # Use processor path as reference
            )
            converter = converter.replaced(tokenizer=tokenizer)

        if hasattr(tokenizer, "vocab") and converter.tokenizer is not None:
            if tokenizer.vocab != converter.tokenizer.vocab:
                logger.warning("The tokenizers appear to be different. You may want to check this.")

        if config.use_hf_model_config:
            config.model = LlavaOnevisionConfig.from_hf_config(converter.default_hf_config)
            logger.info(
                f"Using HF model config: vision_layers={config.model.vision_config.num_hidden_layers}, "
                f"text_layers={config.model.text_config.num_layers}, "
                f"hidden_dim={config.model.text_config.hidden_dim}"
            )
    else:
        # Use processor path as reference checkpoint to get tokenizer
        converter = config.model.hf_checkpoint_converter(ref_checkpoint=config.data.processor)
        converter = converter.replaced(tokenizer=tokenizer)

    # === VALIDATION: Check vision_feature_height consistency ===
    # Ensure data config's vision_feature_height matches model's expected output
    model_vision_feature_height = config.model.vision_feature_height
    data_vision_feature_height = getattr(config.data, "vision_feature_height", None)
    if data_vision_feature_height is not None and data_vision_feature_height != model_vision_feature_height:
        raise ValueError(
            f"vision_feature_height mismatch between data config and model config!\n"
            f"  Data config: vision_feature_height={data_vision_feature_height} "
            f"(features_per_patch={data_vision_feature_height**2})\n"
            f"  Model config: vision_feature_height={model_vision_feature_height} "
            f"(features_per_patch={model_vision_feature_height**2})\n"
            f"  Model vision encoder: image_size={config.model.vision_config.image_size}, "
            f"patch_size={config.model.vision_config.patch_size}\n"
            f"Please update data config's vision_feature_height to {model_vision_feature_height} "
            f"to match the model's vision encoder output."
        )
    elif data_vision_feature_height is not None:
        logger.info(
            f"vision_feature_height validated: {data_vision_feature_height} "
            f"(matches model's {config.model.vision_config.image_size}//{config.model.vision_config.patch_size})"
        )

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Create loss function with optional freezing
    if config.freeze_vision_encoder or config.freeze_llm:
        # Wrap loss function to apply stop_gradient to frozen components
        def compute_vlm_loss_with_freezing(model, example, **kwargs):
            # Collect frozen components to replace in a single dataclasses.replace call
            frozen_updates = {}
            if config.freeze_vision_encoder:
                frozen_updates["vision_tower"] = jax.lax.stop_gradient(model.vision_tower)
            if config.freeze_llm:
                frozen_updates["language_model"] = jax.lax.stop_gradient(model.language_model)

            if frozen_updates:
                model = dataclasses.replace(model, **frozen_updates)

            return compute_vlm_loss(model, example, **kwargs)

        loss_fn = compute_vlm_loss_with_freezing
    else:
        loss_fn = compute_vlm_loss

    # Using the trainer as a context manager
    with Trainer(config.trainer, optimizer, loss_fn) as trainer:
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        parameter_axis_mapping = trainer.parameter_axis_mapping

        # Get batch axes
        Batch = config.trainer.TrainBatch

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # Compute max_num_patches early from model config (needed for streaming dataset creation)
        # This ensures BatchImageProcessor pads to the correct size before the loader sees the data
        max_num_patches = _compute_max_num_patches(config, first_ex=None)

        # Build datasets - only build eval if no_eval not set
        # Pass max_num_patches for streaming mode to ensure correct padding
        if config.no_eval:
            eval_datasets = {}
        else:
            eval_datasets = config.data.validation_sets(max_num_patches=max_num_patches)
        train_dataset_mixture = config.data.train_set(
            key=data_key, epochs=config.epoch, max_num_patches=max_num_patches
        )

        # Define axes from config (deterministic across all workers)
        Pos = hax.Axis("position", config.data.max_length)

        # Get max_num_patches from packing config if available (deterministic)
        # This avoids non-determinism from _get_first_example() and respects packed data
        if hasattr(train_dataset_mixture, '_config') and train_dataset_mixture._config is not None:
            # PackedVLMDataset: use packing config's max_patches
            max_num_patches = train_dataset_mixture._config.max_patches - 1  # -1 for base patch
            logger.info(f"Using max_num_patches={max_num_patches} from packing config")
        elif hasattr(train_dataset_mixture, 'datasets'):
            # MixtureDataset: check if any underlying dataset is packed
            for ds in train_dataset_mixture.datasets.values():
                if hasattr(ds, '_config') and ds._config is not None:
                    max_num_patches = ds._config.max_patches - 1
                    logger.info(f"Using max_num_patches={max_num_patches} from packing config (via mixture)")
                    break
            else:
                max_num_patches = _compute_max_num_patches(config, first_ex=None)
                logger.info(f"Using max_num_patches={max_num_patches} from model config")
        else:
            # Non-packed mode: use config-based calculation
            max_num_patches = _compute_max_num_patches(config, first_ex=None)
            logger.info(f"Using max_num_patches={max_num_patches} from model config")

        # Image shape always from config (deterministic)
        channels, height, width = _get_image_shape_from_config(config)
        logger.info(f"Using config-based image shape: C={channels}, H={height}, W={width}")

        # Total patches = max_num_patches (grid) + 1 (base)
        NumPatches = hax.Axis("num_patches", max_num_patches + 1)
        Channels = hax.Axis("channels", channels)
        Height = hax.Axis("height", height)
        Width = hax.Axis("width", width)

        # Determine pixel dtype based on trainer's compute precision
        # This ensures data is transferred to TPU in the correct dtype to save memory
        compute_dtype = trainer.mp.compute_dtype
        logger.info(f"Using compute dtype {compute_dtype} for pixel values")

        # Note: We use train_dataset_mixture (raw ImageTextDict) directly with ImageDataLoader
        # instead of wrapping it in ImageTextDataset. The ImageDataLoader handles
        # the conversion to ImageTextExample during batching.

        # Determine vocab size - use HF checkpoint vocab if loading from HF, otherwise use tokenizer
        vocab_size, vocab_source = _determine_vocab_size(config, converter, tokenizer)
        if vocab_source:
            logger.info(f"Using {vocab_source}")

        # Round vocab size for partitioning
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Initialize model
        def model_init():
            return LlavaOnevisionModel.init(Vocab, config.model, key=model_key)

        # For freezing, we use is_trainable=True and handle gradient zeroing separately
        # This avoids haliax partitioning issues with non-trivial is_trainable filters
        state = trainer.initial_state(training_key, model_init=model_init, is_trainable=True)

        # Log freezing info if requested
        if config.freeze_vision_encoder or config.freeze_llm:
            frozen_parts = []
            if config.freeze_vision_encoder:
                frozen_parts.append("vision encoder")
            if config.freeze_llm:
                frozen_parts.append("LLM")
            logger.info(f"Freezing {' and '.join(frozen_parts)} - only projector will be trained")
            logger.info("Note: Freezing is implemented via gradient zeroing during training.")

        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            logger.info(f"Initializing from Levanter checkpoint: {config.initialize_from_checkpoint_path}")
            state = load_checkpoint(state, config.initialize_from_checkpoint_path)
            # Reset step to 0 - we're just initializing weights for a new training stage
            state = dataclasses.replace(state, step=jnp.array(0))
            logger.info("Loaded Levanter checkpoint and reset step to 0")

        if int(state.step) == 0:
            if config.initialize_from_hf:
                assert converter is not None
                logger.info(
                    f"No training checkpoint found. Initializing model from HF checkpoint "
                    f"'{converter.reference_checkpoint}'"
                )
                state = dataclasses.replace(state, model=None)
                # Load with resize_vocab_to_match_tokenizer=False since we already use HF vocab size
                model = converter.load_pretrained(
                    LlavaOnevisionModel,
                    axis_mapping=parameter_axis_mapping,
                    resize_vocab_to_match_tokenizer=False,  # Keep HF vocab size (already set in model_init)
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            elif config.vlm_checkpoint:
                # Load complete VLM from a single checkpoint
                logger.info(f"Loading complete VLM from: {config.vlm_checkpoint}")

                from levanter.compat.hf_checkpoints import HFCheckpointConverter
                from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig

                # For GCS paths, we need to explicitly pass HfConfigClass since
                # AutoConfig.from_pretrained() doesn't support gs:// paths
                hf_config_class = HfLlavaOnevisionConfig if config.vlm_checkpoint.startswith("gs://") else None

                hf_converter = HFCheckpointConverter(
                    LlavaOnevisionConfig,
                    reference_checkpoint=config.vlm_checkpoint,
                    trust_remote_code=True,
                    tokenizer=tokenizer,
                    HfConfigClass=hf_config_class,
                )

                model = hf_converter.load_pretrained(
                    LlavaOnevisionModel,
                    ref=config.vlm_checkpoint,
                    config=config.model,
                    axis_mapping=parameter_axis_mapping,
                    dtype=jnp.bfloat16,
                    resize_vocab_to_match_tokenizer=False,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
                logger.info("Loaded complete VLM weights (vision encoder + projector + LLM)")

            elif config.vision_checkpoint or config.llm_checkpoint:
                # Load separate vision and LLM checkpoints
                logger.info("Loading weights from separate checkpoints...")
                model = state.model

                if config.vision_checkpoint:
                    logger.info(f"Loading vision encoder from: {config.vision_checkpoint}")
                    model = _load_vision_weights(
                        model, config.vision_checkpoint, parameter_axis_mapping, trainer.mp, tokenizer
                    )

                if config.llm_checkpoint:
                    logger.info(f"Loading LLM from: {config.llm_checkpoint}")
                    model = _load_llm_weights(
                        model, config.llm_checkpoint, parameter_axis_mapping, trainer.mp, tokenizer
                    )

                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
                logger.info("Custom weight loading completed.")
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # Add eval hooks unless no_eval is set
        if config.no_eval:
            logger.info("Evaluation disabled (--no_eval). Skipping eval hooks to save memory.")
        elif len(eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            for name, eval_dataset in eval_datasets.items():
                hax_eval_dataset = ImageTextDataset(
                    eval_dataset,
                    Position=Pos,
                    NumPatches=NumPatches,
                    Channels=Channels,
                    Height=Height,
                    Width=Width,
                    ignore_index=config.data.pad_token_id,
                    pixel_dtype=compute_dtype,  # Use same compute precision for eval
                    grid_pinpoints=config.model.image_grid_pinpoints,
                    patch_size=config.model.vision_config.image_size,
                )
                trainer.add_eval_hook(hax_eval_dataset, name=name)

        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)

        if config.hf_save_path is not None and config.hf_save_steps is not None:
            assert converter is not None, "converter must be set when saving HF checkpoints"
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload or False),
                every=config.hf_save_steps,
            )

        # Create data loader - ImageDataLoader converts raw ImageTextDict to ImageTextExample
        # during batching, handling grid_mask computation and NamedArray creation
        pixel_dtype = np.dtype(compute_dtype)

        # Check if streaming mode for loader configuration
        is_streaming = hasattr(config.data, "use_cache") and not config.data.use_cache

        # Build loader kwargs with common parameters
        loader_kwargs = {
            "Pos": Pos,
            "NumPatches": NumPatches,
            "Channels": Channels,
            "Height": Height,
            "Width": Width,
            "mesh": trainer.device_mesh,
            "axis_resources": trainer.compute_axis_mapping,
            "batch_axis_name": Batch.name,
            "allow_nondivisible_batch_size": trainer.config.allow_nondivisible_batch_size,
            "pixel_dtype": pixel_dtype,
        }

        if is_streaming:
            # For streaming mode, use configurable prefetch settings
            loader_kwargs.update(
                {
                    "batch_size": trainer.config.train_batch_size,
                    "max_buffered_batches": config.streaming_max_buffered_batches,
                    "prefetch_size": config.streaming_prefetch_size,
                }
            )
            logger.info(
                f"Using streaming mode with ImageDataLoader (prefetch_size={config.streaming_prefetch_size}, max_buffered={config.streaming_max_buffered_batches})"
            )
        else:
            loader_kwargs["batch_size"] = Batch

        # Set resume positions on StreamingImageDatasets when resuming from checkpoint
        if state.step > 0 and isinstance(train_dataset_mixture, MixtureDataset):
            total_items = int(state.step) * trainer.config.train_batch_size
            per_ds_items = train_dataset_mixture.cumulative_items_per_dataset(total_items)
            num_processes = jax.process_count()

            for name, ds in train_dataset_mixture.datasets.items():
                actual_ds = ds
                # Unwrap EpochDataset if present
                if hasattr(ds, 'dataset'):
                    actual_ds = ds.dataset
                if isinstance(actual_ds, StreamingImageDataset):
                    items_per_process = per_ds_items.get(name, 0) // num_processes
                    actual_ds.set_resume_item_count(items_per_process)
                    logger.info(
                        f"Resume: {name} -> seek to item {items_per_process}/process "
                        f"(total={per_ds_items[name]}, step={state.step}, "
                        f"batch_size={trainer.config.train_batch_size})"
                    )

        train_loader = ImageDataLoader(train_dataset_mixture, **loader_kwargs).iter_from_step(state.step)

        # Run training
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
