# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Training script for Vision-Language Models (VLM) like LLaVA OneVision.

This module provides training functionality for multimodal models that combine
vision encoders with language models.
"""

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
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data.image import (
    ImageIODatasetConfig,
    ImageDataLoader,
    ImageMixtureDatasetConfig,
    ImageTextDataset,
)
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)

# Constants for VLM training configuration
DEFAULT_NUM_PATCHES = 3 * 3 + 1  # 3x3 grid + base image for default anyres_max_9 config
STREAMING_MAX_BUFFERED_BATCHES = 4  # Memory-efficient buffering for streaming mode
STREAMING_PREFETCH_SIZE = 2  # Minimal prefetch to avoid OOM in streaming mode


def _load_vision_weights(model, checkpoint_path, axis_mapping, mp):
    """Load vision encoder weights from a separate HuggingFace checkpoint.

    Args:
        model: The LlavaOnevisionModel to load weights into
        checkpoint_path: HuggingFace checkpoint path (e.g., 'google/siglip-so400m-patch14-384')
        axis_mapping: Axis mapping for sharding
        mp: Mixed precision policy

    Returns:
        Model with vision weights loaded
    """
    from transformers import SiglipConfig as HfSiglipConfig

    # Create converter to load state dict from HF checkpoint
    vision_config = model.config.vision_config
    converter = HFCheckpointConverter(
        vision_config.__class__,
        reference_checkpoint=checkpoint_path,
        trust_remote_code=True,
        tokenizer="gpt2",  # Dummy tokenizer for vision-only model
        HfConfigClass=HfSiglipConfig,
    )

    # Load state dict from HF checkpoint
    state_dict = converter.load_state_dict()

    # The HF SigLIP model has weights under "vision_model." prefix
    # Our SiglipVisionModel also uses "vision_model." prefix, so they should match
    # Use the existing vision_tower as template and load weights into it
    vision_tower = model.vision_tower
    vision_tower = from_torch_compatible_state_dict(vision_tower, state_dict, prefix=None)

    # Replace vision tower in the model
    model = dataclasses.replace(model, vision_tower=vision_tower)
    logger.info(f"Loaded vision weights from {checkpoint_path}")
    return model


def _load_llm_weights(model, checkpoint_path, axis_mapping, mp, Vocab):
    """Load language model weights from a separate HuggingFace checkpoint.

    Args:
        model: The LlavaOnevisionModel to load weights into
        checkpoint_path: HuggingFace checkpoint path (e.g., 'Qwen/Qwen3-1.7B')
        axis_mapping: Axis mapping for sharding
        mp: Mixed precision policy
        Vocab: Vocabulary axis

    Returns:
        Model with LLM weights loaded
    """
    from transformers import Qwen3Config as HfQwen3Config

    # Create converter to load state dict from HF checkpoint
    text_config = model.config.text_config
    converter = HFCheckpointConverter(
        text_config.__class__,
        reference_checkpoint=checkpoint_path,
        trust_remote_code=True,
        HfConfigClass=HfQwen3Config,
    )

    # Load state dict from HF checkpoint
    state_dict = converter.load_state_dict()

    # The HF Qwen3 model has weights under "model." prefix for the transformer
    # and "lm_head." for the output layer
    # Use the existing language_model as template and load weights into it
    language_model = model.language_model
    language_model = from_torch_compatible_state_dict(language_model, state_dict, prefix=None)

    # Replace language model in the model
    model = dataclasses.replace(model, language_model=language_model)
    logger.info(f"Loaded LLM weights from {checkpoint_path}")
    return model


def _compute_max_num_patches(config, first_ex=None):
    """Compute maximum number of grid patches for anyres image processing.

    This returns the max number of GRID patches (excluding the base patch).
    The total patches = max_num_patches + 1 (for base) is computed in _pad_pixel_values().

    Args:
        config: VLM training config with model.image_grid_pinpoints and vision_config
        first_ex: Optional first example from dataset for fallback

    Returns:
        Maximum number of grid patches (excluding base)
    """
    grid_pinpoints = config.model.image_grid_pinpoints
    patch_size = config.model.vision_config.image_size

    if grid_pinpoints:
        max_resolution = max(max(h, w) for h, w in grid_pinpoints)
        max_patches_per_dim = max_resolution // patch_size
        # Return grid patches only; +1 for base is added in _pad_pixel_values()
        return max_patches_per_dim * max_patches_per_dim
    elif first_ex is not None:
        # first_ex has total patches (including base), subtract 1 for grid patches only
        return first_ex["pixel_values"].shape[0] - 1
    else:
        return DEFAULT_NUM_PATCHES


def _get_vocab_size_from_hf_config(hf_config):
    """Extract vocab_size from HuggingFace config, handling nested text_config."""
    vocab_size = getattr(hf_config, "vocab_size", None)
    if vocab_size is None and hasattr(hf_config, "text_config"):
        vocab_size = hf_config.text_config.vocab_size
    return vocab_size


def _get_first_example(dataset):
    """Extract the first example from a dataset (cached or streaming).

    This is used to determine image axes (Channels, Height, Width) from actual data.
    For streaming datasets, this uses get_batch which processes data on-the-fly
    without affecting subsequent iteration.

    Args:
        dataset: An AsyncDataset or MixtureDataset

    Returns:
        The first example dict, or None if extraction failed
    """
    import asyncio

    try:
        # MixtureDataset case - get from first underlying dataset
        if hasattr(dataset, "datasets"):
            first_ds = next(iter(dataset.datasets.values()))
            return _get_first_example(first_ds)

        # ProcessedImageCache case - use cache directly
        if hasattr(dataset, "cache"):
            return dataset.cache.get_batch_sync([0])[0]

        # StreamingImageDataset or other AsyncDataset - use get_batch
        if hasattr(dataset, "get_batch"):
            # Run async get_batch synchronously
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(dataset.get_batch([0]))
                return result[0]
            finally:
                loop.close()

        return None
    except Exception as e:
        logger.warning(f"Failed to extract first example: {e}")
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

    # Use forward_with_activations for blockwise computation
    activations, lm_head = model.forward_with_activations(
        example.input_ids,
        pixel_values=example.pixel_values,
        grid_mask=grid_mask,
        unpad_indices=unpad_indices,
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
    vision_checkpoint: Optional[str] = None
    """HuggingFace checkpoint for vision encoder (e.g., 'google/siglip-so400m-patch14-384')"""
    llm_checkpoint: Optional[str] = None
    """HuggingFace checkpoint for language model (e.g., 'Qwen/Qwen3-1.7B')"""

    # Evaluation control
    no_eval: bool = False
    """If True, disable evaluation completely to save memory."""

    # Epoch control
    epoch: int = 0
    """Number of epochs to train. If 0, train indefinitely until num_train_steps is reached."""


def main(config: TrainVLMConfig):
    """Main training function for VLM."""
    tokenizer = config.data.the_tokenizer

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

        # Get shape info from first example (required for axes setup)
        first_ex = _get_first_example(train_dataset_mixture)
        if first_ex is None:
            raise RuntimeError(
                "Could not extract first example from dataset. "
                "This is required to determine image axes (Channels, Height, Width)."
            )

        # Define axes from config (works for both cached and streaming modes)
        Pos = hax.Axis("position", config.data.max_length)

        # Recompute max_num_patches with first_ex for fallback (if grid_pinpoints not configured)
        max_num_patches = _compute_max_num_patches(config, first_ex)

        # Total patches = max_num_patches (grid) + 1 (base)
        NumPatches = hax.Axis("num_patches", max_num_patches + 1)
        Channels = hax.Axis("channels", first_ex["pixel_values"].shape[1])
        Height = hax.Axis("height", first_ex["pixel_values"].shape[2])
        Width = hax.Axis("width", first_ex["pixel_values"].shape[3])

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
            elif config.vision_checkpoint or config.llm_checkpoint:
                # Custom weight loading for hybrid models (e.g., SigLIP + Qwen3)
                logger.info("Loading weights from separate checkpoints...")
                model = state.model

                if config.vision_checkpoint:
                    logger.info(f"Loading vision encoder from: {config.vision_checkpoint}")
                    model = _load_vision_weights(model, config.vision_checkpoint, parameter_axis_mapping, trainer.mp)

                if config.llm_checkpoint:
                    logger.info(f"Loading LLM from: {config.llm_checkpoint}")
                    model = _load_llm_weights(model, config.llm_checkpoint, parameter_axis_mapping, trainer.mp, Vocab)

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

        if config.hf_save_path is not None:
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
            # For streaming mode, use minimal prefetch to avoid OOM
            loader_kwargs.update(
                {
                    "batch_size": trainer.config.train_batch_size,
                    "max_buffered_batches": STREAMING_MAX_BUFFERED_BATCHES,
                    "prefetch_size": STREAMING_PREFETCH_SIZE,
                }
            )
            logger.info(
                f"Using streaming mode with ImageDataLoader (prefetch_size={STREAMING_PREFETCH_SIZE}, max_buffered={STREAMING_MAX_BUFFERED_BATCHES})"
            )
        else:
            loader_kwargs["batch_size"] = Batch

        train_loader = ImageDataLoader(train_dataset_mixture, **loader_kwargs).iter_from_step(state.step)

        # Run training
        trainer.train(state, train_loader)


if __name__ == "__main__":
    levanter.config.main(main)()
