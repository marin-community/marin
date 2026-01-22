# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLM Image Caption Demo

Generate image descriptions using a trained Vision-Language Model.

Architecture:
- Vision Encoder: SigLIP (384x384, patch16)
- Language Model: Qwen3-1.7B
- Projector: 2-layer MLP

Usage:
    # Basic usage with default checkpoint
    python experiments/VLM/demo_vlm_caption.py --image_path path/to/image.jpg

    # With custom prompt
    python experiments/VLM/demo_vlm_caption.py \
        --image_path path/to/image.jpg \
        --prompt "What objects are in this image?"

    # With different checkpoint
    python experiments/VLM/demo_vlm_caption.py \
        --checkpoint gs://your-bucket/checkpoint-path \
        --image_path path/to/image.jpg
"""

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Tuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from PIL import Image
from transformers import AutoTokenizer

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.data.image import BatchImageProcessor, create_custom_processor, CustomVLMProcessor
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llava_onevision import (
    LlavaInferenceEngine,
    LlavaOnevisionConfig,
    LlavaOnevisionModel,
    VLMRequest,
)
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)

# Enable JAX compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class VLMCaptionConfig:
    """Configuration for VLM Image Caption generation."""

    # Checkpoint path (supports GCS)
    checkpoint: str = "gs://marin-us-east1/checkpoints/vlm-official-qwen3-1.7b-8-c3e151/hf/vlm-official-qwen3-1.7b-8-c3e151/step-544"

    # Image input
    image_path: str = ""
    prompt: str = "Describe this image in detail."

    # Tokenizer and processor
    tokenizer: str = "Qwen/Qwen3-1.7B"
    processor: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

    # Generation parameters
    max_tokens: int = 256
    temperature: float = 0.0  # Use greedy decoding (like test)
    seed: int = 42

    # Trainer config for mesh/precision
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            mp=jmp.get_policy("p=f32,c=bfloat16"),
        )
    )


# ============================================================================
# MODEL CONFIGURATION (matches demo_vlm_train.py)
# ============================================================================


def build_vlm_config(tokenizer_name: str) -> LlavaOnevisionConfig:
    """Build LlavaOnevisionConfig matching demo_vlm_train.py."""

    # Vision encoder: SigLIP (384x384, patch16)
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=384,
        patch_size=16,
    )

    # Language model: Qwen3-1.7B
    text_config = Qwen3Config(
        max_seq_len=2048,
        hidden_dim=2048,
        intermediate_dim=6144,
        num_heads=16,
        num_kv_heads=8,
        num_layers=28,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
    )

    # Get image token ID from Qwen3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    image_token_index = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # Combined VLM config
    return LlavaOnevisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        vision_encoder_type="siglip",
        vision_feature_select_strategy="full",
        vision_aspect_ratio="single",  # Single resolution (no anyres)
        disable_anyres=True,
        image_token_index=image_token_index,
    )


# ============================================================================
# MODEL LOADING
# ============================================================================


def _is_hf_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is HuggingFace format (has config.json) vs Levanter format."""
    import fsspec
    fs, _ = fsspec.core.url_to_fs(checkpoint_path)
    # HF checkpoints have config.json, Levanter checkpoints have metadata.json
    config_json = os.path.join(checkpoint_path, "config.json")
    return fs.exists(config_json)


def load_model(
    config: VLMCaptionConfig,
    vlm_config: LlavaOnevisionConfig,
) -> Tuple[LlavaOnevisionModel, any, Axis]:
    """Load VLM model from checkpoint.

    Supports both Levanter and HuggingFace checkpoint formats.
    - Levanter format: Uses load_checkpoint() with metadata.json
    - HuggingFace format: Uses HFCheckpointConverter.load_pretrained() with config.json

    Args:
        config: Caption configuration
        vlm_config: VLM model configuration

    Returns:
        Tuple of (model, tokenizer, Vocab axis)
    """
    logger.info(f"Loading model from checkpoint: {config.checkpoint}")

    key = jrandom.PRNGKey(config.seed)

    # Check if this is a HuggingFace format checkpoint
    is_hf = _is_hf_checkpoint(config.checkpoint)

    import json
    import tempfile
    import fsspec
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig

    if is_hf:
        logger.info("Detected HuggingFace format checkpoint, using HFCheckpointConverter")

        # Download tokenizer files from GCS to temp directory
        fs, _ = fsspec.core.url_to_fs(config.checkpoint)
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
            "added_tokens.json", "chat_template.jinja"
        ]

        # Create a persistent temp directory for the tokenizer
        tokenizer_tmpdir = tempfile.mkdtemp(prefix="vlm_tokenizer_")
        logger.info(f"Downloading tokenizer to {tokenizer_tmpdir}")

        for fname in tokenizer_files:
            src = os.path.join(config.checkpoint, fname)
            dst = os.path.join(tokenizer_tmpdir, fname)
            if fs.exists(src):
                fs.get(src, dst)
                logger.info(f"  Downloaded {fname}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_tmpdir, trust_remote_code=True)
        logger.info(f"Loaded tokenizer with vocab_size={len(tokenizer)}")

        # Load HF config from GCS checkpoint
        config_path = os.path.join(config.checkpoint, "config.json")
        with fs.open(config_path, "r") as f:
            config_dict = json.load(f)
        hf_config = HfLlavaOnevisionConfig.from_dict(config_dict)

        # Use vocab_size from HF config (model's actual vocab size)
        # Note: Qwen3 model vocab_size (151936) is larger than tokenizer vocab_size (151669)
        # This is by design - the model's embedding table is larger to accommodate potential token additions
        vocab_size = hf_config.text_config.vocab_size
        logger.info(f"Using model vocab_size={vocab_size} (tokenizer has {len(tokenizer)} tokens)")

        # Convert HF config to Levanter config
        lev_config = LlavaOnevisionConfig.from_hf_config(hf_config)

        # Switch to dot-product attention to avoid VMEM OOM with ragged_paged_attention
        # The paged attention kernel requires >16MB VMEM which exceeds TPU limits
        text_config_updated = replace(
            lev_config.text_config,
            attn_backend="dot",
            flash_attention_block_size=None
        )
        lev_config = replace(lev_config, text_config=text_config_updated)
    else:
        # For Levanter checkpoints, use the configured tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        vocab_size = len(tokenizer)
        tokenizer_tmpdir = None

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size),
            config.trainer.compute_axis_mapping,
        )

        if is_hf:
            # Create converter - use the downloaded tokenizer
            converter = HFCheckpointConverter(
                LlavaOnevisionConfig,
                reference_checkpoint=config.checkpoint,
                trust_remote_code=True,
                tokenizer=tokenizer,  # Use downloaded tokenizer
                HfConfigClass=HfLlavaOnevisionConfig,
            )

            # Load pretrained model with the converted config
            # Use bfloat16 to match model weights
            model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=config.checkpoint,
                config=lev_config,  # Pass the converted config
                axis_mapping=config.trainer.parameter_axis_mapping,
                dtype=jnp.bfloat16,
                resize_vocab_to_match_tokenizer=False,
            )
        else:
            # Load Levanter checkpoint
            # Create model shape template (without allocating memory)
            with use_cpu_device():
                model = eqx.filter_eval_shape(vlm_config.build, Vocab, key=key)

            # Load checkpoint
            model = load_checkpoint(model, config.checkpoint, subpath="model")

            # Cast to compute precision
            model = config.trainer.mp.cast_to_compute(model)

    logger.info("Model loaded successfully")
    return model, tokenizer, Vocab


# ============================================================================
# IMAGE PROCESSING
# ============================================================================


def process_image(
    config: VLMCaptionConfig,
    vlm_config: LlavaOnevisionConfig,
    tokenizer,
) -> Tuple[VLMRequest, list]:
    """Process image and create VLMRequest.

    Uses Levanter's BatchImageProcessor for consistent processing with training.
    This ensures proper single-resolution mode (1 patch per image, no anyres).

    Args:
        config: Caption configuration
        vlm_config: VLM model configuration
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (VLMRequest, prompt_tokens list)
    """
    logger.info(f"Processing image: {config.image_path}")

    # Configuration matching training
    # SigLIP: image_size=384, patch_size=16 -> vision_feature_height = 384/16 = 24
    PATCH_SIZE = 384
    VISION_FEATURE_HEIGHT = 24  # 384 // 16 for SigLIP
    GRID_PINPOINTS = [[384, 384]]  # Single resolution only
    MAX_NUM_PATCHES = 0  # For disable_anyres with 1 image: max_images_per_sample - 1 = 0
    MAX_LENGTH = 2048

    # Create base processor using Levanter's create_custom_processor
    base_processor = create_custom_processor(
        config.processor,
        do_pad=True,
        image_grid_pinpoints=GRID_PINPOINTS,
        max_image_tiles=1,  # One tile per image (disable_anyres)
        vision_aspect_ratio="single",  # Disable anyres
    )

    # Wrap with custom tokenizer for consistent tokenization
    lev_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        base_processor, tokenizer, use_full_padded_tokens=False
    )

    # Create BatchImageProcessor for proper grid_mask handling
    batch_processor = BatchImageProcessor(
        processor=lev_processor,
        max_length=MAX_LENGTH,
        padding=True,
        max_num_patches=MAX_NUM_PATCHES,
        grid_pinpoints=GRID_PINPOINTS,
        patch_size=PATCH_SIZE,
        vision_feature_height=VISION_FEATURE_HEIGHT,
        add_generation_prompt=True,  # For inference
        disable_anyres=True,
    )

    # Load image
    image = Image.open(config.image_path).convert("RGB")

    # Build conversation for chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": config.prompt},
            ],
        }
    ]

    # Process using BatchImageProcessor
    # It expects a list of examples with "messages" and "images" keys
    example = {"messages": messages, "images": [image]}
    results = batch_processor([example])
    result = results[0]  # ImageTextDict

    # Extract outputs from ImageTextDict
    input_ids_full = result["input_ids"].astype(np.int32)
    pixel_values = result["pixel_values"]
    grid_mask = result["grid_mask"]
    attention_mask = result["attention_mask"]

    # Get actual content length (non-padded tokens)
    actual_content_len = int(attention_mask.sum())
    input_ids = input_ids_full[:actual_content_len]

    # Get unpad_indices and num_unpadded_features for spatial unpadding
    unpad_indices = result.get("unpad_indices")
    num_unpadded_features = result.get("num_unpadded_features")

    total_patches = pixel_values.shape[0]
    actual_patches = int(grid_mask.sum())

    logger.info(f"Input tokens: {len(input_ids)} (actual), Total patches: {total_patches}, Actual patches: {actual_patches}")

    # Create NamedArrays for VLMRequest
    # All tensors need batch dimension
    Batch = Axis("batch", 1)
    Position = Axis("position", len(input_ids))
    NumPatches = Axis("num_patches", total_patches)
    Channels = Axis("channels", pixel_values.shape[1])
    Height = Axis("height", pixel_values.shape[2])
    Width = Axis("width", pixel_values.shape[3])

    # Add batch dimension to input_ids: (batch, position)
    input_ids_array = jnp.array(input_ids, dtype=jnp.int32).reshape(1, -1)
    input_ids_named = hax.named(input_ids_array, (Batch, Position))

    # Add batch dimension to pixel_values: (batch, num_patches, C, H, W)
    # Use bfloat16 to match model weights
    pixel_values_batched = pixel_values[np.newaxis, ...]  # (1, num_patches, C, H, W)
    pixel_values_bf16 = jnp.array(pixel_values_batched, dtype=jnp.bfloat16)
    pixel_values_named = hax.named(
        pixel_values_bf16,
        (Batch, NumPatches, Channels, Height, Width),
    )

    # Add batch dimension to grid_mask: (batch, num_patches)
    grid_mask_batched = grid_mask[np.newaxis, ...]  # (1, num_patches)
    grid_mask_named = hax.named(grid_mask_batched, (Batch, NumPatches))

    # Create unpad_indices NamedArray if present
    unpad_indices_named = None
    if unpad_indices is not None:
        NumImageTokens = Axis("num_image_tokens", len(unpad_indices))
        unpad_indices_batched = unpad_indices[np.newaxis, ...]
        unpad_indices_named = hax.named(
            jnp.array(unpad_indices_batched, dtype=jnp.int32),
            (Batch, NumImageTokens)
        )

    # Configure stop tokens
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
    else:
        stop_tokens = None

    # Create decode params - use plain int/float (not jnp.array)
    # max_num_tokens is TOTAL sequence length (prompt + new tokens), not just new tokens
    estimated_max_seq_len = len(input_ids) + config.max_tokens + 64
    decode_params = SeqDecodingParams(
        max_num_tokens=estimated_max_seq_len,
        stop_tokens=stop_tokens,
        temperature=config.temperature,
        key=jrandom.PRNGKey(config.seed),
    )

    # Create VLMRequest with unpad_indices and num_unpadded_features
    vlm_request = VLMRequest(
        prompt_tokens=input_ids.tolist(),
        request_id=0,
        decode_params=decode_params,
        n_generations=1,
        pixel_values=pixel_values_named,
        grid_mask=grid_mask_named,
        input_ids=input_ids_named,
        unpad_indices=unpad_indices_named,
        num_unpadded_features=num_unpadded_features,
    )

    return vlm_request, input_ids.tolist()


# ============================================================================
# CAPTION GENERATION
# ============================================================================


def generate_caption(
    model: LlavaOnevisionModel,
    tokenizer,
    Vocab: Axis,
    vlm_request: VLMRequest,
    config: VLMCaptionConfig,
) -> str:
    """Generate image caption using LlavaInferenceEngine.

    Args:
        model: Loaded VLM model
        tokenizer: Tokenizer instance
        Vocab: Vocabulary axis
        vlm_request: Processed VLM request
        config: Caption configuration

    Returns:
        Generated caption string
    """
    logger.info("Starting caption generation...")

    # Create engine config
    # Use bfloat16 to match model weights
    # Use smaller page_size and max_pages to reduce KV cache HBM usage
    engine_config = InferenceEngineConfig(
        max_seq_len=2048,
        page_size=16,   # Smaller pages (test uses 16)
        max_pages=256,  # Fewer pages to fit in HBM
        max_seqs=1,
        max_queued_tokens=512,
        max_seqs_in_prefill=1,
        max_prefill_size=2048,
        max_rounds=64,
        compute_dtype=jnp.bfloat16,
    )

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        # Get mesh from trainer config (required for proper sharding)
        mesh = config.trainer.device_mesh

        # Create inference engine with explicit mesh
        engine = LlavaInferenceEngine.from_model_with_config(
            model=model,
            tokenizer=tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )

        # Generate
        result = engine.generate([vlm_request])

    # Decode output tokens
    raw_tokens = result.tokens[0]  # First request
    logger.info(f"Raw output tokens: {raw_tokens[:20]}...")  # Debug: show first 20 raw tokens

    # Filter out invalid tokens
    output_tokens = [t for t in raw_tokens if t >= 0]
    logger.info(f"Filtered tokens: {output_tokens[:20]}...")  # Debug: show first 20 filtered tokens

    # Decode without skip_special_tokens first for debugging
    raw_caption = tokenizer.decode(output_tokens, skip_special_tokens=False)
    logger.info(f"Raw decoded (first 100 chars): {raw_caption[:100]}")

    caption = tokenizer.decode(output_tokens, skip_special_tokens=True)

    logger.info(f"Generated {len(output_tokens)} tokens")
    return caption


# ============================================================================
# MAIN
# ============================================================================


def main(config: VLMCaptionConfig):
    """Main entry point for VLM Image Caption demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if not config.image_path:
        raise ValueError("--image_path is required")
    if not os.path.exists(config.image_path):
        raise FileNotFoundError(f"Image not found: {config.image_path}")

    print("\n" + "=" * 60)
    print("VLM Image Caption Demo")
    print("=" * 60)
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Image: {config.image_path}")
    print(f"Prompt: {config.prompt}")
    print("=" * 60 + "\n")

    # Build model config
    vlm_config = build_vlm_config(config.tokenizer)

    # Load model
    model, tokenizer, Vocab = load_model(config, vlm_config)

    # Process image
    vlm_request, prompt_tokens = process_image(config, vlm_config, tokenizer)

    # Generate caption
    caption = generate_caption(model, tokenizer, Vocab, vlm_request, config)

    # Output result
    print("\n" + "=" * 60)
    print("Generated Caption:")
    print("=" * 60)
    print(caption)
    print("=" * 60 + "\n")

    return caption


if __name__ == "__main__":
    levanter.config.main(main)()
