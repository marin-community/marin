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

import os

# Set temporary directory to /dev/shm before other imports
# This fixes "No usable temporary directory found" error
os.environ["TMPDIR"] = "/dev/shm"

import logging
from dataclasses import dataclass, field
from typing import Tuple

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

import levanter
from levanter.data.image import BatchImageProcessor, CustomVLMProcessor, create_custom_processor
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

logger = logging.getLogger(__name__)

# Enable JAX compilation cache (use /dev/shm for faster access)
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
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
    temperature: float = 0.7
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
        tokenizer=tokenizer_name,  # Set tokenizer to avoid inference from GCS path
    )


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model(
    config: VLMCaptionConfig,
    vlm_config: LlavaOnevisionConfig,
) -> Tuple[LlavaOnevisionModel, any, Axis]:
    """Load VLM model from HuggingFace-format checkpoint.

    The checkpoint should be in HuggingFace format (with config.json, model.safetensors, etc.)
    Reference: lib/levanter/tests/test_llava_onevision.py

    Args:
        config: Caption configuration
        vlm_config: VLM model configuration

    Returns:
        Tuple of (model, tokenizer, Vocab axis)
    """
    logger.info(f"Loading model from HF checkpoint: {config.checkpoint}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    vocab_size = len(tokenizer)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size),
            config.trainer.compute_axis_mapping,
        )

        # Use HFCheckpointConverter to load HuggingFace format checkpoint
        converter = vlm_config.hf_checkpoint_converter(ref_checkpoint=config.checkpoint)
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

        model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=config.checkpoint,
            config=vlm_config,
            axis_mapping=parameter_axis_mapping,
            dtype=config.trainer.mp.compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )

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

    Uses BatchImageProcessor for consistent processing with training pipeline.
    Reference: lib/levanter/tests/test_image_utils.py

    Args:
        config: Caption configuration
        vlm_config: VLM model configuration
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (VLMRequest, prompt_tokens list)
    """
    logger.info(f"Processing image: {config.image_path}")

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

    # Create processor using create_custom_processor (same as test utils)
    # patch_size and vision_feature_height must match model config
    patch_size = vlm_config.vision_config.image_size  # 384
    vision_feature_height = patch_size // vlm_config.vision_config.patch_size  # 384/16=24

    base_processor = create_custom_processor(
        config.processor,
        do_pad=True,
        image_grid_pinpoints=[[patch_size, patch_size]],  # Single resolution
        max_image_tiles=5,  # For disable_anyres, one tile per image
        vision_aspect_ratio="single",
    )

    lev_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        base_processor,
        tokenizer,
        use_full_padded_tokens=False,  # HF-style unpadded tokens
    )

    # Create BatchImageProcessor for consistent grid_mask handling
    batch_processor = BatchImageProcessor(
        processor=lev_processor,
        max_length=2048,
        padding=True,
        max_num_patches=0,  # disable_anyres: each image uses only base patch
        grid_pinpoints=[[patch_size, patch_size]],
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        add_generation_prompt=True,
        disable_anyres=True,
    )

    # Process image using BatchImageProcessor
    example = {"messages": messages, "images": [image]}
    processed = batch_processor([example])[0]

    # Extract outputs from processed dict
    input_ids = np.array(processed["input_ids"], dtype=np.int32)
    pixel_values = np.array(processed["pixel_values"], dtype=np.float32)
    grid_mask = np.array(processed["grid_mask"], dtype=bool)
    attention_mask = np.array(processed["attention_mask"], dtype=np.int32)

    # Get actual content length (non-padding tokens)
    actual_content_len = int(attention_mask.sum())
    prompt_tokens = input_ids[:actual_content_len].tolist()

    logger.info(f"Input tokens: {len(prompt_tokens)}, Total padded: {len(input_ids)}, Patches: {pixel_values.shape[0]}")

    # Create NamedArrays for VLMRequest (with batch dimension)
    batch_size = 1
    total_patches = pixel_values.shape[0]
    seq_len = len(input_ids)

    Batch = Axis("batch", batch_size)
    NumPatches = Axis("num_patches", total_patches)
    Channels = Axis("channels", pixel_values.shape[1])
    Height = Axis("height", pixel_values.shape[2])
    Width = Axis("width", pixel_values.shape[3])
    Position = Axis("position", seq_len)
    GridMaskAxis = Axis("grid_mask", total_patches)

    # Add batch dimension and create NamedArrays
    pixel_values_batched = pixel_values.reshape(1, total_patches, pixel_values.shape[1], pixel_values.shape[2], pixel_values.shape[3])
    grid_mask_batched = grid_mask.reshape(1, -1)
    input_ids_batched = input_ids.reshape(1, -1)

    pixel_values_named = hax.named(
        jnp.array(pixel_values_batched, dtype=jnp.bfloat16),
        (Batch, NumPatches, Channels, Height, Width),
    )
    grid_mask_named = hax.named(jnp.array(grid_mask_batched), (Batch, GridMaskAxis))
    input_ids_named = hax.named(jnp.array(input_ids_batched, dtype=jnp.int32), (Batch, Position))

    # Create decode params with EOS token as stop token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        stop_tokens = hax.named(jnp.array([[eos_token_id]], dtype=jnp.int32), ("stop_seq", "position"))
    else:
        stop_tokens = None

    decode_params = SeqDecodingParams(
        max_num_tokens=jnp.array(config.max_tokens + actual_content_len, dtype=jnp.int32),
        stop_tokens=stop_tokens,
        temperature=jnp.array(config.temperature, dtype=jnp.float32),
        key=jrandom.PRNGKey(config.seed),
    )

    # Create VLMRequest
    vlm_request = VLMRequest(
        prompt_tokens=prompt_tokens,
        request_id=0,
        decode_params=decode_params,
        n_generations=1,
        pixel_values=pixel_values_named,
        grid_mask=grid_mask_named,
        input_ids=input_ids_named,
        unpad_indices=None,  # None for disable_anyres mode
        num_unpadded_features=None,
    )

    return vlm_request, prompt_tokens


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

    Reference: lib/levanter/tests/test_llava_onevision.py::test_llava_onevision_generation_with_inference_engine

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

    # Estimate max sequence length for inference
    prompt_len = len(vlm_request.prompt_tokens)
    estimated_max_seq_len = prompt_len + config.max_tokens + 64
    page_size = 16

    # Create engine config (following test patterns)
    engine_config = InferenceEngineConfig(
        max_seq_len=estimated_max_seq_len,
        page_size=page_size,
        max_seqs=1,
        max_rounds=32,
        max_stop_seqs=1,
        max_stop_tokens=4,
        max_pages=800,
        compute_dtype=jnp.bfloat16,
    )

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        mesh = config.trainer.device_mesh

        # Create inference engine
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
    output_tokens = result.tokens[0]  # First request
    # Filter out invalid tokens
    output_tokens = [t for t in output_tokens if t >= 0]
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
