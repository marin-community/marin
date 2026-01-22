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
Demo VLM Training Experiment

Train a Vision-Language Model (LLaVA-OneVision architecture):
- Vision Encoder: SigLIP (384x384, patch14)
- Language Model: Qwen3-1.7B
- Projector: 2-layer MLP

Data Format: LLaVA conversation format (messages + images)

Usage:
    # For Cloud TPU via Marin/Ray:
    python experiments/VLM/demo_vlm_train.py

    # For Local TPU VM, use vlm_config.yaml instead:
    python -m levanter.main.train_vlm --config_path experiments/VLM/vlm_config.yaml
"""

import os

from huggingface_hub import login

login(token="YOUR_HF_TOKEN_HERE")

from fray.cluster import ResourceConfig
from levanter.data.image import ConversationDatasetSourceConfig, ImageMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llava_onevision import LlavaOnevisionConfig
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from marin.execution.executor import executor_main

from experiments.defaults import default_train_vlm
from experiments.simple_vlm_train_config import SimpleVlmTrainConfig

# ============================================================================
# 1. RESOURCE CONFIGURATION
# ============================================================================
# Options:
#   - ResourceConfig.with_tpu("v4-8")   # Small TPU slice
#   - ResourceConfig.with_tpu("v4-32")  # Medium TPU slice
#   - ResourceConfig.with_tpu("v4-128") # Large TPU slice
#   - ResourceConfig.with_gpu("H100", count=8)  # GPU cluster
#   - ResourceConfig.with_cpu()  # CPU only (for testing)
# Can be overridden via TPU_TYPE environment variable (e.g., -e TPU_TYPE v4-128)
TPU_TYPE = os.environ.get("TPU_TYPE", "v5p-64")
RESOURCES = ResourceConfig.with_tpu(TPU_TYPE)

# Extract TPU chip count from TPU_TYPE (e.g., "v5p-64" -> 64)
TPU_CHIPS = int(TPU_TYPE.split("-")[-1])

# Gradient accumulation configuration
# - per_device_parallelism: samples processed per device at a time (limited by memory)
# - gradient_accumulation_steps: how many micro-batches to accumulate before updating
# - effective batch size = TPU_CHIPS * per_device_parallelism * gradient_accumulation_steps
PER_DEVICE_PARALLELISM = 2  # 1 sample per device (memory-safe for VLM with large images)
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate 4 micro-batches
BATCH_SIZE = TPU_CHIPS * PER_DEVICE_PARALLELISM * GRADIENT_ACCUMULATION_STEPS  # Effective batch = 256 for v5p-64

# ============================================================================
# 2. MODEL CONFIGURATION
# ============================================================================

# Flash attention block size (set to None to disable flash attention)
FLASH_ATTENTION_BLOCK_SIZE = 1024

# Vision encoder: SigLIP-like (matches google/siglip-so400m-patch14-384)
vision_config = SiglipVisionConfig(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    image_size=384,
    patch_size=16,  # Must match vision_checkpoint (siglip2-so400m-patch16-384)
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,
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
    flash_attention_block_size=FLASH_ATTENTION_BLOCK_SIZE,
)

# Get image token ID from Qwen3 tokenizer (different from default which is for Qwen2)
from transformers import AutoTokenizer
_qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
IMAGE_TOKEN_INDEX = _qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

# Combined VLM config
vlm_config = LlavaOnevisionConfig(
    vision_config=vision_config,
    text_config=text_config,
    vision_encoder_type="siglip",
    vision_feature_select_strategy="full",
    vision_aspect_ratio="single",
    # Set disable_anyres=True to use single resolution (base patch only).
    # This reduces memory usage and speeds up training but may lose image details.
    disable_anyres=True,
    # Use Qwen3's <|image_pad|> token ID (default 151646 is for Qwen2)
    image_token_index=IMAGE_TOKEN_INDEX,
)

# ============================================================================
# 3. DATA CONFIGURATION (LLaVA Conversation Format)
# ============================================================================
# Your data should be parquet files with columns:
#   - "messages": list of {role, content} conversation turns
#   - "images": list of image paths/URLs
#
# Example data format:
# {
#     "messages": [
#         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]},
#         {"role": "assistant", "content": [{"type": "text", "text": "This image shows..."}]}
#     ],
#     "images": ["path/to/image.jpg"]
# }

data_source = ConversationDatasetSourceConfig(
    # >>> EDIT THIS PATH to point to your training data <<<
    train_urls=["gs://marin-vlm/stage2_sharded/*.parquet"],
    messages_key="messages",
    images_key="images",
)

# Compute vision_feature_height from model config to match data pipeline with model
# This is critical: the HF processor assumes 27x27=729 tokens (384//14), but our
# SigLIP vision encoder with patch_size=16 outputs 24x24=576 tokens (384//16).
# Without this override, training loss won't decrease due to index wrapping.
VISION_FEATURE_HEIGHT = vision_config.image_size // vision_config.patch_size  # 384 // 16 = 24

data_config = ImageMixtureDatasetConfig(
    cache_dir="cache/vlm_demo",
    # Processor for image preprocessing
    processor="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    # Custom tokenizer for text processing (uses CustomVLMProcessor internally)
    tokenizer="Qwen/Qwen3-1.7B",
    configs={"train": data_source},
    train_weights={"train": 1.0},
    use_cache=False,  # Streaming mode (no disk caching)
    max_length=2048,  # Match model's max_seq_len to avoid truncation issues
    vision_feature_height=VISION_FEATURE_HEIGHT,  # Override: use model's actual feature size
    # Disable anyres to match model config (disable_anyres=True sets vision_aspect_ratio="single")
    # Without this, the HF processor uses anyres_max_9 which calculates extra tokens for grid patches
    vision_aspect_ratio="single",
    image_grid_pinpoints=[[384, 384]],  # Single resolution only
)

# ============================================================================
# 4. TRAINING CONFIGURATION
# ============================================================================
# Dataset size: 558K samples
DATASET_SIZE = 10*1000*1000
NUM_EPOCHS = 1
NUM_TRAIN_STEPS = (DATASET_SIZE // BATCH_SIZE) * NUM_EPOCHS

train_config = SimpleVlmTrainConfig(
    resources=RESOURCES,
    train_batch_size=BATCH_SIZE,
    per_device_parallelism=PER_DEVICE_PARALLELISM,
    num_train_steps=NUM_TRAIN_STEPS,
    epoch=0,  # Disable epoch mode (use num_train_steps instead)
    learning_rate=1e-5,
    warmup=0.002,  # 3% warmup
    weight_decay=0.0,
    min_lr_ratio=0.1,  # Final LR = 1% of peak LR

    # Full bfloat16: params and compute both in bfloat16 (saves memory)
    mp="bfloat16",

    # Streaming mode: double the default prefetch for better throughput
    streaming_max_buffered_batches=16,
    streaming_prefetch_size=8,

    # Checkpointing
    steps_per_export=1000,
    steps_per_eval=500,

    # Load complete VLM weights from GCS HF checkpoint (vision encoder + projector + LLM)
    # This checkpoint contains trained weights from stage 1
    vlm_checkpoint="gs://marin-eu-west4/checkpoints/vlm-official-qwen3-1.7b-8-c3e151/hf/vlm-official-qwen3-1.7b-8-c3e151/step-544",
    # vision_checkpoint and llm_checkpoint not needed when using vlm_checkpoint

    # New training stage - data starts from beginning
    reset_data_loader_on_init=True,

    # Freeze components during training (only train projector)
    freeze_vision_encoder=False,
    freeze_llm=False,

    # Profiler configuration
    profiler=True,
    profiler_start_step=10,
    profiler_num_steps=20,

    # Disable evaluation to save memory
    no_eval=True,
)

# ============================================================================
# 5. EXPERIMENT NAME (via environment variable)
# ============================================================================
# Can be set via: -e EXP_NAME my-experiment-name
EXP_NAME = os.environ.get("EXP_NAME", "vlm-demo21-qwen2-1.7b")

# ============================================================================
# 6. CREATE TRAINING STEP
# ============================================================================
vlm_training = default_train_vlm(
    name=EXP_NAME,
    data_config=data_config,
    model_config=vlm_config,
    train_config=train_config,
    tags=["vlm", "demo", "qwen3-1.7b", "siglip"],
)

# ============================================================================
# 7. RUN
# ============================================================================
if __name__ == "__main__":
    executor_main(steps=[vlm_training])
