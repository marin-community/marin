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
SFT script for Llama 8B using DSPy format adaptation trace data.

This script configures supervised fine-tuning for Llama 8B using trace data
collected from DSPy modules (HotpotQA, HoVer, FHIR) for format adaptation.

The trace data should be stored in GCS as JSONL files with chat format:
- Each line is a JSON object with a "chat" field containing messages
- Messages follow OpenAI chat format: [{"role": "system/user/assistant", "content": "..."}]

If your data is in JSON array format (from process_data.py), convert it to JSONL:
  python -c "import json; [print(json.dumps(item)) for item in json.load(open('format_adaptation_dataset.json'))]" > traces.jsonl

Then upload to GCS:
  gsutil cp traces.jsonl gs://your-bucket/path/to/traces.jsonl

Usage:
  uv run levanter.train experiments.dspy.expxxx_dspy_baml_sft:DSPyFormatAdaptationSFTConfig \
    --chat_train_urls '["gs://your-bucket/path/to/traces*.jsonl.gz"]'
"""

from dataclasses import dataclass, field
from typing import Optional

import jmp
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.sft import DatasetType, SFTConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig

from experiments.llama import llama3_instruct_tokenizer


@dataclass
class DSPyFormatAdaptationSFTConfig(SFTConfig):
    """Configuration for SFT on DSPy format adaptation traces."""

    # Model configuration - Llama 8B
    model: LlamaConfig = field(
        default_factory=lambda: LlamaConfig(
            seq_len=4096,
            hidden_dim=4096,
            intermediate_dim=14336,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            flash_attention_block_size=512,
            use_bias=False,
            use_layer_norm_weight=True,
            initializer_range=0.02,
            rope=Llama3RotaryEmbeddingsConfig(),
        )
    )

    # Trainer configuration
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            tracker=WandbConfig(project="marin-dspy-format-adaptation", tags=["dspy", "format-adaptation", "llama-8b"]),
            num_train_steps=5000,
            train_batch_size=64,
            tensor_parallel_axes=["mlp", "heads"],
            fsdp_axis="embed",
            batch_axis="batch",
            steps_per_eval=500,
        )
    )

    # Optimizer configuration
    optimizer: AdamConfig = field(
        default_factory=lambda: AdamConfig(
            learning_rate=2e-5,
            weight_decay=0.0,
            min_lr_ratio=0.1,
            warmup=100,
        )
    )

    # Dataset configuration
    dataset_type: DatasetType = DatasetType.CHAT_JSONL
    chat_train_urls: Optional[list[str]] = None
    messages_field: str = "chat"  # Field name in JSONL containing messages array (matches trace data format)
    input_role: str = "user"
    output_role: str = "assistant"

    # Model initialization
    initialize_from_hf: bool = True
    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer: str = llama3_instruct_tokenizer
    max_seq_len: int = 4096

    # Supervised data cache
    supervised_data: Optional[dict] = field(
        default_factory=lambda: dict(
            cache_dir="gs://marin-us-central2/scratch/dspy-format-adaptation-sft-cache",
        )
    )

    # Reinitialize tokens for Llama 3 tokenizer
    reinit_tokens: bool = True
    reinit_lm_head: bool = True
    reinit_embeddings: bool = True

    # Checkpointing
    hf_save_steps: int = 1000
    hf_save_path: Optional[str] = None


def main():
    """Main entry point for SFT training."""
    import draccus

    # Parse config from command line or use defaults
    config = draccus.parse(DSPyFormatAdaptationSFTConfig)

    # Validate that chat_train_urls is provided
    if config.chat_train_urls is None:
        raise ValueError(
            "chat_train_urls must be provided. "
            "Example: --chat_train_urls '[\"gs://bucket/path/to/traces/*.jsonl.gz\"]'"
        )

    # Set default HF save path if not provided
    if config.hf_save_path is None:
        config.hf_save_path = "gs://marin-us-central2/checkpoints/dspy-format-adaptation-sft/llama-8b"

    # Import and run training
    from levanter.main.sft import train

    train(config)


if __name__ == "__main__":
    main()

