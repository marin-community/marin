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
  python -c "import json; \
[print(json.dumps(item)) for item in json.load(open('format_adaptation_dataset.json'))]" \
> traces.jsonl

Then upload to GCS:
  gsutil cp traces.jsonl gs://your-bucket/path/to/traces.jsonl

Usage:
  uv run marin.execution.executor:executor_main experiments.dspy.expxxx_dspy_baml_sft \
    --chat_train_urls '["gs://your-bucket/path/to/traces*.jsonl.gz"]'
"""

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.llama import llama3_instruct_tokenizer, llama_8b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config
from marin.resources import TpuPodConfig

# Default GCS path for trace data - override via command line or modify this variable
DEFAULT_CHAT_TRAIN_URLS = "gs://marin-us-central2/scratch/dspy-format-adaptation/traces/*.jsonl.gz"

# Chat format configuration - uses "chat" field instead of default "messages"
dspy_chat_format = ChatLmDatasetFormat(
    messages_field="chat",  # Field name in JSONL containing messages array
    single_turn=False,
    pack=True,
    mask_user_turns=True,
)

# Tokenize chat JSONL data
tokenize_step = default_tokenize(
    name="dspy-format-adaptation-tokenize",
    dataset=DEFAULT_CHAT_TRAIN_URLS,
    tokenizer=llama3_instruct_tokenizer,
    format=dspy_chat_format,
)

# Create data config from tokenized data
tokenized_data = lm_data_config(tokenize_step, permutation_type="linear")

# SFT configuration
sft_config = SimpleSFTConfig(
    resources=TpuPodConfig(tpu_type="v5p-8"),
    train_batch_size=64,
    num_train_steps=5000,
    learning_rate=2e-5,
    weight_decay=0.0,
    tokenizer=llama3_instruct_tokenizer,
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_seq_len=4096,
    warmup=0.02,  # 100 steps / 5000 steps
    cooldown=0.0,
    lr_schedule="linear",
    min_lr_ratio=0.1,
    steps_per_eval=500,
    steps_per_checkpoint=1000,
    steps_per_hf_export=1000,
    reinit_tokens=True,
    seed=0,
)

# Create SFT step
sft_step = default_sft(
    name="dspy-format-adaptation-sft",
    tokenized=tokenized_data,
    model_config=llama_8b,
    sft_config=sft_config,
    tags=["dspy", "format-adaptation", "llama-8b"],
).with_output_path("checkpoints/dspy-format-adaptation-sft")

# Pipeline entry point
if __name__ == "__main__":
    executor_main(
        steps=[tokenize_step, sft_step],
        description=(
            "SFT for Llama 8B using DSPy format adaptation trace data. "
            "Traces are collected from HotpotQA, HoVer, and FHIR datasets."
        ),
    )
