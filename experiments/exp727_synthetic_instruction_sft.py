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

from experiments.defaults import default_sft, default_tokenize
from experiments.exp964_custom_chat_tokenizer import llama3_instruct_trainable_chat_template
from experiments.llama import llama3_instruct_tokenizer, llama3_tokenizer, llama_8b
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.processing.tokenize import lm_data_config

# Get instruction dataset
synthetic_instruction_dataset = get_instruction_dataset("sherryy/tulu-3-sft-personas-instruction-following-expanded")

# TODO: tune this for a good number of steps
NUM_TRAIN_STEPS = 2500

# Add tokenization step
synthetic_instruction_llama_tokenized = default_tokenize(
    name="synthetic_instruction_llama3_tokenizer",
    dataset=synthetic_instruction_dataset / "**/*.jsonl.gz",
    tokenizer=llama3_instruct_tokenizer,
    format=llama3_instruct_trainable_chat_template,
)
seed = 1

tulu3_sft_8b_synthetic_instruction_model = default_sft(
    name=f"checkpoints/tulu3_sft_synthetic_instruction{seed}",
    tokenized=lm_data_config(synthetic_instruction_llama_tokenized, permutation_type="linear"),
    model_config=llama_8b,
    sft_config=SimpleSFTConfig(
        train_batch_size=64,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=5e-6,
        resources=TpuPodConfig(tpu_type="v4-64"),
        tokenizer=llama3_tokenizer,
        model_name_or_path="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        seed=seed,
        initialize_from_hf=False,
        initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/llama3.1_8b_tulu_3-12305c/checkpoints/step-9980/",
    ),
    tags=["llama", "8b", "synthetic_instruction", "exp727"],
)


if __name__ == "__main__":
    executor_main(steps=[synthetic_instruction_llama_tokenized, tulu3_sft_8b_synthetic_instruction_model])
