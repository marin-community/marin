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

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.llama import llama_8b
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.processing.tokenize import lm_data_config

# Get instruction dataset
tulu2_mixture_olmo = get_instruction_dataset("allenai/tulu-v2-sft-mixture-olmo-4096")

# Number of tokens in the SFT dataset below
NUM_TRAIN_TOKENS = 150849275
# number of epochs over the dataset set to reproduce Olmo SFT
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 3 epochs

# Add tokenization step
tulu2_olmo_tokenized = default_tokenize(
    name="olmo702024_sft_4096_3eps",
    dataset=tulu2_mixture_olmo / "**/*.jsonl.gz",
    tokenizer="stanford-crfm/marin-olmo2-tokenizer",
    format=ChatLmDatasetFormat(),
)
train_step = default_sft(
    name="checkpoints/olmo7_072024_sft_4096_3eps",
    tokenized=lm_data_config(tulu2_olmo_tokenized, permutation_type="linear"),
    model_config=llama_8b,
    sft_config=SimpleSFTConfig(
        train_batch_size=128,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=2e-6,  # 2x10^-6
        resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
        tokenizer="EleutherAI/gpt-neox-20b",
        model_name_or_path="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",
        max_seq_len=4096,
        seed=0,
        weight_decay=0.0,
        warmup=0.03,
        cooldown=0.0,
        lr_schedule="linear",
        min_lr_ratio=0.0,
        steps_per_hf_export=500,
        max_grad_norm=None,
    ),
)


if __name__ == "__main__":
    executor_main(steps=[tulu2_olmo_tokenized, train_step])
