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

from experiments.exp808_sft_mixture import create_tokenization_step
from experiments.defaults import default_sft
from experiments.llama import llama_3_2_1b
from experiments.simple_sft_config import SimpleSFTConfig
from marin.resources import TpuPodConfig
from marin.processing.tokenize import lm_mixture_data_config
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import executor_main

NUM_TRAIN_STEPS = 7338  # 1 epoch over tulu-3-sft-mixture

tulu_3_sft_mixture = create_tokenization_step("allenai/tulu-3-sft-mixture")

# Define an SFT config appropriate for mixture training
mixture_sft_config = SimpleSFTConfig(
    train_batch_size=16,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6 / 3,  # 4x smaller batch size, 2x smaller laerning rate
    resources=TpuPodConfig(tpu_type="v5p-8"),
    tokenizer=marin_tokenizer,
    # model_name_or_path="marin-community/marin-8b-base",
    model_name_or_path="meta-llama/Llama-3.2-1B",
    initialize_from_hf="meta-llama/Llama-3.2-1B",
    max_seq_len=4096,
    seed=0,
)

mixture_config = lm_mixture_data_config(
    {"tulu_3_sft_mixture": tulu_3_sft_mixture},
    {"tulu_3_sft_mixture": 1.0},
    permutation_type="feistel",
    shuffle=True,
    missing_weights_are_validation=True,
)

# Configure mixture-based SFT training
training_step = default_sft(
    name="llama-3.2-1b-tulu-3-sft-mixture",
    tokenized=mixture_config,
    model_config=llama_3_2_1b,
    sft_config=mixture_sft_config,
    tags=["llama-3.2-1b", "mixture"],
)

if __name__ == "__main__":
    executor_main(steps=[tulu_3_sft_mixture, training_step])
