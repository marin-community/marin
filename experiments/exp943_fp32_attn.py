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
Experiment 943: Check the performance impact of FP32 attention vs not FP32 attention
"""

import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_old
from experiments.defaults import default_train
from experiments.llama import llama_1_4b, llama_1_4b_train_config, llama_8b, llama_8b_train_config
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

# Default upcast_attn is False in LlamaConfig

# Create a version with upcast_attn=True
llama_1_4b_fp32_attn = dataclasses.replace(
    llama_1_4b,
    upcast_attn=True,
)

# Create a version of the 8B config with upcast_attn=True
llama_8b_quick_fp32_attn = dataclasses.replace(
    llama_8b,
    upcast_attn=True,
)

# Modify the train config for a quick run
llama_8b_quick_train_config = dataclasses.replace(
    llama_8b_train_config,
    num_train_steps=1000,
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
    # 1024 doesn't fit on v4-128
    train_batch_size=512,
)


dclm_mix_model = default_train(
    name="dclm_mix-1.4b-default",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
)

dclm_mix_model_fp32_attn = default_train(
    name="dclm_mix-1.4b-fp32-attn",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_1_4b_fp32_attn,
    train_config=llama_1_4b_train_config,
)

dclm_mix_model_8b_quick_fp32_attn = default_train(
    name="dclm_mix-8b-quick-fp32-attn",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_8b_quick_fp32_attn,
    train_config=llama_8b_quick_train_config,
)

dclm_mix_model_8b_quick_baseline = default_train(
    name="dclm_mix-8b-quick-baseline",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_8b,
    train_config=llama_8b_quick_train_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_mix_model,
            dclm_mix_model_fp32_attn,
            dclm_mix_model_8b_quick_fp32_attn,
            dclm_mix_model_8b_quick_baseline,
        ],
        description="Train 1.4B and 8B models on dclm with and without FP32 attention.",
    )
