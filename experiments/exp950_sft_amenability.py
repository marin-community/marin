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
Experiment 950: Learning Rate Schedule Comparison for 1.4B Llama Models
Link: https://github.com/marin-community/marin/issues/950

We have observed that Tootsie seems to be harder to SFT than OLMo, despite getting better
scores on MMLU and similar scores on Paloma. One hypothesis is that the higher LR used
or the use of WSD-S makes the model more difficult to finetune. We want to test that!

This experiment evaluates the impact of different learning rate schedules and configurations
on 1.4B parameter Llama models trained on a DCLM mixture followed by supervised fine-tuning.

The experiment compares:
- Linear vs. cosine learning rate schedules
- Higher (1e-3) vs. lower (3e-4) learning rates
- The effect of z_loss with weight 1e-4

Three pre-training configurations are tested:
1. Linear schedule with high learning rate (1e-3) and z_loss
2. Cosine schedule with high learning rate (1e-3) and z_loss
3. Cosine schedule with lower learning rate (3e-4) and z_loss

Each resulting model is then fine-tuned using supervised fine-tuning (SFT)
with the Tulu SFT configuration.

Author: Will Held
"""

import dataclasses

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_old
from experiments.defaults import default_sft, default_train
from experiments.exp606_sft import tulu3_llama_data_old, tulu_sft_config
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import executor_main, output_path_of

llama_1_4b_wsd_high_lr_train_config = dataclasses.replace(
    llama_1_4b_train_config,
    num_train_steps=238418,  # 4096 * 1024 * 238418 = 1T tokens
    weight_decay=0.05,
    decay=0.2,
    learning_rate=1e-3,
    lr_schedule="linear",
    ema_beta=0.995,
    z_loss_weight=1e-4,
)


dclm_mix_model_wsd = default_train(
    name="lr_tests_wsd",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_1_4b,
    train_config=llama_1_4b_wsd_high_lr_train_config,
)

dclm_mix_model_cos_high = default_train(
    name="lr_tests_cosin_high",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_1_4b,
    train_config=dataclasses.replace(llama_1_4b_wsd_high_lr_train_config, lr_schedule="cosine", decay=None),
)

dclm_mix_model_cos_low = default_train(
    name="lr_tests_cosin_low",
    tokenized=dclm_mixture_config_llama3_old,
    model_config=llama_1_4b,
    train_config=dataclasses.replace(
        llama_1_4b_wsd_high_lr_train_config, learning_rate=3e-4, lr_schedule="cosine", decay=None
    ),
)


sft_model_wsd = default_sft(
    name="sft/tulu_sft_wsd_linear_lr",
    tokenized=tulu3_llama_data_old,
    model_config=llama_1_4b,
    sft_config=dataclasses.replace(
        tulu_sft_config,
        model_name_or_path=output_path_of(dclm_mix_model_wsd, "hf/238417/"),
    ),
    tags=["llama", "1.4b", "exp934", "linear_lr", "sft", "z_loss"],
).with_output_path("checkpoints/sft/tulu_sft_wsd_linear_lr")

sft_model_cos_high = default_sft(
    name="sft/tulu_sft_cos_high_lr",
    tokenized=tulu3_llama_data_old,
    model_config=llama_1_4b,
    sft_config=dataclasses.replace(
        tulu_sft_config,
        model_name_or_path=output_path_of(dclm_mix_model_cos_high, "hf/238417/"),
    ),
    tags=["llama", "1.4b", "exp934", "cosine_lr", "high_lr", "sft", "z_loss"],
).with_output_path("checkpoints/sft/tulu_sft_cos_high_lr")

sft_model_cos_low = default_sft(
    name="sft/tulu_sft_cos_low_lr",
    tokenized=tulu3_llama_data_old,
    model_config=llama_1_4b,
    sft_config=dataclasses.replace(
        tulu_sft_config,
        model_name_or_path=output_path_of(dclm_mix_model_cos_low, "hf/238417/"),
    ),
    tags=["llama", "1.4b", "exp934", "cosine_lr", "low_lr", "sft", "z_loss"],
).with_output_path("checkpoints/sft/tulu_sft_cos_low_lr")

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_mix_model_wsd,
            dclm_mix_model_cos_high,
            dclm_mix_model_cos_low,
            sft_model_wsd,
            sft_model_cos_high,
            sft_model_cos_low,
        ],
        description="Train 1.4B models on dclm using varying learning rates, then SFT the resulting models.",
    )
