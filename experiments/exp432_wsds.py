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
Train 1.4B models on standard datasets (e.g., SlimPajama) using WSD-S.
https://github.com/marin-community/marin/issues/432
"""

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config
from marin.resources import TpuPodConfig

llama_1_4b_wsds_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=2),
    train_batch_size=1024,
    learning_rate=3e-4,
    weight_decay=0.1,
    num_train_steps=10000,  # 4096 * 1024 * 10000 = 42B tokens
    cycle_length=2000,  # 5 cycles with 2000 steps/cycle
    warmup=0.05,  # 5%  * 2000 = 100 steps
    decay=0.1,  # 10% * 2000 = 200 steps
    lr_schedule="inv",
    steps_per_eval=2000,
)

fineweb_edu_model = default_train(
    name="fineweb-edu-1.4b-wsds",
    tokenized=lm_data_config(fineweb_edu_tokenized, permutation_type="linear"),
    model_config=llama_1_4b,
    train_config=llama_1_4b_wsds_train_config,
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_model],
        description="Train 1.4B models on FineWebEdu using WSD-S.",
    )
