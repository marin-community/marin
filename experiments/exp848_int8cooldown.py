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

"""An experiment comparing standard v.s. int8 cooldown

We cooldown a 8B model on DCLM baine using both bf16 and int8 training to compare the results.
Link to issue: https://github.com/marin-community/marin/issues/848
"""

import dataclasses

from haliax.quantization import QuantizationConfig

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig

dclm = dclm_components_llama3["dclm_baseline"]

NUM_ANNEAL_TOKENS = 50_000_000_000

control_dataset_config = lm_mixture_data_config(
    components={"dclm": dclm},
    weights={"dclm": 1.0},
    permutation_type="linear",
)
control_anneal_config = AnnealConfig(
    dataset_config=control_dataset_config,
    num_anneal_training_tokens=NUM_ANNEAL_TOKENS,
    resources=TpuPodConfig(tpu_type="v5litepod-128"),
)

control_model = default_anneal(name="llama-8b-anneal-bf16-control", anneal_config=control_anneal_config)

# Keep everything fixed except for int8 training.
int8_trainer = dataclasses.replace(control_model.config.train_config.trainer, quantization=QuantizationConfig(int8=True))
int8_config = dataclasses.replace(control_model.config.train_config, trainer=int8_trainer)
int8_model = dataclasses.replace(control_model, config=int8_config, name="llama-8b-anneal-int8")


if __name__ == "__main__":
    executor_main(
        steps=[control_model, int8_model],
    )
