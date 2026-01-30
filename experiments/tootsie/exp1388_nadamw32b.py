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
Train 32B on Nemotron with Starcoderdata and Proofpile 2 using NAdamW
"""

import dataclasses

from levanter.optim import AdamConfig

from experiments.defaults import default_train
from experiments.llama import llama_32b
from experiments.tootsie.exp1295_32b import llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from fray.v2 import ResourceConfig
from marin.execution import executor_main

warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-77096/").nonblocking()


nadamw_config = AdamConfig(
    # Yolo hypers for nadamw from Kaiyue
    learning_rate=llama_32b_train_config.learning_rate,
    weight_decay=llama_32b_train_config.weight_decay,
    beta1=0.95,
    beta2=0.95,
    nesterov=True,
)

llama_32b_warmstart_train = dataclasses.replace(
    llama_32b_train_config,
    initialize_from_checkpoint_path=warmstart_checkpoint,
    optimizer_config=nadamw_config,
    resources=ResourceConfig.with_tpu("v4-2048", slice_count=1),
    reset_data_loader_on_init=False,
)

llama_32b_nadamw = default_train(
    name="marin-32b-nadamw-4",
    tokenized=nemotron_mix,
    model_config=llama_32b,
    train_config=llama_32b_warmstart_train,
    tags=["llama", "32b", "ema", "exp1388", "tootsie", "nadamw"],
    eval_harness_tasks=[],
).with_output_path("checkpoints/marin-32b-nadamw-4")


if __name__ == "__main__":
    executor_main(
        [
            llama_32b_nadamw,
        ],
        description="Give nadamw a shot on 32B with Nemotron and Starcoderdata",
    )
