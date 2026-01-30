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
Speedrun code for a 75M parameter model based on the Llama architecture, and the Adamax optimizer.
"""

import logging
from dataclasses import dataclass

import optax
from levanter.optim import OptimizerConfig

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.v2 import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


@dataclass(frozen=True)
class AdamaxConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        print(f"Building optimizer: {self.__class__.__name__}")

        # try to register the class if it's not already registered
        try:
            OptimizerConfig.register_subclass("adamax")(AdamaxConfig)
        except ValueError:
            # ignore and use the already registered class
            pass

        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(
                optax.scale_by_adamax(
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                )
            )

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)
            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


# --------------------- speedrun Using Adamax ------------------------

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Nikil Ravi",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/nikilravi/",
    ),
    description="75M parameter model with Adamax optimizer",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-128"),
        train_batch_size=512,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.0,
        steps_per_eval=2000,
        optimizer_config=AdamaxConfig(),
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_75m_adamax", speedrun_config))
