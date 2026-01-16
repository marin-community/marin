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

"""Learning rate sweeps comparing MuP and non-MuP variants of small Llama models."""

import logging

from experiments.defaults import default_train
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import MuonRemezConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")

llama_300m = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=768,
    intermediate_dim=3072,
    num_heads=12,
    num_kv_heads=12,  # following the same ratio as the original code
    num_layers=32,
)


def _format_lr(lr: float) -> str:
    return f"{lr:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace("+", "")


def _lr_sweep() -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Construct a set of training steps for the provided hyperparameter sweep."""

    steps = []
    num_train_step = 11520
    weight_decay = 0.1  # Fixed weight decay, no sweeping
    lrs = [0.001, 0.002, 0.004, 0.008, 0.012]
    for lr in lrs:
        optimizer_config = MuonRemezConfig(
            learning_rate=lr,
            warmup=0,
            min_lr_ratio=0.0,
            lr_schedule="cosine",
            adam_lr=0.4 * lr,
            momentum=0.95,
            nesterov=True,
            backend_steps=7,
            weight_decay=weight_decay,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            muon_epsilon=1e-8,
            max_grad_norm=1.0,
            use_kimi_scaling=False,
        )
        train_config = SimpleTrainConfig(
            resources=ResourceConfig.with_tpu("v5litepod-64"),
            train_batch_size=128,
            num_train_steps=num_train_step,
            learning_rate=lr,
            watch=WatchConfig(watch_targets=["grads", "params"], interval=10),
            optimizer_config=optimizer_config,
        )
        step = default_train(
            name=f"test-remez-300m-lr{_format_lr(lr)}-wd{weight_decay}-warmup0-alr0.4",
            tokenized=dclm_mixture_config_llama3,
            model_config=llama_300m,
            train_config=train_config,
            tags=("remez", "300m", "lr_sweep", "cosine"),
            eval_harness_tasks=(),
        )

        steps.append(step)

    return steps


def build_steps():
    all_steps = []
    all_steps.extend(_lr_sweep())

    return all_steps


if __name__ == "__main__":
    executor_main(steps=build_steps())
