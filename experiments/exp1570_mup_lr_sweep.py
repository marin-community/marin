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

import dataclasses
import logging
import numpy as np
from collections.abc import Iterable

from experiments.defaults import default_train
from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from levanter.callbacks.watch import WatchConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.resources import TpuPodConfig
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")

LR_CHOICES = np.logspace(-13, -5, num=15, base=2)
"""Learning rates to sweep for each configuration."""


BASE_BATCH_SIZE = 128
BASE_NUM_TOKS = 100_000_000_000
BASE_WEIGHT_DECAY = 0.1


def _scale_llama_config(base: LlamaConfig, hidden_dim: int) -> LlamaConfig:
    """Create a new LlamaConfig scaled from ``base`` by adjusting the hidden size."""

    intermediate_ratio = base.intermediate_dim / base.hidden_dim
    intermediate_dim = int(intermediate_ratio * hidden_dim)

    logger.info("Scaling Llama config: hidden_dim=%s, intermediate_dim=%s", hidden_dim, intermediate_dim)

    return dataclasses.replace(
        base,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )


def _format_lr(lr: float) -> str:
    return f"{lr:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace("+", "")


def _lr_sweep(
    *,
    hidden_dim: int,
    use_mup: bool,
    learning_rates: Iterable[float],
) -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Construct a set of training steps for the provided hyperparameter sweep."""

    scaled_config = _scale_llama_config(llama_30m, hidden_dim)
    sweep_kind = "mup" if use_mup else "baseline"
    base_name = f"exp1570_hidden{hidden_dim}-{sweep_kind}-wsd-long"
    stable_sweep_kind = "mup-stable"
    stable_base_name = f"exp1570_hidden{hidden_dim}-{stable_sweep_kind}-wsd-long"

    steps = []
    for lr in learning_rates:
        optimizer_config = AdamConfig(
            learning_rate=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            use_mup=use_mup,
            warmup=0.1,
            decay=0.1,
            lr_schedule="linear",
        )

        train_config = SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type="v4-16"),
            train_batch_size=BASE_BATCH_SIZE,
            num_train_steps=BASE_NUM_TOKS / (BASE_BATCH_SIZE * scaled_config.seq_len),
            learning_rate=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            watch=WatchConfig(watch_targets=["grads", "params", "updates", "opt_state"], interval=1),
            use_mup=use_mup,
            optimizer_config=optimizer_config,
        )

        step = default_train(
            name=f"{base_name}-lr{_format_lr(lr)}",
            tokenized=nemotron_mix,
            model_config=scaled_config,
            train_config=train_config,
            tags=("exp1570", f"hidden_dim_{hidden_dim}", sweep_kind, "lr_sweep"),
            eval_harness_tasks=(),
        )

        steps.append(step)

        if use_mup:
            independent_weight_decay = BASE_WEIGHT_DECAY * (2 ** (-7.2))
            oc = dataclasses.replace(
                optimizer_config, decoupled_weight_decay=True, weight_decay=independent_weight_decay
            )
            mc = dataclasses.replace(scaled_config, use_layer_norm_weight=False)
            train_config = SimpleTrainConfig(
                resources=TpuPodConfig(tpu_type="v4-16"),
                train_batch_size=BASE_BATCH_SIZE,
                num_train_steps=BASE_NUM_TOKS / (BASE_BATCH_SIZE * scaled_config.seq_len),
                learning_rate=lr,
                weight_decay=independent_weight_decay,
                watch=WatchConfig(watch_targets=["grads", "params", "updates", "opt_state"], interval=1),
                use_mup=use_mup,
                optimizer_config=oc,
            )

            step = default_train(
                name=f"{stable_base_name}-lr{_format_lr(lr)}",
                tokenized=nemotron_mix,
                model_config=mc,
                train_config=train_config,
                tags=("exp1570", f"hidden_dim_{hidden_dim}", sweep_kind, "lr_sweep"),
                eval_harness_tasks=(),
            )
            steps.append(step)

    return steps


def build_steps():
    hidden_dims = [llama_30m.hidden_dim]
    hidden_dims.append(hidden_dims[-1] * 2)
    hidden_dims.append(hidden_dims[-1] * 2)

    all_steps = []
    for use_mup in (True, False):
        for hidden_dim in hidden_dims:
            all_steps.extend(_lr_sweep(hidden_dim=hidden_dim, use_mup=use_mup, learning_rates=LR_CHOICES))

    return all_steps


if __name__ == "__main__":
    executor_main(steps=build_steps())
