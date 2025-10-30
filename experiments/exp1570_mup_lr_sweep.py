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
from collections.abc import Iterable

import numpy as np

from experiments.defaults import default_train
from experiments.llama import llama_30m
from experiments.qwen3 import qwen3_30m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from levanter.callbacks.watch import WatchConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig, AdamHConfig, MuonHConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.resources import TpuPodConfig
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")

LR_CHOICES = np.logspace(-13, -5, num=15, base=2)
INDEPENDENT_WD_CHOICES = np.logspace(-14, -5, num=10, base=2)

BASE_BATCH_SIZE = 128
BASE_NUM_TOKS = 100_000_000
BASE_WEIGHT_DECAY = 0.1
TPU_TYPE = "v5p-8"
WATCH_TARGETS = ["grads", "params", "updates", "opt_state"]
STABLE_WD_RATIO = 2 ** (-7.2)


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


def _make_train_config(
    *,
    scaled_config: LlamaConfig,
    lr: float,
    weight_decay: float,
    use_mup: bool,
    optimizer_config,
) -> SimpleTrainConfig:
    """Helper to construct a SimpleTrainConfig with our common defaults."""
    return SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE),
        train_batch_size=BASE_BATCH_SIZE,
        num_train_steps=BASE_NUM_TOKS / (BASE_BATCH_SIZE * scaled_config.seq_len),
        learning_rate=lr,
        weight_decay=weight_decay,
        watch=WatchConfig(watch_targets=WATCH_TARGETS, interval=1),
        use_mup=use_mup,
        optimizer_config=optimizer_config,
    )


def _lr_sweep(
    *, hidden_dim: int, use_mup: bool, learning_rates: Iterable[float], base_model: LlamaConfig
) -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Construct a set of training steps for the provided hyperparameter sweep."""

    scaled_config = _scale_llama_config(base_model, hidden_dim)
    model_type = "-qwen3" if isinstance(base_model, Qwen3Config) else ""
    sweep_kind = f"mup{model_type}" if use_mup else f"baseline{model_type}"
    base_name = f"exp1570_hidden{hidden_dim}-{sweep_kind}-wsd"
    stable_sweep_kind = f"mup-stable{model_type}"
    stable_base_name = f"exp1570_hidden{hidden_dim}-{stable_sweep_kind}-wsd"

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

        train_config = _make_train_config(
            scaled_config=scaled_config,
            lr=lr,
            weight_decay=BASE_WEIGHT_DECAY,
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
            independent_weight_decay = BASE_WEIGHT_DECAY * STABLE_WD_RATIO
            oc = dataclasses.replace(
                optimizer_config, decoupled_weight_decay=True, weight_decay=independent_weight_decay
            )
            mc = dataclasses.replace(scaled_config, use_layer_norm_weight=False)
            train_config = _make_train_config(
                scaled_config=scaled_config,
                lr=lr,
                weight_decay=independent_weight_decay,
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


def _lr_sweep_muonh(
    *, hidden_dim: int, learning_rates: Iterable[float], base_model: LlamaConfig
) -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Construct a set of training steps for MuonH across learning rates."""

    scaled_config = _scale_llama_config(base_model, hidden_dim)
    model_type = "-qwen3" if isinstance(base_model, Qwen3Config) else ""
    sweep_kind = f"muonh{model_type}"
    base_name = f"exp1570_hidden{hidden_dim}-{sweep_kind}-wsd"

    steps: list[ExecutorStep[TrainLmOnPodConfig]] = []
    for lr in learning_rates:
        optimizer_config = MuonHConfig(
            learning_rate=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            warmup=0.1,
            decay=0.1,
            lr_schedule="linear",
        )

        train_config = _make_train_config(
            scaled_config=scaled_config,
            lr=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            use_mup=False,
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

    return steps


def _lr_sweep_adamh(
    *, hidden_dim: int, learning_rates: Iterable[float], base_model: LlamaConfig
) -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Construct a set of training steps for AdamH across learning rates."""

    scaled_config = _scale_llama_config(base_model, hidden_dim)
    model_type = "-qwen3" if isinstance(base_model, Qwen3Config) else ""
    sweep_kind = f"adamh{model_type}"
    base_name = f"exp1570_hidden{hidden_dim}-{sweep_kind}-wsd"

    steps: list[ExecutorStep[TrainLmOnPodConfig]] = []
    for lr in learning_rates:
        optimizer_config = AdamHConfig(
            learning_rate=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            warmup=0.1,
            decay=0.1,
            lr_schedule="linear",
        )

        train_config = _make_train_config(
            scaled_config=scaled_config,
            lr=lr,
            weight_decay=BASE_WEIGHT_DECAY,
            use_mup=False,
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

    return steps


def build_steps(num_dim_mult=3, base_model=llama_30m):
    hidden_dims = [base_model.hidden_dim * (2**i) for i in range(num_dim_mult)]
    all_steps = []
    for use_mup in (True, False):
        for hidden_dim in hidden_dims:
            all_steps.extend(
                _lr_sweep(hidden_dim=hidden_dim, use_mup=use_mup, learning_rates=LR_CHOICES, base_model=base_model)
            )

    # Add MuonH and AdamH sweeps (non-MuP) for the same hidden sizes
    for hidden_dim in hidden_dims:
        all_steps.extend(_lr_sweep_muonh(hidden_dim=hidden_dim, learning_rates=LR_CHOICES, base_model=base_model))
        all_steps.extend(_lr_sweep_adamh(hidden_dim=hidden_dim, learning_rates=LR_CHOICES, base_model=base_model))

    return all_steps


def build_wd_sweep(base_model=llama_30m) -> list[ExecutorStep[TrainLmOnPodConfig]]:
    """Compare decoupled vs coupled weight decay at two LRs for the base 30M model.

    For each learning rate in [0.006, 0.001] and each weight decay value in
    `INDEPENDENT_WD_CHOICES`, launch two runs:
    - decoupled (AdamW-style): `decoupled_weight_decay=True`.
    - coupled (L2-style): standard coupled decay with `decoupled_weight_decay=False`.
    """

    hidden_dim = base_model.hidden_dim
    scaled_config = _scale_llama_config(base_model, hidden_dim)

    model_type = "-qwen3" if isinstance(base_model, Qwen3Config) else ""
    decoupled_kind = f"decoupled{model_type}"
    coupled_kind = f"coupled{model_type}"
    decoupled_base_name = f"exp1570_hidden{hidden_dim}-{decoupled_kind}-wsd"
    coupled_base_name = f"exp1570_hidden{hidden_dim}-{coupled_kind}-wsd"

    steps: list[ExecutorStep[TrainLmOnPodConfig]] = []
    for fixed_lr in (0.006, 0.001):
        for wd in INDEPENDENT_WD_CHOICES:
            # 1) Decoupled WD (AdamW-style): use decoupled WD
            decoupled_opt = AdamConfig(
                learning_rate=fixed_lr,
                weight_decay=wd,
                use_mup=True,
                warmup=0.1,
                decay=0.1,
                lr_schedule="linear",
            )
            decoupled_opt = dataclasses.replace(decoupled_opt, decoupled_weight_decay=True)

            decoupled_model_cfg = scaled_config

            decoupled_train_cfg = _make_train_config(
                scaled_config=scaled_config,
                lr=fixed_lr,
                weight_decay=wd,
                use_mup=True,
                optimizer_config=decoupled_opt,
            )

            steps.append(
                default_train(
                    name=f"{decoupled_base_name}-lr{_format_lr(fixed_lr)}-wd{_format_lr(wd)}",
                    tokenized=nemotron_mix,
                    model_config=decoupled_model_cfg,
                    train_config=decoupled_train_cfg,
                    tags=("exp1570", f"hidden_dim_{hidden_dim}", decoupled_kind, "wd_sweep", "decoupled"),
                    eval_harness_tasks=(),
                )
            )
            coupled_opt = dataclasses.replace(
                decoupled_opt,
                decoupled_weight_decay=False,
            )

            coupled_train_cfg = _make_train_config(
                scaled_config=scaled_config,
                lr=fixed_lr,
                weight_decay=wd,
                use_mup=True,
                optimizer_config=coupled_opt,
            )

            steps.append(
                default_train(
                    name=f"{coupled_base_name}-lr{_format_lr(fixed_lr)}-wd{_format_lr(wd)}",
                    tokenized=nemotron_mix,
                    model_config=scaled_config,
                    train_config=coupled_train_cfg,
                    tags=("exp1570", f"hidden_dim_{hidden_dim}", coupled_kind, "wd_sweep", "coupled"),
                    eval_harness_tasks=(),
                )
            )

    return steps


if __name__ == "__main__":
    # executor_main(steps=build_steps(4))
    executor_main(steps=build_steps(4, base_model=qwen3_30m) + build_wd_sweep(base_model=qwen3_30m))
