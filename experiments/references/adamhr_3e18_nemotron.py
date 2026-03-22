# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamHR vs AdamH comparison at 3e18 FLOPs on Nemotron mix.

Launches 4 runs using the smallest Completed AdamH candidate at 3e18 FLOPs
(hidden_dim=512, num_layers=6, batch_size=32):
  - adamH baseline at heuristic LR
  - adamHR at LR multipliers 1.0, 1.5, 2.0

See: https://github.com/marin-community/marin/issues/3964
"""

from dataclasses import replace

from levanter.optim import AdamHConfig, AdamHRConfig

from experiments.defaults import default_train
from experiments.pretraining_datasets.nemotron import nemotron_mix
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main

BUDGET = 3e18
SEQ_LEN = 4096
EXPERIMENT_NAME = "adamhr-vs-adamh-3e18"
LR_MULTIPLIERS = (1.0, 1.5, 2.0)


def _get_reference_candidate():
    """Get the smallest candidate at 3e18 FLOPs from the Completed AdamH heuristic."""
    candidates = list(completed_adamh_heuristic.candidates_for_budget(BUDGET, SEQ_LEN))
    return min(candidates, key=lambda c: c.model_config.hidden_dim)


def build_steps() -> list[ExecutorStep]:
    candidate = _get_reference_candidate()
    model_config = candidate.model_config
    base_optimizer = candidate.optimizer_config
    batch_size = candidate.batch_size
    train_steps = candidate.train_steps

    base_train_config_kwargs = dict(
        resources=ResourceConfig.with_tpu("v4-8"),
        train_batch_size=batch_size,
        num_train_steps=train_steps,
        learning_rate=base_optimizer.learning_rate,
        z_loss_weight=1e-7,
    )

    steps: list[ExecutorStep] = []

    # 1. AdamH baseline
    adamh_config = base_optimizer
    adamh_train = default_train(
        name=f"{EXPERIMENT_NAME}-adamh-baseline",
        tokenized=nemotron_mix,
        model_config=model_config,
        train_config=replace(
            _make_simple_train_config(**base_train_config_kwargs, optimizer_config=adamh_config),
        ),
        tags=("adamh", "baseline", f"FLOPs={BUDGET:.0e}"),
        eval_harness_tasks=[],
    )
    steps.append(adamh_train)

    # 2. AdamHR at each LR multiplier
    base_lr = base_optimizer.learning_rate
    base_adam_lr = base_optimizer.adam_lr
    for mult in LR_MULTIPLIERS:
        adamhr_config = AdamHRConfig(
            learning_rate=base_lr * mult,
            adam_lr=base_adam_lr * mult,
            min_lr_ratio=base_optimizer.min_lr_ratio,
            warmup=base_optimizer.warmup,
            beta1=base_optimizer.beta1,
            beta2=base_optimizer.beta2,
            epsilon=base_optimizer.epsilon,
            max_grad_norm=base_optimizer.max_grad_norm,
            lr_schedule=base_optimizer.lr_schedule,
            decay=base_optimizer.decay,
        )
        # Use embedding gain for constrained embeddings
        model_config_with_gain = replace(model_config, use_embedding_gain=True)
        adamhr_train = default_train(
            name=f"{EXPERIMENT_NAME}-adamhr-lr{mult:.1f}x",
            tokenized=nemotron_mix,
            model_config=model_config_with_gain,
            train_config=_make_simple_train_config(
                **base_train_config_kwargs, optimizer_config=adamhr_config
            ),
            tags=("adamhr", f"lr_mult={mult}", f"FLOPs={BUDGET:.0e}"),
            eval_harness_tasks=[],
        )
        steps.append(adamhr_train)

    return steps


def _make_simple_train_config(**kwargs):
    from experiments.simple_train_config import SimpleTrainConfig

    return SimpleTrainConfig(**kwargs)


steps = build_steps()

if __name__ == "__main__":
    executor_main(steps)
