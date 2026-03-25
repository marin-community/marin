# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonHR vs MuonH comparison at 3e18 FLOPs on Nemotron mix.

Launches 4 runs using the smallest Completed AdamH candidate shape at 3e18 FLOPs
(hidden_dim=512, num_layers=6, batch_size=32) with MuonH-family optimizers:
  - muonH baseline at reference LR
  - muonHR at LR multipliers 1.0, 1.5, 2.0

The model shape comes from the AdamH heuristic (architecture is optimizer-agnostic);
MuonH hyperparameters are taken from the 130M Qwen3 speedrun reference point.

See: https://github.com/marin-community/marin/issues/4110
"""

from dataclasses import replace

from levanter.optim import MuonHConfig, MuonHRConfig

from experiments.defaults import default_train
from experiments.pretraining_datasets.nemotron import nemotron_mix
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main

BUDGET = 3e18
SEQ_LEN = 4096
EXPERIMENT_NAME = "muonhr-vs-muonh-3e18"
LR_MULTIPLIERS = (1.0, 1.5, 2.0)

# MuonH reference hyperparameters from the 130M Qwen3 speedrun.
MUONH_LR = 0.02
MUONH_ADAM_LR = 0.008
MUONH_MOMENTUM = 0.95
MUONH_BETA1 = 0.9
MUONH_BETA2 = 0.98
MUONH_EPSILON = 1e-15
MUONH_MUON_EPSILON = 1e-5
MUONH_MAX_GRAD_NORM = 1.0
MUONH_WARMUP = 1000


def _get_reference_candidate():
    """Get the smallest candidate at 3e18 FLOPs from the Completed AdamH heuristic."""
    candidates = list(completed_adamh_heuristic.candidates_for_budget(BUDGET, SEQ_LEN))
    return min(candidates, key=lambda c: c.model_config.hidden_dim)


def build_steps() -> list[ExecutorStep]:
    candidate = _get_reference_candidate()
    model_config = candidate.model_config
    batch_size = candidate.batch_size
    train_steps = candidate.train_steps

    base_train_config_kwargs = dict(
        resources=ResourceConfig.with_tpu("v4-8"),
        train_batch_size=batch_size,
        num_train_steps=train_steps,
        learning_rate=MUONH_LR,
        z_loss_weight=1e-7,
    )

    steps: list[ExecutorStep] = []

    # 1. MuonH baseline
    muonh_config = MuonHConfig(
        learning_rate=MUONH_LR,
        adam_lr=MUONH_ADAM_LR,
        momentum=MUONH_MOMENTUM,
        beta1=MUONH_BETA1,
        beta2=MUONH_BETA2,
        epsilon=MUONH_EPSILON,
        muon_epsilon=MUONH_MUON_EPSILON,
        max_grad_norm=MUONH_MAX_GRAD_NORM,
        warmup=MUONH_WARMUP,
        min_lr_ratio=0,
    )
    muonh_train = default_train(
        name=f"{EXPERIMENT_NAME}-muonh-baseline",
        tokenized=nemotron_mix,
        model_config=model_config,
        train_config=_make_simple_train_config(**base_train_config_kwargs, optimizer_config=muonh_config),
        tags=("muonh", "baseline", f"FLOPs={BUDGET:.0e}"),
        eval_harness_tasks=[],
    )
    steps.append(muonh_train)

    # 2. MuonHR at each LR multiplier
    for mult in LR_MULTIPLIERS:
        muonhr_config = MuonHRConfig(
            learning_rate=MUONH_LR * mult,
            adam_lr=MUONH_ADAM_LR * mult,
            momentum=MUONH_MOMENTUM,
            beta1=MUONH_BETA1,
            beta2=MUONH_BETA2,
            epsilon=MUONH_EPSILON,
            muon_epsilon=MUONH_MUON_EPSILON,
            max_grad_norm=MUONH_MAX_GRAD_NORM,
            warmup=MUONH_WARMUP,
            min_lr_ratio=0,
        )
        model_config_with_gain = replace(model_config, use_embedding_gain=True)
        muonhr_train = default_train(
            name=f"{EXPERIMENT_NAME}-muonhr-lr{mult:.1f}x",
            tokenized=nemotron_mix,
            model_config=model_config_with_gain,
            train_config=_make_simple_train_config(**base_train_config_kwargs, optimizer_config=muonhr_config),
            tags=("muonhr", f"lr_mult={mult}", f"FLOPs={BUDGET:.0e}"),
            eval_harness_tasks=[],
        )
        steps.append(muonhr_train)

    return steps


def _make_simple_train_config(**kwargs):
    from experiments.simple_train_config import SimpleTrainConfig

    return SimpleTrainConfig(**kwargs)


steps = build_steps()

if __name__ == "__main__":
    executor_main(steps)
