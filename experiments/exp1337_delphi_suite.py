# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scaling Ladder Analysis for Completed AdamH v6 heuristic on Nemotron.

This experiment runs scaling ladder analysis on the isoflop training sweeps
for Nemotron using the CompletedAdamH v6 heuristic (sqrt batch LR, no /H on adam_lr).

The scaling ladder:
1. Fits scaling laws from IsoFLOP sweep data to find compute-optimal configurations
2. Generates visualization plots (isoflop curves and scaling fit plots)
3. Optionally trains compute-optimal models at larger target budgets
"""

import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LMMixtureDatasetConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.scaling_laws import ScalingFit, predict_optimal_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.defaults import default_validation_sets
from experiments.isoflop_sweep import (
    MARIN_SCALING_SUITES,
    IsoFlopAnalysisConfig,
    nemotron_mix,
    run_isoflop_analysis_step,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get training steps from the isoflop sweep
adamh_training, _ = MARIN_SCALING_SUITES["nemotron-completed-adamh"]

# --- Configuration ---
TARGET_BUDGETS: dict[float, tuple[str, int]] = {
    1e21: ("v4-128", 512),
    1e22: ("v4-512", 1024),
    1e23: ("v4-1024", 2048),
}
EXPERIMENT_NAME = "adamh-scaling-ladder-nemotron"
LABEL = "adamh_scaling_v6"
SEQ_LEN = 4096


@dataclass(frozen=True)
class OptimalTrainingConfig:
    """Config for training a compute-optimal model based on scaling law analysis."""

    analysis_output_path: str
    target_budget: float
    tpu_type: str
    batch_size: int
    label: str
    output_path: str
    tokenized: LMMixtureDatasetConfig
    seed: int = 0
    validation_configs: dict[str, DatasetComponent] | None = None


def run_optimal_training(config: OptimalTrainingConfig) -> None:
    """Run compute-optimal training at the given budget."""
    result_path = os.path.join(config.analysis_output_path, "isoflop_analysis_result.json")
    fs, _, _ = fsspec.get_fs_token_paths(result_path)

    with fs.open(result_path, "r") as f:
        analysis_result = json.load(f)

    scaling_fits: dict[str, ScalingFit] = {}
    for key, value in analysis_result["scaling_fits"].items():
        if len(value) != 2:
            raise ValueError(f"Expected 2 scaling fit values for '{key}', got {len(value)}")
        scaling_fits[key] = ScalingFit(float(value[0]), float(value[1]))

    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=config.target_budget,
        label=config.label,
        heuristic=completed_adamh_heuristic,
        seq_len=SEQ_LEN,
    )

    if candidate is None:
        raise RuntimeError(
            f"Could not find optimal config for budget {config.target_budget:.2e} and label '{config.label}'"
        )

    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    hidden_dim = candidate.model_config.hidden_dim
    tpu_type = config.tpu_type
    cores = int(tpu_type.split("-")[1])
    chips = cores // 2

    # Compute minimum TP needed so hidden_dim is divisible by data-axis size
    tp = 1
    while hidden_dim % (chips // tp) != 0:
        tp *= 2

    batch_size = config.batch_size
    tokens = candidate.tokens
    train_steps = round(tokens / (batch_size * SEQ_LEN))

    optimizer_config = completed_adamh_heuristic.build_optimizer_config(batch_size, tokens)
    candidate = replace(candidate, batch_size=batch_size, train_steps=train_steps, optimizer_config=optimizer_config)

    print(
        f"Optimal config for {config.target_budget:.2e} FLOPs:\n"
        f"  hidden_dim={hidden_dim}, layers={candidate.model_config.num_layers}\n"
        f"  params={params:.2e}, tokens={candidate.tokens:.2e}\n"
        f"  batch_size={candidate.batch_size}, train_steps={candidate.train_steps}\n"
        f"  tpu_type={tpu_type}, tp={tp}"
    )

    model_config = candidate.model_config

    data = config.tokenized
    if config.validation_configs:
        new_components = {
            **data.components,
            **{k: v for k, v in config.validation_configs.items() if k not in data.components},
        }
        if isinstance(data.train_weights, dict):
            new_weights = {
                **data.train_weights,
                **{name: 0.0 for name in config.validation_configs if name not in data.train_weights},
            }
        else:
            new_weights = [
                (step_idx, {**weights, **{name: 0.0 for name in config.validation_configs if name not in weights}})
                for step_idx, weights in data.train_weights
            ]
        data = replace(data, components=new_components, train_weights=new_weights)

    inner_config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    "optimal-training",
                    "completed-adamh",
                    f"FLOPs={config.target_budget:.1e}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                    f"seed={config.seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=candidate.batch_size,
            per_device_parallelism=-1,
            num_train_steps=candidate.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tp},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=config.seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=model_config,
        optimizer=candidate.optimizer_config,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu(tpu_type),
        output_path=config.output_path,
    )

    logger.info(f"Launching training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# --- Step 1: IsoFLOP Analysis ---
analysis_step = ExecutorStep(
    name=f"{EXPERIMENT_NAME}-analysis",
    fn=run_isoflop_analysis_step,
    config=IsoFlopAnalysisConfig(
        training_runs=[r.as_input_name() for r in adamh_training],
        output_path=this_output_path(),
    ),
)

# --- Create validation configs ---
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}

# --- Step 2: Optimal Training Runs ---
# Seeds per budget: 1e21 and 1e22 get 3 seeds (0, 42, 62746); 1e23 gets seed 0 only
SEEDS_PER_BUDGET: dict[float, list[int]] = {
    1e21: [0, 42, 62746],
    1e22: [0, 42, 62746],
    1e23: [0],
}

optimal_runs: list[ExecutorStep] = []
for budget, (tpu_type, batch_size) in TARGET_BUDGETS.items():
    for seed in SEEDS_PER_BUDGET[budget]:
        suffix = f"-seed{seed}" if seed != 0 else ""
        step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}-optimal-{budget:.0e}-v5{suffix}",
            fn=run_optimal_training,
            config=OptimalTrainingConfig(
                analysis_output_path=analysis_step.as_input_name(),
                target_budget=budget,
                tpu_type=tpu_type,
                batch_size=batch_size,
                label=LABEL,
                output_path=this_output_path(),
                tokenized=nemotron_mix,
                seed=seed,
                validation_configs=validation_configs,
            ),
        )
        optimal_runs.append(step)

all_steps = [analysis_step, *optimal_runs]

if __name__ == "__main__":
    executor_main(steps=all_steps)
