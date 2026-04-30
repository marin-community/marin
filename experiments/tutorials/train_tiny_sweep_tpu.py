# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial: LR/WD hyper-parameter sweep over a tiny model on TPU using TinyStories.

A single Iris job submits ``NUM_WORKERS`` CPU-only coordinator replicas; each
races on ``step_lock`` to claim grid targets and dispatch child TPU training
jobs. ``SWEEP_NAME`` is the stable lock-path key — bump it to start a fresh
sweep over the same grid.
"""
import dataclasses
import os
from dataclasses import dataclass

import levanter.main.train_lm as levanter_train_lm
from fray import client as fray_client
from fray.cluster import ResourceConfig
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.main.train_lm import TrainLmConfig
from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run
from rigging.filesystem import marin_prefix

from experiments.defaults import _submit_train_job, prepare_lm_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu("v4-8")
EVALS = CORE_TASKS

# Stable sweep identifier — derives the lock root so workers from different
# `iris job run` invocations converge on the same target set. Bump for a fresh sweep.
SWEEP_NAME = "train-tiny-sweep"

NUM_WORKERS = 3  # CPU coordinator replicas; workers exit when all targets are claimed.

small_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=10000,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
)

sweep_configs = [
    dataclasses.replace(
        small_train_config,
        learning_rate=lr,
        weight_decay=wd,
    )
    for lr in [3e-4, 6e-4, 1e-3]
    for wd in [0.0, 0.1, 0.2]
]


@dataclass(frozen=True)
class SweepTrial:
    name: str
    inner_config: TrainLmConfig
    output_path: str
    resources: ResourceConfig
    env_vars: dict[str, str]


# Build all trials at submission time so coordinators do no config work.
trials = []
for sc in sweep_configs:
    _name = f"tutorial-slimpajama_6b-30m-sweep-lr{sc.learning_rate}-wd{sc.weight_decay}"
    _job_name, _inner_config, _output_path = prepare_lm_train(
        name=_name,
        tokenized=tokenized["slimpajama_6b"],
        model_config=versioned(llama_30m),
        train_config=sc,
        tags=["llama", "30m", "slimpajama_6b", "tutorial", "sweep", "test20251117"],
        eval_harness_tasks=CORE_TASKS,
    )
    trials.append(
        SweepTrial(
            name=_job_name,
            inner_config=_inner_config,
            output_path=_output_path,
            resources=sc.resources,
            env_vars=dict(sc.env_vars or {}),
        )
    )

targets = [SweepTarget(target_id=t.name, config=t) for t in trials]


def _run_one(target: SweepTarget) -> None:
    """Submit one sweep trial as a child Iris training job and block on it."""
    trial: SweepTrial = target.config
    _submit_train_job(
        name=trial.name,
        train_config=trial.inner_config,
        resources=trial.resources,
        env_vars=trial.env_vars,
        worker_fn=levanter_train_lm.main,
    )


def _sweep_worker_entrypoint(sweep_root: str) -> None:
    """One CPU sweep coordinator: loop, claim a target, dispatch a child TPU job.

    ``sweep_root`` is resolved once in the submitter (where ``marin_prefix()``
    reflects the user's region) and baked into the entrypoint args. All N CPU
    replicas thus contend on the same lock namespace regardless of where Iris
    schedules them.
    """
    claim_and_run(sweep_root, targets, _run_one)


if __name__ == "__main__":
    sweep_root = os.path.join(marin_prefix(), "sweeps", SWEEP_NAME)
    client = fray_client.current_client()
    handle = client.submit(
        JobRequest(
            name=SWEEP_NAME,
            entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[sweep_root]),
            resources=ResourceConfig.with_cpu(replicas=NUM_WORKERS),
            environment=create_environment(),
        )
    )
    handle.wait(raise_on_failure=True)
