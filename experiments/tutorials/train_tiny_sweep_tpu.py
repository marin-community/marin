# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial: hyper-parameter sweep over a tiny model on TinyStories using TPU hardware.

The script defines a 9-element learning-rate by weight-decay grid. Each grid point
becomes one ``SweepTarget`` whose payload is a fully-built ``TrainingPlan``;
``claim_and_run`` races workers across processes / machines for unclaimed targets
via the executor's distributed ``step_lock``.

Submission model: a single ``iris job run`` invocation submits ONE Iris job
whose ``ResourceConfig`` requests ``NUM_WORKERS`` CPU-only replicas (tasks).
Each replica runs the same entrypoint, calls ``claim_and_run``, and loops over
the target list — competing with its peers for the next unclaimed target via
``step_lock``. Whichever worker wins a target submits a child Iris training
job (TPU) via ``run_train`` and blocks on it; when that returns, the worker
moves on to the next target.

The number of workers is independent of the sweep size: workers run in a loop
until the target list is exhausted. ``NUM_WORKERS = 3`` against 9 grid points
gives roughly three trials per worker; bump it for more parallelism, lower it
to be polite to the TPU pool. ``SWEEP_NAME`` is the stable lock-path key —
bump it to start a fresh sweep over the same grid.
"""
import dataclasses
import os

from fray import client as fray_client
from fray.cluster import ResourceConfig
from fray.types import Entrypoint, JobRequest, create_environment
from rigging.filesystem import marin_prefix

from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run

from experiments.defaults import prepare_train, run_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu("v4-8")
EVALS = CORE_TASKS

# Stable sweep identifier — derives the lock root so workers from different
# `iris job run` invocations converge on the same target set. Bump for a fresh sweep.
SWEEP_NAME = "train-tiny-sweep"

# Number of CPU-only sweep coordinator tasks to run in parallel. Each worker
# loops over targets and submits a child TPU training job per claim. The grid
# below has 9 points; 3 workers gives ~3 trials per worker. Independent of the
# sweep size — workers exit when no unclaimed targets remain.
NUM_WORKERS = 3

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

plans = []
for config in sweep_configs:
    lr, wd = config.learning_rate, config.weight_decay
    plan = prepare_train(
        # Marin will automatically create unique ids for runs b/c the model_config is versioned;
        # the explicit name keeps each run identifiable in W&B.
        name=f"tutorial-slimpajama_6b-30m-sweep-lr{lr}-wd{wd}",
        tokenized=tokenized["slimpajama_6b"],
        model_config=versioned(llama_30m),
        train_config=config,
        # wandb tags
        tags=["llama", "30m", "slimpajama_6b", "tutorial", "sweep", "test20251117"],
        eval_harness_tasks=CORE_TASKS,
    )
    plans.append(plan)

# `prepare_train` names plans `checkpoints/<name>`; the slash is not a valid
# path component for the lock root, so flatten it. The `(name, lr, wd)` tuple is
# unique within the sweep so the resulting target_id is unique too.
targets = [SweepTarget(target_id=plan.name.replace("/", "-"), config=plan) for plan in plans]


def _run_one(target: SweepTarget) -> None:
    """Submit one sweep trial as a child Iris training job and block on it."""
    run_train(target.config)


def _sweep_worker_entrypoint() -> None:
    """One CPU sweep coordinator: loop, claim a target, dispatch a child TPU job."""
    sweep_root = os.path.join(marin_prefix(), "sweeps", SWEEP_NAME)
    claim_and_run(sweep_root, targets, _run_one)


if __name__ == "__main__":
    # Single Iris submission, NUM_WORKERS gang-scheduled CPU replicas. Each
    # replica runs `_sweep_worker_entrypoint` and competes with its peers via
    # `step_lock` for the next unclaimed sweep target.
    client = fray_client.current_client()
    handle = client.submit(
        JobRequest(
            name=SWEEP_NAME,
            entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint),
            resources=ResourceConfig.with_cpu(replicas=NUM_WORKERS),
            environment=create_environment(),
        )
    )
    handle.wait(raise_on_failure=True)
