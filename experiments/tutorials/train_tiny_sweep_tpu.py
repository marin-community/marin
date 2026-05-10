# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial: LR/WD hyper-parameter sweep over a tiny model on TPU using TinyStories.

Submits ``NUM_WORKERS`` independent TPU jobs; each worker races on
``step_lock`` to claim grid targets and trains inline on its own TPU. There is
no CPU coordinator. ``SWEEP_NAME`` is the stable lock-path key — bump it to
start a fresh sweep over the same grid.
"""
import dataclasses
from dataclasses import dataclass

from fray import client as fray_client
from fray.cluster import ResourceConfig
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.main.train_lm import TrainLmConfig
from marin.execution.executor import versioned
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.training.training import extras_for_resources, resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu("v4-8")
EVALS = CORE_TASKS

# Stable sweep identifier — derives the lock root so workers from different
# `iris job run` invocations converge on the same target set. Bump for a fresh sweep.
SWEEP_NAME = "train-tiny-sweep"

# Sweep lock root lives in a fixed region (matches MARIN_REMOTE_STATE_DIR
# in iris). Workers in any region contend on the same path, and re-submitting
# the same sweep from a different region resumes against the same locks
# instead of starting a new claim namespace.
SWEEP_ROOT = f"gs://marin-us-central2/sweeps/{SWEEP_NAME}"

# Each TPU worker claims one target at a time and trains inline on its own
# TPU, so NUM_WORKERS sets the parallelism — three trials run concurrently
# here. Workers exit when no unclaimed targets remain.
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


@dataclass(frozen=True)
class SweepTrial:
    name: str
    raw_config: TrainLmConfig


# Build all trials at submission time so workers do no config work. Configs
# carry placeholders (OutputName, InputName) until resolved on the worker, so
# checkpoint paths land in the *worker's* region after a cross-region
# preemption.
trials = []
for sc in sweep_configs:
    # Marin will automatically create unique ids for runs b/c the model_config is versioned
    # however, we can give each run a unique name for easier identification
    _name = f"tutorial-slimpajama_6b-30m-sweep-lr{sc.learning_rate}-wd{sc.weight_decay}"
    _job_name, _raw_config = prepare_lm_train(
        name=_name,
        tokenized=tokenized["slimpajama_6b"],
        model_config=versioned(llama_30m),
        train_config=sc,
        tags=["llama", "30m", "slimpajama_6b", "tutorial", "sweep", "test20251117"],
        eval_harness_tasks=CORE_TASKS,
    )
    trials.append(SweepTrial(name=_job_name, raw_config=_raw_config))

targets = [SweepTarget(target_id=t.name, config=t) for t in trials]


def _run_one(target: SweepTarget) -> None:
    """Resolve the trial's config under this worker's region and train inline."""
    trial: SweepTrial = target.config
    _run_training_on_worker(
        name=trial.name,
        raw_config=trial.raw_config,
        override_output_path=None,
        resources=RESOURCES,
    )


def _sweep_worker_entrypoint(sweep_root: str) -> None:
    """One TPU sweep worker: loop, claim a target, train inline.

    ``sweep_root`` is the canonical (region-pinned) lock path baked into the
    entrypoint args. All TPU replicas — across regions, across resubmissions —
    contend on the same lock namespace regardless of where Iris schedules
    them.
    """
    claim_and_run(sweep_root, targets, _run_one)


if __name__ == "__main__":
    client = fray_client.current_client()

    env = resolve_training_env(base_env=None, resources=RESOURCES)
    handles = []
    for i in range(NUM_WORKERS):
        handle = client.submit(
            JobRequest(
                name=f"{SWEEP_NAME}-{i}",
                entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[SWEEP_ROOT]),
                resources=RESOURCES,
                environment=create_environment(env_vars=env, extras=extras_for_resources(RESOURCES)),
            )
        )
        handles.append(handle)
    for h in handles:
        h.wait(raise_on_failure=True)
