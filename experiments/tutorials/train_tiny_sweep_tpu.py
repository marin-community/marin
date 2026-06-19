# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial: LR/WD hyper-parameter sweep over a tiny model on TPU using TinyStories.

Submits ``NUM_WORKERS`` independent TPU jobs; each worker races on
``step_lock`` to claim grid targets and trains inline on its own TPU. There is
no CPU coordinator. ``SWEEP_NAME`` is the stable lock-path key — bump it to
start a fresh sweep over the same grid.

Run it directly from a dev box; ``--tpu_type`` / ``--region`` pick the
accelerator (default ``v4-8``)::

    uv run python experiments/tutorials/train_tiny_sweep_tpu.py --cluster=marin

The default ``v4-8`` is a single-host slice, so each worker is one process. On a
multi-host slice the whole gang acts as one worker: its leader (task 0) claims
targets and the other hosts train alongside it — see
``marin.execution.sweep_coordination``. A multi-host sweep must also pass
``ports=["actor"]`` in its ``JobRequest`` so the leader's coordination actor is
reachable by its followers.
"""
import dataclasses
from dataclasses import dataclass

import draccus
from fray import client as fray_client
from fray.cluster import ResourceConfig
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.main.train_lm import TrainLmConfig
from marin.execution.sweep import SweepTarget, claim_and_run
from marin.execution.types import versioned
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.launch import LaunchConfig, launch, override_resources
from experiments.llama import llama_30m
from experiments.pretraining_datasets.simple import tokenized
from experiments.simple_train_config import SimpleTrainConfig

# Default accelerator; --tpu_type / --region override it (see override_resources).
DEFAULT_TPU_TYPE = "v4-8"

# Stable sweep identifier — derives the lock root so workers from different
# invocations converge on the same target set. Bump for a fresh sweep.
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


@dataclass(frozen=True)
class SweepTrial:
    name: str
    raw_config: TrainLmConfig


def build_targets(resources: ResourceConfig) -> list[SweepTarget]:
    """Build the LR/WD grid as placeholder-bearing trial configs for ``resources``.

    Configs carry placeholders (OutputName, InputName) until resolved on the
    worker, so checkpoint paths land in the *worker's* region after a
    cross-region preemption. All workers in a submission call this with the same
    ``resources`` and thus agree on the same target set and hardware.
    """
    base = SimpleTrainConfig(
        resources=resources,
        train_batch_size=128,
        num_train_steps=10000,
        learning_rate=6e-4,
        weight_decay=0.1,
    )
    targets = []
    for lr in [3e-4, 6e-4, 1e-3]:
        for wd in [0.0, 0.1, 0.2]:
            train_config = dataclasses.replace(base, learning_rate=lr, weight_decay=wd)
            # A human-readable name per trial; Marin versions the run via model_config.
            name = f"tutorial-slimpajama_6b-30m-sweep-lr{lr}-wd{wd}"
            job_name, raw_config = prepare_lm_train(
                name=name,
                tokenized=tokenized["slimpajama_6b"],
                model_config=versioned(llama_30m),
                train_config=train_config,
                tags=["llama", "30m", "slimpajama_6b", "tutorial", "sweep", "test20251117"],
                eval_harness_tasks=CORE_TASKS,
            )
            targets.append(SweepTarget(target_id=job_name, config=SweepTrial(name=job_name, raw_config=raw_config)))
    return targets


def _run_one(target: SweepTarget, resources: ResourceConfig) -> None:
    """Resolve the trial's config under this worker's region and train inline."""
    trial: SweepTrial = target.config
    _run_training_on_worker(
        name=trial.name,
        raw_config=trial.raw_config,
        override_output_path=None,
        resources=resources,
    )


def _sweep_worker_entrypoint(sweep_root: str, resources: ResourceConfig) -> None:
    """One TPU sweep worker: build the grid, then loop claiming and training targets.

    ``sweep_root`` is the canonical (region-pinned) lock path and ``resources``
    is the hardware the job was scheduled on — both baked into the entrypoint
    args. All TPU replicas — across regions, across resubmissions — contend on
    the same lock namespace regardless of where Iris schedules them.
    """
    targets = build_targets(resources)
    claim_and_run(sweep_root, targets, lambda target: _run_one(target, resources))


def _launch_sweep(resources: ResourceConfig) -> None:
    """Submit ``NUM_WORKERS`` independent TPU sweep workers and block until they finish.

    Runs on the coordinator (or locally): each worker races on ``step_lock`` to
    claim grid targets and trains inline on its own TPU.
    """
    client = fray_client.current_client()
    env = resolve_training_env(base_env=None, resources=resources)
    handles = []
    for i in range(NUM_WORKERS):
        handle = client.submit(
            JobRequest(
                name=f"{SWEEP_NAME}-{i}",
                entrypoint=Entrypoint.from_callable(_sweep_worker_entrypoint, args=[SWEEP_ROOT, resources]),
                resources=resources,
                environment=create_environment(env_vars=env, extras=extras_for_resources(resources)),
            )
        )
        handles.append(handle)
    for h in handles:
        h.wait(raise_on_failure=True)


@draccus.wrap()
def main(config: LaunchConfig):
    resources = override_resources(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE), config)
    launch(config, _launch_sweep, resources)


if __name__ == "__main__":
    main()
