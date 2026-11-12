# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a lowered task's verifier in its real sandbox, with no agent.

The verifier-liveness phase of the no-oracle procedure. Each sampled task gets one
machine through ``rolloutengine.task_machine``, the same path a rollout uses: the
resources are installed and ``environment_requirements.setup_commands`` run before
the body executes. The body then runs the task's own verifier against the
workspace exactly as the setup left it, which is what makes the verdict mean
"this environment can grade", not "this agent solved it".

A verifier that runs writes a reward below 1 for the unsolved workspace. A crash,
timeout, missing reward or positive reward is not a pass.

Run it with the RL Daytona credential installed::

    uv run python -m experiments.post_training.task_curation.no_oracle_liveness \
        --lowered /path/to/lowered-tasks.parquet \
        --manifest /path/to/manifest.json \
        --output /path/to/liveness.jsonl \
        --sample-per-base 20

Snapshot names are written beside each result for cleanup; this repository has
no code path that deletes Daytona snapshots (see ``.agents/ops/daytona.md``).
"""

import argparse
import asyncio
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import HostImage, MachineSpec, NetworkPolicy
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.grading import grade_empty_in_sandbox


def _resources(cpus: int | None, memory_mb: int | None, storage_mb: int | None, gpus: int):
    """The ``Resources`` the Daytona factory builds from the same numbers."""
    from daytona import Resources  # noqa: PLC0415  (optional: only the Daytona placement needs it)

    return Resources(
        cpu=cpus,
        memory=math.ceil(memory_mb / 1024) if memory_mb is not None else None,
        disk=math.ceil(storage_mb / 1024) if storage_mb is not None else None,
        gpu=gpus or None,
    )


VERIFIER_COMMAND = "bash /tests/test.sh"
VERIFIER_TIMEOUT = 900.0
REWARD_PATH = "/logs/verifier/reward.txt"


@dataclass(frozen=True)
class LivenessResult:
    task_id: str
    base: str
    outcome: str
    exit_code: int | None
    reward: float | None
    snapshot: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == "negative_verdict"


def load_specs(path: Path) -> dict[str, dict]:
    specs = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["path", "lowered_task_spec"]):
        columns = batch.to_pydict()
        for task_id, spec in zip(columns["path"], columns["lowered_task_spec"], strict=True):
            specs[task_id] = json.loads(spec)
    return specs


def sample_by_base(specs: dict[str, dict], bases: dict[str, str], per_base: int, seed: int = 0) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(specs):
        grouped[bases.get(task_id, "<unknown>")].append(task_id)
    rng = random.Random(seed)
    chosen: list[str] = []
    for base in sorted(grouped):
        chosen.extend(rng.sample(grouped[base], min(per_base, len(grouped[base]))))
    return chosen


async def run_one(
    spec: dict, base: str, factory=None, lock_url: str | None = None, cgroup_parent: Path | None = None
) -> LivenessResult:
    """Grade an empty submission with the task's own grader, on a fresh grading machine."""
    task = TaskSpec.model_validate(spec["task"])
    if lock_url is not None and getattr(task.grader, "environment", None) is not None:
        # Grade against the lock this run built, so the cohort is qualified on the artifact
        # the pipeline produces rather than on a path recorded somewhere else.
        task = task.model_copy(
            update={
                "grader": task.grader.model_copy(
                    update={"environment": task.grader.environment.model_copy(update={"packages_lock": lock_url})}
                )
            }
        )
    runtime = MachineRuntimeSpec.model_validate(
        {
            **(spec["runtime"].get("verifier_machine") or {}),
            "backend": "daytona",
            "network": NetworkPolicy.ALLOW,
            "gpus": 0,
            "user": None,
            "startup_timeout": 900.0,
            "cleanup_timeout": 120.0,
        }
    )
    requirements = task.environment_requirements
    grader_environment = getattr(task.grader, "environment", None)
    locked = grader_environment is not None and grader_environment.packages_lock is not None
    if locked:
        # A packaged grader runs in the local bubblewrap runtime; no sandbox backend and
        # no snapshot identity are involved, so the Daytona pieces are never imported.
        snapshot = ""
        spec_machine = MachineSpec(
            source=HostImage(),
            workdir=requirements.working_directory or "/workspace",
            network=NetworkPolicy.ALLOW,
            cpus=runtime.cpus,
            memory_mb=runtime.memory_mb,
            # The local backend provides no disk quota; declaring one is refused.
            storage_mb=None,
        )
        # A worker whose own cgroup is threaded cannot host the grading cgroups
        # (cgroup.procs is unwritable there), so an explicit parent overrides discovery.
        grading_factory = LocalMachineFactory(cgroup_parent=cgroup_parent) if cgroup_parent else LocalMachineFactory()
    else:
        from harbor.environments.daytona import machine_spec  # noqa: PLC0415
        from harbor_config.models.task.config import EnvironmentConfig  # noqa: PLC0415
        from shellbox.backends.daytona.machine import (  # noqa: PLC0415
            DaytonaMachineFactory,
            snapshot_identity,
        )
        from shellbox.image import RegistryImage  # noqa: PLC0415

        snapshot = snapshot_identity(
            RegistryImage(requirements.docker_image),
            _resources(runtime.cpus, runtime.memory_mb, runtime.storage_mb, runtime.gpus),
        )
        spec_machine = machine_spec(
            EnvironmentConfig(
                docker_image=requirements.docker_image,
                workdir=requirements.working_directory,
                cpus=runtime.cpus,
                memory_mb=runtime.memory_mb,
                storage_mb=runtime.storage_mb,
            ),
            Path("/nonexistent"),
            {},
            cpus=runtime.cpus,
            memory_mb=runtime.memory_mb,
            storage_mb=runtime.storage_mb,
        )
        grading_factory = factory or DaytonaMachineFactory()
    try:
        result = await grade_empty_in_sandbox(task, grading_factory, spec_machine, timeout=VERIFIER_TIMEOUT)
    except TimeoutError as error:
        return LivenessResult(task.id, base, "timeout", None, None, snapshot, str(error))
    except Exception as error:
        return LivenessResult(
            task.id, base, "infra_error", None, None, snapshot, f"{type(error).__name__}: {error}"[:500]
        )
    if result.status is not Outcome.SUCCESS:
        return LivenessResult(
            task.id, base, "infra_error", None, result.reward, snapshot, (result.error or str(result.status))[:300]
        )
    if result.reward is None:
        return LivenessResult(task.id, base, "missing_reward", None, None, snapshot)
    outcome = "negative_verdict" if result.reward < 1.0 else "positive_reward"
    return LivenessResult(task.id, base, outcome, 0, result.reward, snapshot)


async def run(
    specs: dict[str, dict],
    bases: dict[str, str],
    task_ids: list[str],
    output: Path,
    lock_url: str | None = None,
    cgroup_parent: Path | None = None,
) -> None:
    with output.open("a") as handle:
        for index, task_id in enumerate(task_ids, start=1):
            result = await run_one(
                specs[task_id], bases.get(task_id, "<unknown>"), lock_url=lock_url, cgroup_parent=cgroup_parent
            )
            handle.write(json.dumps(asdict(result)) + "\n")
            handle.flush()
            print(f"[{index}/{len(task_ids)}] {task_id}: {result.outcome}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowered", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="rebuild manifest naming each task's base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-base", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="cap the sample for a pilot run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lock-url", help="grade packaged graders against this built lock")
    parser.add_argument(
        "--cgroup-parent",
        type=Path,
        help="cgroup v2 directory for the local placement when the worker's own cgroup cannot host processes",
    )
    args = parser.parse_args()
    specs = load_specs(args.lowered)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        bases = {
            row["task_id"]: row["base_ref"] for row in manifest["tasks"] if row.get("status", "").startswith("rebuilt")
        }
    else:
        # A published cohort carries each task's own base in its spec.
        bases = {task_id: spec["task"]["environment_requirements"]["docker_image"] for task_id, spec in specs.items()}
    task_ids = sample_by_base(specs, bases, args.sample_per_base, args.seed)
    if args.limit:
        task_ids = task_ids[: args.limit]
    print(f"sampled {len(task_ids)} of {len(specs)} tasks", flush=True)
    asyncio.run(run(specs, bases, task_ids, args.output, args.lock_url, args.cgroup_parent))


if __name__ == "__main__":
    main()
