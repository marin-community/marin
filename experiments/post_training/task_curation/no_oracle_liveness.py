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
from rolloutengine.machines import task_machine
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory, snapshot_identity
from shellbox.machine import Command, ExitReason, NetworkPolicy
from shellbox.image import RegistryImage
from daytona import Resources
from taskcompendium.models import TaskSpec

def _resources(cpus: int | None, memory_mb: int | None, storage_mb: int | None, gpus: int) -> Resources:
    """The ``Resources`` the Daytona factory builds from the same numbers."""
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


async def run_one(spec: dict, base: str, factory: DaytonaMachineFactory) -> LivenessResult:
    task = TaskSpec.model_validate(spec["task"])
    runtime = MachineRuntimeSpec.model_validate(
        spec["runtime"]["task_machine"] | {"backend": "daytona", "network": NetworkPolicy.ALLOW}
    )
    snapshot = snapshot_identity(
        RegistryImage(task.environment_requirements.docker_image),
        _resources(runtime.cpus, runtime.memory_mb, runtime.storage_mb, runtime.gpus),
    )
    workdir = task.environment_requirements.working_directory or "/app"
    try:
        async with task_machine(task, factory, runtime, cleanup_timeout=runtime.cleanup_timeout or 120.0) as machine:
            await machine.run(Command(("mkdir", "-p", "/logs/verifier"), timeout=60))
            result = await machine.run(Command(("bash", "-lc", VERIFIER_COMMAND), cwd=workdir, timeout=VERIFIER_TIMEOUT))
            reward = None
            read = await machine.run(Command(("cat", REWARD_PATH), timeout=60))
            if read.exit_code == 0:
                text = read.stdout.decode(errors="replace").strip().splitlines()
                reward = float(text[-1]) if text else None
    except TimeoutError as error:
        return LivenessResult(task.id, base, "timeout", None, None, snapshot, str(error))
    except Exception as error:  # noqa: BLE001
        return LivenessResult(task.id, base, "infra_error", None, None, snapshot, f"{type(error).__name__}: {error}"[:500])
    if result.reason == ExitReason.TIMED_OUT:
        return LivenessResult(task.id, base, "timeout", None, reward, snapshot)
    if result.exit_code != 0:
        return LivenessResult(task.id, base, "infra_error", result.exit_code, reward, snapshot, result.stderr.decode(errors="replace")[-300:])
    if reward is None:
        return LivenessResult(task.id, base, "missing_reward", result.exit_code, None, snapshot)
    outcome = "negative_verdict" if reward < 1.0 else "positive_reward"
    return LivenessResult(task.id, base, outcome, result.exit_code, reward, snapshot)


async def run(specs: dict[str, dict], bases: dict[str, str], task_ids: list[str], output: Path) -> None:
    factory = DaytonaMachineFactory()
    with output.open("a") as handle:
        for index, task_id in enumerate(task_ids, start=1):
            result = await run_one(specs[task_id], bases.get(task_id, "<unknown>"), factory)
            handle.write(json.dumps(asdict(result)) + "\n")
            handle.flush()
            print(f"[{index}/{len(task_ids)}] {task_id}: {result.outcome}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowered", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-base", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="cap the sample for a pilot run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    specs = load_specs(args.lowered)
    manifest = json.loads(args.manifest.read_text())
    bases = {row["task_id"]: row["base_ref"] for row in manifest["tasks"] if row.get("status", "").startswith("rebuilt")}
    task_ids = sample_by_base(specs, bases, args.sample_per_base, args.seed)
    if args.limit:
        task_ids = task_ids[: args.limit]
    print(f"sampled {len(task_ids)} of {len(specs)} tasks", flush=True)
    asyncio.run(run(specs, bases, task_ids, args.output))


if __name__ == "__main__":
    main()
