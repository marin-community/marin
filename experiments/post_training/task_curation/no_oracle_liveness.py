# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a lowered task's verifier in its real sandbox, with no agent.

The verifier-liveness phase of the no-oracle procedure grades an empty submission
in each task's declared environment. Lock-based graders run locally; other
graders use a sandbox. No agent solution is executed.

A qualifying rejection requires exit code zero, reward below one, and a positive
structured executed-test count in the grader result. Numeric reward files without
test execution evidence report ``unverified_execution``. The published runtime
shape and grader environment are used without lock substitution.

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
import base64
import json
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pyarrow.parquet as pq
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import HostImage, MachineSpec, NetworkPolicy
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage
from taskcompendium.runtime.grading import grade_empty_in_sandbox, grade_in_sandbox

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
    tests_executed: int = 0

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
    spec: dict,
    base: str,
    factory=None,
    cgroup_parent: Path | None = None,
    submission_files: dict[str, bytes] | None = None,
) -> LivenessResult:
    """Grade an empty submission with the task's own grader, on a fresh grading machine."""
    with tempfile.TemporaryDirectory(prefix="liveness-build-") as context:
        task = TaskSpec.model_validate(spec["task"])
        runtime = MachineRuntimeSpec.model_validate(
            {
                "backend": "daytona",
                "network": NetworkPolicy.ALLOW,
                "gpus": 0,
                "user": None,
                "startup_timeout": 900.0,
                "cleanup_timeout": 120.0,
                **(spec["runtime"].get("verifier_machine") or {}),
            }
        )
        requirements = getattr(task.grader, "environment", None)
        if requirements is None:
            return LivenessResult(task.id, base, "infra_error", None, None, "", "Grader declares no sandbox environment")
        if runtime.gpus:
            return LivenessResult(
                task.id, base, "infra_error", None, None, "", "CPU qualification cannot provision GPUs"
            )
        if runtime.user is not None:
            return LivenessResult(
                task.id, base, "infra_error", None, None, "", "Explicit runtime users require rollout execution"
            )
        locked = requirements.packages_lock is not None
        if locked:
            # A packaged grader runs in the local bubblewrap runtime; no sandbox backend and
            # no snapshot identity are involved, so the Daytona pieces are never imported.
            snapshot = ""
            spec_machine = MachineSpec(
                source=HostImage(),
                workdir=requirements.working_directory or "/workspace",
                network=runtime.network,
                cpus=runtime.cpus,
                memory_mb=runtime.memory_mb,
                storage_mb=runtime.storage_mb,
            )
            # A worker whose own cgroup is threaded cannot host the grading cgroups
            # (cgroup.procs is unwritable there), so an explicit parent overrides discovery.
            grading_factory = (
                LocalMachineFactory(cgroup_parent=cgroup_parent) if cgroup_parent else LocalMachineFactory()
            )
        else:
            from harbor.environments.daytona import machine_spec  # noqa: PLC0415
            from harbor_config.models.task.config import EnvironmentConfig  # noqa: PLC0415
            from shellbox.backends.daytona.machine import (  # noqa: PLC0415
                DaytonaMachineFactory,
                snapshot_identity,
                snapshot_resources,
            )

            if requirements.docker_build is not None:
                context_dir = Path(context)
                for resource in requirements.docker_build.files:
                    target = context_dir / resource.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base64.b64decode(resource.source.content_base64))
                environment_dir = context_dir
            else:
                environment_dir = Path("/nonexistent")
            spec_machine = machine_spec(
                EnvironmentConfig(
                    docker_image=requirements.docker_image,
                    allow_internet=runtime.network == NetworkPolicy.ALLOW,
                    workdir=requirements.working_directory,
                    cpus=runtime.cpus,
                    memory_mb=runtime.memory_mb,
                    storage_mb=runtime.storage_mb,
                ),
                environment_dir,
                {},
                cpus=runtime.cpus,
                memory_mb=runtime.memory_mb,
                storage_mb=runtime.storage_mb,
            )
            snapshot = snapshot_identity(spec_machine.source, snapshot_resources(spec_machine))
            grading_factory = factory or DaytonaMachineFactory()
        spec_machine = replace(spec_machine, startup_timeout=runtime.startup_timeout)
        try:
            if submission_files is None:
                result = await grade_empty_in_sandbox(task, grading_factory, spec_machine, timeout=VERIFIER_TIMEOUT)
            else:
                attempt = GradingAttempt(
                    ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))),
                    files=submission_files,
                )
                result = await grade_in_sandbox(task, attempt, grading_factory, spec_machine, timeout=VERIFIER_TIMEOUT)
        except TimeoutError as error:
            return LivenessResult(task.id, base, "timeout", None, None, snapshot, str(error))
        except Exception as error:
            return LivenessResult(
                task.id, base, "infra_error", None, None, snapshot, f"{type(error).__name__}: {error}"[:500]
            )
        return classify_result(task.id, base, snapshot, result)


def classify_result(task_id: str, base: str, snapshot: str, result: GradeResult) -> LivenessResult:
    """Require a completed verifier and a structured executed-test count before qualification."""
    exit_code = result.diagnostics.get("exit_code")
    detail = (result.error or json.dumps(result.diagnostics, default=str))[-1500:]
    tests_executed = (result.detail or {}).get("total", 0)
    if not isinstance(tests_executed, int) or tests_executed < 0:
        tests_executed = 0
    if result.status is not Outcome.GRADED or exit_code != 0:
        outcome = "verifier_error"
    elif result.reward is None:
        outcome = "missing_reward"
    elif tests_executed == 0:
        outcome = "unverified_execution"
    else:
        outcome = "negative_verdict" if result.reward < 1.0 else "positive_reward"
    return LivenessResult(task_id, base, outcome, exit_code, result.reward, snapshot, detail, tests_executed)


async def run(
    specs: dict[str, dict],
    bases: dict[str, str],
    task_ids: list[str],
    output: Path,
    cgroup_parent: Path | None = None,
) -> None:
    with output.open("a") as handle:
        for index, task_id in enumerate(task_ids, start=1):
            result = await run_one(specs[task_id], bases.get(task_id, "<unknown>"), cgroup_parent=cgroup_parent)
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
    asyncio.run(run(specs, bases, task_ids, args.output, args.cgroup_parent))


if __name__ == "__main__":
    main()
