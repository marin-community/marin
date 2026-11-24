# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acceptance controls for a published cohort-conversion release.

Ruling 2026-11-24: the converted graders must behave exactly like the Harbor
harness they replace before a release enters admission. This stage replays the
recorded negative controls against the converted graders on the same backend
the Harbor trials used (Daytona): per sampled task it stages the task
environment (the recorded bug state), grades an empty submission through the
converted sandbox grader, and requires the reward to equal the recorded bug
control's reward. Anything else — a mismatch, an unscored outcome, an
infrastructure error, a missing recorded control — fails the run.

The oracle (reference) tier sources each family's gold state separately; this
module is its fail-closed negative half plus the comparison scaffolding both
tiers share. Judge-harness families resolve their verifier env through the host
environment: the job exports OPENAI_API_KEY/JUDGE_MODEL, never the task data.
"""

import argparse
import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _machine_spec, _prepare_machine
from rolloutengine.models import MachineRuntimeSpec
from shellbox.image import DockerfileSource
from shellbox.machine import NetworkPolicy
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.docker_build import materialize_build_directories
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from taskcompendium.runtime.resources import resource_bytes

LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLE = 12
DAYTONA_BACKEND = "daytona"


@dataclass(frozen=True)
class ControlOutcome:
    task_id: str
    expected: float
    observed: float | None
    outcome: str
    detail: str


def recorded_bug_rewards(original_repo: str, original_revision: str) -> dict[str, float]:
    """The recorded Harbor bug-control rewards, keyed by the controls' own task ids."""
    path = Path(hf_hub_download(original_repo, "controls.jsonl", repo_type="dataset", revision=original_revision))
    rewards = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            controls = row.get("controls") or {}
            bug = controls.get("bug") or {}
            if bug.get("status") != "scored":
                raise ValueError(
                    f"The recorded bug control for {row.get('task_id')} is not scored: {bug.get('status')!r}"
                )
            rewards[row["task_id"]] = float(bug["reward"])
    if not rewards:
        raise ValueError(f"No recorded bug controls at {original_repo}@{original_revision[:12]}")
    return rewards


def original_pin(cohort_repo: str, cohort_revision: str) -> tuple[str, str]:
    """The original release pin the cohort manifest records."""
    manifest_path = Path(hf_hub_download(cohort_repo, "manifest.json", repo_type="dataset", revision=cohort_revision))
    manifest = json.loads(manifest_path.read_bytes())
    source_id, source_revision = manifest.get("source_dataset_id"), manifest.get("source_revision")
    if not source_id or not source_revision:
        raise ValueError(f"The cohort manifest names no original release: {cohort_repo}@{cohort_revision[:12]}")
    return source_id, source_revision


def _runtime(task: TaskSpec, *, allow_network: bool) -> MachineRuntimeSpec:
    execution = task.execution_requirements
    return MachineRuntimeSpec(
        backend=DAYTONA_BACKEND,
        network=NetworkPolicy.ALLOW if allow_network else NetworkPolicy.DENY,
        cpus=execution.cpus if execution is not None and execution.cpus else None,
        memory_mb=execution.memory_mb if execution is not None and execution.memory_mb else None,
        storage_mb=execution.storage_mb if execution is not None and execution.storage_mb else None,
        gpus=0,
        user=None,
        startup_timeout=600.0,
        cleanup_timeout=120.0,
    )


def _docker_build_source(task: TaskSpec, workdir: Path) -> DockerfileSource | None:
    build = task.environment_requirements.docker_build
    if build is None:
        return None
    context = workdir / "build-context"
    context.mkdir(parents=True, exist_ok=True)
    materialize_build_directories(build, context)
    for file in build.files:
        target = context / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(file))
    dockerfile = next((path for path in ("Dockerfile", "dockerfile") if (context / path).exists()), None)
    if dockerfile is None:
        raise ValueError(f"{task.id}: the build context carries no Dockerfile")
    return DockerfileSource(context=context, dockerfile=context / dockerfile)


async def _grade_empty(task: TaskSpec, expected: float) -> ControlOutcome:
    """The negative control: the recorded bug state, graded by the converted grader."""
    from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415

    factory = DaytonaMachineFactory()
    runtime = _runtime(task, allow_network=True)
    with TemporaryDirectory(prefix="acceptance-") as directory:
        workdir = Path(directory)
        build_source = _docker_build_source(task, workdir)
        with AsyncExitStack() as owned:
            task_machine = await _prepare_machine(
                task.environment_requirements,
                runtime,
                (),
                {DAYTONA_BACKEND: factory},
                _Cleanup(runtime.cleanup_timeout or 120.0),
                owned,
                docker_build_source=build_source,
            )
            if task_machine is None:
                return ControlOutcome(task.id, expected, None, "infra_error", "the task machine did not start")
            grading_spec = _machine_spec(task.grader.environment, runtime, build_source)
            grade = await grade_empty_in_sandbox(
                task, factory, grading_spec, timeout=task.grader.timeout, task_machine=task_machine
            )
    if grade.outcome is Outcome.GRADED:
        return ControlOutcome(task.id, expected, grade.reward, "graded", "")
    return ControlOutcome(task.id, expected, grade.reward, grade.outcome.value, str(grade.detail or "")[:300])


def _expected_reward(task_id: str, rewards: dict[str, float]) -> float:
    if task_id in rewards:
        return rewards[task_id]
    matches = {reward: key for key, reward in rewards.items() if key.startswith(task_id) or task_id.startswith(key)}
    if len(matches) == 1:
        return next(iter(matches))
    raise ValueError(f"No recorded bug control covers {task_id}")


def run_acceptance(tasks: list[TaskSpec], rewards: dict[str, float], *, sample: int) -> dict:
    """Grade the sampled empty states and require every recorded bug reward to reproduce."""
    sampled = tasks[:sample]
    outcomes = asyncio.run(_run_all(sampled, rewards))
    failures = [outcome for outcome in outcomes if outcome.outcome != "graded" or outcome.observed != outcome.expected]
    return {
        "sampled": len(outcomes),
        "passed": len(outcomes) - len(failures),
        "failures": [outcome.__dict__ for outcome in failures],
        "state": "ok" if not failures else "mismatched",
    }


async def _run_all(tasks: list[TaskSpec], rewards: dict[str, float]) -> list[ControlOutcome]:
    outcomes = []
    for task in tasks:
        expected = _expected_reward(task.id, rewards)
        try:
            outcomes.append(await _grade_empty(task, expected))
        except Exception as error:
            outcomes.append(ControlOutcome(task.id, expected, None, "infra_error", str(error)[:300]))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The cohort source id.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--release-revision", required=True, help="The published conversion release revision.")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="How many tasks to accept.")
    parser.add_argument("--report", type=Path, help="Write the result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from experiments.post_training.task_curation.sources import all_sources  # noqa: PLC0415

    source = all_sources()[args.source_id]
    dataset = source.dataset
    assert dataset is not None
    original_repo, original_revision = original_pin(dataset.name, dataset.revision)
    rewards = recorded_bug_rewards(original_repo, original_revision)

    converted = hf_hub_download(args.release_repo, "tasks.parquet", repo_type="dataset", revision=args.release_revision)
    rows = pq.read_table(converted, columns=["task_spec"]).to_pylist()
    tasks = [TaskSpec.model_validate_json(row["task_spec"]) for row in rows]
    result = {"source_id": args.source_id, "release_repo": args.release_repo, "release_revision": args.release_revision}
    result.update(run_acceptance(tasks, rewards, sample=args.sample))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)
    if result["state"] != "ok":
        raise SystemExit(f"acceptance mismatched: {result['failures'][:3]}")


if __name__ == "__main__":
    main()
