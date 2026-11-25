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
from taskcompendium.models import ArtifactKind, TaskSpec, VerifierArtifact
from taskcompendium.runtime.docker_build import materialize_docker_build
from taskcompendium.runtime.grading import grade_empty_in_sandbox

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
    """The recorded Harbor bug-control rewards, keyed by the controls' own task ids.

    Some cohort families recorded no per-task controls.jsonl (their negative
    control is the campaign's verifier-liveness sample — an empty submission
    scores 0.0, recorded in the recipe QA history); an empty mapping means the
    acceptance expects 0.0 for every task.
    """
    try:
        path = Path(hf_hub_download(original_repo, "controls.jsonl", repo_type="dataset", revision=original_revision))
    except Exception:
        return {}
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
    return rewards


def original_pin(cohort_repo: str, cohort_revision: str) -> tuple[str, str]:
    """The original release pin the cohort manifest records (top-level or under ``plan``).

    Cohort families without a manifest (their recorded controls are the
    campaign's verifier-liveness sample) resolve to an empty original pin and
    the acceptance expects the zero-reward sample.
    """
    try:
        manifest_path = Path(
            hf_hub_download(cohort_repo, "manifest.json", repo_type="dataset", revision=cohort_revision)
        )
    except Exception:
        return "", ""
    manifest = json.loads(manifest_path.read_bytes())
    plan = manifest.get("plan") or {}
    source_id = manifest.get("source_dataset_id") or plan.get("source_dataset_id")
    source_revision = manifest.get("source_revision") or plan.get("source_revision")
    if not source_id or not source_revision:
        return "", ""
    return source_id, source_revision


def _runtime(task: TaskSpec, *, allow_network: bool) -> MachineRuntimeSpec:
    execution = task.execution_requirements
    # The provider's per-sandbox row maxima bound every request (a larger
    # request fails at creation); the control's semantics do not depend on the
    # allocation size above what the harness uses.
    return MachineRuntimeSpec(
        backend=DAYTONA_BACKEND,
        network=NetworkPolicy.ALLOW if allow_network else NetworkPolicy.DENY,
        cpus=min(execution.cpus, 8) if execution is not None and execution.cpus else None,
        memory_mb=min(execution.memory_mb, 32_768) if execution is not None and execution.memory_mb else None,
        storage_mb=min(execution.storage_mb, 10_240) if execution is not None and execution.storage_mb else None,
        gpus=0,
        user=None,
        startup_timeout=600.0,
        cleanup_timeout=120.0,
    )


def _docker_build_source(task: TaskSpec, workdir: Path) -> DockerfileSource | None:
    build = task.environment_requirements.docker_build
    if build is None:
        return None
    # The canonical materializer restores the declared bytes, modes and mtimes;
    # the Daytona build admission rejects any other context as changed.
    return materialize_docker_build(build, workdir / "build-context")


async def _grade_empty(task: TaskSpec, expected: float) -> ControlOutcome:
    """The negative control: the recorded bug state, graded by the converted grader."""
    import os  # noqa: PLC0415

    from shellbox.backends.daytona.connection import (  # noqa: PLC0415
        DEFAULT_DAYTONA_API_URL,
        DaytonaConnection,
    )
    from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415
    from shellbox.machine import Command  # noqa: PLC0415

    connection = DaytonaConnection(
        api_key=os.environ["DAYTONA_API_KEY"],
        api_url=os.environ.get("DAYTONA_API_URL", DEFAULT_DAYTONA_API_URL),
        target=os.environ.get("DAYTONA_TARGET", "us"),
    )
    factory = DaytonaMachineFactory(client_factory=connection.client_factory())
    runtime = _runtime(task, allow_network=True)
    with TemporaryDirectory(prefix="acceptance-") as directory:
        workdir = Path(directory)
        build_source = _docker_build_source(task, workdir)
        async with AsyncExitStack() as owned:
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
            if grade.status is not Outcome.GRADED:
                # The on-machine post-mortem: the harness's own exit code, the
                # log directory state, and the artifact-download probe result,
                # for a typed failure report.
                probe = await task_machine.run(
                    Command(
                        (
                            "bash",
                            "-c",
                            "ls -la /logs/verifier 2>&1 | tail -6; "
                            "echo '--- exit-code:'; cat /logs/verifier/exit-code.txt 2>&1 | tail -1; "
                            "echo '--- stdout tail:'; tail -3 /logs/verifier/test-stdout.txt 2>&1; "
                            "echo '--- reward probe:'; "
                            'if [ -f /logs/verifier/reward.txt ]; then printf "reward.txt=file "; '
                            "cat /logs/verifier/reward.txt; "
                            'else printf "reward.txt=missing\\n"; fi; '
                            'if [ -f /logs/verifier/reward.json ]; then printf "reward.json=file\\n"; '
                            'else printf "reward.json=missing\\n"; fi',
                        ),
                        timeout=60.0,
                    )
                )
                postmortem = probe.stdout.decode(errors="replace")[-600:]
                # Reproduce the grading machinery's own artifact download for the
                # reward file: a working download with the right bytes means the
                # loss happens in the submission staging/install, not here.
                from taskcompendium.runtime.grading import (  # noqa: PLC0415
                    _download_artifact,
                )

                local = workdir / "probe-reward.txt"
                download_ok = await _download_artifact(
                    task_machine,
                    VerifierArtifact(
                        source="/logs/verifier/reward.txt", target="/probe-reward.txt", kind=ArtifactKind.FILE
                    ),
                    local,
                    120.0,
                )
                content = local.read_bytes()[:40] if download_ok and local.exists() else b"<not downloaded>"
                detail = (
                    f"{grade.error or grade.detail or ''} | machine: {postmortem} | "
                    f"download_ok={download_ok} content={content!r}"
                )
                return ControlOutcome(task.id, expected, grade.reward, grade.status.value, detail[:900])
    return ControlOutcome(task.id, expected, grade.reward, "graded", "")


def _expected_reward(task_id: str, rewards: dict[str, float]) -> float:
    """The recorded bug reward for the task; families without controls.jsonl expect 0.0."""
    if not rewards:
        return 0.0
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
