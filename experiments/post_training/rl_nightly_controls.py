# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run nightly control cohorts through each model's regular launcher."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from html import escape
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.rl.rl_progress import AttemptSummary, completed_controls
from finestore.rl.run_records import AttemptRecord, LaunchState
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient
from iris.cluster.client.job_info import resolve_job_user
from iris.cluster.types import JobName
from iris.resources.state import JobState, TaskState, is_job_finished
from marin.execution.remote import sanitize_job_name
from marin.rl.skyrl import IRIS_HUB_CLUSTER_CONFIG
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_compose, rl_gate, rl_index, rl_publication
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.rl_presets import Preset
from experiments.post_training.snowball_rl import SNOWBALL

_TABLES = {table.model: table for table in (SNOWBALL, MOCHI)}
_OBSERVATION_SECONDS = 600
_POLL_SECONDS = 30
_SUBMISSION_GRACE = timedelta(minutes=15)


class _Observation(Enum):
    WAITING = auto()
    VISIBLE = auto()
    FINISHED = auto()
    FAILED = auto()


def select_presets(selection: str) -> list[tuple[str, Preset]]:
    gated = {
        f"{table.model}/{preset.name}": (table.model, preset)
        for table in _TABLES.values()
        for preset in table.presets
        if preset.gated
    }
    selected = set()
    for token in selection.split():
        matches = {key for key, (model, _) in gated.items() if token == key or token == model}
        if not matches:
            raise ValueError(f"Unknown or ungated RL preset: {token!r}")
        selected.update(matches)
    return [value for key, value in gated.items() if not selected or key in selected]


def dispatch_comment(event_path: Path) -> None:
    event = json.loads(event_path.read_text())
    command = event["comment"]["body"].split(maxsplit=1)
    if not command or command[0] != "/rl-controls" or "pull_request" not in event["issue"]:
        return
    repository = f"repos/{os.environ['GITHUB_REPOSITORY']}"
    number = event["issue"]["number"]
    login = event["comment"]["user"]["login"]
    # The 2026-03-10 dispatch API returns the run ID and links in its response.
    api = ["gh", "api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2026-03-10"]
    try:
        pull = json.loads(subprocess.check_output([*api, f"{repository}/pulls/{number}"]))
        if pull["merged"] is not True:
            raise ValueError("RL controls require a merged pull request")
        permission = json.loads(subprocess.check_output([*api, f"{repository}/collaborators/{login}/permission"]))
        if permission["permission"] not in {"write", "admin"}:
            raise ValueError("RL controls require repository write permission")
        if len(command) != 2:
            raise ValueError("Provide at least one model or model/preset after /rl-controls")
        selection = command[1]
        select_presets(selection)
    except json.JSONDecodeError:
        raise
    except ValueError as error:
        reply = f"🤖 <code>{escape(str(error))}</code>"
    else:
        response = subprocess.check_output(
            [*api, "-X", "POST", f"{repository}/actions/workflows/marin-skyrl-nightly.yaml/dispatches", "--input", "-"],
            input=json.dumps({"ref": "main", "inputs": {"controls": "true", "presets": selection}}).encode(),
        )
        if not response:
            raise RuntimeError("GitHub dispatch returned no run details")
        reply = f"🤖 [RL control run]({json.loads(response)['html_url']})"
    subprocess.check_output([*api, f"{repository}/issues/{number}/comments", "-f", f"body={reply}"])


def _read_submission(path: StoragePath) -> rl_publication.ControlSubmission | None:
    try:
        submission = rl_publication.ControlSubmission.model_validate_json(path.read_text())
    except FileNotFoundError:
        return None
    if submission.recorded_at is None:
        raise ValueError(f"Pending control submission at {path} has no recording time")
    return submission


def _job_state(client: IrisClient, job: JobName) -> JobState | None:
    try:
        return client.job_state(job)
    except ConnectError as error:
        if error.code != Code.NOT_FOUND:
            raise
        return None


def _job_was_interrupted(client: IrisClient, job: JobName) -> bool:
    try:
        status = client.job_status(job)
    except ConnectError as error:
        if error.code == Code.NOT_FOUND:
            return False
        raise
    if status.state == JobState.KILLED:
        return True
    # An exhausted worker failure can come from the workload or a lost pod.
    reason = status.error_message.partition(":")[0]
    return (
        status.state == JobState.WORKER_FAILED
        and not status.task_state_counts.get(TaskState.FAILED)
        and (
            bool(status.task_state_counts.get(TaskState.PREEMPTED))
            or reason in {"PodDeleted", "Evicted", "Preempting"}
            or reason.startswith("WorkloadEvicted")
        )
    )


def _control_was_interrupted(
    client: IrisClient, coordinator: JobName, receipt: AttemptRecord, coordinator_interrupted: bool
) -> bool:
    result = receipt.native.result
    job_id = result.iris_job_id or receipt.iris_job_id
    job = JobName.from_wire(job_id) if job_id else coordinator
    if job.root_job != coordinator:
        return False
    if result.state.is_terminal:
        return result.state == LaunchState.FAILED and job != coordinator and _job_was_interrupted(client, job)
    # A parent failure can kill its children. An already failed child retains its failure state.
    if receipt.interruption is None or not coordinator_interrupted:
        return False
    state = _job_state(client, job) if job != coordinator else None
    if state == JobState.WORKER_FAILED:
        return _job_was_interrupted(client, job)
    return state not in {JobState.FAILED, JobState.UNSCHEDULABLE}


def _publish_submission(
    submission: rl_publication.ControlSubmission,
    model: str,
    preset: Preset,
    client: IrisClient,
    published: set[tuple[str, str]],
    *,
    pending_file: StoragePath | None = None,
) -> _Observation:
    table = _TABLES[model]
    job = JobName.from_wire(submission.job)
    name_prefix = rl_compose.control_run_name_prefix(table.namespace, preset.name)
    prefix = StoragePath(rl_compose.run_prefix(preset)) / name_prefix
    job_prefix = sanitize_job_name(name_prefix) + "-"
    runs = [StoragePath(output).parent for output in submission.outputs]
    if (
        not job.is_root
        or job.user != resolve_job_user()
        or not job.name.startswith(job_prefix)
        or any(run.parent != prefix.parent or not run.name.startswith(prefix.name) for run in runs)
    ):
        raise ValueError(f"Control submission for {model}/{preset.name} has a foreign job or output")
    state = _job_state(client, job)
    if state is None and submission.recorded_at is not None:
        # A submit RPC can outlast its caller. Keep its handoff until the grace expires.
        grace_end = submission.recorded_at.astimezone(UTC) + _SUBMISSION_GRACE
        if datetime.now(UTC) < grace_end:
            print(
                f"Pending record {pending_file}: Iris job {job} not found yet. "
                f"Grace ends at {grace_end.isoformat()}.",
                flush=True,
            )
            return _Observation.WAITING

    if state is not None and not is_job_finished(state):
        visible = [
            rl_publication.publish_running(output, preset, table, submission.job, None, published)
            for output in submission.outputs
        ]
        return _Observation.VISIBLE if all(visible) else _Observation.WAITING
    outcome = rl_publication.publish_runs(
        submission.outputs,
        preset,
        table=table,
        coordinator=submission.job,
        interrupted=state != JobState.SUCCEEDED,
        allow_unprepared=True,
        fail_incomplete_control=pending_file is None and state == JobState.SUCCEEDED,
    )
    print(outcome.model_dump_json(), flush=True)
    if outcome.unserved_records:
        raise RuntimeError(f"Terminal publication remains pending for {submission.job}")
    coordinator_interrupted = state == JobState.KILLED or (
        state == JobState.WORKER_FAILED and _job_was_interrupted(client, job)
    )
    failed = any(
        not _control_was_interrupted(client, job, receipt, coordinator_interrupted)
        for receipt in outcome.ineligible_controls
    )
    if not outcome.attempts and state is not None:
        failed = not coordinator_interrupted
    if pending_file is not None:
        pending_file.rm()
    return _Observation.FAILED if failed else _Observation.FINISHED


def _running_submissions(
    records: Sequence[AttemptSummary], model: str, preset: str
) -> list[rl_publication.ControlSubmission]:
    """Recover unfinished batches from dashboard rows written by an earlier workflow."""
    jobs = defaultdict(set)
    owner = resolve_job_user()
    for row in records:
        if (
            row.is_control
            and row.branch == "main"
            and row.dirty is False
            and row.status == "running"
            and (row.model, row.series) == (model, preset)
            and row.artifact_uri
            and row.iris_job_id
        ):
            job = JobName.from_wire(row.iris_job_id).root_job
            if job.user == owner:
                jobs[str(job)].add(row.artifact_uri)
    return [rl_publication.ControlSubmission(job=job, outputs=tuple(sorted(outputs))) for job, outputs in jobs.items()]


def _observe_startup(
    started: dict[tuple[str, str], tuple[Preset, StoragePath, subprocess.Popen]],
    client: IrisClient,
    published: set[tuple[str, str]],
) -> bool:
    failed = False
    errors = {}
    visible = set()
    deadline = time.monotonic() + _OBSERVATION_SECONDS
    while started:
        for target, (preset, path, child) in list(started.items()):
            exit_code = child.poll()
            failed |= exit_code not in (None, 0)
            if target in visible:
                if exit_code is not None:
                    del started[target]
                continue
            try:
                submission = _read_submission(path)
                if submission is None:
                    if exit_code is not None:
                        if exit_code == 0:
                            print(f"{'/'.join(target)}: launcher exited without a pending handoff", file=sys.stderr)
                            failed = True
                        del started[target]
                    continue
                status = _publish_submission(submission, target[0], preset, client, published, pending_file=path)
            except Exception as error:
                errors[target] = error
                continue
            errors.pop(target, None)
            failed |= status is _Observation.FAILED
            if status is not _Observation.WAITING:
                visible.add(target)
                if exit_code is not None:
                    del started[target]
        remaining = deadline - time.monotonic()
        if not started or remaining <= 0:
            break
        time.sleep(min(_POLL_SECONDS, remaining))
        if time.monotonic() >= deadline:
            break
    for target, (_, _, child) in started.items():
        exit_code = child.poll()
        if exit_code is None:
            child.kill()
            child.wait(timeout=10)
            print(f"{'/'.join(target)}: launcher exceeded the startup window", file=sys.stderr)
        failed |= exit_code != 0
    for target, error in errors.items():
        print(f"{'/'.join(target)}: {error}", file=sys.stderr)
    return failed or bool(errors)


def _run_controls(batch: str, selected: Sequence[tuple[str, Preset]], state_dir: StoragePath) -> int:
    # The nightly workflow's concurrency group serializes writes to each pending file.
    configure_coreweave_s3()
    started = {}
    published = set()
    failed = False
    with open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=rl_compose.REPO_ROOT) as client:
        records = rl_index.list_attempts()
        blocked = set()
        for model, preset in select_presets(""):
            target = (model, preset.name)
            path = state_dir / f"{model}-{preset.name}.json"
            try:
                stored = _read_submission(path)
            except Exception as error:
                print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                blocked.add(target)
                failed = True
                stored = None
            sources = [(stored, path)] if stored is not None else []
            sources.extend(
                (submission, None)
                for submission in _running_submissions(records, *target)
                if stored is None or submission.job != stored.job
            )
            for submission, pending_file in sources:
                try:
                    status = _publish_submission(submission, model, preset, client, published, pending_file=pending_file)
                    failed |= status is _Observation.FAILED
                    if status in {_Observation.WAITING, _Observation.VISIBLE}:
                        blocked.add(target)
                except Exception as error:
                    print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                    blocked.add(target)
                    failed = True
        records = rl_index.list_attempts()
        for model, preset in selected:
            path = state_dir / f"{model}-{preset.name}.json"
            try:
                if (model, preset.name) in blocked:
                    print(f"{model}/{preset.name}: waiting for the pending control batch", flush=True)
                    continue
                version = rl_gate.config_version(rl_gate.compose_preset(preset))
                cohort = completed_controls(records, model=model, preset=preset.name, version=version)
                count = max(1, rl_gate.CONTROL_COUNT - len(cohort))
                print(
                    json.dumps({"model": model, "preset": preset.name, "version": version, "requested": count}),
                    flush=True,
                )
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        f"experiments.post_training.{model}_rl",
                        "--preset",
                        preset.name,
                        "--control",
                        batch,
                        "--replicates",
                        str(count),
                        "--version",
                        rl_compose.RUN_ARTIFACT_VERSION,
                        "--run",
                    ],
                    env={**os.environ, rl_publication.CONTROL_SUBMISSION_ENV: str(path)},
                )
                started[model, preset.name] = (preset, path, child)
            except (ValueError, OSError, RuntimeError) as error:
                print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                failed = True
        failed |= _observe_startup(started, client, published)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", help="Unique schedule or dispatch identifier.")
    source.add_argument("--comment-event", type=Path, help="GitHub issue_comment event payload.")
    source.add_argument("--check-presets", help="Validate a selector list before launch setup.")
    parser.add_argument("--presets", default="", help="Space-separated models or model/preset selectors.")
    parser.add_argument("--state-dir", help="Shared object storage directory for pending control batches.")
    args = parser.parse_args()
    try:
        if args.comment_event is not None:
            dispatch_comment(args.comment_event)
            return 0
        if args.check_presets is not None:
            select_presets(args.check_presets)
            return 0
        selected = select_presets(args.presets)
    except ValueError as error:
        parser.error(str(error))
    if not args.state_dir:
        parser.error("--batch requires --state-dir")
    return _run_controls(args.batch, selected, StoragePath(args.state_dir))


if __name__ == "__main__":
    sys.exit(main())
