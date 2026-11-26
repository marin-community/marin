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
from collections.abc import Callable, Sequence
from functools import partial
from html import escape
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.rl.rl_progress import AttemptSummary, completed_controls
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import resolve_job_user
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
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


def _publish_submission(
    submission: rl_publication.ControlSubmission,
    model: str,
    preset: Preset,
    job_state: Callable[[JobName], JobState],
    published: set[tuple[str, str]],
    *,
    new: bool = False,
) -> tuple[bool, bool]:
    """Publish a snapshot and report whether the batch ended and its outputs are visible."""
    try:
        state = job_state(JobName.from_wire(submission.job))
    except ConnectError as error:
        if error.code != Code.NOT_FOUND:
            raise
        if new:
            return False, False
        # A previous runner can stop between recording the handoff and submitting.
        state = JobState.FAILED

    table = _TABLES[model]
    if not is_job_finished(state):
        visible = [
            rl_publication.publish_running(output, preset, table, submission.job, None, published)
            for output in submission.outputs
        ]
        return False, all(visible)
    outcome = rl_publication.publish_runs(
        submission.outputs,
        preset,
        table=table,
        coordinator=submission.job,
        interrupted=state != JobState.SUCCEEDED,
        allow_unprepared=True,
    )
    print(outcome.model_dump_json(exclude={"attempts"}), flush=True)
    if outcome.unserved_records:
        raise RuntimeError(f"Terminal publication remains pending for {submission.job}")
    return True, True


def _publish_pending(
    path: StoragePath,
    model: str,
    preset: Preset,
    job_state: Callable[[JobName], JobState],
    published: set[tuple[str, str]],
    *,
    new: bool = False,
) -> bool:
    submission = rl_publication.ControlSubmission.model_validate_json(path.read_text())
    finished, visible = _publish_submission(submission, model, preset, job_state, published, new=new)
    if finished:
        path.rm()
    return visible


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


def _run_controls(batch: str, selection: str, state_dir: StoragePath) -> int:
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
                if path.exists():
                    _publish_pending(path, model, preset, client.job_state, published)
                else:
                    for submission in _running_submissions(records, *target):
                        finished, _ = _publish_submission(submission, model, preset, client.job_state, published)
                        if not finished:
                            blocked.add(target)
            except (ConnectError, OSError, RuntimeError, ValueError) as error:
                print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                blocked.add(target)
                failed = True
        records = rl_index.list_attempts()
        for model, preset in select_presets(selection):
            path = state_dir / f"{model}-{preset.name}.json"
            try:
                if (model, preset.name) in blocked or path.exists():
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
                started[model, preset.name] = (
                    child,
                    partial(_publish_pending, path, model, preset, client.job_state, published, new=True),
                )
            except (ValueError, OSError, RuntimeError) as error:
                print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                failed = True
        pending = {}
        for target, (child, publish) in started.items():
            if child.wait() == 0:
                pending[target] = publish
            else:
                failed = True

        deadline = time.monotonic() + _OBSERVATION_SECONDS
        while pending:
            for target, publish in list(pending.items()):
                try:
                    if publish():
                        del pending[target]
                except (ConnectError, OSError, RuntimeError, ValueError) as error:
                    print(f"{'/'.join(target)}: {error}", file=sys.stderr)
                    failed = True
                    del pending[target]
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(_POLL_SECONDS)
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
        select_presets(args.presets)
    except ValueError as error:
        parser.error(str(error))
    if not args.state_dir:
        parser.error("--batch requires --state-dir")
    return _run_controls(args.batch, args.presets, StoragePath(args.state_dir))


if __name__ == "__main__":
    sys.exit(main())
