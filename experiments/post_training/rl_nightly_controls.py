# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run nightly control cohorts through each model's regular launcher."""

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.rl.rl_progress import completed_controls
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
_MOCHI_RACKS = 3
_SNOWBALL_BUDGET = 60_000  # Leave 15,000 of runner's 75,000 units for the nightly GPU gates.
_ACTIVE_STATES = (JobState.PENDING, JobState.BUILDING, JobState.RUNNING)


def _attempt_cost(model: str, preset: Preset) -> int:
    # Snowball: 8 * 1,000 GPU + 32 * 5 CPU + 1,800 GiB per node, plus 36 for the coordinator.
    return 9_960 * preset.recipe.layout.workload.plan.num_nodes + 36 if model == SNOWBALL.model else 1


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


def _failed_job(client: IrisClient, state: str | None, job_id: str | None) -> bool:
    if state != JobState.WORKER_FAILED or job_id is None:
        return state != JobState.KILLED
    name = JobName.from_wire(job_id)
    job = client.job_status(name)
    if job.task_state_counts.get(TaskState.FAILED):
        return True
    reasons = " ".join([job.error_message, *(task.error_message for task in client.list_tasks(name))]).lower()
    pattern = (
        r"\boom\s*killed\b|out of memory|host crash|worker reconcile failure threshold exceeded|unhealthy worker failed"
    )
    return re.search(pattern, reasons) is not None


def _run_controls(batch: str, selected: Sequence[tuple[str, Preset]], state_dir: StoragePath) -> int:
    configure_coreweave_s3()
    failed = False
    blocked = set()
    pending_jobs = set()
    capacity = {MOCHI.model: _MOCHI_RACKS, SNOWBALL.model: _SNOWBALL_BUDGET}
    owner = resolve_job_user()
    budget_models = set()
    with open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=rl_compose.REPO_ROOT) as client:
        for model, preset in select_presets(""):
            target = (model, preset.name)
            for suffix in (".json", ".migration.json"):
                path = state_dir / f"{model}-{preset.name}{suffix}"
                try:
                    try:
                        submission = rl_publication.ControlSubmission.model_validate_json(path.read_text())
                    except FileNotFoundError:
                        continue
                    job = JobName.from_wire(submission.job)
                    try:
                        state = client.job_state(job)
                    except ConnectError as error:
                        if error.code != Code.NOT_FOUND:
                            raise
                        state = None
                    if state is not None and not is_job_finished(state):
                        blocked.add(target)
                        pending_jobs.add(job)
                        capacity[model] -= len(submission.outputs) * _attempt_cost(model, preset)
                        if job.user == owner:
                            budget_models.add(model)
                        published = set()
                        for output in submission.outputs:
                            rl_publication.publish_running(
                                output, preset, _TABLES[model], submission.job, None, published
                            )
                        continue
                    result = rl_publication.publish_runs(
                        submission.outputs,
                        preset,
                        table=_TABLES[model],
                        coordinator=submission.job,
                        interrupted=True,
                        allow_unprepared=True,
                    )
                    print(result.model_dump_json(), flush=True)
                    if result.unserved_records:
                        raise RuntimeError(f"Publication remains pending for {submission.job}")
                    path.rm()
                    check_coordinator = not result.attempts
                    for attempt in result.ineligible_controls:
                        native = attempt.native.result
                        if native.state.is_terminal:
                            failed |= _failed_job(client, native.iris_job_state, native.iris_job_id)
                        else:
                            check_coordinator = True
                    failed |= check_coordinator and state is not None and _failed_job(client, state, submission.job)
                except Exception as error:
                    print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                    blocked.add(target)
                    failed = True
        if capacity[MOCHI.model] > 0 and any(model == MOCHI.model for model, _ in selected):
            try:
                jobs = (
                    job.job_id
                    for user in ("runner", "atqamar")
                    for state in _ACTIVE_STATES
                    for job in client.list_jobs(state=state, prefix=f"/{user}/")
                    if job.job_id.root_job not in pending_jobs
                )
                for job in jobs:
                    detail = client.job_status(job)
                    device = detail.resources.device
                    if device and device.variant.upper() == "GB200":
                        capacity[MOCHI.model] -= max(1, (detail.task_count + 17) // 18)
                        if job.user == owner:
                            budget_models.add(MOCHI.model)
                    elif job.is_root and f"checkpoints-{MOCHI.namespace}-" in job.name:
                        # A coordinator without a handoff can still submit its replicas.
                        capacity[MOCHI.model] = 0
            except Exception as error:
                print(f"Mochi rack scan failed: {error}", file=sys.stderr)
                capacity[MOCHI.model] = 0
        try:
            prefix = f"/{owner}/checkpoints-{SNOWBALL.namespace}-controls-"
            if any(job.root_job not in pending_jobs for job in client.active_job_names_for_prefix(prefix)):
                capacity[SNOWBALL.model] = 0
                budget_models.add(SNOWBALL.model)
            if client.active_job_names_for_prefix(f"/{owner}/checkpoints-{MOCHI.namespace}-"):
                budget_models.add(MOCHI.model)
        except Exception as error:
            print(f"Snowball coordinator scan failed: {error}", file=sys.stderr)
            capacity[SNOWBALL.model] = 0
        records = rl_index.list_attempts()
        for position, (model, preset) in enumerate(selected):
            try:
                target = f"{model}/{preset.name}"
                if (model, preset.name) in blocked:
                    print(f"Skipping {target}: pending record")
                    continue
                prefix = sanitize_job_name(f"checkpoints/{_TABLES[model].namespace}/controls/{preset.name}") + "-"
                if client.active_job_names_for_prefix(f"/{owner}/{prefix}"):
                    print(f"Skipping {target}: active coordinator")
                    continue
                cost = _attempt_cost(model, preset)
                slots = capacity[model] // cost
                if slots <= 0:
                    reason = "rack" if model == MOCHI.model else "Iris budget"
                    print(f"Skipping {target}: no free {reason}")
                    continue
                if budget_models - {model}:
                    print(f"Skipping {target}: shared Iris budget reserved for the other model")
                    continue
                version = rl_gate.config_version(rl_gate.compose_preset(preset))
                cohort = completed_controls(records, model=model, preset=preset.name, version=version)
                count = min(slots, max(1, rl_gate.CONTROL_COUNT - len(cohort)))
                if model == MOCHI.model:
                    later = sum(
                        name == MOCHI.model and (name, following.name) not in blocked
                        for name, following in selected[position + 1 :]
                    )
                    count = min(count, max(1, slots - later))
                capacity[model] -= count * cost
                budget_models.add(model)
                path = state_dir / f"{model}-{preset.name}.json"
                module = [sys.executable, "-m", f"experiments.post_training.{model}_rl"]
                options = ["--preset", preset.name, "--control", batch, "--replicates", str(count)]
                result = subprocess.run(
                    [*module, *options, "--version", rl_compose.RUN_ARTIFACT_VERSION, "--run"],
                    env={**os.environ, rl_publication.CONTROL_SUBMISSION_ENV: str(path)},
                    check=False,
                )
                failed |= result.returncode != 0
            except (ValueError, OSError, RuntimeError, ConnectError) as error:
                print(f"{model}/{preset.name}: {error}", file=sys.stderr)
                failed = True
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch", help="Unique schedule or dispatch identifier.")
    source.add_argument("--comment-event", type=Path, help="GitHub issue_comment event payload.")
    source.add_argument("--check-presets", help="Validate a selector list before launch setup.")
    parser.add_argument("--presets", default="", help="Space-separated models or model/preset selectors.")
    parser.add_argument(
        "--state-dir",
        default="s3://marin-us-east-08a/marin/ci/rl-controls/pending",
        help="Shared directory for pending control batches.",
    )
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
        parser.error("--state-dir cannot be empty")
    # Rotate the first preset daily so bounded cohorts share admission across ticks.
    start = datetime.now(UTC).toordinal() % len(selected)
    selected = selected[start:] + selected[:start]
    return _run_controls(args.batch, selected, StoragePath(args.state_dir))


if __name__ == "__main__":
    sys.exit(main())
