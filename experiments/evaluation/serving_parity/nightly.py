# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded Iris runs and classify the nightly score check."""

import argparse
import json
import os
from enum import StrEnum
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.serving_parity.score_check import check_scores


class Outcome(StrEnum):
    PASS = "pass"
    GAP = "gap"
    CRASH = "crash"
    NO_CAPACITY = "no_capacity"


def classify(arms: list[dict]) -> Outcome:
    """Errors take precedence over an unstarted peer's capacity timeout."""
    if len(arms) != 2 or {arm["hardware"] for arm in arms} != {"h100", "v4"}:
        return Outcome.CRASH
    waiting = False
    for arm in arms:
        if arm["state"] == "unschedulable" and not arm["started"] and arm["scheduling_timeout"]:
            waiting = True
            continue
        if arm["state"] != "succeeded" or "result" not in arm:
            return Outcome.CRASH
        try:
            result = arm["result"]
            items = result["items"]
            if (
                result["hardware"] != arm["hardware"]
                or len(items) != 100
                or len({item["id"] for item in items}) != 100
                or any(item["correct"] not in (0, 1) for item in items)
            ):
                return Outcome.CRASH
        except (KeyError, TypeError):
            return Outcome.CRASH
    if waiting:
        return Outcome.NO_CAPACITY
    by_hardware = {arm["hardware"]: arm["result"] for arm in arms}
    try:
        return Outcome.PASS if check_scores(by_hardware["h100"], by_hardware["v4"]) else Outcome.GAP
    except (KeyError, TypeError, ValueError):
        return Outcome.CRASH


def collect(job_id: str, hardware: str, output: str, timeout: int) -> dict:
    report = dict(job_id=job_id, hardware=hardware, state="error", started=False, scheduling_timeout=False)
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            job = client.job(JobName.from_wire(job_id))
            terminal = False
            try:
                status = job.wait(timeout=timeout, poll_interval=60, raise_on_failure=False)
                terminal = status.state in TERMINAL_JOB_STATES
                started = status.started_at is not None or any(
                    task.started_at is not None or any(attempt.started_at is not None for attempt in task.attempts)
                    for task in status.tasks
                )
                report.update(
                    state=status.state.value,
                    started=started,
                    scheduling_timeout="Scheduling timeout exceeded" in (status.error_message + status.pending_reason),
                )
                if status.state == JobState.SUCCEEDED:
                    report["result"] = json.loads((StoragePath(output) / "result.json").read_text())
            except Exception as error:
                report.update(state="error", error_type=type(error).__name__)
            finally:
                if not terminal:
                    job.cancel()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--job-id-file", type=Path, required=True)
    wait.add_argument("--hardware", choices=("h100", "v4"), required=True)
    wait.add_argument("--output", required=True)
    wait.add_argument("--timeout", type=int, required=True)
    wait.add_argument("--report", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("reports", type=Path)
    args = parser.parse_args()
    if args.command == "wait":
        configure_coreweave_s3()
        report = collect(args.job_id_file.read_text().strip(), args.hardware, args.output, args.timeout)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"{report['job_id']}: {report['state']}")
        return
    arms = [json.loads(path.read_text()) for path in sorted(args.reports.glob("*.json"))]
    outcome = classify(arms)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"outcome={outcome}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"## Score check: {outcome}\n\n")
        for arm in arms:
            result = arm.get("result", {})
            summary.write(f"- `{arm['job_id']}`: {arm['state']}; score={result.get('score', 'unavailable')}\n")
        if outcome == Outcome.NO_CAPACITY:
            summary.write("\nNo capacity: a scheduling timeout occurred before startup. No paired score is available.\n")
    if outcome == Outcome.NO_CAPACITY:
        print("::warning::No capacity: an Iris job expired in the queue without starting")
    raise SystemExit(1 if outcome in (Outcome.CRASH, Outcome.GAP) else 0)


if __name__ == "__main__":
    main()
