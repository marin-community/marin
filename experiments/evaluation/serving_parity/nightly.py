# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the two bounded score-check jobs and report their outcome."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.serving_parity.launch import H100_WEIGHTS, V4_WEIGHTS
from experiments.evaluation.serving_parity.score_check import scored_items


def classify(states: list[str], score_exit: int | None) -> str:
    """A failed arm takes precedence over its peer having no capacity."""
    if len(states) != 2 or any(state not in ("succeeded", "no_capacity") for state in states):
        return "crash"
    if "no_capacity" in states:
        return "no_capacity"
    return {0: "pass", 1: "gap"}.get(score_exit, "crash")


def cleanup() -> None:
    """Cancel only this run's unfinished arms; completed jobs retain their status."""
    paths = [Path(f"{hardware}.job") for hardware in ("h100", "v4") if Path(f"{hardware}.job").exists()]
    if not paths:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            for path in paths:
                job = client.job(JobName.from_wire(path.read_text()))
                if job.state not in TERMINAL_JOB_STATES:
                    job.cancel()


def run(run_id: str, scheduling_timeout: int) -> None:
    configure_coreweave_s3()
    module = "experiments.evaluation.serving_parity"
    outputs, states = {}, []
    for hardware, weights, bucket in (
        ("h100", H100_WEIGHTS, "s3://marin-us-east-02a"),
        ("v4", V4_WEIGHTS, "gs://marin-us-central2"),
    ):
        outputs[hardware] = f"{bucket}/ci/serving-score/{run_id}/{hardware}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"{module}.launch",
                f"--hardware={hardware}",
                f"--name={run_id}-{hardware}",
                f"--weights={weights}",
                f"--output={outputs[hardware]}",
                f"--scheduling-timeout={scheduling_timeout}",
                f"--job-id-file={hardware}.job",
                "--submit",
            ],
            check=True,
            timeout=180,
        )
    deadline = time.monotonic() + scheduling_timeout + 3720
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            for hardware in outputs:
                job = client.job(JobName.from_wire(Path(f"{hardware}.job").read_text()))
                status = job.wait(timeout=max(1, deadline - time.monotonic()), poll_interval=60, raise_on_failure=False)
                started = status.started_at or any(
                    t.started_at or any(a.started_at for a in t.attempts) for t in status.tasks
                )
                no_capacity = (
                    status.state == "unschedulable"
                    and not started
                    and "Scheduling timeout exceeded" in (status.error_message + status.pending_reason)
                )
                state = "no_capacity" if no_capacity else status.state.value
                if state == "succeeded":
                    result = json.loads((StoragePath(outputs[hardware]) / "result.json").read_text())
                    scored_items(result, hardware)
                    Path(f"{hardware}.json").write_text(json.dumps(result))
                states.append(state)
                with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
                    summary.write(f"- `{job}`: {state}\n")
    score_exit = None
    if states == ["succeeded", "succeeded"]:
        score = subprocess.run(
            [sys.executable, "-m", f"{module}.score_check", "--h100", "h100.json", "--tpu", "v4.json"],
            capture_output=True,
            text=True,
        )
        score_exit = score.returncode
        details = score.stdout + score.stderr
        print(details)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(details)
    outcome = classify(states, score_exit)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"outcome={outcome}\n")
    if outcome == "no_capacity":
        print("::warning::No capacity: a job expired before starting; no paired score available")
    raise SystemExit(1 if outcome in ("crash", "gap") else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--scheduling-timeout", type=int, default=3600)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if not args.cleanup and not args.run_id:
        parser.error("--run-id is required unless cleaning up")
    try:
        if not args.cleanup:
            run(args.run_id, args.scheduling_timeout)
    finally:
        cleanup()
