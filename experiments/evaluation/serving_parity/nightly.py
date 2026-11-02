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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def classify(states: list[str], score_exit: int | None) -> str:
    """A failed arm takes precedence over its peer having no capacity."""
    if len(states) != 2 or any(state not in ("succeeded", "no_capacity") for state in states):
        return "crash"
    if "no_capacity" in states:
        return "no_capacity"
    return {0: "pass", 1: "gap"}.get(score_exit, "crash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scheduling-timeout", type=int, default=3600)
    args = parser.parse_args()
    configure_coreweave_s3()
    module = "experiments.evaluation.serving_parity"
    h100 = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
    v4 = "gs://marin-us-central2/scratch/karan/rsi-merge/parents/S--open-athena_Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02@50fe2f488aee/"
    outputs, states = {}, []
    for hardware, weights, bucket in (("h100", h100, "s3://marin-us-east-02a"), ("v4", v4, "gs://marin-us-central2")):
        outputs[hardware] = f"{bucket}/ci/serving-score/{args.run_id}/{hardware}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"{module}.launch",
                "--hardware",
                hardware,
                "--name",
                f"{args.run_id}-{hardware}",
                "--weights",
                weights,
                "--output",
                outputs[hardware],
                "--scheduling-timeout",
                str(args.scheduling_timeout),
                "--job-id-file",
                f"{hardware}.job",
                "--submit",
            ],
            check=True,
            timeout=180,
        )
    deadline = time.monotonic() + args.scheduling_timeout + 3720
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
                    if len(result["items"]) != 100 or len({item["id"] for item in result["items"]}) != 100:
                        state = "partial"
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
        print(score.stdout)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(score.stdout)
    outcome = classify(states, score_exit)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"outcome={outcome}\n")
    if outcome == "no_capacity":
        print("::warning::No capacity: a job expired before starting; no paired score available")
    raise SystemExit(1 if outcome in ("crash", "gap") else 0)


if __name__ == "__main__":
    main()
