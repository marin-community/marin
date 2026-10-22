# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit campaign SWE-bench errors without changing or sealing source results."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

TRANSPORT_ERRORS = (
    "ReadTimeout",
    "ConnectTimeout",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "AuthenticationError",
    "ConnectError",
)


def inspect_trial(item):
    uri, fs, path = item
    payload = fs.cat(path)
    data = json.loads(payload)
    error = (data.get("exception_info") or {}).get("exception_type")
    reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    row = {
        "task_name": data["task_name"],
        "source_uri": uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "reward": reward,
        "exception_type": error,
        "agent_execution": data.get("agent_execution"),
        "verifier": data.get("verifier"),
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
    }
    if error not in {"NonZeroAgentExitCodeError", "AgentTimeoutError", "VerifierTimeoutError"}:
        return row
    trial_dir = path.rsplit("/", 1)[0]
    attempts = fs.ls(trial_dir + "/attempts", detail=False)
    logs = []
    for attempt in attempts:
        log_path = attempt.rstrip("/") + "/agent/mini-swe-agent.txt"
        try:
            content = fs.cat(log_path)
        except FileNotFoundError:
            continue
        text = content.decode(errors="replace")
        logs.append(
            {
                "source_uri": "s3://" + log_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "transport_errors": [name for name in TRANSPORT_ERRORS if name in text],
                "terminal_error_classes": re.findall(r"(?m)^\s*([A-Za-z_.]*(?:Error|Timeout)):", text)[-10:],
            }
        )
    row["agent_logs"] = logs
    row["transport_failure_observed"] = any(log["transport_errors"] for log in logs)
    return row


def audit(source, output):
    fs, summary_path = filesystem_for(source["summary_uri"])
    summary_payload = fs.cat(summary_path)
    summary = json.loads(summary_payload)
    job_path = summary_path.rsplit("/", 1)[0]
    root = job_path.split("/harbor_jobs/")[0]
    identity = json.loads(fs.cat(root + "/harbor_resume_identity.json"))
    config_payload = fs.cat(job_path + "/config.json")
    directories = [p for p in fs.ls(job_path, detail=False) if "__" in p.rsplit("/", 1)[-1]]
    items = []
    for directory in directories:
        path = directory + "/result.json"
        if fs.exists(path):
            items.append(("s3://" + path, fs, path))
    with ThreadPoolExecutor(max_workers=12) as pool:
        trials = list(pool.map(inspect_trial, items))
    tasks = [trial["task_name"] for trial in trials]
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate task outcomes require explicit selection")
    report = {
        "observed_at": datetime.now(UTC).isoformat(),
        **source,
        "runtime": identity["runtime"],
        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "completed": len(trials),
        "active_at_start": summary["stats"]["n_running_trials"],
        "transport_affected": sum(t.get("transport_failure_observed", False) for t in trials),
        "trials": sorted(trials, key=lambda t: t["task_name"]),
    }
    filename = re.sub(r"[^A-Za-z0-9_-]", "-", source["model"]) + ".json"
    (output / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in json.loads(args.sources.read_text()):
        audit(source, args.output)


if __name__ == "__main__":
    main()
