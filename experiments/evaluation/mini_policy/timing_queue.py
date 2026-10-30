# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen research timing queue locally; all evaluators execute on Iris."""

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.connect import open_iris_client
from marin.evaluation.evalchemy.runtime import evalchemy_runtime_source_hash
from marin.evaluation.harbor.driver_config import harbor_runtime_descriptor

ACTIVE = {"pending", "running"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature():
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "launcher": digest(Path("experiments/evaluation/mini_policy/launch_timing.py")),
        "evalchemy": evalchemy_runtime_source_hash(),
        "harbor": harbor_runtime_descriptor(),
        "marin_changes": (
            hashlib.sha256(
                subprocess.check_output(
                    [
                        "git",
                        "diff",
                        "--",
                        "lib/marin",
                        "experiments/evaluation/launch.py",
                        "experiments/evaluation/evals.py",
                    ]
                )
            ).hexdigest()
        ),
    }


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    frozen = signature()
    queue_hash = hashlib.sha256(plan_bytes).hexdigest()
    entries = []
    for item in plan["launches"]:
        receipt = args.state.parent / "queue-receipts" / (item["run_id"] + ".json")
        entries.append({**item, "receipt": str(receipt), "job": item.get("existing_job"), "state": "waiting_for_slot"})
    args.state.parent.mkdir(exist_ok=True, parents=True)
    prefix = f"/{getpass.getuser()}/eval-mini-policy-"
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        while True:
            if args.plan.read_bytes() != plan_bytes or signature() != frozen:
                raise RuntimeError("Frozen queue inputs/runtime changed; review provenance before restarting launches")
            jobs = {str(job.job_id): job for job in client.list_jobs(prefix=prefix, limit=5000)}
            roots = {name: row for name, row in jobs.items() if name.count("/") == 2}
            for item in entries:
                receipt = Path(item["receipt"])
                if item["job"] is None and receipt.exists():
                    item["job"] = json.loads(receipt.read_text())["job"]
                if item["job"] is None:
                    matches = [name for name in roots if name.startswith(f"/{getpass.getuser()}/eval-{item['run_id']}-")]
                    if len(matches) > 1:
                        raise RuntimeError(f"Duplicate root for {item['run_id']}")
                    if matches:
                        item["job"] = matches[0]
                if item["job"] is not None:
                    if item["job"] not in roots:
                        roots[item["job"]] = client.job_status(item["job"])
                    item["state"] = str(roots[item["job"]].state)
            live = sum(str(row.state) in ACTIVE for row in roots.values())
            state = {
                "observed_utc": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "plan_sha256": queue_hash,
                "runtime": frozen,
                "active_roots": live,
                "entries": entries,
            }
            save(args.state, state)
            if any(item["state"] == "failed" for item in entries):
                raise RuntimeError("A timing root failed; remediate before submitting more arms")
            pending = [item for item in entries if item["job"] is None]
            if not pending:
                print(json.dumps({"queue_submitted": True, "active_roots": live}), flush=True)
                return
            if live < plan["max_in_flight"]:
                item = pending[0]
                manifest = Path(item["configs_dir"]) / "manifest.json"
                if (
                    digest(manifest) != item["manifest_sha256"]
                    or digest(Path(item["model_config"])) != item["model_config_sha256"]
                ):
                    raise RuntimeError(f"Frozen config changed: {item['run_id']}")
                command = [
                    sys.executable,
                    "-m",
                    "experiments.evaluation.mini_policy.launch_timing",
                    "--component",
                    item["component"],
                    "--model-config",
                    item["model_config"],
                    "--configs-dir",
                    item["configs_dir"],
                    "--run-id",
                    item["run_id"],
                    "--output",
                    item["receipt"],
                ]
                for option in ("judge_model_config", "finalization_config"):
                    if option in item:
                        command += ["--" + option.replace("_", "-"), item[option]]
                log = Path("/tmp") / (item["run_id"] + ".launch.log")
                with log.open("w") as output:
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    item["state"] = "launch_error"
                    state["error_log"] = str(log)
                    save(args.state, state)
                    raise RuntimeError(f"Launch failed for {item['run_id']}; inspect {log}")
                item["job"] = json.loads(Path(item["receipt"]).read_text())["job"]
                item["state"] = "submitted"
                save(args.state, state)
                print(
                    json.dumps(
                        {
                            "submitted": item["job"],
                            "component": item["component"],
                            "method": item["method"],
                            "budget": item["budget"],
                        }
                    ),
                    flush=True,
                )
                continue
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
