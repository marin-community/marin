# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit retained and recovery results without publishing a campaign score."""

import argparse
import hashlib
import json
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from marin.evaluation.harbor.driver_config import preflight_harbor_configs
from marin.evaluation.harbor.runner import _read_trial
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def combine(manifest: dict, retained: list[dict], recovered: list[dict], outages: list[dict]) -> dict:
    """Combine disjoint task outcomes while keeping missing and unscored counts."""
    expected = {Path(task["path"]).name for task in manifest["task_descriptors"]}
    old_tasks = set(manifest["retained_tasks"])
    new_tasks = set(manifest["recovery_tasks"])
    if old_tasks & new_tasks or old_tasks | new_tasks != expected:
        raise ValueError("manifest task partition is not disjoint and exhaustive")
    seen = set()
    for records, allowed in ((retained, old_tasks), (recovered, new_tasks)):
        for record in records:
            task = record["task_name"]
            if task not in allowed or task in seen:
                raise ValueError(f"unexpected or duplicate task: {task}")
            seen.add(task)
    if {row["task_name"] for row in retained} != old_tasks:
        raise ValueError("retained evidence is missing")
    records = [dict(row) for row in retained + recovered]
    for record in records:
        phase = record.get("agent_execution") or {}
        if not phase.get("started_at") or not phase.get("finished_at"):
            continue
        start = datetime.fromisoformat(phase["started_at"])
        finish = datetime.fromisoformat(phase["finished_at"])
        for outage in outages:
            if not record["source_uri"].startswith(outage["results_root"].rstrip("/") + "/"):
                continue
            if finish <= datetime.fromisoformat(outage["started_at"]):
                continue
            if outage["finished_at"] and start >= datetime.fromisoformat(outage["finished_at"]):
                continue
            record["scored"] = False
            record["exclusion_reason"] = "inference_service_outage"
            break
    rewards = [row["reward"] for row in records if row["scored"]]
    return {
        "expected_tasks": len(expected),
        "completed_tasks": len(records),
        "scored_tasks": len(rewards),
        "unscored_tasks": len(records) - len(rewards),
        "outage_excluded_tasks": sum(row.get("exclusion_reason") == "inference_service_outage" for row in records),
        "pending_tasks": sorted(expected - seen),
        "all_tasks_scored": len(rewards) == len(expected),
        "mean_reward": statistics.mean(rewards) if rewards else None,
        "reward_stderr": statistics.stdev(rewards) / len(rewards) ** 0.5 if len(rewards) > 1 else None,
        "trials": sorted(records, key=lambda row: row["task_name"]),
    }


def audit(manifest_path: Path, recovery_results: str, outages_path: Path) -> dict:
    """Read frozen source evidence and new results using the pinned taxonomy."""
    manifest = json.loads(manifest_path.read_text())
    outages_payload = outages_path.read_bytes()
    outages = json.loads(outages_payload)
    config_path = manifest_path.with_name("harbor.yaml")
    taxonomy = preflight_harbor_configs([(config_path, {})])[0].error_taxonomy
    if taxonomy.commit != manifest["harbor_commit"]:
        raise ValueError("recovery taxonomy differs from the manifest pin")
    filesystem, root = filesystem_for(recovery_results.rstrip("/"))
    identity = json.loads(filesystem.cat(root + "/harbor_resume_identity.json"))
    if taxonomy.commit not in identity["runtime"]:
        raise ValueError("recovery runtime differs from the manifest pin")
    jobs = filesystem.ls(root + "/harbor_jobs", detail=False)
    if len(jobs) != 1:
        raise ValueError("expected exactly one recovery Harbor job")
    job = jobs[0]
    config = json.loads(filesystem.cat(job + "/config.json"))
    if set(config["datasets"][0]["task_names"]) != set(manifest["recovery_tasks"]):
        raise ValueError("launched recovery task selection differs from the manifest")
    source = [row for row in manifest["observed_trials"] if row["recovery_reason"] is None]
    directories = [path for path in filesystem.ls(job, detail=False) if "__" in path.rsplit("/", 1)[-1]]
    with tempfile.TemporaryDirectory(prefix="swebench-combine-") as temporary:

        def read(uri: str, expected: dict | None) -> dict | None:
            fs, path = filesystem_for(uri)
            try:
                payload = fs.cat(path)
            except FileNotFoundError:
                if expected is not None:
                    raise
                return None
            digest = hashlib.sha256(payload).hexdigest()
            if expected is not None and digest != expected["sha256"]:
                raise ValueError(f"retained source changed: {expected['task_name']}")
            data = json.loads(payload)
            snapshot = Path(temporary) / (digest + ".json")
            snapshot.write_bytes(payload)
            trial = _read_trial(StoragePath.parse(str(snapshot)), taxonomy)
            if expected is not None and (not trial.scored or data["task_name"] != expected["task_name"]):
                raise ValueError("retained source is no longer a scored matching task")
            return {
                "task_name": data["task_name"],
                "source_uri": uri,
                "sha256": digest,
                "reward": trial.reward,
                "scored": trial.scored,
                "error_type": trial.error["type"] if trial.error else None,
                "agent_execution": data.get("agent_execution"),
            }

        with ThreadPoolExecutor(max_workers=12) as pool:
            retained = list(pool.map(lambda row: read(row["source_uri"], row), source))
            recovered = list(pool.map(lambda path: read("s3://" + path + "/result.json", None), directories))
    report = combine(
        manifest, [row for row in retained if row is not None], [row for row in recovered if row is not None], outages
    )
    return {
        "manifest": str(manifest_path),
        "recovery_results": recovery_results,
        "service_outages_sha256": hashlib.sha256(outages_payload).hexdigest(),
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recovery-results", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-outages", type=Path, default=Path(__file__).with_name("service_outages.json"))
    args = parser.parse_args()
    report = audit(args.manifest, args.recovery_results, args.service_outages)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("trials", "pending_tasks")}))


if __name__ == "__main__":
    main()
