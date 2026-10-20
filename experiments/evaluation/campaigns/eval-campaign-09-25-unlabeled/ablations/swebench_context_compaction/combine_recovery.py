# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit retained and recovery results without publishing a campaign score."""

import argparse
import hashlib
import json
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from marin.evaluation.harbor.driver_config import preflight_harbor_configs
from marin.evaluation.harbor.runner import _read_trial
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath


def combine(manifest: dict, retained: list[dict], recovered: list[dict]) -> dict:
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
    records = retained + recovered
    rewards = [row["reward"] for row in records if row["scored"]]
    return {
        "expected_tasks": len(expected),
        "completed_tasks": len(records),
        "scored_tasks": len(rewards),
        "unscored_tasks": len(records) - len(rewards),
        "pending_tasks": sorted(expected - seen),
        "all_tasks_scored": len(rewards) == len(expected),
        "mean_reward": statistics.mean(rewards) if rewards else None,
        "reward_stderr": statistics.stdev(rewards) / len(rewards) ** 0.5 if len(rewards) > 1 else None,
        "trials": sorted(records, key=lambda row: row["task_name"]),
    }


def audit(manifest_path: Path, recovery_results: str) -> dict:
    """Read frozen source evidence and new results using the pinned taxonomy."""
    manifest = json.loads(manifest_path.read_text())
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
            }

        with ThreadPoolExecutor(max_workers=12) as pool:
            retained = list(pool.map(lambda row: read(row["source_uri"], row), source))
            recovered = list(pool.map(lambda path: read("s3://" + path + "/result.json", None), directories))
    report = combine(
        manifest, [row for row in retained if row is not None], [row for row in recovered if row is not None]
    )
    return {"manifest": str(manifest_path), "recovery_results": recovery_results, **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recovery-results", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.recovery_results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("trials", "pending_tasks")}))


if __name__ == "__main__":
    main()
