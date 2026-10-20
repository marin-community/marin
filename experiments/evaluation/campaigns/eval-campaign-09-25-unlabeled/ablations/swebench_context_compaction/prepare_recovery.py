# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot retained SWE-bench results and materialize a fixed-runtime task subset.

This command does not cancel or launch jobs. Run it again after the source job
is terminal before using its manifest for the final combined aggregate.
"""

import argparse
import concurrent.futures
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from marin.evaluation.harbor.driver_config import preflight_harbor_configs
from marin.evaluation.harbor.runner import _read_trial
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

EXPECTED_TASKS = 500


def recovery_reason(result: dict, scored: bool, agent_deadline: float) -> str | None:
    """Exclude outcomes affected by the old cleanup bug or setup failures."""
    exception = (result.get("exception_info") or {}).get("exception_type")
    phase = result.get("agent_execution") or {}
    if phase.get("started_at") and phase.get("finished_at"):
        elapsed = (
            datetime.fromisoformat(phase["finished_at"]) - datetime.fromisoformat(phase["started_at"])
        ).total_seconds()
        if elapsed >= agent_deadline:
            return "old_runtime_agent_deadline"
    if exception == "AgentTimeoutError":
        return "old_runtime_agent_deadline"
    if exception == "NonZeroAgentExitCodeError" and not phase:
        return "agent_setup_failed_before_execution"
    if not scored:
        return f"unscored:{exception or 'missing_verifier_result'}"
    return None


def partition_tasks(task_names: set[str], evidence: list[dict]) -> tuple[list[str], list[str]]:
    """Partition the original cohort without duplicates or unknown tasks."""
    seen = set()
    retained = set()
    for trial in evidence:
        task = trial["task_name"]
        if task not in task_names or task in seen:
            raise ValueError(f"unknown or duplicate source task: {task}")
        seen.add(task)
        if trial["recovery_reason"] is None:
            retained.add(task)
    return sorted(retained), sorted(task_names - retained)


def registry_tasks(dataset: dict) -> list[dict]:
    response = httpx.get(dataset["registry_url"], timeout=60)
    response.raise_for_status()
    matches = [row for row in response.json() if row["name"] == dataset["name"] and row["version"] == dataset["version"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one pinned dataset entry")
    return sorted(matches[0]["tasks"], key=lambda task: task["path"])


def followup_manifest(previous: dict, report: dict) -> dict:
    """Preserve scored evidence across generations without retrying valid timeouts."""
    task_names = {Path(task["path"]).name for task in previous["task_descriptors"]}
    evidence = [
        {
            **row,
            "recovery_reason": (
                None
                if row["scored"]
                else row.get("exclusion_reason") or f"unscored:{row['error_type'] or 'missing_verifier_result'}"
            ),
        }
        for row in report["trials"]
    ]
    retained, recovery = partition_tasks(task_names, evidence)
    if not set(previous["retained_tasks"]).issubset(row["task_name"] for row in evidence):
        raise ValueError("combined audit is missing previously retained evidence")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "source_must_be_terminal_before_final_use": True,
        "harbor_commit": previous["harbor_commit"],
        "task_descriptors": previous["task_descriptors"],
        "retained_tasks": retained,
        "recovery_tasks": recovery,
        "observed_trials": sorted(evidence, key=lambda trial: trial["task_name"]),
    }


def write_recovery(config: dict, manifest: dict, destination: Path) -> dict:
    """Materialize the manifest and its exact task subset without launching."""
    if not manifest["recovery_tasks"]:
        raise ValueError("no tasks require recovery; do not emit an empty task filter")
    config["datasets"][0]["task_names"] = manifest["recovery_tasks"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "harbor.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "retained": len(manifest["retained_tasks"]),
        "recovery": len(manifest["recovery_tasks"]),
        "destination": str(destination),
    }


def prepare_followup(audit_path: Path, base_config: Path, destination: Path) -> dict:
    """Prepare another recovery from a combined audit of terminal source runs."""
    audit_payload = audit_path.read_bytes()
    report = json.loads(audit_payload)
    previous_path = Path(report["manifest"])
    previous_payload = previous_path.read_bytes()
    previous = json.loads(previous_payload)
    config = yaml.safe_load(base_config.read_text())
    if config["n_attempts"] != 1 or len(config["datasets"]) != 1:
        raise ValueError("this recovery requires one attempt and one dataset")
    previous_config = yaml.safe_load(previous_path.with_name("harbor.yaml").read_text())
    previous_config["datasets"][0].pop("task_names", None)
    config["datasets"][0].pop("task_names", None)
    if config != previous_config:
        raise ValueError("follow-up recovery must preserve benchmark settings apart from task selection")
    taxonomy = preflight_harbor_configs([(base_config, {})])[0].error_taxonomy
    if taxonomy.commit != previous["harbor_commit"]:
        raise ValueError("follow-up recovery must preserve the audited runtime pin")
    tasks = registry_tasks(config["datasets"][0])
    if tasks != previous["task_descriptors"] or len(tasks) != EXPECTED_TASKS:
        raise ValueError("follow-up recovery must preserve the original 500 task descriptors")
    manifest = followup_manifest(previous, report)
    manifest.update(
        source_audit=str(audit_path),
        source_audit_sha256=hashlib.sha256(audit_payload).hexdigest(),
        previous_manifest=str(previous_path),
        previous_manifest_sha256=hashlib.sha256(previous_payload).hexdigest(),
        base_config_sha256=hashlib.sha256(base_config.read_bytes()).hexdigest(),
    )
    return write_recovery(config, manifest, destination)


def prepare(source: str, base_config: Path, destination: Path) -> dict:
    """Write a sanitized evidence manifest and canonical recovery config."""
    config = yaml.safe_load(base_config.read_text())
    if config["n_attempts"] != 1 or len(config["datasets"]) != 1:
        raise ValueError("this recovery requires one attempt and one dataset")
    taxonomy = preflight_harbor_configs([(base_config, {})])[0].error_taxonomy
    filesystem, root = filesystem_for(source.rstrip("/"))
    jobs = filesystem.ls(root + "/harbor_jobs", detail=False)
    if len(jobs) != 1:
        raise ValueError("expected one source Harbor job")
    job = jobs[0]
    original_config = json.loads(filesystem.cat(job + "/config.json"))
    old_tasks = registry_tasks(original_config["datasets"][0])
    new_tasks = registry_tasks(config["datasets"][0])
    if old_tasks != new_tasks or len(new_tasks) != EXPECTED_TASKS:
        raise ValueError("source and recovery must resolve the same 500 task descriptors")
    task_names = {Path(task["path"]).name for task in new_tasks}
    deadline = original_config["agents"][0]["override_timeout_sec"]
    directories = [path for path in filesystem.ls(job, detail=False) if "__" in path.rsplit("/", 1)[-1]]

    with tempfile.TemporaryDirectory(prefix="swebench-recovery-") as temporary:

        def inspect(directory: str) -> dict | None:
            key = directory + "/result.json"
            try:
                payload = filesystem.cat(key)
            except FileNotFoundError:
                return None
            digest = hashlib.sha256(payload).hexdigest()
            result = json.loads(payload)
            snapshot = Path(temporary) / f"{digest}.json"
            snapshot.write_bytes(payload)
            trial = _read_trial(StoragePath.parse(str(snapshot)), taxonomy)
            return {
                "task_name": result["task_name"],
                "trial_name": result["trial_name"],
                "task_checksum": result["task_checksum"],
                "source_uri": "s3://" + key,
                "sha256": digest,
                "reward": trial.reward,
                "recovery_reason": recovery_reason(result, trial.scored, deadline),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            evidence = [row for row in pool.map(inspect, directories) if row is not None]

    retained, recovery = partition_tasks(task_names, evidence)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_results": source,
        "source_must_be_terminal_before_final_use": True,
        "harbor_commit": taxonomy.commit,
        "base_config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
        "task_descriptors": new_tasks,
        "retained_tasks": retained,
        "recovery_tasks": recovery,
        "observed_trials": sorted(evidence, key=lambda trial: trial["task_name"]),
    }
    return write_recovery(config, manifest, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-results")
    source.add_argument("--combined-audit", type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.combined_audit:
        result = prepare_followup(args.combined_audit, args.base_config, args.destination)
    else:
        result = prepare(args.source_results, args.base_config, args.destination)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
