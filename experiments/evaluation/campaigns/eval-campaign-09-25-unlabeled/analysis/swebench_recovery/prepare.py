# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a disjoint recovery subset from the campaign's saved trace audit."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from marin.evaluation.harbor.driver_config import harbor_runtime_descriptor
from rigging.filesystem.buckets import filesystem_for

TRANSPORT_CLASSES = {
    "ReadTimeout",
    "ConnectTimeout",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "AuthenticationError",
    "ConnectError",
}
SCOREABLE = {None, "AgentTimeoutError", "ContextLengthExceededError", "NonZeroAgentExitCodeError"}


def recovery_reason(row: dict) -> str | None:
    error = row["exception_type"]
    if any(
        name.rsplit(".", 1)[-1] in TRANSPORT_CLASSES
        for log in row.get("agent_logs", [])
        for name in log["terminal_error_classes"] + log.get("retry_error_classes", [])
    ):
        return "agent_log_transport_failure"
    if error not in SCOREABLE or not isinstance(row["reward"], int | float):
        return f"unscored:{error or 'missing_reward'}"
    reviewed = row.get("manual_review", {}).get("decision") == "retain_scored_model_failure"
    if row.get("transport_failure_observed") and not reviewed:
        raise ValueError(f"transport text needs manual review: {row['task_name']}")
    if error == "NonZeroAgentExitCodeError":
        if not reviewed:
            raise ValueError(f"nonzero exit needs manual review: {row['task_name']}")
    return None


def apply_reviews(report: dict, reviews: list[dict]) -> None:
    """Bind manual decisions to the exact audited trial and agent log."""
    by_task = {row["task_name"]: row for row in report["trials"]}
    for review in reviews:
        if review["model"] != report["model"]:
            continue
        row = by_task[review["task_name"]]
        if row["sha256"] != review["result_sha256"] or review["agent_log_sha256"] not in {
            log["sha256"] for log in row.get("agent_logs", [])
        }:
            raise ValueError(f"review evidence differs from audit: {row['task_name']}")
        if review["decision"] != "retain_scored_model_failure" or not review["evidence"]:
            raise ValueError(f"unsupported review decision: {row['task_name']}")
        row["manual_review"] = review


def enrich_retry_evidence(report: dict) -> None:
    """Recognize transport retries in the exact logs captured by the audit."""
    for row in report["trials"]:
        if not row.get("transport_failure_observed"):
            continue
        for log in row.get("agent_logs", []):
            if not log["transport_errors"]:
                continue
            fs, path = filesystem_for(log["source_uri"])
            payload = fs.cat(path)
            if hashlib.sha256(payload).hexdigest() != log["sha256"]:
                raise ValueError(f"source log changed: {log['source_uri']}")
            log["retry_error_classes"] = re.findall(
                r"(?m)^Retrying <unknown> in [0-9.]+ seconds as it raised ([A-Za-z_.]+):",
                payload.decode(errors="replace"),
            )


def partition(report: dict, task_names: set[str]) -> tuple[list[dict], list[str]]:
    observed = set()
    retained = []
    for row in report["trials"]:
        name = row["task_name"]
        if name in observed or name not in task_names:
            raise ValueError(f"duplicate or unknown task: {name}")
        observed.add(name)
        if recovery_reason(row) is None:
            retained.append(row)
    return retained, sorted(task_names - {row["task_name"] for row in retained})


def prepare(audit: Path, base_config: Path, output: Path, reviews: Path | None = None) -> dict:
    payload = audit.read_bytes()
    report = json.loads(payload)
    if reviews is not None:
        apply_reviews(report, json.loads(reviews.read_text()))
    enrich_retry_evidence(report)
    config = yaml.safe_load(base_config.read_text())
    dataset = config["datasets"][0]
    response = httpx.get(dataset["registry_url"], timeout=60)
    response.raise_for_status()
    matches = [d for d in response.json() if d["name"] == dataset["name"] and d["version"] == dataset["version"]]
    if len(matches) != 1:
        raise ValueError("expected one pinned dataset")
    tasks = matches[0]["tasks"]
    names = {Path(task["path"]).name for task in tasks}
    if len(names) != 500 or len(config["datasets"]) != 1 or config["n_attempts"] != 1:
        raise ValueError("expected the 500-task single-attempt SWE-bench cohort")
    retained, recovery = partition(report, names)
    if not recovery:
        raise ValueError("no recovery needed")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": report["model"],
        "source_root": report["root"],
        "source_summary": report["summary_uri"],
        "source_runtime": report["runtime"],
        "recovery_runtime": harbor_runtime_descriptor(),
        "source_must_be_terminal_before_launch": True,
        "audit_sha256": hashlib.sha256(payload).hexdigest(),
        "base_config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
        "retained_trials": retained,
        "recovery_tasks": recovery,
        "excluded_trials": [
            dict(row, recovery_reason=recovery_reason(row)) for row in report["trials"] if row["task_name"] in recovery
        ],
    }
    config["datasets"][0]["task_names"] = recovery
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    (output / "swebench-verified-recovery.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {"retained": len(retained), "recovery": len(recovery), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.base_config, args.output, args.reviews)))


if __name__ == "__main__":
    main()
