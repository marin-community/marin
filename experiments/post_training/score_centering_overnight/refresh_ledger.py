# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile safe controller CSV snapshots with frozen configs and all attempt costs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from analysis_json import read_json, write_json


def stage(job_id: str) -> str:
    if "sc-snowball-main-noise" in job_id:
        return "Snowball noise main"
    if "sc-snowball-main-fresh" in job_id:
        return "Snowball fresh controls"
    if "sc-snowball-main-stale" in job_id:
        return "Snowball staleness main"
    if "sc-snowball-incumbent" in job_id:
        return "Snowball incumbent qualification" if "runtime-qual" in job_id else "Snowball incumbent main"
    if "sc-core-runtime" in job_id:
        return "core GPU qualification"
    if "sc-snowball-runtime" in job_id:
        return "Snowball GPU qualification"
    if "sc-snowball-noise-calibration" in job_id:
        return "Snowball noise calibration"
    if "sc-snowball-stale-calibration" in job_id:
        return "Snowball staleness calibration"
    if "stale-path" in job_id:
        return "SkyRL staleness qualification"
    if "sc-skyrl-countdown-main" in job_id:
        return "SkyRL main"
    if "sc-qwen-transfer" in job_id:
        if "runtime-qual" in job_id:
            return "Qwen GPU qualification"
        return "Qwen transfer pilot" if "pilot" in job_id else "Qwen transfer main"
    return "SkyRL qualification"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    args = parser.parse_args()
    campaign = Path(__file__).resolve().parent
    repository = campaign.parents[2]
    registry = json.loads((campaign / "registry.json").read_text())
    jobs = {row["job_id"]: row for row in registry["jobs"]}
    frozen = {}
    for path in campaign.glob("**/*.yaml"):
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict) or config.get("run", {}).get("submission") != "detach":
            continue
        job_id = "/romain/" + config["run"]["id"]
        assert job_id not in frozen, f"ambiguous submitted configuration for {job_id}"
        frozen[job_id] = (path, config)

    audits = {}
    for path in (campaign / "results").glob("*.json*"):
        if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
            continue
        value = read_json(path)
        if isinstance(value, dict) and value.get("audit_passed") is True:
            audits[value["run_id"]] = {
                "path": str(path.relative_to(repository)),
                "archive_after_ms": value.get("archive_after_ms", 0),
                "completed_updates": value["completed_updates"],
            }

    for observed in csv.DictReader(args.jobs.open()):
        job_id = observed["job_id"]
        row = jobs.setdefault(job_id, {"job_id": job_id})
        if job_id in frozen:
            path, config = frozen[job_id]
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if "configuration" in row:
                assert row["configuration_sha256"] == checksum, f"submitted config changed: {path}"
            row.update(
                cluster=args.cluster,
                priority=config["iris"]["priority"],
                h100_nodes=config["iris"]["allocation"]["num_nodes"],
                h100_per_node=config["iris"]["allocation"]["gpus_per_node"],
                source=config["runtime"]["launcher_commit"],
                configuration=str(path.relative_to(repository)),
                configuration_sha256=checksum,
                configuration_sha256_basis="published launcher-input bytes",
                artifacts=config["artifacts"]["attempts_root"].removesuffix("/attempts"),
            )
        device = json.loads(observed["res_device_json"])["gpu"]
        assert device["variant"] == "H100" and device["count"] == row["h100_per_node"]
        assert int(observed["num_tasks"]) == row["h100_nodes"]
        state = int(observed["state"])
        row.update(
            controller_state=state,
            status="running" if state in (1, 2, 3, 9) else "terminal",
            priority_band=int(observed["priority_band"]),
            bundle_id=observed["bundle_id"],
            task_image=observed["task_image"],
        )
        assert row["priority_band"] == {"interactive": 2, "batch": 3}[row["priority"]]
        run_id = job_id.rsplit("/", 1)[-1]
        if run_id in audits:
            row["retained_audit"] = audits[run_id]["path"]
            row["scientific_completed_updates"] = audits[run_id]["completed_updates"]

    now = datetime.now(UTC)
    registry.update(observed_at=now.isoformat(), jobs=list(jobs.values()))
    costs = read_json(campaign / "costs.json.gz")
    attempts = {(row["task_id"], row["attempt_uid"]): row for row in costs["attempts"]}
    active = [row for row in costs.get("active_attempts", []) if row["cluster"] != args.cluster]
    observed_attempts = list(csv.DictReader(args.attempts.open()))
    starts = defaultdict(list)
    for observed in observed_attempts:
        if observed["started_at_ms"]:
            starts[(observed["task_id"].rsplit("/", 1)[0], observed["attempt_id"])].append(
                int(observed["started_at_ms"])
            )
    for observed in observed_attempts:
        job = jobs[observed["task_id"].rsplit("/", 1)[0]]
        key = observed["task_id"], observed["attempt_uid"]
        row = {**attempts.get(key, {}), **observed}
        row.update(
            stage=stage(job["job_id"]),
            cluster=args.cluster,
            priority=job["priority"],
            gpus=job["h100_per_node"],
            source=job["source"],
            configuration_sha256=job["configuration_sha256"],
        )
        start = int(row["started_at_ms"] or 0)
        finish = int(row["finished_at_ms"] or 0)
        if finish:
            assert not start or finish >= start
            row["allocated_seconds"] = (finish - start) / 1000 if start else 0
            row["allocated_h100_hours"] = row["allocated_seconds"] * row["gpus"] / 3600
            row["outcome"] = (
                "task process succeeded; audit pending" if int(row["state"]) == 4 else "failed engineering attempt"
            )
            if row["terminal_reason"].startswith("PodDeleted:"):
                row["outcome"] = "infrastructure interruption; controller restarted the job"
            run_id = job["job_id"].rsplit("/", 1)[-1]
            audit = audits.get(run_id)
            attempt_starts = starts[(job["job_id"], row["attempt_id"])]
            selected_attempt = bool(audit and attempt_starts and min(attempt_starts) == audit["archive_after_ms"])
            row.pop("retained_audit", None)
            row["scientific_attempt_selected"] = selected_attempt
            if selected_attempt:
                row["retained_audit"] = audit["path"]
                row["scientific_outputs_complete"] = True
                if int(row["state"]) == 4:
                    row["outcome"] = "task process succeeded in selected audited scientific attempt"
            else:
                row.pop("scientific_outputs_complete", None)
            attempts[key] = row
        elif start:
            row["observed_at"] = now.isoformat()
            row["allocated_h100_hours_so_far"] = (now.timestamp() - start / 1000) * row["gpus"] / 3600
            active.append(row)
    totals = defaultdict(float)
    for row in attempts.values():
        totals[row["stage"]] += row["allocated_h100_hours"]
    costs.update(
        observed_at=now.isoformat(),
        attempts=list(attempts.values()),
        active_attempts=active,
        terminal_h100_hours=sum(totals.values()),
        terminal_h100_hours_by_stage=dict(totals),
        active_costs="Active intervals are partial observations and excluded from terminal stage totals.",
    )
    (campaign / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    write_json(campaign / "costs.json.gz", costs)
    print(
        json.dumps(
            {"jobs": len(jobs), "terminal_attempts": len(attempts), "active_attempts": len(active), "hours": totals}
        )
    )


if __name__ == "__main__":
    main()
