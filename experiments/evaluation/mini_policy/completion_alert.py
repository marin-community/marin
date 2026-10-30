# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wake the issue-update workflow after the submitted timing roots terminate."""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.connect import open_iris_client

ACTIVE = {"pending", "running"}
TERMINAL = {"succeeded", "failed", "killed", "cancelled"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    folder = args.directory
    if (folder / "finished").exists():
        return
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        jobs = client.list_jobs(prefix="/benfeuer/eval-mini-policy-", limit=5000)
        roots = {str(job.job_id): str(job.state) for job in jobs if str(job.job_id).count("/") == 2}
    baseline_path = folder / "baseline.json"
    if args.initialize:
        baseline_path.write_text(
            json.dumps({"roots": roots, "watched": [j for j, s in roots.items() if s in ACTIVE]}, indent=2)
        )
    baseline = json.loads(baseline_path.read_text())
    watched = set(baseline["watched"]) | (set(roots) - set(baseline["roots"]))
    missing = watched - set(roots)
    if missing:
        raise RuntimeError(f"Tracked roots missing from controller: {sorted(missing)}")
    current = {j: roots[j] for j in sorted(watched)}
    snapshot = {"observed_utc": datetime.now(UTC).isoformat(), "watched": current, "all_roots": roots}
    (folder / "latest.json").write_text(json.dumps(snapshot, indent=2))
    logging.info("Watched roots: %s", current)
    if args.check_only or not current or any(s not in TERMINAL for s in current.values()):
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with (folder / f"followup-{stamp}.log").open("w") as log:
        subprocess.run(
            [
                "/opt/homebrew/bin/codex",
                "exec",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                "--sandbox",
                "danger-full-access",
                "-c",
                "approval_policy=never",
                "--output-last-message",
                str(folder / "followup-result.md"),
                "-",
            ],
            input=(folder / "FOLLOWUP.md").read_text(),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=True,
        )
    if not (folder / "finished").exists():
        raise RuntimeError("Follow-up returned without verifying and marking the issue update")


if __name__ == "__main__":
    main()
