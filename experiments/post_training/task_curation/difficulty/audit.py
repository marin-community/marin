# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate a regional backfill on the complete captured legacy reason audit."""

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from experiments.post_training.task_curation.difficulty import legacy
from experiments.post_training.task_curation.difficulty.launch import cpu_job_command


@dataclass(frozen=True)
class ReasonAudit:
    run_roots: tuple[str, ...]
    original_reason_count: int
    report_sha256: str
    audit_log_sha256: str
    reason_mapping_sha256: str


def audit_captured_runs(report_path: Path, audit_log_path: Path) -> ReasonAudit:
    """Validate captured reasons against current mappings; reject incomplete scans."""
    report_bytes = report_path.read_bytes()
    log_bytes = audit_log_path.read_bytes()
    report = json.loads(report_bytes)
    log = log_bytes.decode()
    if "Complete reason audit failed:" not in log:
        raise ValueError("The raw-receipt audit did not reach its complete failure report")
    roots = {run["root"] for run in report["unattributed_runs"]}
    reasons = set()
    for solver in report["solvers"].values():
        for run in solver["runs"]:
            roots.add(run["root"])
            for field in ("pending_reasons", "not_measured_reasons"):
                reasons.update(run["catalog"][field])
    if len(roots) != report["runs_discovered"]:
        raise ValueError("The catalog report does not cover every discovered run root")
    roots.update(re.findall(re.escape(report["base"].rstrip("/")) + r"/[^/\s]+/[^/\s\"']+", log))
    for literal in re.findall(r"Unmapped difficulty gap reason: (.+)", log):
        reasons.add(ast.literal_eval(literal))
    for reason in sorted(reasons):
        legacy.legacy_gaps(reason)
    return ReasonAudit(
        tuple(sorted(roots)),
        len(reasons),
        hashlib.sha256(report_bytes).hexdigest(),
        hashlib.sha256(log_bytes).hexdigest(),
        hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
    )


def launch_import(
    report_path: Path, audit_log_path: Path, base: str, ledger: str, job_name: str, refresh_roots: tuple[str, ...]
) -> ReasonAudit:
    """Submit only after the full captured audit exits successfully in this process."""
    audit = audit_captured_runs(report_path, audit_log_path)
    print(
        "DIFFICULTY_REASON_AUDIT_OK "
        + json.dumps(
            {
                "run_roots": len(audit.run_roots),
                "original_reason_count": audit.original_reason_count,
                "report_sha256": audit.report_sha256,
                "audit_log_sha256": audit.audit_log_sha256,
                "reason_mapping_sha256": audit.reason_mapping_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    command = cpu_job_command(job_name)
    command.extend(
        [
            "--",
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty",
            "import",
            "--discover-root",
            base,
            "--ledger",
            ledger,
        ]
    )
    for root in refresh_roots:
        command.extend(["--refresh-run-root", root])
    subprocess.run(command, check=True)
    return audit
