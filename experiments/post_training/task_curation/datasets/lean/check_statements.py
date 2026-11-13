# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate every ProofNet-Verified contract with the Lean grader's reference check, and report the result.

Each row is converted into the grader's contract and its reference, the statement proved by ``sorry``
followed by the check ``lean_grade.py`` runs, is compiled with the given toolchain under the grader's
reference timeout. Run it from a Marin checkout on a machine with the Lean and Mathlib the grader image
installs, against the pinned ``proofnet-verified.jsonl``::

    uv run python -m experiments.post_training.task_curation.datasets.lean.check_statements \\
        proofnet-verified.jsonl --lean LEAN --lean-path LEAN_PATH --mathlib-commit COMMIT

The first output line records what was checked; each further line is one row's result. It exits non-zero
when a row other than the expected rejections fails. At the pinned revision, with the grader image's
toolchain, all 365 converted contracts validate; rerun it when the revision or the toolchain moves.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.lean.lean_grade import (
    LEAN_THREADS,
    MODULE_FILE,
    OBJECT_FILE,
    STEP_SECONDS,
    reference_source,
)
from experiments.post_training.task_curation.datasets.lean.proofs import PROOFNET_VERIFIED_REVISION, proofnet_contract

EXPECTED_REJECTIONS = {35: "no_sorry_proof", 351: "repeated_exercise"}
"""Rows the converter rejects at the pinned revision: row 35's statement carries its proof, and row 351
repeats row 168's exercise."""


def check_row(row: dict, lean: str, lean_path: str) -> dict:
    """One row's result: its rejection reason, or whether its reference validates and how long it took."""
    index = row.get("index")
    contract = proofnet_contract(row)
    if isinstance(contract, ImportRejection):
        return {"index": index, "status": "rejected", "reason": contract.reason}
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, MODULE_FILE).write_text(reference_source(contract))
        start = time.monotonic()
        try:
            result = subprocess.run(
                [lean, LEAN_THREADS, "-o", OBJECT_FILE, MODULE_FILE],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=STEP_SECONDS["reference"],
                env={"LEAN_PATH": lean_path, "PATH": "/usr/bin:/bin"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"index": index, "status": "timeout"}
    status = "validates" if result.returncode == 0 else "fails"
    output = (result.stdout + result.stderr)[-2000:]
    return {"index": index, "status": status, "seconds": round(time.monotonic() - start, 1), "output": output}


def failed_unexpectedly(result: dict) -> bool:
    if result["status"] == "validates":
        return False
    return not (result["status"] == "rejected" and EXPECTED_REJECTIONS.get(result["index"]) == result.get("reason"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("rows", type=Path)
    parser.add_argument("--lean", required=True)
    parser.add_argument("--lean-path", required=True)
    parser.add_argument("--mathlib-commit", required=True, help="the Mathlib commit LEAN_PATH points into")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    payload = args.rows.read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    version = subprocess.run([args.lean, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    provenance = {
        "revision": PROOFNET_VERIFIED_REVISION,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "lean": version,
        "mathlib": args.mathlib_commit,
    }
    print(json.dumps({"checked": provenance}), flush=True)
    failures = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(lambda row: check_row(row, args.lean, args.lean_path), rows):
            failures += failed_unexpectedly(result)
            print(json.dumps(result), flush=True)
    if failures:
        print(f"{failures} rows failed unexpectedly", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
