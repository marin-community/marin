# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate every contract a Lean source converts with the grader's reference check, and report the result.

Each ProofNet-Verified or NuminaMath-LEAN row is converted into the grader's contract, and its reference,
the statement proved by ``sorry`` followed by the check ``lean_grade.py`` runs, is compiled with the given
toolchain under the grader's reference timeout. Run it from a Marin checkout on a machine with the Lean
and Mathlib the grader image installs, against the source's pinned file; NuminaMath-LEAN also reads
Prime Intellect's two lists from a directory holding the files its recipe stages::

    uv run python -m experiments.post_training.task_curation.datasets.lean.check_statements proofnet \\
        proofnet-verified.jsonl --lean LEAN --lean-path LEAN_PATH --mathlib-commit COMMIT
    uv run python -m experiments.post_training.task_curation.datasets.lean.check_statements numina \\
        train-00000-of-00001.parquet --prime-lists DIR --lean LEAN --lean-path LEAN_PATH --mathlib-commit COMMIT

The first output line records what was checked; each further line is one row's result. It exits non-zero
when a reference does not validate or ProofNet-Verified rejects a row other than ``EXPECTED_REJECTIONS``.
At the pinned revisions, with the grader image's toolchain, all 365 ProofNet-Verified contracts and all
7,628 NuminaMath-LEAN contracts validate; rerun it when a revision or the toolchain moves.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.lean import numina
from experiments.post_training.task_curation.datasets.lean.lean_grade import (
    LEAN_THREADS,
    MODULE_FILE,
    OBJECT_FILE,
    STEP_SECONDS,
    Contract,
    reference_source,
)
from experiments.post_training.task_curation.datasets.lean.proofs import PROOFNET_VERIFIED_REVISION, proofnet_contract

EXPECTED_REJECTIONS = {35: "no_sorry_proof", 351: "repeated_exercise"}
"""ProofNet-Verified rows the converter rejects at the pinned revision: row 35's statement carries its
proof, and row 351 repeats row 168's exercise. NuminaMath-LEAN rejects many rows by design."""


def proofnet_contracts(path: Path, _args: argparse.Namespace) -> Iterator[tuple[object, Contract | ImportRejection]]:
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            yield row.get("index"), proofnet_contract(row)


def numina_contracts(path: Path, args: argparse.Namespace) -> Iterator[tuple[object, Contract | ImportRejection]]:
    lists = {
        name: frozenset(json.loads(gzip.decompress((args.prime_lists / f"{name}.json.gz").read_bytes())))
        for name in numina.PRIME_LISTS
    }

    def listed(uuid: str) -> str | None:
        return next((name for name, uuids in lists.items() if uuid in uuids), None)

    for index, row in enumerate(pq.read_table(path).to_pylist()):
        if numina.is_human_proof_row(row, None):
            yield f"{index}:{row['uuid']}", numina.row_contract(row, listed)


SOURCES = {
    "proofnet": (PROOFNET_VERIFIED_REVISION, proofnet_contracts),
    "numina": (numina.NUMINA_REVISION, numina_contracts),
}


def check_contract(row: object, contract: Contract | ImportRejection, lean: str, lean_path: str) -> dict:
    """One row's result: its rejection reason, or whether its reference validates and how long it took."""
    if isinstance(contract, ImportRejection):
        return {"row": row, "status": "rejected", "reason": contract.reason}
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
            return {"row": row, "status": "timeout"}
    status = "validates" if result.returncode == 0 else "fails"
    output = (result.stdout + result.stderr)[-2000:]
    return {"row": row, "status": status, "seconds": round(time.monotonic() - start, 1), "output": output}


def failed_unexpectedly(source: str, result: dict) -> bool:
    if result["status"] == "validates":
        return False
    if result["status"] != "rejected":
        return True
    return source == "proofnet" and EXPECTED_REJECTIONS.get(result["row"]) != result.get("reason")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("rows", type=Path)
    parser.add_argument("--prime-lists", type=Path, help="NuminaMath-LEAN: the directory with Prime Intellect's lists")
    parser.add_argument("--lean", required=True)
    parser.add_argument("--lean-path", required=True)
    parser.add_argument("--mathlib-commit", required=True, help="the Mathlib commit LEAN_PATH points into")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.source == "numina" and args.prime_lists is None:
        parser.error("numina needs --prime-lists")
    revision, contracts = SOURCES[args.source]
    version = subprocess.run([args.lean, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    provenance = {
        "source": args.source,
        "revision": revision,
        "sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "lean": version,
        "mathlib": args.mathlib_commit,
    }
    print(json.dumps({"checked": provenance}), flush=True)
    failures = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(
            lambda item: check_contract(*item, args.lean, args.lean_path), contracts(args.rows, args)
        ):
            failures += failed_unexpectedly(args.source, result)
            print(json.dumps(result), flush=True)
    if failures:
        print(f"{failures} rows failed unexpectedly", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
