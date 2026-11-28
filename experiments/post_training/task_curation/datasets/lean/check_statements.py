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
    uv run python -m experiments.post_training.task_curation.datasets.lean.check_statements numina \\
        train-00000-of-00001.parquet --prime-lists DIR --lean LEAN --lean-path LEAN_PATH --mathlib-commit COMMIT \\
        --write-proofs experiments/post_training/task_curation/datasets/lean/numina_proofs.json
    uv run python -m experiments.post_training.task_curation.datasets.lean.check_statements numina \\
        train-00000-of-00001.parquet --prime-lists DIR --lean LEAN --lean-path LEAN_PATH --mathlib-commit COMMIT \\
        --write-replies numina_replies.json

The first output line records what was checked; each further line is one row's result. It exits non-zero
when a reference does not validate or ProofNet-Verified rejects a row other than ``EXPECTED_REJECTIONS``.
At the pinned revisions, with the grader image's toolchain, all 365 ProofNet-Verified contracts and all
4,035 NuminaMath-LEAN contracts validate; rerun it when a revision or the toolchain moves. ``--write-proofs``
checks each NuminaMath-LEAN row's human proof as its golden control submits it, compiled like the
grader's candidate step, and writes the rows whose proof compiles without ``sorry`` to ``numina_proofs.json``.
``--write-replies`` compiles ``numina.CONTRADICTION_REPLY`` and ``numina.FIXED_REPLY`` on every NuminaMath-LEAN
task in the same way and writes their outcomes to ``numina_replies.json``, which the recipe fetches from
``numina.REPLIES_URL``: publish a new file there and update ``REPLIES_URL`` and ``REPLIES_SHA256``. Its
compile of ``FIXED_REPLY`` takes about 30 CPU-seconds and 2.3 GB beside Mathlib's shared pages when no tactic
proves the task, mostly ``exact?`` indexing Mathlib, so size ``--jobs`` to the machine's memory.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.pipeline.models import ImportRejection

from experiments.post_training.task_curation.datasets.lean import numina
from experiments.post_training.task_curation.datasets.lean.lean_grade import (
    LEAN_ERROR,
    LEAN_THREADS,
    MODULE_FILE,
    OBJECT_FILE,
    STEP_SECONDS,
    Contract,
    candidate_source,
    declares_target,
    reference_source,
    without_comments,
)
from experiments.post_training.task_curation.datasets.lean.numina import REPLIES, ReplyOutcome
from experiments.post_training.task_curation.datasets.lean.proofs import PROOFNET_VERIFIED_REVISION, proofnet_contract

EXPECTED_REJECTIONS = {35: "no_sorry_proof", 351: "repeated_exercise"}
SORRY_WARNING = "declaration uses 'sorry'"
"""ProofNet-Verified rows the converter rejects at the pinned revision: row 35's statement carries its
proof, and row 351 repeats row 168's exercise. NuminaMath-LEAN rejects many rows by design."""
RECORDED = ("revision", "sha256", "lean", "mathlib")
"""What a written file records of the check that produced it."""


def proofnet_contracts(path: Path, _args: argparse.Namespace) -> Iterator[tuple[object, Contract | ImportRejection]]:
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            yield row.get("index"), proofnet_contract(row)


def numina_rows(path: Path, args: argparse.Namespace) -> Iterator[tuple[str, dict, Contract | ImportRejection]]:
    """Each row with a human statement and proof: its id, its data and its contract. The contract requires a
    listed passing proof except while ``--write-proofs`` builds that list."""
    lists = {
        name: frozenset(json.loads(gzip.decompress((args.prime_lists / f"{name}.json.gz").read_bytes())))
        for name in numina.PRIME_LISTS
    }

    def listed(uuid: str) -> str | None:
        return next((name for name, uuids in lists.items() if uuid in uuids), None)

    for index, row in enumerate(pq.read_table(path).to_pylist()):
        if numina.is_human_proof_row(row, None):
            contract = numina.row_contract(row, listed, require_passing_proof=args.write_proofs is None)
            yield f"{index}:{row['uuid']}", row, contract


def numina_contracts(path: Path, args: argparse.Namespace) -> Iterator[tuple[object, Contract | ImportRejection]]:
    for row_id, _data, contract in numina_rows(path, args):
        yield row_id, contract


SOURCES = {
    "proofnet": (PROOFNET_VERIFIED_REVISION, proofnet_contracts),
    "numina": (numina.NUMINA_REVISION, numina_contracts),
}


def compile_source(
    source: str, lean: str, lean_path: str, step: str
) -> tuple[subprocess.CompletedProcess | None, float]:
    """Compile ``source`` as the grader's ``step`` would, under that step's allowance; ``None`` on a timeout."""
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, MODULE_FILE).write_text(source)
        start = time.monotonic()
        try:
            result = subprocess.run(
                [lean, LEAN_THREADS, "-o", OBJECT_FILE, MODULE_FILE],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=STEP_SECONDS[step],
                env={"LEAN_PATH": lean_path, "PATH": "/usr/bin:/bin"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, round(time.monotonic() - start, 1)
    return result, round(time.monotonic() - start, 1)


def check_contract(row: object, contract: Contract | ImportRejection, lean: str, lean_path: str) -> dict:
    """One row's result: its rejection reason, or whether its reference validates and how long it took."""
    if isinstance(contract, ImportRejection):
        return {"row": row, "status": "rejected", "reason": contract.reason}
    result, seconds = compile_source(reference_source(contract), lean, lean_path, "reference")
    if result is None:
        return {"row": row, "status": "timeout"}
    status = "validates" if result.returncode == 0 else "fails"
    output = (result.stdout + result.stderr)[-2000:]
    return {"row": row, "status": status, "seconds": seconds, "output": output}


def check_golden(row: object, contract: Contract | ImportRejection, data: dict, lean: str, lean_path: str) -> dict:
    """Whether a NuminaMath-LEAN row's human proof, as its golden control submits it, compiles without ``sorry``
    in the grader's candidate step."""
    if isinstance(contract, ImportRejection):
        return {"row": row, "status": "rejected", "reason": contract.reason}
    block = without_comments(data["formal_ground_truth"]).strip()
    result, seconds = compile_source(candidate_source(contract, block), lean, lean_path, "candidate")
    if result is None:
        return {"row": row, "status": "timeout", "seconds": seconds}
    output = result.stdout + result.stderr
    passes = result.returncode == 0 and SORRY_WARNING not in output and declares_target(block, contract.target)
    return {"row": row, "key": numina.proof_key(data), "status": "passes" if passes else "fails", "seconds": seconds}


def reply_outcome(result: subprocess.CompletedProcess | None) -> ReplyOutcome:
    """How a candidate compile of a fixed reply ended, read as the grader reads it; ``None`` is a timeout."""
    if result is None:
        return ReplyOutcome.UNKNOWN
    output = result.stdout + result.stderr
    if result.returncode == 0:
        return ReplyOutcome.NOT_PROVED if SORRY_WARNING in output else ReplyOutcome.PROVED
    if result.returncode == 1 and LEAN_ERROR.search(output.encode()):
        return ReplyOutcome.NOT_PROVED
    return ReplyOutcome.UNKNOWN


def check_replies(row: object, contract: Contract | ImportRejection, data: dict, lean: str, lean_path: str) -> dict:
    """Each of ``REPLIES``' outcomes on one NuminaMath-LEAN task, compiled like the grader's candidate step."""
    if isinstance(contract, ImportRejection):
        return {"row": row, "status": "rejected", "reason": contract.reason}
    result: dict[str, object] = {"row": row, "key": numina.proof_key(data), "status": "checked"}
    for name, reply in REPLIES.items():
        compiled, seconds = compile_source(candidate_source(contract, reply), lean, lean_path, "candidate")
        result[name] = reply_outcome(compiled).value
        result[f"{name}_seconds"] = seconds
    return result


def replies_record(provenance: dict, results: Iterable[dict]) -> dict:
    """The contents of ``numina_replies.json``: the tasks ``CONTRADICTION_REPLY`` proves, those it leaves
    undecided, and every other task by ``FIXED_REPLY``'s outcome."""
    checked = [result for result in results if result["status"] == "checked"]
    contradictory = {result["key"] for result in checked if result["contradiction"] == ReplyOutcome.PROVED}
    others = [result for result in checked if result["key"] not in contradictory]
    return {
        "checked": {**{key: provenance[key] for key in RECORDED}, "replies": REPLIES},
        "contradictory": sorted(contradictory),
        "contradiction_unknown": sorted(
            {result["key"] for result in checked if result["contradiction"] == ReplyOutcome.UNKNOWN}
        ),
        "fixed_reply": {
            outcome.value: sorted({result["key"] for result in others if result["fixed_reply"] == outcome})
            for outcome in ReplyOutcome
        },
    }


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
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument("--write-proofs", type=Path, help="NuminaMath-LEAN: check human proofs and write this file")
    writes.add_argument("--write-replies", type=Path, help="NuminaMath-LEAN: check the fixed replies, write this file")
    args = parser.parse_args()
    if args.source == "numina" and args.prime_lists is None:
        parser.error("numina needs --prime-lists")
    if (args.write_proofs is not None or args.write_replies is not None) and args.source != "numina":
        parser.error("--write-proofs and --write-replies are for numina")
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
    if args.write_proofs is not None:
        write_proofs(args, provenance)
        return
    if args.write_replies is not None:
        write_replies(args, provenance)
        return
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


def checked_numina_rows(
    args: argparse.Namespace, check: Callable[[object, Contract | ImportRejection, dict, str, str], dict]
) -> Iterator[dict]:
    """``check``'s result on every converted NuminaMath-LEAN row, ``args.jobs`` rows at a time, each also printed."""
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(
            lambda item: check(item[0], item[2], item[1], args.lean, args.lean_path), numina_rows(args.rows, args)
        ):
            print(json.dumps(result), flush=True)
            yield result


def write_proofs(args: argparse.Namespace, provenance: dict) -> None:
    """Check every converted row's human proof and write the keys of those that pass to ``args.write_proofs``."""
    passing = [result["key"] for result in checked_numina_rows(args, check_golden) if result["status"] == "passes"]
    record = {"checked": {key: provenance[key] for key in RECORDED}}
    args.write_proofs.write_text(json.dumps({**record, "passing": sorted(passing)}, indent=1) + "\n")


def write_replies(args: argparse.Namespace, provenance: dict) -> None:
    """Compile ``REPLIES`` on every converted row and write their outcomes to ``args.write_replies``."""
    results = list(checked_numina_rows(args, check_replies))
    args.write_replies.write_text(json.dumps(replies_record(provenance, results), indent=1) + "\n")


if __name__ == "__main__":
    main()
