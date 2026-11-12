# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify a source that ships no oracle solutions.

A source whose tasks carry no oracle solution (``has_solution`` false, or an
empty ``solution_binary``) cannot be qualified by executing its oracle. This
module checks payload preservation and setup execution for cohorts rebuilt onto
shared environments. Empty-submission grading runs separately through
``no_oracle_liveness``:

1. **Payload preservation** (offline). Every file the native recipe copies into
   the workspace must appear in the rebuilt task's setup payload with identical
   bytes and mode. This catches a rebuild that drops, rewrites or re-modes
   fixture data.
2. **Setup execution** (local Docker). Run the rebuilt setup script in the task's
   own base image and compare the files it writes against the same expectation
   set. Preservation proves the bytes travel; this proves the script lands them.
   Package installation runs by default. ``--stub-apt`` skips installs for a
   payload-only diagnostic and does not establish dependency provisioning.
3. **Empty-submission grading** (separate command). ``no_oracle_liveness``
   samples the grader on a fresh machine. Qualification requires a successful
   verifier exit, a reward below 1.0, and a positive structured executed-test
   count.

Preservation always runs; setup execution runs with ``--docker`` (locally in Docker or,
with ``--lowered``, against the lowered specs). The report records pass rates by base
and overall. Verifier results require an explicit ``--liveness`` receipt file. A
reward-only sample is insufficient to establish ``oracle_validation=verifier_executed``.

Run the offline phase with::

    uv run python -m experiments.post_training.task_curation.no_oracle_qualification \
        --cohort /path/to/native-tasks.parquet \
        --rebuild-out /path/to/rebuild-output \
        --output /path/to/qc-report.json

Attach the separate liveness JSONL with ``--liveness`` to combine receipts.
Use ``no_oracle_liveness`` through the procedure in
``docs/references/task-curation.md`` for the separate reward sample.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.task_curation.no_oracle_liveness import (
    UNKNOWN_BASE,
    load_specs,
    manifest_bases,
)
from experiments.post_training.task_curation.no_oracle_preservation import PreservationResult, check_preservation
from experiments.post_training.task_curation.no_oracle_setup import SetupExecution, _execute_lowered, _execute_setups

PASSING_VERIFIER_OUTCOMES = ("negative_verdict",)


@dataclass(frozen=True)
class VerifierResult:
    """One verifier execution against the untouched workspace."""

    task_id: str
    outcome: str
    exit_code: int | None
    detail: str = ""
    tests_executed: int = 0

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING_VERIFIER_OUTCOMES and self.exit_code == 0 and self.tests_executed > 0


@dataclass
class QualificationReport:
    """Per-task checks plus the pass rates admission reads."""

    preservation: list[PreservationResult] = field(default_factory=list)
    execution: list[SetupExecution] = field(default_factory=list)
    verifier: list[VerifierResult] = field(default_factory=list)
    bases: dict[str, str] = field(default_factory=dict)

    def _rate(self, results) -> float | None:
        if not results:
            return None
        return sum(1 for result in results if result.passed) / len(results)

    @property
    def preservation_rate(self) -> float | None:
        return self._rate(self.preservation)

    @property
    def execution_rate(self) -> float | None:
        return self._rate(self.execution)

    @property
    def verifier_rate(self) -> float | None:
        return self._rate(self.verifier)

    def by_base(self) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[PreservationResult]] = defaultdict(list)
        for result in self.preservation:
            grouped[self.bases.get(result.task_id, UNKNOWN_BASE)].append(result)
        return {
            base: {"tasks": len(results), "preservation_rate": self._rate(results)}
            for base, results in sorted(grouped.items())
        }

    def verdict(self) -> str:
        """``verifier_executed`` only when every check ran and every sample passed."""
        rates = [self.preservation_rate, self.execution_rate, self.verifier_rate]
        if any(rate is None for rate in rates):
            return "skipped"
        if any(rate != 1.0 for rate in rates):
            return "skipped"
        return "verifier_executed"

    def to_dict(self) -> dict:
        return {
            "preservation_rate": self.preservation_rate,
            "execution_rate": self.execution_rate,
            "verifier_rate": self.verifier_rate,
            "by_base": self.by_base(),
            "verdict": self.verdict(),
            "preservation": [
                {
                    "task_id": result.task_id,
                    "passed": result.passed,
                    "missing": list(result.missing),
                    "changed": list(result.changed),
                    "remoded": list(result.remoded),
                }
                for result in self.preservation
            ],
            "execution": [
                {
                    "task_id": result.task_id,
                    "outcome": result.outcome,
                    "written": result.written,
                    "missing": list(result.missing),
                    "altered": list(result.altered),
                    "extra": list(result.extra),
                    "remoded": list(result.remoded),
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.execution
            ],
            "verifier": [
                {
                    "task_id": result.task_id,
                    "outcome": result.outcome,
                    "exit_code": result.exit_code,
                    "tests_executed": result.tests_executed,
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.verifier
            ],
        }


def summarize(report: QualificationReport) -> dict:
    """Admission summary with the pass rates and the recorded verdict."""
    return {
        "checked": len(report.preservation),
        "verified": len(report.verifier),
        "preservation_rate": report.preservation_rate,
        "verifier_rate": report.verifier_rate,
        "minimum_rate": 1.0,
        "executed": len(report.execution),
        "execution_rate": report.execution_rate,
        "setup_failures": dict(Counter(result.outcome for result in report.execution if not result.passed)),
        "verifier_failures": dict(Counter(result.outcome for result in report.verifier if not result.passed)),
        "by_base": report.by_base(),
        "oracle_validation": report.verdict(),
    }


def _native_tasks(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob)


def _wanted_task_ids(args: argparse.Namespace, rebuilt: dict[str, bytes]) -> set[str]:
    """The task ids this run should cover: explicit ids plus a previous report's failures."""
    wanted = set(args.task)
    if args.retry_failed is not None:
        previous = json.loads(args.retry_failed.read_text())
        wanted |= {row["task_id"] for row in previous["execution"] if not row["passed"]}
        wanted |= {row["task_id"] for row in previous["preservation"] if not row["passed"]}
    return wanted & set(rebuilt)


def _merge_liveness_receipts(report: QualificationReport, path: Path, native: dict[str, bytes]) -> None:
    """Append each liveness receipt as a verifier result; every task must be known once."""
    seen: set[str] = set()
    for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
        if row["task_id"] not in native or row["task_id"] in seen:
            raise ValueError(f"unknown or duplicate liveness task: {row['task_id']}")
        seen.add(row["task_id"])
        report.verifier.append(
            VerifierResult(
                row["task_id"], row["outcome"], row["exit_code"], row.get("detail", ""), row.get("tests_executed", 0)
            )
        )


def main() -> None:
    """Check preservation, execute setup in Docker or the lowered runtime, and merge liveness receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="native cohort parquet (path, task_binary)")
    parser.add_argument("--rebuild-out", type=Path, required=True, help="rebuild output directory")
    parser.add_argument("--output", type=Path, required=True, help="qualification report JSON")
    parser.add_argument("--docker", action="store_true", help="also execute every rebuilt setup script in Docker")
    parser.add_argument("--stub-apt", action="store_true", help="skip package installation in the setup script")
    parser.add_argument("--workers", type=int, default=4, help="concurrent setup executions")
    parser.add_argument("--docker-dns", action="append", default=[], help="resolver for container DNS")
    parser.add_argument("--retry-failed", type=Path, help="re-run the tasks a previous report failed")
    parser.add_argument("--lowered", action="store_true", help="execute the lowered specs instead of the tar setups")
    parser.add_argument("--task", action="append", default=[], help="limit to these task ids")
    parser.add_argument("--liveness", type=Path, help="JSONL receipts from no_oracle_liveness")
    args = parser.parse_args()

    bases = manifest_bases(args.rebuild_out / "manifest.json")
    rebuilt = dict(_native_tasks(args.rebuild_out / "pending-tasks.parquet"))

    report = QualificationReport(bases=bases)
    wanted = _wanted_task_ids(args, rebuilt)
    native = {
        task_id: payload
        for task_id, payload in _native_tasks(args.cohort)
        if task_id in rebuilt and (not wanted or task_id in wanted)
    }
    for task_id, native_payload in native.items():
        report.preservation.append(check_preservation(task_id, native_payload, rebuilt[task_id]))

    if args.docker and args.lowered:
        lowered_path = args.rebuild_out / "lowered-tasks.parquet"
        lowered = {task_id: spec for task_id, spec in load_specs(lowered_path).items() if task_id in native}
        report.execution = _execute_lowered(lowered, native, args.workers, dns=tuple(args.docker_dns))
    elif args.docker:
        report.execution = _execute_setups(
            native, rebuilt, args.workers, stub_apt=args.stub_apt, dns=tuple(args.docker_dns)
        )

    if args.liveness is not None:
        _merge_liveness_receipts(report, args.liveness, native)

    summary = summarize(report)
    args.output.write_text(json.dumps({"summary": summary, **report.to_dict()}, indent=1) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_base"}))


if __name__ == "__main__":
    main()
