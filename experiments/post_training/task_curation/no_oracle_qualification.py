# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify a source that ships no oracle solutions.

A source whose tasks carry no oracle solution (``has_solution`` false, or an
empty ``solution_binary``) cannot be qualified by executing its oracle. This
module runs the two checks that replace it when a cohort has been rebuilt onto
shared environments:

1. **Payload preservation** (offline). Every file the native recipe copies into
   the workspace must appear in the rebuilt task's setup payload with identical
   bytes and mode. This catches a rebuild that drops, rewrites or re-modes
   fixture data.
2. **Verifier liveness** (sandbox). The task's own verifier must execute and
   return a well-formed negative verdict on the untouched workspace. A crash, a
   timeout or unparsable output is an infrastructure error, not a pass.

The report records the pass rate per base reference and overall, and its verdict
becomes ``oracle_validation=verifier_executed`` on the declaration. Until the
report exists the declaration must say ``skipped``: absence of oracle solutions
is not evidence that execution works.

Run the offline phase with::

    uv run python -m experiments.post_training.task_curation.no_oracle_qc \
        --cohort /path/to/native-tasks.parquet \
        --rebuild-out /path/to/rebuild-output \
        --output /path/to/qc-report.json

Add ``--sandbox`` to include the verifier-liveness phase; it needs a Daytona
credential and creates one sandbox per sampled task. Snapshot deletion has no
code path in this repository (see ``.agents/ops/daytona.md``); hand the created
snapshot names to an operator for cleanup.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

ENVIRONMENT_PREFIX = "environment/"
SETUP_ROOT = "/setup_files"
FIXTURE_ARCHIVE = "calibforge-fixtures.tar.gz"
COPY_SOURCE = re.compile(r"^\s*COPY\s+(?:--\S+\s+)*(?P<sources>.+)$", re.IGNORECASE | re.MULTILINE)
ADD_SOURCE = re.compile(r"^\s*ADD\s", re.IGNORECASE | re.MULTILINE)

PASSING_VERIFIER_OUTCOMES = ("negative_verdict",)


@dataclass(frozen=True)
class FileExpectation:
    """One file the native recipe copies, with the bytes it must keep."""

    path: str
    sha256: str
    mode: int


@dataclass(frozen=True)
class PreservationResult:
    """Whether the rebuilt payload preserved every copied file."""

    task_id: str
    expected: tuple[FileExpectation, ...]
    missing: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    remoded: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (self.missing or self.changed or self.remoded)


@dataclass(frozen=True)
class VerifierResult:
    """One verifier execution against the untouched workspace."""

    task_id: str
    outcome: str
    exit_code: int | None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING_VERIFIER_OUTCOMES


@dataclass
class QualificationReport:
    """Per-task checks plus the pass rates admission reads."""

    preservation: list[PreservationResult] = field(default_factory=list)
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
    def verifier_rate(self) -> float | None:
        return self._rate(self.verifier)

    def by_base(self) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[PreservationResult]] = defaultdict(list)
        for result in self.preservation:
            grouped[self.bases.get(result.task_id, "<unknown>")].append(result)
        return {
            base: {"tasks": len(results), "preservation_rate": self._rate(results)}
            for base, results in sorted(grouped.items())
        }

    def verdict(self, *, minimum_rate: float = 1.0) -> str:
        """``verifier_executed`` only when both checks ran and every sample passed."""
        if self.preservation_rate is None or self.verifier_rate is None:
            return "skipped"
        if self.preservation_rate < minimum_rate or self.verifier_rate < minimum_rate:
            return "skipped"
        return "verifier_executed"

    def to_dict(self) -> dict:
        return {
            "preservation_rate": self.preservation_rate,
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
            "verifier": [
                {
                    "task_id": result.task_id,
                    "outcome": result.outcome,
                    "exit_code": result.exit_code,
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in self.verifier
            ],
        }


def _read_tar(payload: bytes) -> tarfile.TarFile:
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return tarfile.open(fileobj=io.BytesIO(raw))


def copied_paths(dockerfile: str) -> list[str]:
    """The context-relative sources a native recipe copies into the workspace."""
    if ADD_SOURCE.search(dockerfile):
        raise ValueError("ADD recipes have no preservation contract")
    paths: list[str] = []
    for match in COPY_SOURCE.finditer(dockerfile):
        *sources, _destination = match.group("sources").split()
        for source in sources:
            if source.startswith("--") or "$" in source or "*" in source or "?" in source:
                continue
            paths.append(source.rstrip("/"))
    return paths


def preservation_expectations(native_payload: bytes) -> tuple[FileExpectation, ...]:
    """Files and modes the native recipe copies, read from its own tar."""
    with _read_tar(native_payload) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        dockerfile_member = members.get(f"{ENVIRONMENT_PREFIX}Dockerfile")
        if dockerfile_member is None:
            raise ValueError("native task has no environment/Dockerfile")
        dockerfile = archive.extractfile(dockerfile_member).read().decode(errors="replace")
        expectations: list[FileExpectation] = []
        for source in copied_paths(dockerfile):
            source_path = f"{ENVIRONMENT_PREFIX}{source}"
            prefix = source_path + "/"
            matches = [name for name in members if name == source_path or name.startswith(prefix)]
            for name in sorted(matches):
                content = archive.extractfile(members[name]).read()
                expectations.append(
                    FileExpectation(
                        path=name,
                        sha256=hashlib.sha256(content).hexdigest(),
                        mode=members[name].mode & 0o7777,
                    )
                )
    return tuple(expectations)


def check_preservation(task_id: str, native_payload: bytes, rebuilt_payload: bytes) -> PreservationResult:
    """Compare the native recipe's copied files with the rebuilt setup payload."""
    expected = preservation_expectations(native_payload)
    payload_member = f"{SETUP_ROOT.lstrip(chr(47))}/{FIXTURE_ARCHIVE}"
    with _read_tar(rebuilt_payload) as archive:
        member = next((item for item in archive.getmembers() if item.name == payload_member), None)
        if member is None:
            return PreservationResult(task_id=task_id, expected=expected, missing=tuple(item.path for item in expected))
        payload = archive.extractfile(member).read()
    staged = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for item in archive.getmembers():
            if item.isfile():
                staged[f"{ENVIRONMENT_PREFIX}{item.name}"] = (archive.extractfile(item).read(), item.mode & 0o7777)
    missing, changed, remoded = [], [], []
    for item in expected:
        found = staged.get(item.path)
        if found is None:
            missing.append(item.path)
            continue
        if hashlib.sha256(found[0]).hexdigest() != item.sha256:
            changed.append(item.path)
        if found[1] != item.mode:
            remoded.append(item.path)
    return PreservationResult(
        task_id=task_id, expected=expected, missing=tuple(missing), changed=tuple(changed), remoded=tuple(remoded)
    )


def classify_verifier_run(*, exit_code: int | None, timed_out: bool, stdout: str) -> str:
    """Classify one verifier execution against an untouched workspace.

    A verifier that ran returns a negative verdict: it executed and rejected the
    unsolved submission. A timeout, crash or unparsable report is an
    infrastructure error and never counts as a pass.
    """
    if timed_out:
        return "timeout"
    if exit_code is None:
        return "infra_error"
    text = stdout.strip()
    if not text:
        return "infra_error"
    reward_line = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    try:
        reward = float(reward_line.split()[-1])
    except ValueError:
        return "malformed"
    if exit_code != 0:
        return "infra_error"
    return "negative_verdict" if reward < 1.0 else "positive_reward"


def summarize(report: QualificationReport, *, minimum_rate: float = 1.0) -> dict:
    """Admission summary with the pass rates and the recorded verdict."""
    return {
        "checked": len(report.preservation),
        "verified": len(report.verifier),
        "preservation_rate": report.preservation_rate,
        "verifier_rate": report.verifier_rate,
        "minimum_rate": minimum_rate,
        "verifier_failures": dict(Counter(result.outcome for result in report.verifier if not result.passed)),
        "by_base": report.by_base(),
        "oracle_validation": report.verdict(minimum_rate=minimum_rate),
    }


def _native_tasks(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            yield task_id, bytes(blob.as_py() if hasattr(blob, "as_py") else blob)


def main() -> None:
    """Run the offline preservation phase and write the qualification report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="native cohort parquet (path, task_binary)")
    parser.add_argument("--rebuild-out", type=Path, required=True, help="rebuild output directory")
    parser.add_argument("--output", type=Path, required=True, help="qualification report JSON")
    parser.add_argument("--sandbox", action="store_true", help="also run verifier liveness (needs Daytona)")
    parser.add_argument("--sample-per-base", type=int, default=20, help="verifier samples per base reference")
    args = parser.parse_args()

    rebuilt_path = args.rebuild_out / "pending-tasks.parquet"
    manifest = json.loads((args.rebuild_out / "manifest.json").read_text())
    bases = {row["task_id"]: row["base_ref"] for row in manifest["tasks"] if row.get("status", "").startswith("rebuilt")}

    rebuilt = {}
    for batch in pq.ParquetFile(rebuilt_path).iter_batches(batch_size=64, columns=["path", "task_binary"]):
        columns = batch.to_pydict()
        for task_id, blob in zip(columns["path"], columns["task_binary"], strict=True):
            rebuilt[task_id] = bytes(blob.as_py() if hasattr(blob, "as_py") else blob)

    report = QualificationReport(bases=bases)
    for task_id, native_payload in _native_tasks(args.cohort):
        if task_id not in rebuilt:
            continue
        report.preservation.append(check_preservation(task_id, native_payload, rebuilt[task_id]))

    if args.sandbox:
        raise SystemExit(
            "Verifier liveness needs a Daytona credential and creates one sandbox per sampled task; "
            "run it through the reviewed procedure in docs/references/task-curation.md"
        )

    summary = summarize(report)
    args.output.write_text(json.dumps({"summary": summary, **report.to_dict()}, indent=1) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_base"}))


if __name__ == "__main__":
    main()
