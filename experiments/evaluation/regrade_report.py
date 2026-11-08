# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the grades of a regraded result root with the run whose stored model responses it reused."""

import argparse
import dataclasses
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from finestore.reader import ReadView
from marin.rollouts.evaluation import ATTEMPTS_TABLE, REUSED_FROM_RUN, RUN_OBJECT, Attempt, AttemptPhase, run_identity
from marin.rollouts.evaluation_results import SCORED_OUTCOMES
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_result import Outcome

BLOB_BATCH = 256

type AttemptKey = tuple[str, str, int]
"""Benchmark, source row and trial."""


@dataclass(frozen=True)
class Grade:
    outcome: Outcome
    score: float | None
    extracted: str | None


@dataclass(frozen=True)
class Change:
    benchmark: str
    source_row: str
    trial: int
    before: Grade
    after: Grade


@dataclass(frozen=True)
class Summary:
    benchmark: str
    attempts: int
    """Terminal attempts of the regraded run."""
    reused: int
    """Attempts graded on the source run's model response for the same source row and trial."""
    paired: int
    """Reused attempts with a score in both runs; ``before`` and ``after`` are their mean rewards."""
    before: float | None
    after: float | None
    raised: int
    lowered: int


@dataclass(frozen=True)
class Terminal:
    result_blob: str
    grade: Grade
    reused_from: tuple[str, str] | None
    """Run identity and evidence blob of the stored model response this attempt reused, if any."""


def terminal_attempts(root: str, benchmarks: Collection[str] | None = None) -> tuple[str, dict[AttemptKey, Terminal]]:
    """A result root's run identity, and the grade of each benchmark, source row and trial's latest finished attempt."""
    reader = ReadView(root)
    saved = reader.read_blob(RUN_OBJECT)
    if saved is None:
        raise ValueError(f"No evaluation run is stored in {root}")
    run = json.loads(saved)
    names = {entry["dataset"]: entry["protocol"]["benchmark"] for entry in run["datasets"]}
    latest: dict[AttemptKey, Attempt] = {}
    for row in reader.iter_rows(ATTEMPTS_TABLE):
        attempt = Attempt.model_validate({name: row[name] for name in Attempt.model_fields})
        key = (names[attempt.dataset], attempt.source_row, attempt.trial)
        if (
            attempt.phase is AttemptPhase.FINISHED
            and (benchmarks is None or key[0] in benchmarks)
            and (key not in latest or attempt.retry > latest[key].retry)
        ):
            latest[key] = attempt
    blobs = {key: attempt.result_blob for key, attempt in sorted(latest.items()) if attempt.result_blob}
    keys = list(blobs)
    terminal = {}
    for start in range(0, len(keys), BLOB_BATCH):
        batch = keys[start : start + BLOB_BATCH]
        evidence = reader.read_blobs([blobs[key] for key in batch])
        for key in batch:
            raw = json.loads(evidence[blobs[key]])
            outcome = latest[key].outcome
            assert outcome is not None
            grade = raw.get("grade") or {}
            extracted = (grade.get("detail") or {}).get("extracted")
            reused = raw.get(REUSED_FROM_RUN)
            terminal[key] = Terminal(
                result_blob=blobs[key],
                grade=Grade(
                    outcome=outcome,
                    score=grade.get("reward") if outcome in SCORED_OUTCOMES else None,
                    extracted=extracted if extracted is None or isinstance(extracted, str) else json.dumps(extracted),
                ),
                reused_from=(reused["run"], reused["result_blob"]) if reused else None,
            )
    return run_identity(run), terminal


def compare(source: str, regrade: str) -> tuple[list[Summary], list[Change]]:
    """Pair each attempt of ``regrade`` that reused a model response of ``source`` with that source attempt."""
    _, after = terminal_attempts(regrade)
    source_run, before = terminal_attempts(source, {key[0] for key in after})
    summaries, changes = [], []
    for benchmark in sorted({key[0] for key in after}):
        keys = [key for key in sorted(after) if key[0] == benchmark]
        reused = [
            (key, before[key].grade, after[key].grade)
            for key in keys
            if key in before and after[key].reused_from == (source_run, before[key].result_blob)
        ]
        changes += [Change(*key, before=old, after=new) for key, old, new in reused if old != new]
        scores = [(old.score, new.score) for _, old, new in reused if old.score is not None and new.score is not None]
        summaries.append(
            Summary(
                benchmark=benchmark,
                attempts=len(keys),
                reused=len(reused),
                paired=len(scores),
                before=sum(old for old, _ in scores) / len(scores) if scores else None,
                after=sum(new for _, new in scores) / len(scores) if scores else None,
                raised=sum(new > old for old, new in scores),
                lowered=sum(new < old for old, new in scores),
            )
        )
    return summaries, changes


def summary_table(summaries: Sequence[Summary]) -> str:
    rows = [("benchmark", "attempts", "reused", "paired", "before", "after", "raised", "lowered")]
    for summary in summaries:
        rows.append(
            (
                summary.benchmark,
                str(summary.attempts),
                str(summary.reused),
                str(summary.paired),
                "-" if summary.before is None else f"{summary.before:.4f}",
                "-" if summary.after is None else f"{summary.after:.4f}",
                str(summary.raised),
                str(summary.lowered),
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip() for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Result root whose stored model responses were reused.")
    parser.add_argument("--regrade", required=True, help="Result root written by task_driver --responses-from.")
    parser.add_argument(
        "--changes", help="Write each reused attempt whose outcome, score or extraction changed as JSONL."
    )
    arguments = parser.parse_args()
    configure_coreweave_s3()
    summaries, changes = compare(arguments.source, arguments.regrade)
    print(summary_table(summaries))
    if arguments.changes:
        StoragePath(arguments.changes).write_text(
            "".join(json.dumps(dataclasses.asdict(change)) + "\n" for change in changes)
        )


if __name__ == "__main__":
    main()
