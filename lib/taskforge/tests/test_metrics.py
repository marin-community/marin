# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.effects.events import Event, Note, Resolved, StageCompleted
from taskforge.effects.log import JsonlSink, Record, Status
from taskforge.metrics import Origin, workpiece_metrics
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Accept, BandOutcome, Cause, TrialResult, ValidationResult
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Submission, Terminal, Workpiece
from taskforge.report import report
from taskforge.resources.env import TIMEOUT_EXIT_CODE, CommandResult, RunCommand
from taskforge.stages.author import AUTHOR_IMAGE_FALLBACK


def evidence(submission: Submission, solved: int) -> ValidationResult:
    # The last trial passes but hit its generation limit, so it never counts as solved.
    trials = tuple(TrialResult(i, GradeResult(Outcome.GRADED, float(i < solved))) for i in range(3))
    limited = TrialResult(3, GradeResult(Outcome.GRADED, 1.0), Cause.GENERATION_LIMIT)
    return ValidationResult(submission.digest, (*trials, limited), (), ())


def repaired() -> Workpiece:
    source = Submission(arithmetic_task(), (), "source task")
    revised = Submission(arithmetic_task().model_copy(update={"instruction": "Add the numbers."}), (), "revised")
    return Workpiece(
        "repaired",
        source=source,
        submission=revised,
        round=1,
        evidence=(evidence(source, 1), evidence(revised, 2)),
        decisions=(Accept(BandOutcome.IN_BAND),),
        terminal=Terminal.ACCEPTED,
    )


def records(events: list[tuple[float, Event[object]]]) -> tuple[Record, ...]:
    return tuple(Record(i, stamp, event) for i, (stamp, event) in enumerate(events, 1))


def test_source_task_metrics_count_one_session_per_round_and_measure_both_submissions() -> None:
    timed_out = CommandResult(TIMEOUT_EXIT_CODE, "", "", 30, False)
    log = records(
        [
            (100, StageCompleted(Stage.VALIDATE, "", "Validated", 0)),
            (110, StageCompleted(Stage.REVIEW, "", "Modify", 1)),
            (120, Note(f"{AUTHOR_IMAGE_FALLBACK}: setup.sh exited 1")),
            (130, Resolved(RunCommand("pytest", 30), (), timed_out)),
            (140, StageCompleted(Stage.AUTHOR, "", "Submitted", 1)),
            # A resumed process completes the same author round again.
            (150, StageCompleted(Stage.AUTHOR, "", "Submitted", 1)),
            (190, StageCompleted(Stage.REVIEW, "", "Accept", 1, Terminal.ACCEPTED)),
        ]
    )
    metrics = workpiece_metrics(Status(terminal=Terminal.ACCEPTED), repaired(), log, since=-math.inf)
    assert metrics.origin is Origin.SOURCE
    assert (metrics.author_sessions, metrics.author_image_fallbacks, metrics.command_timeouts) == (1, 1, 1)
    assert (metrics.initial_solve_rate, metrics.final_solve_rate) == (0.25, 0.5)
    assert (metrics.started, metrics.finished, metrics.accepted_at) == (100, 190, 190)


async def test_report_splits_lines_by_origin(tmp_path: Path, proposal: TaskProposal) -> None:
    untouched = Submission(arithmetic_task(), (), "source task")
    workpieces = {
        "generated": (
            Workpiece("generated", proposal, submission=untouched, terminal=Terminal.ACCEPTED),
            [(0, StageCompleted(Stage.AUTHOR, "", "Submitted", 0))],
        ),
        "untouched": (
            Workpiece("untouched", source=untouched, submission=untouched, terminal=Terminal.ACCEPTED),
            [],
        ),
    }
    for name, (workpiece, events) in workpieces.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "workpiece.json").write_bytes(WORKPIECE.dump_json(workpiece))
        (directory / "status.json").write_text(json.dumps({"terminal": Terminal.ACCEPTED}))
        accepted = (1, StageCompleted(Stage.REVIEW, "", "Accept", 0, Terminal.ACCEPTED))
        await JsonlSink(str(directory / "events.jsonl")).write(records([*events, accepted]))
    lines = await report(str(tmp_path))
    assert "origin=all workpieces count=2" in lines
    assert "origin=all accepted_without_author count=1" in lines
    assert "origin=proposal accepted_without_author count=0" in lines
    assert "origin=source accepted_without_author count=1" in lines
    assert "origin=source terminal=accepted count=1" in lines
