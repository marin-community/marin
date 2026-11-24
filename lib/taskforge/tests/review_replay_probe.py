# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the two-process replay test: ``record DIR`` reviews a recorded task and persists the journal;
``replay DIR`` reloads it, takes the task and trial from the reloaded solver rollout, and reviews again. Each prints
how many reviewer calls it made."""

import asyncio
import sys
from pathlib import Path

from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.stage_fakes import transcript
from lib.taskforge.tests.test_solve_review import (
    LONG,
    TRIALS,
    journalled,
    recorded_task,
    reloaded,
    report,
    reviewed,
)
from rolloutengine.models import GradedRollout, Rollout
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.effects.log import BufferSink, EventLog, decode
from taskforge.model.review import TrialResult


async def main(mode: str, directory: Path) -> None:
    sink, journal = BufferSink(), directory / "events.jsonl"
    if mode == "record":
        submission, grade = recorded_task(), GradeResult(Outcome.GRADED, 0.0)
        rollout = GradedRollout(Rollout("run-1", (transcript(LONG, "I answer 9."),), "stop"), grade)
        log, reviewer = EventLog(sink=sink), FakeInference((report(),))
        await log.record(journalled(submission, rollout))
    else:
        records = decode(journal.read_bytes())
        (submission, rollout), log, reviewer = reloaded(records), EventLog(replay_from=records), FakeInference(())
    await reviewed(reviewer, (TRIALS[0], (TrialResult(1, rollout.grade), rollout)), log, submission=submission)
    if mode == "record":
        await log.persist()
        journal.write_bytes(sink.data)
    print(f"{mode}: {len(reviewer.requests)} reviewer call(s)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
