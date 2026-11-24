# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the two-process confirmation replay test: ``record DIR`` runs a workpiece through validation and
its confirmation step and persists the journal; ``replay DIR`` reloads it in a fresh process and runs the workpiece
again. Each prints the terminal, how many solver slots it paid for and how many reviewer calls it made."""

import asyncio
import sys
from pathlib import Path

from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.test_confirmation import Reviewer, Trials, bundle, setup

from taskforge.coordinator import run_workpiece
from taskforge.effects.log import EventLog, JsonlSink
from taskforge.wiring import install


async def main(mode: str, directory: Path) -> None:
    sink = JsonlSink(str(directory / "events.jsonl"))
    engine, reviewer = Trials(), Reviewer()
    if mode == "record":
        log, live = EventLog(sink=sink), bundle(engine, reviewer)
    else:
        log, live = EventLog(replay_from=local_records(sink)), bundle(engine, reviewer, ())
    wp = await run_workpiece(sample_proposal(), setup(), live, log, install=install)
    await log.persist()
    print(f"{mode}: {wp.terminal}, {len(engine.runs)} solve(s), {len(reviewer.requests)} review call(s)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
