# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a context refusal in one process and replay its journal in another.

``launch`` and ``resume`` currently use the same replay log. Later driver revisions
can give those modes different replay rules.
"""

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resources.inference_fakes import Backend, provider_completion
from lib.taskforge.tests.test_end_to_end import Envs, config, resources

from taskforge.coordinator import run_workpiece
from taskforge.effects.log import EventLog, JsonlSink, Record
from taskforge.llm.client import ContextExhausted
from taskforge.model.review import Abandon
from taskforge.resources.inference import GlmInference
from taskforge.stages.classify import replayable_grade
from taskforge.wiring import install


def driver_log(previous: Sequence[Record], *, resume: bool) -> EventLog:
    """Build the replay log used by this revision's driver."""
    del resume  # this revision's driver replays the same way with or without the flag
    return EventLog(replay_from=previous, replayable=replayable_grade)


async def main(mode: str, directory: Path) -> None:
    sink = JsonlSink(str(directory / "events.jsonl"))
    if mode == "record":
        backend, log = Backend((ContextExhausted("maximum context length exceeded", ()),)), EventLog(sink=sink)
    else:
        backend, log = Backend((provider_completion(),)), driver_log(await sink.read(), resume=mode == "resume")
    live = replace(resources(Envs(), ()), inference=GlmInference(backend, "provider"))
    wp = await run_workpiece(sample_proposal(), config("S"), live, log, install=install)
    (stop,) = wp.decisions
    assert isinstance(stop, Abandon)
    print(f"{mode}: {wp.terminal}, {stop.reasons}, {len(backend.calls)} provider call(s)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
