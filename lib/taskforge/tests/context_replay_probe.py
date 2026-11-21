# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the two-process context-refusal replay test: ``record DIR`` runs a workpiece whose provider refuses
the author's first request for a full context window and persists the journal; ``launch DIR`` and ``resume DIR``
reload it in a fresh process whose provider would answer and run the workpiece again under the replay rule the driver
builds without and with ``--resume-held``. Each prints the terminal, the author's stop and how many provider calls it
made."""

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resources.inference_fakes import Backend, provider_completion
from lib.taskforge.tests.test_end_to_end import Envs, config, resources

from taskforge.coordinator import run_workpiece
from taskforge.effects.log import EventLog, JsonlSink, Record
from taskforge.llm.client import ContextExhausted
from taskforge.model.review import Abandon
from taskforge.resources.inference import GlmInference
from taskforge.stages.classify import replay_rule
from taskforge.wiring import install


def driver_log(previous: Sequence[Record], *, resume: bool) -> EventLog:
    """The log ``driver.one`` builds over a published stream; ``resume`` is its ``--resume-held``."""
    return EventLog(replay_from=previous, replayable=replay_rule(previous, resume))


async def main(mode: str, directory: Path) -> None:
    sink = JsonlSink(str(directory / "events.jsonl"))
    if mode == "record":
        backend, log = Backend((ContextExhausted("maximum context length exceeded", ()),)), EventLog(sink=sink)
    else:
        backend, log = Backend((provider_completion(),)), driver_log(local_records(sink), resume=mode == "resume")
    live = replace(resources(Envs(), ()), inference=GlmInference(backend, "provider"))
    wp = await run_workpiece(sample_proposal(), config(), live, log, install=install)
    (stop,) = wp.decisions
    assert isinstance(stop, Abandon)
    print(f"{mode}: {wp.terminal}, {stop.reasons}, {len(backend.calls)} provider call(s)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
