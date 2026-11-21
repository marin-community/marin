# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the two-process context-refusal replay test: ``record DIR`` runs a workpiece whose provider refuses
the author's first request for a full context window and persists the journal; ``replay DIR`` reloads it in a fresh
process whose provider would answer, and runs the workpiece again. Each prints the terminal, the author's stop and how
many provider calls it made."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resources.inference_fakes import Backend, provider_completion
from lib.taskforge.tests.test_end_to_end import Envs, config, resources

from taskforge.coordinator import run_workpiece
from taskforge.effects.log import EventLog, JsonlSink
from taskforge.llm.client import ContextExhausted
from taskforge.model.review import Abandon
from taskforge.resources.inference import GlmInference
from taskforge.wiring import install


async def main(mode: str, directory: Path) -> None:
    sink = JsonlSink(str(directory / "events.jsonl"))
    if mode == "record":
        backend, log = Backend((ContextExhausted("maximum context length exceeded", ()),)), EventLog(sink=sink)
    else:
        backend, log = Backend((provider_completion(),)), EventLog(replay_from=await sink.read())
    live = replace(resources(Envs(), ()), inference=GlmInference(backend, "provider"))
    wp = await run_workpiece(sample_proposal(), config(), live, log, install=install)
    (stop,) = wp.decisions
    assert isinstance(stop, Abandon)
    print(f"{mode}: {wp.terminal}, {stop.reasons}, {len(backend.calls)} provider call(s)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
