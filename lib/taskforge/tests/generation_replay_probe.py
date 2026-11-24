# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the multi-process generation-checkpoint tests, through the public driver on a decision-model task.

``record DIR``: the two validation answers are generated, then their grade meets a decision-model outage. ``resume DIR``
relaunches the same root in a fresh process, unflagged, with the decision model back. Each prints one JSON line: the
driver's exit code, the solver requests it dispatched, the candidates of the decision-model reads that answered, the
published terminal, what it awaits and the head's journal format; ``resume`` adds the published generation checkpoints,
the payer of each paid slot result graded from one, and the report's solver and confirmation usage.

The ``sibling-*`` modes run three validation slots two at a time. ``sibling-record`` checkpoints slots 0 and 1 in an
outage. ``sibling-refused``, ``sibling-zero``, ``sibling-missing``, ``sibling-overflow`` and ``sibling-off-scale``
resume unflagged: slot 1 grades, slot 2 generates, and once slot 2's read is in flight the gateway refuses slot 0's
read with HTTP 400, answers it with all-zero probabilities, without the top level's probability or with a probability
too large for a float, or answers it off the scale.
``sibling-recover`` then resumes with ``--resume-held`` and a healthy decision model."""

import asyncio
import json
import os
import sys
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import httpx
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeCommand, FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider, segment
from lib.taskforge.tests.resources.test_decision_model import GRADER, MODEL, FakeReader, VirtualTime, gateway
from lib.taskforge.tests.test_end_to_end import Envs, history, published_workpiece, script
from rolloutengine.test_support import arithmetic_task
from taskcompendium.decision_model import DecisionModelRequest

from taskforge import driver
from taskforge.checkpoint import HEAD
from taskforge.config import TaskforgeConfig
from taskforge.effects.events import Resolved, event_digest
from taskforge.llm.glm_provider import EndpointSource
from taskforge.model.proposal import TASK_PROPOSAL
from taskforge.model.review import Incomplete
from taskforge.policy import Policy
from taskforge.report import report
from taskforge.resources.bundle import Resources
from taskforge.resources.ids import Ids
from taskforge.resources.rollouts import SolverGeneration, SolverRollout
from taskforge.wiring import build_config, build_resources

ANSWERS = ("twelve", "thirteen", "a dozen")
"""The solvers' answers in turn, one right and one wrong, distinct from the controls' 12 and 9, so that only the
solver slots meet the outage."""
POLICY = {
    "proposals.environments": ["container"],
    "solver.k": 2,
    "adversary.roster": [],
    "budget.max_modify_rounds": 1,
    "decision_model.model": MODEL,
    "decision_model.unavailable_wait": 10.0,
    **{f"{role}.model": "test-model" for role in ("author", "solver", "adversary")},
}
SIBLINGS = {**POLICY, "solver.k": 3, "solver.concurrency": 2, "decision_model.concurrency": 4}


class Answered(FakeReader):
    """``FakeReader``, also listing the reads that answered."""

    def __init__(self, down: set[str]) -> None:
        super().__init__({"12": 4.0, "9": 0.0, "twelve": 4.0, "thirteen": 0.0, "a dozen": 4.0}, down)
        self.answered: list[DecisionModelRequest] = []

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        level = await super().read(request)
        self.answered.append(request)
        return level


class Sibling(Answered):
    """Slot 0's read ("twelve") waits for slot 2's ("a dozen") to start, then fails through the default reader as the
    ``gateway`` answers it (or is answered off the scale without one); slot 2's read answers only after that."""

    def __init__(self, gateway: httpx.Response | None) -> None:
        super().__init__(set())
        self.gateway, self.reading, self.failed = gateway, asyncio.Event(), asyncio.Event()

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        if request.candidate == "twelve":
            await self.reading.wait()
            try:
                if (answer := self.gateway) is not None:
                    return await gateway(lambda _: answer).read(request)
                return 7.0
            finally:
                self.failed.set()
        if request.candidate == "a dozen":
            self.reading.set()
            await self.failed.wait()
            for _ in range(10):  # time for a failure beside it to cancel it
                await asyncio.sleep(0)
        return await super().read(request)


class DecisionEnvs(Envs):
    """``Envs`` whose drafted task is graded by the decision model."""

    def __init__(self) -> None:
        super().__init__()
        drafted = self.commands["write-draft"]
        task = {PurePosixPath("/workspace/task.json"): arithmetic_task(grader=GRADER).model_dump_json().encode()}
        self.commands["write-draft"] = FakeCommand(drafted.result, {**drafted.writes, **task})


async def main(mode: str, directory: Path) -> None:
    os.environ.update(GLM_API_KEY="fake")
    source, overlay = directory / "test.proposal.json", directory / "policy.json"
    if mode in ("record", "sibling-record"):
        source.write_bytes(TASK_PROPOSAL.dump_json(sample_proposal()))
        overlay.write_text(json.dumps(SIBLINGS if mode == "sibling-record" else POLICY))
    args = Namespace(
        proposals=str(source),
        policy=str(overlay),
        output=str(directory / "out"),
        resume_held=mode == "sibling-recover",
        concurrency=1,
        buckets=4096,
        bucket_range=None,
        ranks=(),
        glm_base_url="http://fake/v1",
        glm_relay_job=None,
    )
    recording = mode in ("record", "sibling-record")
    time = VirtualTime()
    zero = {"answers": {"whole": {"probabilities": dict.fromkeys(map(str, range(5)), 0.0)}}}
    missing = {"answers": {"whole": {"probabilities": {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0}}}}
    huge = {"answers": {"whole": {"probabilities": {**dict.fromkeys(map(str, range(4)), 0), "4": 10**309}}}}
    failing = {
        "sibling-refused": httpx.Response(400, text="bad state"),
        "sibling-zero": httpx.Response(200, json=zero),
        "sibling-missing": httpx.Response(200, json=missing),
        "sibling-overflow": httpx.Response(200, content=json.dumps(huge).encode()),
    }
    reader = Sibling(failing.get(mode)) if mode in (*failing, "sibling-off-scale") else None
    reader = reader or Answered(set(ANSWERS) if recording else set())
    answers = ["a dozen"] if isinstance(reader, Sibling) else None
    answers = answers or [ANSWERS[n % 2] for n in range(2 if recording else 5)]
    solver = ScriptedProvider([segment(answer) for answer in answers])

    @asynccontextmanager
    async def runtime(
        policy: Policy, source: EndpointSource, parallel_api_key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        del source, parallel_api_key  # the probe's resources reach no gateway
        providers = {"author": ScriptedProvider(()), "adversary": ScriptedProvider(()), "solver": solver}
        built = build_resources(policy, providers, {}, FakeWeb({}, {}), decision_model=reader)
        assert built.decision_model is not None
        author = FakeInference(script() if recording else ())  # resumed, every author call replays
        roles = {role: client for role, client in built.role_inference.items() if role != "author"}
        yield build_config(policy, {}), replace(
            built,
            envs=DecisionEnvs(),
            inference=author,
            ids=Ids(lambda: "workpiece"),
            role_inference={**roles, "solve_review": CleanReviewer()},
            decision_model=replace(built.decision_model, sleep=time.sleep),
        )

    driver.runtime = runtime
    code = await driver.one(args)
    destination = Path(args.output) / driver.proposal_key(args.proposals)
    published = published_workpiece(destination)
    last = published.decisions[-1] if published.decisions else None
    outcome: dict[str, object] = {
        "exit": code,
        "solves": len(solver.bodies),
        "reads": sorted(read.candidate for read in reader.answered),
        "terminal": str(published.terminal),
        "awaits": last.awaits if isinstance(last, Incomplete) else None,
        "format": json.loads((destination / HEAD).read_text())["format"],
    }
    if isinstance(reader, Sibling):
        outcome["reasons"] = list(last.reasons) if isinstance(last, Incomplete) else []
    if mode in ("resume", "sibling-recover"):
        rows = history(destination)
        generated = {
            (event_digest(r.event.request.slot), r.event.fingerprints)
            for r in rows
            if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverGeneration)
        }
        outcome["generations"] = len(generated)
        outcome["graded"] = [
            r.billed
            for r in rows
            if isinstance(r.event, Resolved)
            and isinstance(r.event.request, SolverRollout)
            and not r.event.replayed
            and (event_digest(r.event.request), r.event.fingerprints) in generated
        ]
        lines = await report(args.output)
        for role in ("solver", "confirmation"):
            outcome[role] = next(line for line in lines if line.startswith(f"tokens role={role} "))
    print(json.dumps(outcome))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
