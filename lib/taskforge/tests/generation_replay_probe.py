# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One process of the two-process generation-checkpoint test, through the public driver on a decision-model task.

``record DIR``: the two validation answers are generated, then their grade meets a decision-model outage. ``resume DIR``
relaunches the same root in a fresh process, unflagged, with the decision model back. Each prints one JSON line: the
driver's exit code, the solver requests it dispatched, the candidates of the decision-model reads that answered, the
published terminal and what it awaits; ``resume`` adds the published generation checkpoints and the report's solver
and confirmation usage."""

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

from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeCommand, FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider, segment
from lib.taskforge.tests.resources.test_decision_model import GRADER, MODEL, FakeReader, VirtualTime
from lib.taskforge.tests.test_end_to_end import Envs, history, published_workpiece, script
from rolloutengine.test_support import arithmetic_task
from taskcompendium.decision_model import DecisionModelRequest

from taskforge import driver
from taskforge.config import TaskforgeConfig
from taskforge.effects.events import Resolved
from taskforge.llm.glm_provider import EndpointSource
from taskforge.model.proposal import TASK_PROPOSAL
from taskforge.model.review import Incomplete
from taskforge.policy import Policy
from taskforge.report import report
from taskforge.resources.bundle import Resources
from taskforge.resources.ids import Ids
from taskforge.resources.rollouts import SolverGeneration
from taskforge.wiring import build_config, build_resources

ANSWERS = ("twelve", "thirteen")
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


class Answered(FakeReader):
    """``FakeReader``, also listing the reads that answered."""

    def __init__(self, down: set[str]) -> None:
        super().__init__({"12": 4.0, "9": 0.0, "twelve": 4.0, "thirteen": 0.0}, down)
        self.answered: list[DecisionModelRequest] = []

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        level = await super().read(request)
        self.answered.append(request)
        return level


class DecisionEnvs(Envs):
    """``Envs`` whose drafted task is graded by the decision model."""

    def __init__(self) -> None:
        super().__init__()
        drafted = self.commands["write-draft"]
        task = {PurePosixPath("/workspace/task.json"): arithmetic_task(grader=GRADER).model_dump_json().encode()}
        self.commands["write-draft"] = FakeCommand(drafted.result, {**drafted.writes, **task})


async def main(mode: str, directory: Path) -> None:
    os.environ.update(GLM_BASE_URL="http://fake/v1", GLM_API_KEY="fake")
    os.environ.pop("GLM_RELAY_JOB", None)
    source, overlay = directory / "test.proposal.json", directory / "policy.json"
    if mode == "record":
        source.write_bytes(TASK_PROPOSAL.dump_json(sample_proposal()))
        overlay.write_text(json.dumps(POLICY))
    args = Namespace(
        proposals=str(source),
        policy=str(overlay),
        output=str(directory / "out"),
        work_dir=str(directory / f"work-{mode}"),
        resume_held=False,
        concurrency=1,
        buckets=4096,
        bucket_range=None,
        ranks=(),
    )
    time, reader = VirtualTime(), Answered(set(ANSWERS) if mode == "record" else set())
    solver = ScriptedProvider([segment(ANSWERS[n % 2]) for n in range(2 if mode == "record" else 5)])

    @asynccontextmanager
    async def runtime(
        policy: Policy, source: EndpointSource, parallel_api_key: str, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        del source, parallel_api_key  # the probe's resources reach no gateway
        providers = {"author": ScriptedProvider(()), "adversary": ScriptedProvider(()), "solver": solver}
        built = build_resources(policy, providers, {}, FakeWeb({}, {}), decision_model=reader)
        assert built.decision_model is not None
        author = FakeInference(script() if mode == "record" else ())  # resumed, every author call replays
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
    }
    if mode == "resume":
        rows = history(destination)
        outcome["generations"] = sum(
            isinstance(r.event, Resolved) and isinstance(r.event.request, SolverGeneration) for r in rows
        )
        lines = await report(args.output)
        for role in ("solver", "confirmation"):
            outcome[role] = next(line for line in lines if line.startswith(f"tokens role={role} "))
    print(json.dumps(outcome))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
