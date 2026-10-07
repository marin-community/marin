# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""run_solver: k trials under the draft's convention, resumed per trial from the attempt files."""

from dataclasses import dataclass

from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import JsonAnswer, PlainText

from taskforge.ledger.jsonl import read_entries
from taskforge.llm.client import GlmUnavailable
from taskforge.sandbox.factories import SHELLSIM
from taskforge.validate.outcome import Cause, Graded, Ungraded
from taskforge.validate.solver import run_solver
from taskforge.validate.trials import EngineSettings

PLAIN = PlainText(id="plain")


def settings(factory, conventions=(PLAIN,)) -> EngineSettings:
    return EngineSettings(
        factories={EnvironmentKind.SHELLSIM: factory},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=4,
        command_timeout=10,
        cleanup_timeout=10,
        conventions=conventions,
    )


@dataclass
class UnavailableFirst:
    """Raises ``GlmUnavailable`` on the first ``failures`` requests, then delegates to ``inner``."""

    inner: object
    failures: int
    calls: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.calls += 1
        if self.calls <= self.failures:
            raise GlmUnavailable("router drained", ())
        return await self.inner(request)


async def test_a_re_entered_solver_runs_only_the_unsettled_trials(tmp_path, math_task, rounds, fakes):
    draft = rounds.draft(math_task, (), PLAIN)
    policy = rounds.policy(k=3)
    site = rounds.site(tmp_path)
    flaky = UnavailableFirst(fakes.script_model([fakes.text("395")]), failures=1)

    first = await run_solver(draft, policy, site, settings(fakes.flaky_factory(0, RuntimeError)), lambda _: flaky)

    assert sorted(type(o).__name__ for o in first) == ["Graded", "Graded", "Ungraded"]
    (failed_outcome,) = [o for o in first if isinstance(o, Ungraded)]
    assert failed_outcome.cause is Cause.MODEL_UNAVAILABLE
    failed = str(first.index(failed_outcome))

    resumed = fakes.script_model([fakes.text("395")])
    second = await run_solver(draft, policy, site, settings(fakes.flaky_factory(0, RuntimeError)), lambda _: resumed)

    assert len(resumed.requests) == 1
    assert all(isinstance(o, Graded) and o.reward == 1.0 for o in second)
    assert [p.name for p in sorted((site.evidence_dir / "solver" / failed).iterdir())] == [
        "attempt-0.json",
        "attempt-1.json",
    ]
    steps = [e.step for e in read_entries(tmp_path / "ledger" / "item.jsonl")]
    assert sorted(steps) == sorted({*(f"solver/{i}/0" for i in range(3)), f"solver/{failed}/1"})


async def test_solver_trials_run_under_the_drafts_convention(tmp_path, math_task, rounds, fakes):
    # The run-wide settings list JSON first; the draft was built for plain text, so a plain reply is correct.
    draft = rounds.draft(math_task, (), PLAIN)
    run_wide = settings(fakes.flaky_factory(0, RuntimeError), conventions=(JsonAnswer(id="json"), PLAIN))

    outcomes = await run_solver(
        draft, rounds.policy(k=2), rounds.site(tmp_path), run_wide, lambda _: fakes.script_model([fakes.text("395")])
    )

    assert [o.reward for o in outcomes if isinstance(o, Graded)] == [1.0, 1.0]
