# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""run_trials: deadlines, concurrency, retries on retryable causes, refusals, ledger spans and rollout evidence."""

import asyncio
import json
from dataclasses import dataclass

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import AGENT_TIMEOUT_STOP_REASON, ModelRequest, ModelTurn, RolloutContractError
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import TaskExecution
from taskcompendium.submission import JsonAnswer, JsonValueAnswer, PlainText, SubmissionConvention

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.llm.client import GlmRequestRejected
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.draft import shell_command
from taskforge.validate.outcome import Cause, Graded, TrialKind, Ungraded
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, run_trials

EXECUTION = TaskExecution()
DEADLINES = Deadlines(agent_timeout=30, attempt_timeout=60)


PLAIN = PlainText(id="plain")
JSON_VALUE = JsonValueAnswer(id="json-value")


def settings(factory, capabilities=None, conventions: tuple[SubmissionConvention, ...] = (PLAIN,)) -> EngineSettings:
    return EngineSettings(
        factories={EnvironmentKind.SHELLSIM: factory},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM} if capabilities is None else capabilities,
        max_turns=4,
        command_timeout=10,
        cleanup_timeout=10,
        conventions=conventions,
    )


def plan(
    tmp_path,
    k: int = 3,
    max_retries: int = 2,
    deadlines: Deadlines = DEADLINES,
    token_contract_retries: int = 0,
    first_attempt: int = 0,
) -> TrialPlan:
    return TrialPlan(
        item_id="item",
        round=1,
        kind=TrialKind.SOLVER,
        k=k,
        deadlines=deadlines,
        max_retries=max_retries,
        token_contract_retries=token_contract_retries,
        retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
        evidence_dir=tmp_path / "evidence",
        ledger=JsonlLedger(tmp_path / "ledger"),
        first_attempt=first_attempt,
    )


def ledger(tmp_path):
    return list(read_entries(tmp_path / "ledger" / "item.jsonl"))


async def test_failed_starts_are_retried_and_every_attempt_is_recorded(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=2, error=lambda: RuntimeError("broker refused"))
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcomes = await run_trials(file_task, EXECUTION, plan(tmp_path), settings(factory), model)

    assert [o.reward for o in outcomes if isinstance(o, Graded)] == [1.0, 1.0, 1.0]
    entries = ledger(tmp_path)
    assert sorted(e.cause for e in entries if e.cause) == [Cause.MACHINE_START, Cause.MACHINE_START]
    assert len(entries) == 5 and {e.round for e in entries} == {1}
    records = sorted((tmp_path / "evidence" / "solver").glob("*/attempt-*.json"))
    assert len(records) == 5
    graded = [json.loads(path.read_bytes()) for path in records if json.loads(path.read_bytes())["outcome"] == "graded"]
    assert all(record["rollout"]["grade"]["reward"] == 1.0 for record in graded)


async def test_a_re_entered_trial_numbers_attempts_after_the_ones_on_disk(tmp_path, file_task, fakes):
    earlier = tmp_path / "evidence" / "solver" / "0" / "attempt-1.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"earlier attempt")
    factory = fakes.flaky_factory(failures=1, error=lambda: RuntimeError("broker refused"))
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcomes = await run_trials(file_task, EXECUTION, plan(tmp_path, k=1, first_attempt=2), settings(factory), model)

    assert isinstance(outcomes[0], Graded)
    assert earlier.read_bytes() == b"earlier attempt"
    assert sorted(path.name for path in earlier.parent.iterdir()) == [
        "attempt-1.json",
        "attempt-2.json",
        "attempt-3.json",
    ]
    assert [e.step for e in ledger(tmp_path)] == ["solver/0/2", "solver/0/3"]


async def test_retries_stop_at_the_cap(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=100, error=lambda: RuntimeError("broker down"))

    outcomes = await run_trials(
        file_task, EXECUTION, plan(tmp_path, k=1, max_retries=2), settings(factory), fakes.script_model([])
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.MACHINE_START
    assert factory.creates == 3


async def test_non_retryable_failure_is_not_retried(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)
    model = fakes.raising_model(lambda: GlmRequestRejected(400, "invalid tools"))

    outcomes = await run_trials(file_task, EXECUTION, plan(tmp_path), settings(factory), model)

    assert all(isinstance(o, Ungraded) and o.cause is Cause.MODEL_REJECTED for o in outcomes)
    assert [e.cause for e in ledger(tmp_path)] == [Cause.MODEL_REJECTED] * 3


async def test_a_failing_setup_command_is_a_task_defect_and_is_not_retried(tmp_path, file_task, fakes):
    environment = file_task.environment.model_copy(update={"setup": (shell_command("exit 3", timeout=10),)})
    task = file_task.model_copy(update={"environment": environment})
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)

    outcomes = await run_trials(task, EXECUTION, plan(tmp_path), settings(factory), fakes.script_model([]))

    assert all(isinstance(o, Ungraded) and (o.cause, o.retryable) == (Cause.TASK_SETUP, False) for o in outcomes)
    assert factory.creates == 3


async def test_a_task_the_factories_refuse_never_starts_a_machine(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)

    outcomes = await run_trials(
        file_task, EXECUTION, plan(tmp_path), settings(factory, capabilities={}), fakes.script_model([])
    )

    assert all(isinstance(o, Ungraded) and o.cause is Cause.MACHINE_UNSUPPORTED for o in outcomes)
    assert all(isinstance(o, Ungraded) and "no_factory" in o.detail for o in outcomes)
    assert factory.creates == 0
    assert [e.cause for e in ledger(tmp_path)] == [Cause.MACHINE_UNSUPPORTED] * 3


async def test_trials_run_concurrently(tmp_path, math_task, fakes):
    # Every model call waits until all 20 trials are in flight, so serial trials would never finish.
    model = fakes.script_model([fakes.text("395")], barrier=asyncio.Barrier(20))

    outcomes = await run_trials(
        math_task, EXECUTION, plan(tmp_path, k=20), settings(fakes.flaky_factory(0, RuntimeError)), model
    )

    assert [o.reward for o in outcomes if isinstance(o, Graded)] == [1.0] * 20


async def test_validation_deadlines_replace_the_builders_agent_deadline(tmp_path, file_task, fakes):
    # The builder allows an hour; validation's agent deadline ends the trial after the first shell turn.
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")], hang_from=1)
    deadlines = Deadlines(agent_timeout=0.5, attempt_timeout=30)

    outcomes = await run_trials(
        file_task,
        TaskExecution(agent_timeout=3600),
        plan(tmp_path, k=1, deadlines=deadlines),
        settings(fakes.flaky_factory(0, RuntimeError)),
        model,
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].timed_out
    assert (outcomes[0].reward, outcomes[0].rollout.stop_reason) == (1.0, AGENT_TIMEOUT_STOP_REASON)


async def test_validation_attempt_deadline_bounds_a_trial_without_builder_deadlines(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError, delay=5)
    deadlines = Deadlines(agent_timeout=0.1, attempt_timeout=0.2)

    outcomes = await run_trials(
        file_task,
        EXECUTION,
        plan(tmp_path, k=1, max_retries=0, deadlines=deadlines),
        settings(factory),
        fakes.script_model([]),
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.ATTEMPT_TIMEOUT


async def test_the_ledger_input_hash_covers_the_deadlines(tmp_path, math_task, fakes):
    model = fakes.script_model([fakes.text("395")])
    factory = fakes.flaky_factory(0, RuntimeError)
    short, long = Deadlines(agent_timeout=30, attempt_timeout=60), Deadlines(agent_timeout=60, attempt_timeout=120)

    for deadlines in (short, short, long):
        await run_trials(math_task, EXECUTION, plan(tmp_path, k=1, deadlines=deadlines), settings(factory), model)

    first, again, other = (entry.input_hash for entry in ledger(tmp_path))
    assert first == again != other


async def test_each_task_runs_under_the_first_convention_that_carries_its_answer(tmp_path, json_task, fakes):
    model = fakes.script_model([fakes.text('{"sum": 60}')])
    conventions = (PLAIN, JSON_VALUE)

    outcomes = await run_trials(
        json_task,
        EXECUTION,
        plan(tmp_path, k=1),
        settings(fakes.flaky_factory(0, RuntimeError), None, conventions),
        model,
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    assert "one JSON value" in model.requests[0].messages[-1]["content"]


async def test_a_task_no_convention_carries_is_not_started(tmp_path, json_task, fakes):
    model = fakes.script_model([fakes.text('{"sum": 60}')])

    outcomes = await run_trials(
        json_task, EXECUTION, plan(tmp_path), settings(fakes.flaky_factory(0, RuntimeError)), model
    )

    assert all(
        isinstance(o, Ungraded) and (o.cause, o.retryable) == (Cause.SUBMISSION_UNSUPPORTED, False) for o in outcomes
    )
    assert all(isinstance(o, Ungraded) and "PlainText cannot carry json" in o.detail for o in outcomes)
    assert model.requests == []
    assert [e.cause for e in ledger(tmp_path)] == [Cause.SUBMISSION_UNSUPPORTED] * 3


async def test_a_machine_state_task_runs_under_any_convention(tmp_path, file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    factory = fakes.flaky_factory(0, RuntimeError)

    outcomes = await run_trials(file_task, EXECUTION, plan(tmp_path, k=1), settings(factory, None, (JSON_VALUE,)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0


async def test_the_ledger_input_hash_covers_the_convention(tmp_path, math_task, fakes):
    model = fakes.script_model([fakes.text("395")])
    factory = fakes.flaky_factory(0, RuntimeError)

    for conventions in ((PLAIN,), (PLAIN,), (JsonAnswer(id="json"),)):
        await run_trials(math_task, EXECUTION, plan(tmp_path, k=1), settings(factory, None, conventions), model)

    first, again, other = (entry.input_hash for entry in ledger(tmp_path))
    assert first == again != other


DIVERGENCE = "Served prompt diverges from the replayed prefix at index 6689: sampled (701, 7), served (23482,)"


@dataclass
class ContractBreakingModel:
    """Answers ``first``, then breaks the token contract on the next turn of the first ``broken``
    attempts, as ``GlmRolloutModel`` does when GLM sampled a non-canonical tokenization, then ``last``."""

    first: dict
    last: dict
    broken: int
    breaks: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if not request.prefix_token_ids:
            return ModelTurn(self.first, (10, 11), (20,), (-0.5,), "tool_calls")
        if self.breaks < self.broken:
            self.breaks += 1
            raise RolloutContractError(DIVERGENCE)
        return ModelTurn(self.last, (*request.prefix_token_ids, 90), (21,), (-0.5,), "stop")


def attempt_records(tmp_path) -> list[dict]:
    paths = sorted((tmp_path / "evidence" / "solver" / "0").glob("attempt-*.json"))
    return [json.loads(path.read_bytes()) for path in paths]


async def test_a_sampled_token_contract_break_is_retried_and_each_attempt_keeps_the_divergence(
    tmp_path, file_task, fakes
):
    model = ContractBreakingModel(fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done."), broken=2)
    trial_plan = plan(tmp_path, k=1, max_retries=0, token_contract_retries=2)

    outcomes = await run_trials(file_task, EXECUTION, trial_plan, settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    records = attempt_records(tmp_path)
    assert [record.get("cause") for record in records] == [Cause.TOKEN_CONTRACT, Cause.TOKEN_CONTRACT, None]
    assert all(DIVERGENCE in record["detail"] for record in records[:2])


async def test_token_contract_retries_stop_at_their_cap(tmp_path, file_task, fakes):
    model = ContractBreakingModel(fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done."), broken=100)
    trial_plan = plan(tmp_path, k=1, max_retries=5, token_contract_retries=1)

    outcomes = await run_trials(file_task, EXECUTION, trial_plan, settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.TOKEN_CONTRACT
    assert model.breaks == 2


async def test_the_evidence_counts_failed_cleanup_actions(tmp_path, file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcomes = await run_trials(
        file_task, EXECUTION, plan(tmp_path, k=1), settings(fakes.faulty_factory(close_error=True)), model
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    assert [record["cleanup_errors"] for record in attempt_records(tmp_path)] == [1]
    assert [entry.attrs["cleanup_errors"] for entry in ledger(tmp_path)] == ["1"]
