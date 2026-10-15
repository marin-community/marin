# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""run_trials: deadlines, concurrency, retries on retryable causes, refusals, ledger spans and rollout evidence."""

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Self

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import TOTAL_TURN_TIMEOUT_STOP_REASON, ModelRequest, ModelTurn, RolloutData
from shellbox.machine import Backend
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import JsonAnswer, PlainText

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import ModelRequestRejected, TokenEvidenceError
from taskforge.llm.glm_client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel, RolloutTransport, TokenContractBreak
from taskforge.stage50_validate.outcome import Cause, Graded, TrialKind, Ungraded
from taskforge.stage50_validate.trials import (
    Deadlines,
    EngineSettings,
    TrialPlan,
    contract_retries,
    outcome_json,
    run_trials,
    task_digest,
)
from tests.conftest import fake_source
from tests.sandbox.fixture_images import FixtureImageFactory

DEADLINES = Deadlines(total_turn_timeout=30, attempt_timeout=60)
SHELLSIM_BACKEND = Backend.SHELLSIM.value
DOCKER_BACKEND = Backend.DOCKER.value


def settings(factory) -> EngineSettings:
    """Task machines from ``factory``; verifier machines from the fixture image factory."""
    return EngineSettings(
        factories={SHELLSIM_BACKEND: factory, DOCKER_BACKEND: FixtureImageFactory()},
        max_turns=4,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
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

    outcomes = await run_trials(file_task, plan(tmp_path), settings(factory), model)

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

    outcomes = await run_trials(file_task, plan(tmp_path, k=1, first_attempt=2), settings(factory), model)

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

    outcomes = await run_trials(file_task, plan(tmp_path, k=1, max_retries=2), settings(factory), fakes.script_model([]))

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.MACHINE_START
    assert factory.creates == 3


async def test_non_retryable_failure_is_not_retried(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)
    model = fakes.raising_model(lambda: ModelRequestRejected(400, "invalid tools", ()))

    outcomes = await run_trials(file_task, plan(tmp_path), settings(factory), model)

    assert all(isinstance(o, Ungraded) and o.cause is Cause.MODEL_REJECTED for o in outcomes)
    assert [e.cause for e in ledger(tmp_path)] == [Cause.MODEL_REJECTED] * 3


async def test_misaligned_token_evidence_is_not_retried_even_with_a_contract_budget(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)
    model = fakes.raising_model(lambda: TokenEvidenceError("3 token ids but 2 logprob entries"))

    outcomes = await run_trials(file_task, plan(tmp_path, k=1, token_contract_retries=3), settings(factory), model)

    [outcome] = outcomes
    assert isinstance(outcome, Ungraded) and not outcome.retryable
    assert len(ledger(tmp_path)) == 1


async def test_a_failing_setup_command_is_a_task_defect_and_is_not_retried(tmp_path, file_task_with, fakes):
    task = file_task_with(setup=("exit 3",))
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)

    outcomes = await run_trials(task, plan(tmp_path), settings(factory), fakes.script_model([]))

    assert all(isinstance(o, Ungraded) and (o.cause, o.retryable) == (Cause.TASK_SETUP, False) for o in outcomes)
    assert factory.creates == 3


async def test_a_task_the_factories_refuse_never_starts_a_machine(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError)
    task_machine = file_task.runtime.task_machine.model_copy(update={"cpus": 2})
    refused = file_task.model_copy(
        update={"runtime": file_task.runtime.model_copy(update={"task_machine": task_machine})}
    )

    outcomes = await run_trials(refused, plan(tmp_path), settings(factory), fakes.script_model([]))

    assert all(isinstance(o, Ungraded) and o.cause is Cause.MACHINE_UNSUPPORTED for o in outcomes)
    assert all(isinstance(o, Ungraded) and "ShellSim does not provide host CPU" in o.detail for o in outcomes)
    assert factory.creates == 0
    assert [e.cause for e in ledger(tmp_path)] == [Cause.MACHINE_UNSUPPORTED] * 3


async def test_trials_run_concurrently(tmp_path, math_task, fakes):
    # Every model call waits until all 20 trials are in flight, so serial trials would never finish.
    model = fakes.script_model([fakes.text("395")], barrier=asyncio.Barrier(20))

    outcomes = await run_trials(math_task, plan(tmp_path, k=20), settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert [o.reward for o in outcomes if isinstance(o, Graded)] == [1.0] * 20


async def test_each_trial_of_a_glm_model_records_and_routes_under_its_own_attempt_step(
    tmp_path, math_task, fakes, fake_glm, fast_provider
):
    for _ in range(2):
        fake_glm.stream(
            content="395", prompt_tokens=3, completion_tokens=2, prompt_token_ids=(1, 2, 3), token_ids=(4, 5)
        )
    trial_plan = plan(tmp_path, k=2, max_retries=0)

    async with GlmClient(fake_source(fake_glm), replace(fast_provider, max_attempts=1)) as client:
        record = CallLedger(trial_plan.ledger, trial_plan.item_id, trial_plan.round, str(TrialKind.SOLVER))
        model = GlmRolloutModel(client, LLMPolicy(max_continuations=0), RolloutTransport.CHAT_COMPLETIONS, record)
        outcomes = await run_trials(math_task, trial_plan, settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert [o.reward for o in outcomes if isinstance(o, Graded)] == [1.0, 1.0]
    assert sorted(body["prompt_cache_key"] for body in fake_glm.requests) == ["item:1:solver/0/0", "item:1:solver/1/0"]
    calls = [entry.step for entry in ledger(tmp_path) if entry.kind is EntryKind.LLM_CALL]
    assert sorted(calls) == ["solver/0/0", "solver/1/0"]


async def test_validation_deadlines_replace_the_builders_turn_deadline(tmp_path, file_task, fakes):
    # The builder allows an hour; validation's total-turn deadline ends the trial after the first shell turn.
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")], hang_from=1)
    deadlines = Deadlines(total_turn_timeout=0.5, attempt_timeout=30)
    builder = file_task.model_copy(update={"session": file_task.session.model_copy(update={"total_turn_timeout": 3600})})

    outcomes = await run_trials(
        builder,
        plan(tmp_path, k=1, deadlines=deadlines),
        settings(fakes.flaky_factory(0, RuntimeError)),
        model,
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].timed_out
    assert (outcomes[0].reward, outcomes[0].rollout.stop_reason) == (1.0, TOTAL_TURN_TIMEOUT_STOP_REASON)


async def test_validation_attempt_deadline_bounds_a_trial_without_builder_deadlines(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(failures=0, error=RuntimeError, delay=5)
    deadlines = Deadlines(total_turn_timeout=0.1, attempt_timeout=0.2)

    outcomes = await run_trials(
        file_task,
        plan(tmp_path, k=1, max_retries=0, deadlines=deadlines),
        settings(factory),
        fakes.script_model([]),
    )

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.ATTEMPT_TIMEOUT


async def test_the_ledger_input_hash_covers_the_deadlines(tmp_path, math_task, fakes):
    model = fakes.script_model([fakes.text("395")])
    factory = fakes.flaky_factory(0, RuntimeError)
    short = Deadlines(total_turn_timeout=30, attempt_timeout=60)
    long = Deadlines(total_turn_timeout=60, attempt_timeout=120)

    for deadlines in (short, short, long):
        await run_trials(math_task, plan(tmp_path, k=1, deadlines=deadlines), settings(factory), model)

    first, again, other = (entry.input_hash for entry in ledger(tmp_path))
    assert first == again != other


async def test_a_task_is_presented_in_its_own_answer_format(tmp_path, json_task, fakes):
    model = fakes.script_model([fakes.text('{"sum": 60}')])

    outcomes = await run_trials(json_task, plan(tmp_path, k=1), settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    assert "one JSON value" in model.requests[0].messages[-1]["content"]


async def test_a_task_whose_answer_format_cannot_carry_its_answer_is_not_started(tmp_path, json_task, fakes):
    model = fakes.script_model([fakes.text('{"sum": 60}')])
    # assemble refuses this pairing; a task read from elsewhere can still carry it.
    plain = json_task.model_copy(update={"task": json_task.task.model_copy(update={"answer_format": PlainText()})})

    outcomes = await run_trials(plain, plan(tmp_path), settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert all(
        isinstance(o, Ungraded) and (o.cause, o.retryable) == (Cause.SUBMISSION_UNSUPPORTED, False) for o in outcomes
    )
    assert all(isinstance(o, Ungraded) and "PlainText cannot carry json" in o.detail for o in outcomes)
    assert model.requests == []
    assert [e.cause for e in ledger(tmp_path)] == [Cause.SUBMISSION_UNSUPPORTED] * 3


async def test_the_ledger_input_hash_is_the_digest_of_the_lowered_task_with_its_answer_format(
    tmp_path, math_task, relower, fakes
):
    model = fakes.script_model([fakes.text("395")])
    factory = fakes.flaky_factory(0, RuntimeError)
    as_json = relower(math_task.task.model_copy(update={"answer_format": JsonAnswer()}))

    for task in (math_task, math_task, as_json):
        await run_trials(task, plan(tmp_path, k=1), settings(factory), model)

    first, again, other = (entry.input_hash for entry in ledger(tmp_path))
    assert first == again != other
    assert first == task_digest(settings(factory).apply(DEADLINES.apply(math_task)))


DIVERGENCE = "Served prompt diverges from the replayed prefix at index 6689: sampled (701, 7), served (23482,)"


@dataclass
class ContractBreakingModel:
    """Answers ``first``, then breaks the token contract on the next turn of the first ``broken``
    attempts, as ``GlmRolloutModel`` does when GLM sampled a non-canonical tokenization, then ``last``."""

    first: dict
    last: dict
    broken: int
    breaks: int = 0
    broken_at: tuple[dict, ...] = ()

    def for_step(self, step: str) -> Self:
        return self

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if not request.prefix_token_ids:
            return ModelTurn(self.first, (10, 11), (20,), (-0.5,), "tool_calls")
        if self.breaks < self.broken:
            self.breaks += 1
            self.broken_at = tuple(request.messages)
            raise TokenContractBreak(
                DIVERGENCE, turn=1, messages=self.broken_at, prefix_length=len(request.prefix_token_ids)
            )
        return ModelTurn(self.last, (*request.prefix_token_ids, 90), (21,), (-0.5,), "stop")


def attempt_records(tmp_path) -> list[dict]:
    paths = sorted((tmp_path / "evidence" / "solver" / "0").glob("attempt-*.json"))
    return [json.loads(path.read_bytes()) for path in paths]


async def test_a_sampled_token_contract_break_is_retried_and_each_attempt_keeps_the_divergence(
    tmp_path, file_task, fakes
):
    model = ContractBreakingModel(fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done."), broken=2)
    trial_plan = plan(tmp_path, k=1, max_retries=0, token_contract_retries=2)

    outcomes = await run_trials(file_task, trial_plan, settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    records = attempt_records(tmp_path)
    assert [record.get("cause") for record in records] == [Cause.TOKEN_CONTRACT, Cause.TOKEN_CONTRACT, None]
    assert all(DIVERGENCE in record["detail"] for record in records[:2])
    assert contract_retries(tmp_path / "evidence", TrialKind.SOLVER) == 2


async def test_contract_break_keeps_the_partial_conversation(tmp_path, file_task, fakes):
    model = ContractBreakingModel(fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done."), broken=1)

    await run_trials(file_task, plan(tmp_path, k=1), settings(fakes.flaky_factory(0, RuntimeError)), model)

    (record,) = attempt_records(tmp_path)
    assert record["contract_break"]["messages"] == [dict(message) for message in model.broken_at]
    assert [message["role"] for message in record["contract_break"]["messages"]] == ["user", "assistant", "tool"]
    assert (record["contract_break"]["turn"], record["contract_break"]["prefix_length"]) == (1, 3)


def test_a_candidate_code_error_is_recorded_on_its_graded_attempt():
    grade = GradeResult(Outcome.GRADED, 0.0, detail={"reason": "collection_error"})
    outcome = Graded(RolloutData("task", (), (), (), (), None, grade, "stop"), candidate_code_error=True)

    record = json.loads(outcome_json(outcome))

    assert (record["outcome"], record["reward"], record["candidate_code_error"]) == ("graded", 0.0, True)


def test_captured_output_files_are_recorded_as_base64_on_the_attempt():
    grade = GradeResult(Outcome.GRADED, 1.0)
    rollout = RolloutData("task", (), (), (), (), None, grade, "stop", outputs={"/workspace/sum.txt": b"60\n\x00"})

    record = json.loads(outcome_json(Graded(rollout, candidate_code_error=False)))

    assert record["rollout"]["outputs"] == {"/workspace/sum.txt": "NjAKAA=="}


async def test_token_contract_retries_stop_at_their_cap(tmp_path, file_task, fakes):
    model = ContractBreakingModel(fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done."), broken=100)
    trial_plan = plan(tmp_path, k=1, max_retries=5, token_contract_retries=1)

    outcomes = await run_trials(file_task, trial_plan, settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Ungraded) and outcomes[0].cause is Cause.TOKEN_CONTRACT
    assert model.breaks == 2


async def test_the_evidence_counts_failed_cleanup_actions(tmp_path, file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcomes = await run_trials(file_task, plan(tmp_path, k=1), settings(fakes.faulty_factory(close_error=True)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0
    assert [record["cleanup_errors"] for record in attempt_records(tmp_path)] == [1]
    assert [entry.attrs["cleanup_errors"] for entry in ledger(tmp_path)] == ["1"]


async def test_engine_settings_replace_the_builders_turn_budget(tmp_path, file_task, fakes):
    # The builder allows one turn; validation's four let the agent write the file on its second.
    model = fakes.script_model(
        [fakes.shell("true", "c0"), fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")]
    )
    builder = file_task.model_copy(update={"session": file_task.session.model_copy(update={"max_turns": 1})})

    outcomes = await run_trials(builder, plan(tmp_path, k=1), settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert len(outcomes) == 1 and isinstance(outcomes[0], Graded) and outcomes[0].reward == 1.0


async def test_the_attempt_file_keeps_the_steps(tmp_path, file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcomes = await run_trials(file_task, plan(tmp_path, k=1), settings(fakes.flaky_factory(0, RuntimeError)), model)

    assert isinstance(outcomes[0], Graded) and len(outcomes[0].rollout.steps) == 2
    (record,) = attempt_records(tmp_path)
    steps = record["rollout"]["steps"]
    assert [step["turn"]["message"]["content"] for step in steps] == ["", "Done."]
