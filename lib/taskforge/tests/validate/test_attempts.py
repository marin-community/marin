# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt files read back: the inverse of outcome_json, and which trials are settled."""

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import RolloutContractError
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import TaskExecution
from taskcompendium.submission import PlainText

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmUnavailable
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.draft import shell_command
from taskforge.validate.attempts import load_outcome, trial_files
from taskforge.validate.outcome import Cause, Graded, TrialKind, Ungraded
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, outcome_json, run_trial


def settings(factory) -> EngineSettings:
    return EngineSettings(
        factories={EnvironmentKind.SHELLSIM: factory},
        capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
        max_turns=4,
        command_timeout=10,
        cleanup_timeout=10,
        conventions=(PlainText(id="plain"),),
    )


def plan(tmp_path, kind: TrialKind = TrialKind.SOLVER) -> TrialPlan:
    return TrialPlan(
        item_id="item",
        round=0,
        kind=kind,
        k=1,
        deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
        max_retries=0,
        token_contract_retries=0,
        retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
        evidence_dir=tmp_path,
        ledger=JsonlLedger(tmp_path / "ledger"),
        first_attempt=0,
    )


async def test_an_attempt_file_reads_back_to_the_outcome_it_records(tmp_path, file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    factory = fakes.flaky_factory(0, RuntimeError)
    graded = await run_trial(file_task, TaskExecution(), plan(tmp_path), settings(factory), model, "0")
    failing = fakes.raising_model(lambda: GlmUnavailable("router drained", ()))
    ungraded = await run_trial(file_task, TaskExecution(), plan(tmp_path), settings(factory), failing, "1")

    loaded = [load_outcome(tmp_path / "solver" / trial / "attempt-0.json") for trial in ("0", "1")]

    assert isinstance(loaded[0], Graded) and loaded[0].reward == 1.0
    assert isinstance(loaded[1], Ungraded) and loaded[1].cause is Cause.MODEL_UNAVAILABLE
    for path, original in zip(("0", "1"), (graded, ungraded), strict=True):
        assert outcome_json(load_outcome(tmp_path / "solver" / path / "attempt-0.json")) == outcome_json(original)


async def test_trial_files_take_each_trials_last_attempt_and_tell_settled_from_rerunnable(tmp_path, file_task, fakes):
    factory = fakes.flaky_factory(0, RuntimeError)
    adversary = plan(tmp_path, TrialKind.ADVERSARY)
    unavailable = fakes.raising_model(lambda: GlmUnavailable("router drained", ()))
    solved = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])
    await run_trial(file_task, TaskExecution(), adversary, settings(factory), unavailable, "shortcut/0")
    await run_trial(file_task, TaskExecution(), adversary, settings(factory), unavailable, "leak/0")
    diverged = fakes.raising_model(lambda: RolloutContractError("served prompt diverged"))
    await run_trial(file_task, TaskExecution(), adversary, settings(factory), diverged, "shortcut/1")
    retried = TrialPlan(**{**vars(adversary), "first_attempt": 1})
    await run_trial(file_task, TaskExecution(), retried, settings(factory), solved, "leak/0")
    broken = file_task.model_copy(
        update={"environment": file_task.environment.model_copy(update={"setup": (shell_command("exit 3", 10),)})}
    )
    await run_trial(broken, TaskExecution(), adversary, settings(factory), solved, "ambiguity/0")

    files = trial_files(tmp_path, TrialKind.ADVERSARY)

    assert list(files) == ["ambiguity/0", "leak/0", "shortcut/0", "shortcut/1"]
    assert (files["shortcut/0"].attempts, files["shortcut/0"].settled) == (1, False)
    contract = files["shortcut/1"]
    assert isinstance(contract.last, Ungraded) and contract.last.cause is Cause.TOKEN_CONTRACT and not contract.settled
    assert (files["leak/0"].attempts, files["leak/0"].settled) == (2, True)
    assert isinstance(files["leak/0"].last, Graded)
    ambiguity = files["ambiguity/0"]
    assert isinstance(ambiguity.last, Ungraded) and ambiguity.last.cause is Cause.TASK_SETUP and ambiguity.settled
    assert trial_files(tmp_path, TrialKind.SOLVER) == {}
