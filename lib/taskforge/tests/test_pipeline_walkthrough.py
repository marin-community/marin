# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One toy task carried through every Taskforge surface, on ShellSim, with scripted models.

The program here has three steps: ``numbers`` writes the worker file ``/workspace/numbers.txt``,
``grader`` derives a numeric grader from that file's bytes (so the edge between them is real), and
the sink ``task`` assembles the ``TaskSpec`` from both and the proposal. Solver trials alternate: an
even trial reads the file and replies ``42``, an odd one replies ``13``, so every round solves half.

Reading top to bottom, a reader learns:

1. a proposal is a Markdown document that parses, renders and names its item, and a source proposes
   it given a recorded model;
2. a program is a DAG of declared steps with one sink;
3. ``run_build`` runs it into a draft, and a rebuild is served from the step cache;
4. ``run_validation`` runs ``k`` solver trials into attempt files and ``calibration.json``, and a
   second run reuses every settled trial;
5. ``decide`` is a pure rule table from the summary to a decision, steered by ``BandRules``;
6. a ``Loop`` coordinates the same functions through the default transition table, one event per
   stage, and a relaunch on the same root changes nothing, while one with a changed idea is refused;
7. an unavailable model fails the item at its stage, and a relaunch re-enters that stage.
"""

from dataclasses import replace
from pathlib import Path

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.grader import GraderPackage
from taskcompendium.models import AnswerType, PlainText, Source, TaskResource, TaskSpec, VerifyitGrader
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import NumericSpec

from taskforge.builder.author import PROGRAM_MANIFEST_FILE, adopt
from taskforge.builder.run import DRAFT_DIR, PROVENANCE_FILE, TASK_FILE, item_id_for, load_draft, run_build
from taskforge.builder.sdk import BuildContext
from taskforge.builder.step import CacheStatus, program, step
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import GlmUnavailable, NoModel
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger, RecordedModel
from taskforge.loop.events import EXIT, ProposalOrigin, Stage, Terminal, derive_state
from taskforge.loop.policy import LoopPolicy, install_policy
from taskforge.loop.program import LEDGER_DIR, Loop, evidence_dir, run_idea, run_item
from taskforge.loop.stages import Propose, Stages
from taskforge.proposal.model import TaskProposal, body_sections, parse, render
from taskforge.review.decision import DECISION_FILE, Accept, BandOutcome, Reject, RejectKind
from taskforge.review.rules import BandChoice, BandRules, decide
from taskforge.sandbox.factories import factory_capabilities, machine_factories
from taskforge.spec.draft import answer_grader, assemble, file, requirements
from taskforge.triage.verdict import accept
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import CalibrationBand, FindingKind, load_summary, summarize
from taskforge.validate.outcome import Graded, TrialKind
from taskforge.validate.run import CALIBRATION_FILE, ValidationPolicy, load_validation, run_validation
from taskforge.validate.solver import ModelFactory, ValidationSite
from taskforge.validate.trials import EngineSettings, RetryBackoff, RolloutModel, SessionLimits, task_digest

SOURCE = Source(dataset="walkthrough", revision="r1", row="0", importer_revision="test")
ITEM = "IDEA--0"
BACKOFF = RetryBackoff(initial=0.001, maximum=0.001, factor=1.0, jitter=0.0)
"""A millisecond between attempts, the shortest wait: ``ExponentialBackoff`` refuses zero."""
VALIDATION = ValidationPolicy(
    k=4,
    band=CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875),
    sampling=LLMPolicy(max_continuations=0),
    limits=SessionLimits(
        max_turns=4,
        model_turn_timeout=20.0,
        command_timeout=10.0,
        tool_turn_timeout=20.0,
        total_turn_timeout=30.0,
        attempt_timeout=60.0,
        cleanup_timeout=10.0,
    ),
    max_retries=1,
    token_contract_retries=0,
    retry_backoff=BACKOFF,
)
POLICY = LoopPolicy(
    proposals_per_idea=1,
    max_idea_reproposals=0,
    max_build_revisions=1,
    max_validation_retries=2,
    retry_backoff=BACKOFF,
    band_rules=BandRules(too_easy=BandChoice.ACCEPT, too_hard=BandChoice.REJECT),
    validation=VALIDATION,
)


@step()
async def numbers(b: BuildContext) -> TaskResource:
    return file("workspace/numbers.txt", "6 7\n")


@step(inputs=("numbers",))
async def grader(b: BuildContext, numbers: TaskResource) -> GraderPackage:
    left, right = resource_bytes(numbers).split()
    return answer_grader(NumericSpec(expected=str(int(left) * int(right)), tolerance_abs=0.0, tolerance_rel=0.0))


@step(inputs=("numbers", "grader", "proposal"))
async def task(b: BuildContext, numbers: TaskResource, grader: GraderPackage, proposal: TaskProposal) -> TaskSpec:
    instruction = f"{body_sections(proposal.body)['Task']} Reply with the product alone."
    return assemble(
        b.item_id,
        instruction,
        AnswerType.TEXT,
        PlainText(),
        grader,
        SOURCE,
        environment=requirements(),
        files=(numbers,),
    )


PROGRAM = program(numbers, grader, task)


def scripted(turns: tuple[dict, ...]) -> RolloutModel:
    """A model that serves ``turns`` in order, with token ids that extend each request's prefix."""

    async def model(request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages)
        message = turns[served]
        prompt = (*request.prefix_token_ids, 2 * served + 1)
        stop = "tool_calls" if "tool_calls" in message else "stop"
        return ModelTurn(message, prompt, (2 * served + 2,), None, stop)

    return model


def reply_turns(reply: str) -> tuple[dict, ...]:
    """Read the numbers file with one shell call, then reply ``reply``."""
    command = '{"command": "cat /workspace/numbers.txt"}'
    call = {"id": "call-0", "type": "function", "function": {"name": "shell", "arguments": command}}
    return ({"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "assistant", "content": reply})


def alternating_solver(record: CallLedger, sampling: LLMPolicy) -> RolloutModel:
    """Solver trial ``solver/<index>``: an even index replies the product, an odd one a wrong sum."""
    index = int(record.step.rsplit("/", 1)[-1])
    return scripted(reply_turns("42" if index % 2 == 0 else "13"))


def engine() -> EngineSettings:
    return EngineSettings(factories=machine_factories(), capabilities=factory_capabilities())


async def test_the_stage_functions_carry_a_proposal_to_an_accepted_task_when_called_directly(
    tmp_path, ledger, proposal, proposal_source
):
    # 1. A proposal round-trips through its Markdown form and names its item; a source proposes it.
    assert parse(render(proposal)) == proposal
    assert item_id_for(proposal) == ITEM
    model = RecordedModel(NoModel(), LLMPolicy(), ledger, "idea--IDEA", 0, "propose")
    batch = await proposal_source.propose("idea", 1, model)
    assert batch.proposals == (proposal,)

    # 2. A program is a DAG of declared inputs with one sink, run in dependency order.
    assert (PROGRAM.sink, PROGRAM.order) == ("task", ("numbers", "grader", "task"))
    assert PROGRAM.steps["task"].inputs == ("numbers", "grader", "proposal")

    # 3. The build runs every step once, then serves a rebuild from the cache.
    item_dir = tmp_path / "item"
    first = await run_build(PROGRAM, proposal, item_dir, tmp_path / "cache", NoModel(), LLMPolicy(), ledger)
    again = await run_build(PROGRAM, proposal, item_dir, tmp_path / "cache", NoModel(), LLMPolicy(), ledger)
    assert [(record.name, record.status) for record in first.provenance.steps] == [
        ("numbers", CacheStatus.MISS),
        ("grader", CacheStatus.MISS),
        ("task", CacheStatus.MISS),
    ]
    assert {record.status for record in again.provenance.steps} == {CacheStatus.HIT}
    assert again.task == first.task
    assert isinstance(first.task.grader, VerifyitGrader)
    assert [resource.path for resource in first.task.resources.worker] == ["workspace/numbers.txt"]
    assert load_draft(item_dir / DRAFT_DIR) == again

    # 4. Validation runs k solver trials, one attempt file each, and summarizes them.
    evidence = item_dir / "evidence"
    site = ValidationSite(ITEM, 0, evidence, ledger)
    summary = await run_validation(first, VALIDATION, site, engine(), alternating_solver)
    files = trial_files(evidence, TrialKind.SOLVER)
    assert [(name, trial.attempts) for name, trial in files.items()] == [("0", 1), ("1", 1), ("2", 1), ("3", 1)]
    outcomes = [trial.last for trial in files.values()]
    assert [outcome.reward for outcome in outcomes if isinstance(outcome, Graded)] == [1.0, 0.0, 1.0, 0.0]
    assert (summary.solver.graded, summary.solver.solved, summary.solve_rate) == (4, 2, 0.5)
    assert summary.task_digest == task_digest(first.task)
    trials = sum(entry.kind is EntryKind.TRIAL for entry in ledger.entries)
    assert await run_validation(first, VALIDATION, site, engine(), alternating_solver) == summary
    assert sum(entry.kind is EntryKind.TRIAL for entry in ledger.entries) == trials
    assert load_summary(evidence / CALIBRATION_FILE) == summary

    # 5. Review is a rule table: in the band accepts; outside it, the band rule decides.
    rules = POLICY.band_rules
    assert decide(summary, rules) == Accept(band=BandOutcome.IN_BAND)
    narrow = replace(VALIDATION, band=CalibrationBand(min_solve_rate=0.0, max_solve_rate=0.25))
    too_easy = summarize(load_validation(first, evidence), narrow)
    assert [finding.kind for finding in too_easy.findings] == [FindingKind.TOO_EASY]
    assert decide(too_easy, rules) == Accept(band=BandOutcome.TOO_EASY)
    strict = decide(too_easy, BandRules(too_easy=BandChoice.REJECT, too_hard=BandChoice.REJECT))
    assert isinstance(strict, Reject) and strict.kind is RejectKind.TASK


def loop_for(root: Path, propose: Propose[str], rollout_models: ModelFactory) -> Loop[str]:
    """A loop over the library's own stage functions, with the toy program as every item's program."""
    stages = Stages(
        propose=propose,
        triage=accept,
        author=adopt(PROGRAM),
        build=run_build,
        validate=run_validation,
        review=decide,
    )
    return Loop(
        stages=stages,
        policy=install_policy(root, POLICY),
        root=root,
        ledger=JsonlLedger(root / LEDGER_DIR),
        engine=engine(),
        rollout_models=rollout_models,
        model=NoModel(),
        sampling=LLMPolicy(),
        describe_idea=lambda idea: {"idea": idea},
    )


def item_events(root: Path) -> list[tuple[str, str | None]]:
    """Each event of the item's log, with its exit (or, for ``terminal``, its terminal)."""
    entries = read_entries(JsonlLedger(root / LEDGER_DIR).path_for(ITEM))
    return [
        (entry.step, entry.attrs.get(EXIT, entry.attrs.get("terminal")))
        for entry in entries
        if entry.kind is EntryKind.EVENT
    ]


def ledger_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted((root / LEDGER_DIR).iterdir())}


async def test_the_loop_runs_one_event_per_stage_and_a_relaunch_changes_nothing(tmp_path, proposal_source):
    # 6. The loop calls the same stage functions, records one event per stage, and keeps each round's files.
    root = tmp_path / "run"
    loop = loop_for(root, proposal_source.propose, alternating_solver)
    [proposal] = await run_idea(loop, "IDEA", "idea")
    assert await run_item(loop, proposal, ProposalOrigin.GENERATED) is Terminal.ACCEPTED

    assert item_events(root) == [
        ("opened", None),
        ("triaged", "accept"),
        ("authored", "authored"),
        ("built", "built"),
        ("validated", "validated"),
        ("decided", "accept"),
        ("terminal", "accepted"),
    ]
    state = derive_state(read_entries(JsonlLedger(root / LEDGER_DIR).path_for(ITEM)))
    assert (state.stage, state.terminal, state.round) == (Stage.DONE, Terminal.ACCEPTED, 0)
    round_dir = root / "items" / ITEM / "rounds" / "0"
    assert state.task_digest is not None
    evidence = evidence_dir(round_dir, state.task_digest)
    for path in (
        round_dir / PROGRAM_MANIFEST_FILE,
        round_dir / DRAFT_DIR / TASK_FILE,
        round_dir / DRAFT_DIR / PROVENANCE_FILE,
        evidence / CALIBRATION_FILE,
        evidence / DECISION_FILE,
    ):
        assert path.is_file(), path

    before = ledger_bytes(root)
    relaunched = loop_for(root, proposal_source.propose, alternating_solver)
    assert await run_idea(relaunched, "IDEA", "idea") == (proposal,)
    assert await run_item(relaunched, proposal, ProposalOrigin.GENERATED) is Terminal.ACCEPTED
    assert ledger_bytes(root) == before
    assert proposal_source.calls == 1


async def test_a_relaunch_with_a_changed_idea_under_the_same_id_is_refused(tmp_path, proposal_source):
    # 6. An idea id names one idea per root: a changed record cannot reuse the old batch.
    root = tmp_path / "run"
    await run_idea(loop_for(root, proposal_source.propose, alternating_solver), "IDEA", "idea")
    before = ledger_bytes(root)

    with pytest.raises(ValueError, match=r"idea IDEA changed .* use a new run root"):
        await run_idea(loop_for(root, proposal_source.propose, alternating_solver), "IDEA", "another idea")
    assert ledger_bytes(root) == before
    assert proposal_source.calls == 1


class UnavailableOnce:
    """A solver ``ModelFactory`` whose first request for a model finds the model unavailable."""

    def __init__(self) -> None:
        self.asked = 0

    def __call__(self, record: CallLedger, sampling: LLMPolicy) -> RolloutModel:
        self.asked += 1
        if self.asked == 1:
            raise GlmUnavailable("the hold ran out", ())
        return alternating_solver(record, sampling)


async def test_an_unavailable_model_fails_the_item_at_validation_and_a_relaunch_resumes_there(
    tmp_path, proposal, proposal_source
):
    # 7. GlmUnavailable records TERMINAL(failed) in place; the next run_item re-enters that stage.
    root = tmp_path / "run"
    models = UnavailableOnce()
    with pytest.raises(GlmUnavailable):
        await run_item(loop_for(root, proposal_source.propose, models), proposal, ProposalOrigin.SUPPLIED)
    state = derive_state(read_entries(JsonlLedger(root / LEDGER_DIR).path_for(ITEM)))
    assert (state.stage, state.terminal) == (Stage.VALIDATE, Terminal.FAILED)

    relaunched = loop_for(root, proposal_source.propose, models)
    assert await run_item(relaunched, proposal, ProposalOrigin.SUPPLIED) is Terminal.ACCEPTED
    assert item_events(root) == [
        ("opened", None),
        ("triaged", "accept"),
        ("authored", "authored"),
        ("built", "built"),
        ("terminal", "failed"),
        ("validated", "validated"),
        ("decided", "accept"),
        ("terminal", "accepted"),
    ]
