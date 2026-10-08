# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""summarize: band findings over complete evidence, decisive findings regardless, and adversary trials tiered by
their verifier submissions into repairs (with the accepted candidate as a control), notes and no defect.

Adversary trials run for real against ``fake_glm`` (scripted agent turns) on ShellSim with the task's grader, a
script that runs on the host."""

from collections import Counter
from dataclasses import replace

import pytest
from rigging.timing import ExponentialBackoff
from rolloutengine.spec import LoweredTaskSpec
from shellbox.machine import Backend
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AnswerType, AssistantToolCalls, ConversationToolCall
from taskcompendium.submission import ANSWER_CALL_NAME, ANSWER_FIELD, PlainText

from taskforge.builder.step import StepRole
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.controls import REJECTION_CEILING, ControlKind, Transcript, reply
from taskforge.spec.draft import assemble, script_verifier
from taskforge.validate.adversary import AdversaryRole, ClaimKind, run_adversaries
from taskforge.validate.calibration import (
    DECISIVE,
    AdversarySignals,
    Comparison,
    DefectTier,
    FindingKind,
    RoleStats,
    TaskFacts,
    compare,
    consumed_inputs,
    honest_submissions,
    load_summary,
    summarize,
    task_facts,
    tier_of,
    write_summary,
)
from taskforge.validate.controls import control_outcome, workspace_turn
from taskforge.validate.evidence import Complete, Incomplete
from taskforge.validate.outcome import Cause, Outcome, TrialKind, Ungraded
from taskforge.validate.run import ValidationEvidence
from taskforge.validate.submissions import AdversaryTrial
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, run_trial

DIGEST = "ab" * 32
UNAVAILABLE = Ungraded(Cause.MODEL_UNAVAILABLE, "router drained", None)
SETUP_FAILED = Ungraded(Cause.TASK_SETUP, "setup command exited 3", None)
PLAIN = PlainText(id="plain")
SHORTCUT = AdversaryRole.SHORTCUT
NUMBERS = "/workspace/numbers.txt"
SUM = "/workspace/sum.txt"
SHELLSIM_BACKEND = Backend.SHELLSIM.value


RECOMPUTES = f"captured({SUM!r}).strip() == str(sum(int(n) for n in captured({NUMBERS!r}).split()))"
"""A ``verdict_grader`` condition: the sum file holds the sum of whatever the numbers file holds."""
MENTIONS_60 = f"'60' in captured({SUM!r})"
SOLVE = f"awk '{{s+=$1}} END{{print s}}' {NUMBERS} > {SUM}"
HEDGE = "394 or 395 or 396"


@pytest.fixture
def lenient_text_task(math_task, relower, verdict_grader) -> LoweredTaskSpec:
    """The math question without a machine, graded by a script that accepts any answer containing 395."""
    task = math_task.task
    return relower(
        assemble(
            "validate-lenient-text",
            task.context.events[-1].content,
            AnswerType.NUMBER,
            script_verifier(verdict_grader("'395' in answer"), {}, timeout=30),
            task.source,
            environment=None,
        )
    )


@pytest.fixture
def trial(tmp_path, fakes):
    """Runs ``task`` once with scripted assistant turns (a ``str`` is a shell call, a ``Reply`` a text turn)."""
    count = 0

    async def run(task: LoweredTaskSpec, *turns: str, reply: str | None = "Done.", max_turns: int = 6) -> Outcome:
        nonlocal count
        count += 1
        plan = TrialPlan(
            item_id="item",
            round=0,
            kind=TrialKind.SOLVER,
            k=1,
            deadlines=Deadlines(total_turn_timeout=30, attempt_timeout=60),
            max_retries=0,
            token_contract_retries=0,
            retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
            evidence_dir=tmp_path,
            ledger=JsonlLedger(tmp_path / "ledger"),
            first_attempt=0,
        )
        settings = EngineSettings(
            factories={SHELLSIM_BACKEND: fakes.flaky_factory(0, RuntimeError)},
            capabilities={SHELLSIM_BACKEND: SHELLSIM},
            max_turns=max_turns,
            command_timeout=10,
            tool_turn_timeout=20,
            model_turn_timeout=30,
            cleanup_timeout=10,
            conventions=(PlainText(id="plain"),),
        )
        messages = [*(fakes.shell(command) for command in turns), *(() if reply is None else (fakes.text(reply),))]
        return await run_trial(task, plan, settings, fakes.script_model(messages), str(count))

    return run


@pytest.fixture
def adversary(tmp_path, rounds, fakes, fake_glm, glm_client, turns):
    """Runs one adversary trial of ``task`` with scripted agent turns (see ``conftest.adversary_turns``)."""
    count = 0

    async def run(task: LoweredTaskSpec, *script) -> AdversaryTrial:
        nonlocal count
        count += 1
        turns(fake_glm, *script)
        settings = EngineSettings(
            factories={SHELLSIM_BACKEND: fakes.flaky_factory(0, RuntimeError)},
            capabilities={SHELLSIM_BACKEND: SHELLSIM},
            max_turns=8,
            command_timeout=10,
            tool_turn_timeout=20,
            model_turn_timeout=30,
            cleanup_timeout=10,
            conventions=(PLAIN,),
        )
        site = rounds.site(tmp_path / f"adversary-{count}")
        trials = await run_adversaries(
            rounds.draft(task, (), PLAIN), rounds.policy(adversary_k=1), site, settings, glm_client, ""
        )
        return trials[SHORTCUT][0]

    return run


@pytest.fixture
def answer(trial, math_task):
    """Runs the math task once with a scripted final reply and returns the outcome."""

    async def run(reply: str) -> Outcome:
        return await trial(math_task, reply=reply)

    return run


def evidence(facts: TaskFacts, controls=(), solver=(), adversaries=None) -> ValidationEvidence:
    return ValidationEvidence(DIGEST, tuple(controls), tuple(solver), adversaries or {}, facts)


def failed(outcome: Ungraded) -> AdversaryTrial:
    """An adversary trial that never ran its agent loop."""
    return AdversaryTrial(outcome, "brief", ())


def signals(**changes) -> AdversarySignals:
    """An adversary that read the input, passed with its one file-backed submission and reported no shortcut."""
    base = {
        "submissions": 1,
        "passes": 1,
        "budget_spent": False,
        "first_pass": 1,
        "exploit": 1,
        "claim": ClaimKind.NO_SHORTCUT,
        "why": "",
        "stop_reason": "stop",
        "exhausted": False,
        "output_tokens": 100,
        "turns": 3,
        "shell_calls": 1,
        "inputs_consumed": (NUMBERS,),
        "protected_supplied": (),
        "fileless_passes": (),
        "mismatches": (),
        "comparison": Comparison.UNAVAILABLE,
    }
    return AdversarySignals(**(base | changes))


async def test_band_findings_need_complete_evidence_with_k_graded_solver_trials(answer, rounds, math_facts):
    policy = rounds.policy(k=3)
    right, wrong = await answer("395"), await answer("391")

    too_easy = summarize(evidence(math_facts, solver=(right, right, right)), policy)
    too_hard = summarize(evidence(math_facts, solver=(wrong, wrong, wrong)), policy)
    calibrated = summarize(evidence(math_facts, solver=(right, wrong, wrong)), policy)
    incomplete = summarize(evidence(math_facts, solver=(wrong, wrong, UNAVAILABLE)), policy)
    short = summarize(evidence(math_facts, solver=(wrong, wrong)), policy)

    assert [f.kind for f in too_easy.findings] == [FindingKind.TOO_EASY]
    assert too_easy.findings[0].roles == (StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS)
    assert [f.kind for f in too_hard.findings] == [FindingKind.TOO_HARD]
    assert not too_easy.decisive and not too_hard.decisive
    assert calibrated.findings == () and calibrated.calibrated and calibrated.solve_rate == pytest.approx(1 / 3)
    assert incomplete.status == Incomplete(Counter({Cause.MODEL_UNAVAILABLE: 1}))
    assert incomplete.findings == () and not incomplete.calibrated
    assert isinstance(short.status, Complete) and short.findings == () and short.solver.graded == 2


async def test_decisive_findings_are_reported_when_other_trials_are_ungraded(answer, rounds, math_controls, math_facts):
    correct = next(c for c in math_controls if c.kind is ControlKind.POSITIVE)
    violated = control_outcome(correct, await answer("391"))
    adversaries = {SHORTCUT: (failed(SETUP_FAILED), failed(SETUP_FAILED))}

    summary = summarize(evidence(math_facts, (violated,), (UNAVAILABLE,), adversaries), rounds.policy(k=3))

    assert [f.kind for f in summary.findings] == [FindingKind.CONTROL_VIOLATED, FindingKind.TASK_DEFECT]
    assert summary.decisive == summary.findings and {f.kind for f in summary.findings} <= DECISIVE
    violation, defect = summary.findings
    assert violation.roles == (StepRole.GRADER, StepRole.CONTROLS) and "control/correct" in violation.detail
    assert defect.roles == (StepRole.ENVIRONMENT, StepRole.FIXTURES)
    assert "adversary/shortcut/0" in defect.detail and "adversary/shortcut/1" in defect.detail
    assert summary.controls_violated == ("correct",)
    assert summary.status == Incomplete(Counter({Cause.MODEL_UNAVAILABLE: 1, Cause.TASK_SETUP: 2}))
    assert summary.assessments == () and summary.notes == ()


@pytest.mark.parametrize(
    ("command", "consumed"),
    [
        (f"cat {NUMBERS}", True),
        (f"head -c 200 {NUMBERS}", True),
        (f"od -c {NUMBERS}", True),
        (f"awk '{{s+=$1}} END{{print s}}' {NUMBERS} > {SUM}", True),
        (f"tr '\\n' '+' < {NUMBERS}", True),
        (f"while read n; do s=$((s+n)); done < {NUMBERS}; echo $s > {SUM}", True),
        (f"printf '%s' \"$(awk '{{s+=$1}} END{{print s}}' {NUMBERS})\" > {SUM}", True),
        (f"LC_ALL=C sort -n {NUMBERS}", True),
        (f"sed -i s/12/0/ {NUMBERS}", True),
        (f"wc -l {NUMBERS}", False),
        (f"ls -la {NUMBERS}", False),
        (f"stat {NUMBERS}", False),
        (f"cp {NUMBERS} {NUMBERS}.bak", False),
        (f"cat {NUMBERS}.bak", False),
        (f"seq -5000 15000 >> {SUM}", False),
        (f"printf '0\\n' > {NUMBERS}", False),
        (f"echo 60 > {NUMBERS}", False),
        (f"rm -f {NUMBERS}", False),
    ],
)
def test_command_signals_read_the_transcript(command, consumed):
    assert consumed_inputs([command], [NUMBERS]) == ((NUMBERS,) if consumed else ())


@pytest.mark.parametrize(
    ("submission", "references", "expected"),
    [
        ("ANSWER = 41\nANSWER = 43\nANSWER = 42", ["ANSWER = 42"], Comparison.MISMATCH),
        ("394 or 395 or 396", ["395"], Comparison.MISMATCH),
        ("The answer is 395.", ["395"], Comparison.MATCH),
        ("395", ["395"], Comparison.MATCH),
        ("  The CAPITAL is\nParis ", ["the capital is paris"], Comparison.MATCH),
        ("The capital is Paris", ["Paris"], Comparison.MISMATCH),
        ("60", [], Comparison.UNAVAILABLE),
        (None, ["60"], Comparison.UNAVAILABLE),
    ],
)
def test_compare_by_normalised_text_then_numeric_tokens(submission, references, expected):
    assert compare(submission, references) is expected


async def test_honest_submissions_come_from_solved_solver_trials_and_positive_transcript_controls(
    answer, math_controls, file_controls, math_facts, file_facts
):
    right, wrong = await answer("395"), await answer("391")
    math = [control_outcome(c, right) for c in math_controls]
    files = [control_outcome(c, right) for c in file_controls]

    assert honest_submissions(evidence(math_facts, math, (wrong, right, right))) == ("395",)
    assert honest_submissions(evidence(math_facts, math, (wrong,))) == ("395",)
    assert honest_submissions(evidence(math_facts, (), (await answer("The answer is 395."),))) == ("The answer is 395.",)
    # Only the transcript positive's last reply; the workspace positive and the negatives contribute nothing.
    assert honest_submissions(evidence(file_facts, files)) == ("Done.",)


async def test_an_answer_calls_answer_is_an_honest_submission(answer, math_controls, math_facts):
    right = await answer("395")
    correct = next(c for c in math_controls if c.kind is ControlKind.POSITIVE)
    call = ConversationToolCall(call_id="c1", name=ANSWER_CALL_NAME, arguments={ANSWER_FIELD: "395."})
    called = replace(correct, payload=Transcript((AssistantToolCalls(calls=(call,)),)))

    assert honest_submissions(evidence(math_facts, (control_outcome(called, right),))) == ("395.",)


def test_task_facts_name_inputs_grader_files_and_uniqueness(file_task, math_task, file_facts, math_facts):
    assert task_facts(file_task.task) == file_facts and not file_facts.unique_answer
    assert task_facts(math_task.task) == math_facts and math_facts.unique_answer


FILE_FACTS = TaskFacts(True, False, (NUMBERS,), (NUMBERS, "/tests/grader.py"))
TEXT_FACTS = TaskFacts(False, False, (), ())
UNIQUE_TEXT_FACTS = TaskFacts(False, True, (), ())
TEXT_INPUT_FACTS = TaskFacts(False, False, ("/data/q.txt",), ("/data/q.txt",))
CLAIMED = {"claim": ClaimKind.SHORTCUT, "why": "the grader accepts a hedge"}
NO_CLAIM = {"claim": ClaimKind.NONE}
NO_READ: tuple[str, ...] = ()
REPAIR, NOTED, NONE = DefectTier.REPAIR, DefectTier.NOTED, DefectTier.NONE


@pytest.mark.parametrize(
    ("changes", "facts", "expected"),
    [
        ({"passes": 0, "first_pass": None, "exploit": None}, FILE_FACTS, (NONE, "0", None)),
        # A rejected claim is counted, not believed.
        ({"passes": 0, "first_pass": None, "exploit": None, **CLAIMED}, FILE_FACTS, (NONE, "0", None)),
        # A supplied protected file beats a late claim and a fileless pass.
        ({"protected_supplied": (3,), "fileless_passes": (2,), "exploit": 9, **CLAIMED}, FILE_FACTS, (REPAIR, "1", 3)),
        ({"fileless_passes": (2, 4)}, FILE_FACTS, (REPAIR, "2", 2)),
        # Fileless passes are the norm on a text task.
        ({"fileless_passes": (1,)}, TEXT_FACTS, (NONE, "10", None)),
        # No input read beats the audit guard.
        ({"inputs_consumed": NO_READ, "comparison": Comparison.MATCH, **CLAIMED}, TEXT_INPUT_FACTS, (REPAIR, "3", 1)),
        # The audit guard beats a unique-answer mismatch on an earlier probe.
        (
            {"comparison": Comparison.MATCH, "mismatches": (1,), "exploit": 2, **CLAIMED},
            UNIQUE_TEXT_FACTS,
            (NONE, "4", 2),
        ),
        # A unique-answer mismatch is a repair whatever the claim, before any count rule.
        ({"mismatches": (2,), "comparison": Comparison.MISMATCH, "exploit": 7}, UNIQUE_TEXT_FACTS, (REPAIR, "5", 2)),
        ({"exploit": 2, **CLAIMED}, TEXT_FACTS, (REPAIR, "6", 2)),
        ({"exploit": 3, **CLAIMED}, TEXT_FACTS, (NOTED, "7", 3)),
        ({"exploit": 2, **NO_CLAIM}, TEXT_FACTS, (NOTED, "8", 2)),
        ({"mismatches": (1,), "comparison": Comparison.MISMATCH}, TEXT_FACTS, (NOTED, "9", 1)),
        ({}, FILE_FACTS, (NONE, "10", None)),
    ],
)
def test_tier_table(changes, facts, expected, rounds):
    ruling = tier_of(signals(**changes), facts, rounds.policy(adversary_submissions=10, adversary_repair_submissions=2))

    assert (ruling.tier, ruling.rule, ruling.subject) == expected and ruling.reason
    assert (ruling.kind is None) == (ruling.tier is NONE)


def test_a_zero_threshold_never_repairs_by_count(rounds):
    ruling = tier_of(
        signals(exploit=1, **CLAIMED), TEXT_FACTS, rounds.policy(adversary_submissions=4, adversary_repair_submissions=0)
    )

    assert (ruling.tier, ruling.rule) == (NOTED, "7")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"submissions": 0}, "no verifier submission"),
        ({"submissions": 3}, "all 3 submissions rejected"),
        ({"submissions": 4, "budget_spent": True}, "4 submissions, budget spent, none accepted"),
        (
            {"submissions": 1, **CLAIMED},
            "all 1 submissions rejected; claimed a shortcut without an accepted submission",
        ),
    ],
)
def test_a_trial_without_an_accepted_submission_records_why(changes, reason, rounds):
    rejected = signals(passes=0, first_pass=None, exploit=None, **changes)

    assert tier_of(rejected, FILE_FACTS, rounds.policy()).reason == reason


async def test_a_claimed_shortcut_within_the_threshold_is_a_repair_with_its_candidate_control(
    adversary, trial, lenient_text_task, rounds
):
    hedge = await adversary(lenient_text_task, ("submit", HEDGE), "SHORTCUT: the grader accepts a hedge")
    honest = await trial(lenient_text_task, reply="395")

    summary = summarize(
        evidence(task_facts(lenient_text_task.task), solver=(honest,), adversaries={SHORTCUT: (hedge,)}),
        rounds.policy(k=1, adversary_k=1),
    )

    (finding,) = [f for f in summary.findings if f.kind in DECISIVE]
    assert finding.kind is FindingKind.SHORTCUT_PASSED and finding.roles == (StepRole.GRADER, StepRole.CONTROLS)
    (control,) = finding.new_controls
    assert control.payload == Transcript((reply(HEDGE),))
    assert control.expect.status is GradeStatus.GRADED and control.expect.reward_max == REJECTION_CEILING
    assert control.author == "adversary/shortcut/0#1"
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule, assessment.subject) == (REPAIR, "6", 1)
    assert assessment.signals.why == "the grader accepts a hedge" and "the grader accepts a hedge" in finding.detail


async def test_a_shortcut_above_the_threshold_is_a_note(adversary, trial, lenient_text_task, rounds):
    probes = [("submit", guess) for guess in ("391", "392", "393")]
    late = await adversary(lenient_text_task, *probes, ("submit", HEDGE), "SHORTCUT: the grader accepts a hedge")
    honest, wrong = await trial(lenient_text_task, reply="395"), await trial(lenient_text_task, reply="391")

    summary = summarize(
        evidence(task_facts(lenient_text_task.task), solver=(honest, wrong, wrong), adversaries={SHORTCUT: (late,)}),
        rounds.policy(k=3, adversary_k=1),
    )

    assert summary.findings == () and summary.calibrated
    (note,) = summary.notes
    assert note.kind is FindingKind.SHORTCUT_PASSED and note.new_controls == ()
    assert note.detail.startswith("The shortcut adversary 1 of 1 found an accepted submission. Tier noted (row 7): ")
    assert "\nadversary/shortcut/0: status graded, reward " in note.detail
    (assessment,) = summary.assessments
    assert (assessment.signals.first_pass, assessment.signals.exploit, assessment.signals.submissions) == (4, 4, 4)
    assert assessment.signals.budget_spent


async def test_a_mismatch_on_a_single_answer_grader_is_a_repair_whatever_the_claim(
    adversary, trial, lenient_text_task, rounds
):
    hedge = await adversary(lenient_text_task, ("submit", HEDGE), "NO_SHORTCUT")
    honest = await trial(lenient_text_task, reply="395")

    summary = summarize(
        evidence(UNIQUE_TEXT_FACTS, solver=(honest,), adversaries={SHORTCUT: (hedge,)}),
        rounds.policy(k=1, adversary_k=1),
    )

    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule) == (REPAIR, "5")
    assert [c.payload for f in summary.decisive for c in f.new_controls] == [Transcript((reply(HEDGE),))]


async def test_a_mismatch_on_a_many_answer_grader_reported_as_no_shortcut_is_a_note(
    adversary, trial, lenient_text_task, rounds
):
    hedge = await adversary(lenient_text_task, ("submit", HEDGE), "NO_SHORTCUT")
    honest = await trial(lenient_text_task, reply="395")

    summary = summarize(
        evidence(task_facts(lenient_text_task.task), solver=(honest,), adversaries={SHORTCUT: (hedge,)}),
        rounds.policy(k=1, adversary_k=1),
    )

    assert summary.decisive == ()
    assert [(a.tier, a.rule) for a in summary.assessments] == [(NOTED, "9")]


async def test_a_supplied_protected_file_is_a_repair(adversary, file_task_with, verdict_grader, file_facts, rounds):
    tamper = await adversary(
        file_task_with(grader_script=verdict_grader(RECOMPUTES), output_paths=(NUMBERS, SUM)),
        ("shell", f"printf '0\\n' > {NUMBERS} && echo 0 > {SUM}"),
        ("submit", "Done.", (NUMBERS, SUM)),
        "NO_SHORTCUT",
    )
    assert tamper.submissions[0].passed

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (tamper,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    (control,) = finding.new_controls
    files = tamper.submissions[0].candidate.files
    assert control.payload == Transcript((workspace_turn(0, files[0]), workspace_turn(1, files[1]), reply("Done.")))
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule, assessment.signals.protected_supplied) == (REPAIR, "1", (1,))


async def test_a_fileless_pass_on_a_machine_task_is_a_repair(
    adversary, file_task_with, verdict_grader, file_facts, rounds
):
    untouched = await adversary(file_task_with(grader_script=verdict_grader("True")), ("submit", "Done."), "NO_SHORTCUT")

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (untouched,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert [c.payload for c in finding.new_controls] == [Transcript((reply("Done."),))]
    assert [(a.tier, a.rule) for a in summary.assessments] == [(REPAIR, "2")]


async def test_a_pass_without_reading_the_inputs_is_a_repair(
    adversary, file_task_with, verdict_grader, file_facts, rounds
):
    flood = await adversary(
        file_task_with(grader_script=verdict_grader(MENTIONS_60)),
        ("shell", f"seq 0 100 > {SUM}"),
        ("submit", "Done.", (SUM,)),
        "NO_SHORTCUT",
    )

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (flood,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED and len(finding.new_controls) == 1
    assert [(a.tier, a.rule, a.subject) for a in summary.assessments] == [(REPAIR, "3", 1)]


async def test_an_honest_probe_with_no_shortcut_claim_is_no_defect(adversary, file_task, file_facts, rounds):
    probe = await adversary(file_task, ("shell", SOLVE), ("submit", "Done.", (SUM,)), "NO_SHORTCUT")

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (probe,)}), rounds.policy(adversary_k=1))

    assert summary.findings == () and summary.notes == ()
    assert [(a.tier, a.rule) for a in summary.assessments] == [(NONE, "10")]
    assert summary.roles[SHORTCUT].claims[ClaimKind.NO_SHORTCUT] == 1 and summary.roles[SHORTCUT].passes == 1


async def test_a_claimed_shortcut_that_is_the_honest_answer_is_a_failed_audit(
    adversary, trial, lenient_text_task, rounds
):
    audit = await adversary(lenient_text_task, ("submit", "395"), "SHORTCUT: I answered without checking")
    honest = await trial(lenient_text_task, reply="395")

    summary = summarize(
        evidence(task_facts(lenient_text_task.task), solver=(honest,), adversaries={SHORTCUT: (audit,)}),
        rounds.policy(k=1, adversary_k=1),
    )

    assert summary.decisive == () and summary.notes == ()
    assert [(a.tier, a.rule) for a in summary.assessments] == [(NONE, "4")]
    assert summary.roles[SHORTCUT].failed_audits == 1


async def test_role_stats_count_submissions_claims_budget_and_tiers(adversary, file_task, file_facts, rounds):
    rejected = [("submit", "Done.")] * 4
    spent = await adversary(file_task, *rejected, "NO_SHORTCUT")
    probe = await adversary(file_task, ("shell", SOLVE), ("submit", "Done.", (SUM,)), "Solved it.")
    trials = (spent, probe, failed(UNAVAILABLE))

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: trials}), rounds.policy(adversary_k=3))

    tokens = sum(
        step.turn.metadata["usage"]["completion_tokens"]
        for t in trials
        if t.outcome.rollout
        for step in t.outcome.rollout.steps
    )
    assert summary.roles == {
        SHORTCUT: RoleStats(
            required=3,
            graded=2,
            passes=1,
            submissions=5,
            budget_spent=1,
            claims={ClaimKind.SHORTCUT: 0, ClaimKind.NO_SHORTCUT: 1, ClaimKind.NONE: 1},
            failed_audits=0,
            exhausted=0,
            output_tokens=tokens,
            tiers={REPAIR: 0, NOTED: 1, NONE: 1},
        )
    }
    assert summary.assessments[0].reason == "4 submissions, budget spent, none accepted"


async def test_a_summary_reads_back_from_calibration_json(
    tmp_path, adversary, trial, file_task, file_task_with, verdict_grader, file_controls, file_facts, rounds
):
    correct = next(c for c in file_controls if c.kind is ControlKind.POSITIVE)
    flood = await adversary(
        file_task_with(grader_script=verdict_grader(MENTIONS_60)),
        ("shell", f"seq 0 100 > {SUM}"),
        ("submit", "Done.", (SUM,)),
        "x",
    )
    probe = await adversary(file_task, ("shell", SOLVE), ("submit", "Done.", (SUM,)), "Solved.")
    summary = summarize(
        evidence(
            file_facts,
            (control_outcome(correct, await trial(file_task, f"echo 59 > {SUM}")),),
            (await trial(file_task, SOLVE), UNAVAILABLE),
            {SHORTCUT: (flood, probe, failed(UNAVAILABLE))},
        ),
        rounds.policy(k=2, adversary_k=3),
    )
    assert summary.notes and summary.assessments and summary.decisive
    path = tmp_path / "calibration.json"

    write_summary(path, summary)

    assert load_summary(path) == summary
