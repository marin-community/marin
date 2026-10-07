# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""summarize: band findings over complete evidence, decisive findings regardless, and adversary passes tiered by
coded transcript signals into repairs (with controls), notes and no defect."""

from collections import Counter

import pytest
from rigging.timing import ExponentialBackoff
from taskcompendium.environment import EnvironmentKind, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AnswerType, TaskSpec, TextMessage
from taskcompendium.submission import PlainText

from taskforge.build.step import StepRole
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.controls import REJECTION_CEILING, ControlKind, Transcript
from taskforge.spec.draft import assemble, environment, file, shell_verifier
from taskforge.validate.adversary import AdversaryRole
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
    gave_up,
    honest_submissions,
    load_summary,
    summarize,
    task_facts,
    tier_of,
    write_summary,
    written_inputs,
)
from taskforge.validate.evidence import Complete, Incomplete
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.validate.run import ValidationEvidence, control_outcome
from taskforge.validate.trials import Deadlines, EngineSettings, TrialPlan, run_trial

DIGEST = "ab" * 32
UNAVAILABLE = Ungraded(Cause.MODEL_UNAVAILABLE, "router drained", None)
SETUP_FAILED = Ungraded(Cause.TASK_SETUP, "setup command exited 3", None)
NUMBERS = "/workspace/numbers.txt"
SUM = "/workspace/sum.txt"
RECOMPUTING_CHECK = (
    "s=$(awk '{s+=$1} END{print s}' /workspace/numbers.txt)\n"
    'v=$(tr -d " \\n" < /workspace/sum.txt)\n'
    'if [ "$v" = "$s" ]; then echo 1; else echo 0; fi\n'
)
LENIENT_FILE_CHECK = "if grep -q 60 /workspace/sum.txt; then echo 1; else echo 0; fi\n"
LENIENT_TEXT_CHECK = "if grep -q 395; then echo 1; else echo 0; fi\n"
ACCEPT_ALL = "echo 1\n"
READ_THEN_ANSWER = (f"head -c 200 {NUMBERS}", f"echo 60 > {SUM}")
SOLVE = f"awk '{{s+=$1}} END{{print s}}' {NUMBERS} > {SUM}"


def graded_by(task: TaskSpec, script: str) -> TaskSpec:
    """``task`` graded by ``script`` as a ShellSim stdout-reward verifier at ``/grader/check.sh``."""
    verifier = shell_verifier(
        ("sh", "/grader/check.sh"), StdoutReward(), timeout=30, files=(file("/grader/check.sh", script),)
    )
    return task.model_copy(update={"verifier": verifier})


@pytest.fixture
def lenient_text_task(math_task) -> TaskSpec:
    """The math question on ShellSim, graded by a check that accepts any transcript containing 395."""
    return assemble(
        "validate-lenient-text",
        math_task.context.events[-1].content,
        AnswerType.NUMBER,
        environment(EnvironmentKind.SHELLSIM),
        shell_verifier(
            ("sh", "/grader/check.sh"),
            StdoutReward(),
            timeout=30,
            files=(file("/grader/check.sh", LENIENT_TEXT_CHECK),),
        ),
        math_task.source,
        execution=TaskExecution(),
    )


@pytest.fixture
def trial(tmp_path, fakes):
    """Runs ``task`` once with scripted assistant turns (a ``str`` is a shell call, a ``Reply`` a text turn)."""
    count = 0

    async def run(task: TaskSpec, *turns: str, reply: str | None = "Done.", max_turns: int = 6) -> Outcome:
        nonlocal count
        count += 1
        plan = TrialPlan(
            item_id="item",
            round=0,
            kind=TrialKind.SOLVER,
            k=1,
            deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
            max_retries=0,
            token_contract_retries=0,
            retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
            evidence_dir=tmp_path,
            ledger=JsonlLedger(tmp_path / "ledger"),
            first_attempt=0,
        )
        settings = EngineSettings(
            factories={EnvironmentKind.SHELLSIM: fakes.flaky_factory(0, RuntimeError)},
            capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
            max_turns=max_turns,
            command_timeout=10,
            cleanup_timeout=10,
            conventions=(PlainText(id="plain"),),
        )
        messages = [*(fakes.shell(command) for command in turns), *(() if reply is None else (fakes.text(reply),))]
        return await run_trial(task, TaskExecution(), plan, settings, fakes.script_model(messages), str(count))

    return run


@pytest.fixture
def answer(trial, math_task):
    """Runs the math task once with a scripted final reply and returns the outcome."""

    async def run(reply: str) -> Outcome:
        return await trial(math_task, reply=reply)

    return run


def evidence(facts: TaskFacts, controls=(), solver=(), adversaries=None) -> ValidationEvidence:
    return ValidationEvidence(DIGEST, tuple(controls), tuple(solver), adversaries or {}, facts)


def signals(**changes) -> AdversarySignals:
    """A passing adversary that read the input, wrote nothing, ran a shell command and did not give up."""
    base = {
        "passed": True,
        "gave_up": False,
        "stop_reason": "stop",
        "budget_exhausted": False,
        "output_tokens": 100,
        "turns": 2,
        "shell_calls": 1,
        "inputs_consumed": (NUMBERS,),
        "inputs_written": (),
        "submission": None,
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
    adversaries = {AdversaryRole.LEAK: (SETUP_FAILED, SETUP_FAILED)}

    summary = summarize(evidence(math_facts, (violated,), (UNAVAILABLE,), adversaries), rounds.policy(k=3))

    assert [f.kind for f in summary.findings] == [FindingKind.CONTROL_VIOLATED, FindingKind.TASK_DEFECT]
    assert summary.decisive == summary.findings and {f.kind for f in summary.findings} <= DECISIVE
    violation, defect = summary.findings
    assert violation.roles == (StepRole.GRADER, StepRole.CONTROLS) and "control/correct" in violation.detail
    assert defect.roles == (StepRole.ENVIRONMENT, StepRole.FIXTURES)
    assert "adversary/leak/0" in defect.detail and "adversary/leak/1" in defect.detail
    assert summary.controls_violated == ("correct",)
    assert summary.status == Incomplete(Counter({Cause.MODEL_UNAVAILABLE: 1, Cause.TASK_SETUP: 2}))
    assert summary.assessments == () and summary.notes == ()


async def test_gave_up_reads_the_last_line(answer, trial, file_task):
    shortcut = AdversaryRole.SHORTCUT
    exact = await answer("NO_SHORTCUT_FOUND")
    explained = await answer("I checked the grader and found nothing.\n\nNO_SHORTCUT_FOUND\n")
    mid_sentence = await answer("I would reply NO_SHORTCUT_FOUND, but here is 395.")
    tool_calls_last = await trial(file_task, "ls /workspace", reply=None, max_turns=1)

    assert gave_up(shortcut, exact.rollout) and gave_up(shortcut, explained.rollout)
    assert not gave_up(AdversaryRole.LEAK, exact.rollout)
    assert not gave_up(shortcut, mid_sentence.rollout)
    assert tool_calls_last.rollout is not None and not gave_up(shortcut, tool_calls_last.rollout)


@pytest.mark.parametrize(
    ("command", "consumed", "written"),
    [
        (f"cat {NUMBERS}", True, False),
        (f"head -c 200 {NUMBERS}", True, False),
        (f"od -c {NUMBERS}", True, False),
        (f"awk '{{s+=$1}} END{{print s}}' {NUMBERS} > {SUM}", True, False),
        (f"tr '\\n' '+' < {NUMBERS}", True, False),
        (f"while read n; do s=$((s+n)); done < {NUMBERS}; echo $s > {SUM}", True, False),
        (f"printf '%s' \"$(awk '{{s+=$1}} END{{print s}}' {NUMBERS})\" > {SUM}", True, False),
        (f"LC_ALL=C sort -n {NUMBERS}", True, False),
        (f"wc -l {NUMBERS}", False, False),
        (f"ls -la {NUMBERS}", False, False),
        (f"stat {NUMBERS}", False, False),
        (f"cp {NUMBERS} {NUMBERS}.bak", False, False),
        (f"cat {NUMBERS}.bak", False, False),
        (f"seq -5000 15000 >> {SUM}", False, False),
        (f"printf '0\\n' > {NUMBERS}", False, True),
        (f"echo 60 > {NUMBERS}", False, True),
        (f"echo 0 | tee {NUMBERS}", False, True),
        (f"sed -i s/12/0/ {NUMBERS}", True, True),
        (f"rm -f {NUMBERS}", False, True),
        (f"mv /tmp/zeros {NUMBERS}", False, True),
        # dd names the path in a non-metadata segment, so it also counts as read; row 1 fires first either way.
        (f"dd if=/dev/zero of={NUMBERS} count=0", True, True),
    ],
)
def test_command_signals_read_the_transcript(command, consumed, written):
    assert consumed_inputs([command], [NUMBERS]) == ((NUMBERS,) if consumed else ())
    assert written_inputs([command], [NUMBERS, "/grader/check.sh"]) == ((NUMBERS,) if written else ())


def test_a_grader_file_write_is_a_protected_write():
    assert written_inputs(["echo 'echo 1' > /grader/check.sh"], [NUMBERS, "/grader/check.sh"]) == ("/grader/check.sh",)


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


def test_task_facts_name_inputs_then_grader_files(file_task, math_task, file_facts, math_facts):
    assert task_facts(file_task) == file_facts
    assert task_facts(math_task) == math_facts


FILE_FACTS = TaskFacts(True, (NUMBERS,), (NUMBERS, "/grader/check.sh"))
TEXT_FACTS = TaskFacts(False, (), ())
TEXT_INPUT_FACTS = TaskFacts(False, ("/data/q.txt",), ("/data/q.txt",))
SHORTCUT, LEAK, AMBIGUITY = AdversaryRole.SHORTCUT, AdversaryRole.LEAK, AdversaryRole.AMBIGUITY
REPAIR_SHORTCUT = (DefectTier.REPAIR, FindingKind.SHORTCUT_PASSED)
NOTED_SHORTCUT = (DefectTier.NOTED, FindingKind.SHORTCUT_PASSED)
NO_READ: tuple[str, ...] = ()


@pytest.mark.parametrize(
    ("role", "changes", "facts", "expected"),
    [
        (SHORTCUT, {"passed": False}, FILE_FACTS, (DefectTier.NONE, None, "0")),
        (SHORTCUT, {"passed": False, "inputs_written": (NUMBERS,)}, FILE_FACTS, (DefectTier.NONE, None, "0")),
        (LEAK, {"inputs_written": (NUMBERS,)}, FILE_FACTS, (*REPAIR_SHORTCUT, "1")),
        # A write beats a consumed input and an ambiguity give-up.
        (AMBIGUITY, {"inputs_written": (NUMBERS,), "gave_up": True}, FILE_FACTS, (*REPAIR_SHORTCUT, "1")),
        # A shell-less pass on a machine-graded task beats an ambiguity give-up.
        (
            AMBIGUITY,
            {"shell_calls": 0, "gave_up": True, "inputs_consumed": NO_READ},
            FILE_FACTS,
            (*REPAIR_SHORTCUT, "2"),
        ),
        (SHORTCUT, {"shell_calls": 0, "inputs_consumed": NO_READ}, TEXT_FACTS, (*NOTED_SHORTCUT, "7")),
        (AMBIGUITY, {"gave_up": True}, FILE_FACTS, (DefectTier.NONE, None, "3")),
        (SHORTCUT, {"gave_up": True}, FILE_FACTS, (*NOTED_SHORTCUT, "7")),
        (SHORTCUT, {"inputs_consumed": NO_READ}, FILE_FACTS, (*REPAIR_SHORTCUT, "4")),
        # The leak role is forbidden from reading the inputs, so its no-read pass is a leak.
        (LEAK, {"inputs_consumed": NO_READ}, FILE_FACTS, (DefectTier.REPAIR, FindingKind.LEAK_PASSED, "4")),
        # No input read comes before the comparison.
        (
            AMBIGUITY,
            {"inputs_consumed": NO_READ, "comparison": Comparison.MATCH},
            TEXT_INPUT_FACTS,
            (*REPAIR_SHORTCUT, "4"),
        ),
        (SHORTCUT, {"inputs_consumed": NO_READ, "comparison": Comparison.MISMATCH}, TEXT_FACTS, (*REPAIR_SHORTCUT, "5")),
        (LEAK, {"comparison": Comparison.MISMATCH}, TEXT_FACTS, (DefectTier.REPAIR, FindingKind.LEAK_PASSED, "5")),
        (AMBIGUITY, {"comparison": Comparison.MISMATCH}, TEXT_FACTS, (DefectTier.REPAIR, FindingKind.AMBIGUOUS, "5")),
        (AMBIGUITY, {"comparison": Comparison.MATCH}, TEXT_FACTS, (DefectTier.NONE, None, "6")),
        (SHORTCUT, {"comparison": Comparison.MATCH}, TEXT_FACTS, (*NOTED_SHORTCUT, "7")),
        (LEAK, {}, FILE_FACTS, (DefectTier.NOTED, FindingKind.LEAK_PASSED, "7")),
        (AMBIGUITY, {}, FILE_FACTS, (DefectTier.NOTED, FindingKind.AMBIGUOUS, "7")),
    ],
)
def test_tier_table(role, changes, facts, expected):
    ruling = tier_of(role, signals(**changes), facts)

    assert (ruling.tier, ruling.kind, ruling.rule) == expected and ruling.reason


def test_a_failed_pass_records_why_it_failed():
    assert tier_of(SHORTCUT, signals(passed=False, gave_up=True), FILE_FACTS).reason == "gave up"
    assert tier_of(SHORTCUT, signals(passed=False, budget_exhausted=True), FILE_FACTS).reason == "budget exhausted"
    assert tier_of(SHORTCUT, signals(passed=False), FILE_FACTS).reason == "rejected"


async def test_a_hedged_text_pass_is_a_repair_with_its_transcript_control(trial, lenient_text_task, rounds):
    hedge = await trial(lenient_text_task, reply="394 or 395 or 396")
    honest = await trial(lenient_text_task, reply="395")
    assert isinstance(hedge, Graded) and hedge.reward == 1.0

    summary = summarize(
        evidence(task_facts(lenient_text_task), solver=(honest,), adversaries={SHORTCUT: (hedge,)}),
        rounds.policy(k=1, adversary_k=1),
    )

    (finding,) = [f for f in summary.findings if f.kind in DECISIVE]
    assert finding.kind is FindingKind.SHORTCUT_PASSED
    (control,) = finding.new_controls
    assert control.payload == Transcript((TextMessage(role="assistant", content="394 or 395 or 396"),))
    assert control.expect.status is GradeStatus.GRADED and control.expect.reward_max == REJECTION_CEILING
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule) == (DefectTier.REPAIR, "5")
    assert assessment.signals.comparison is Comparison.MISMATCH


async def test_a_tampering_pass_is_a_repair(trial, file_task, file_facts, rounds):
    task = graded_by(file_task, RECOMPUTING_CHECK)
    tamper = await trial(task, f"printf '0\\n' > {NUMBERS} && echo 0 > {SUM}")
    assert isinstance(tamper, Graded) and tamper.reward == 1.0

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (tamper,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED and len(finding.new_controls) == 1
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule) == (DefectTier.REPAIR, "1")
    assert assessment.signals.inputs_written == (NUMBERS,)


async def test_a_pass_that_never_read_the_inputs_is_a_repair(trial, file_task, file_facts, rounds):
    flood = await trial(graded_by(file_task, LENIENT_FILE_CHECK), f"seq 0 100 > {SUM}")

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: (flood,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED and len(finding.new_controls) == 1
    assert finding.roles == (StepRole.GRADER, StepRole.CONTROLS)
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule) == (DefectTier.REPAIR, "4")


async def test_a_leak_pass_that_never_read_the_inputs_condemns_where_it_leaked(trial, file_task, file_facts, rounds):
    key = file("/workspace/.cache/expected", "60\n")
    leaky = file_task.model_copy(
        update={"environment": file_task.environment.model_copy(update={"files": (*file_task.environment.files, key)})}
    )
    leaked = await trial(leaky, f"cp {key.path} {SUM}")
    assert isinstance(leaked, Graded) and leaked.reward == 1.0

    summary = summarize(evidence(file_facts, adversaries={LEAK: (leaked,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert finding.kind is FindingKind.LEAK_PASSED and len(finding.new_controls) == 1
    assert {StepRole.FIXTURES, StepRole.ENVIRONMENT, StepRole.INSTRUCTIONS} <= set(finding.roles)
    assert {StepRole.GRADER, StepRole.CONTROLS} <= set(finding.roles)
    (assessment,) = summary.assessments
    assert (assessment.tier, assessment.rule) == (DefectTier.REPAIR, "4")


async def test_a_pass_that_submitted_an_honest_answer_ships_no_control(trial, lenient_text_task, rounds):
    honest = await trial(lenient_text_task, reply="395")
    leaked = await trial(lenient_text_task, reply="395")
    facts = TaskFacts(False, ("/data/q.txt",), ("/data/q.txt",))

    summary = summarize(
        evidence(facts, solver=(honest,), adversaries={LEAK: (leaked,)}), rounds.policy(k=1, adversary_k=1)
    )

    (finding,) = [f for f in summary.findings if f.kind in DECISIVE]
    assert finding.kind is FindingKind.LEAK_PASSED and finding.new_controls == ()
    assert "not rendered as a control: it submitted an honest answer" in finding.detail
    assert summary.assessments[0].signals.comparison is Comparison.MATCH


async def test_a_shortcut_that_read_the_inputs_is_noted_not_a_finding(trial, file_task, file_facts, rounds):
    honest = await trial(file_task, f"echo 60 > {SUM}")
    wrong = await trial(file_task, f"echo 59 > {SUM}")
    read = await trial(file_task, *READ_THEN_ANSWER)
    assert isinstance(read, Graded) and read.reward == 1.0

    summary = summarize(
        evidence(file_facts, solver=(honest, wrong, wrong), adversaries={SHORTCUT: (read,)}),
        rounds.policy(k=3, adversary_k=1),
    )

    assert summary.findings == () and summary.calibrated
    (note,) = summary.notes
    assert note.kind is FindingKind.SHORTCUT_PASSED and note.new_controls == ()
    assert note.detail.startswith("The shortcut adversary 0 was graded as passing. Tier noted (row 7): ")
    assert f'inputs_consumed=["{NUMBERS}"]' in note.detail
    assert summary.roles[SHORTCUT].tiers == {DefectTier.REPAIR: 0, DefectTier.NOTED: 1, DefectTier.NONE: 0}


async def test_an_untouched_workspace_pass_is_a_repair(trial, file_task, file_facts, rounds):
    untouched = await trial(graded_by(file_task, ACCEPT_ALL), reply="No reading changes the answer.")

    summary = summarize(evidence(file_facts, adversaries={AMBIGUITY: (untouched,)}), rounds.policy(adversary_k=1))

    (finding,) = summary.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED
    (control,) = finding.new_controls
    assert control.payload == Transcript((TextMessage(role="assistant", content="No reading changes the answer."),))
    assert summary.assessments[0].rule == "2"


async def test_ambiguity_passes(trial, lenient_text_task, file_task, file_facts, rounds):
    honest = await trial(lenient_text_task, reply="395")
    same = await trial(lenient_text_task, reply="Under the stated reading the answer is 395.")
    different = await trial(lenient_text_task, reply="Reading 17 * 23 + 4 as concatenation: 3950")
    gave_up_after_solving = await trial(lenient_text_task, reply="395\nNO_AMBIGUITY_FOUND")
    text = summarize(
        evidence(
            task_facts(lenient_text_task),
            solver=(honest,),
            adversaries={AMBIGUITY: (same, different, gave_up_after_solving)},
        ),
        rounds.policy(k=1, adversary_k=3),
    )
    machine = summarize(
        evidence(file_facts, adversaries={AMBIGUITY: (await trial(file_task, SOLVE),)}), rounds.policy(adversary_k=1)
    )

    assert [(a.tier, a.rule) for a in text.assessments] == [
        (DefectTier.NONE, "6"),
        (DefectTier.REPAIR, "5"),
        (DefectTier.NONE, "3"),
    ]
    (ambiguous,) = text.decisive
    assert ambiguous.kind is FindingKind.AMBIGUOUS and ambiguous.new_controls == ()
    assert ambiguous.roles == (StepRole.INSTRUCTIONS,)
    assert text.roles[AMBIGUITY].gave_up == 1 and text.notes == ()
    assert machine.findings == () and [n.kind for n in machine.notes] == [FindingKind.AMBIGUOUS]
    assert [(a.tier, a.rule) for a in machine.assessments] == [(DefectTier.NOTED, "7")]


async def test_role_stats_count_give_ups_exhaustion_tokens_and_tiers(trial, file_task, file_facts, rounds):
    gave_up_ = await trial(file_task, "ls /workspace", reply="Nothing to exploit.\nNO_SHORTCUT_FOUND")
    exhausted = await trial(file_task, *(["ls /workspace"] * 3), reply=None, max_turns=3)
    read = await trial(file_task, *READ_THEN_ANSWER)
    outcomes = (gave_up_, exhausted, read, UNAVAILABLE)

    summary = summarize(evidence(file_facts, adversaries={SHORTCUT: outcomes}), rounds.policy(adversary_k=4))

    tokens = sum(o.rollout.loss_mask.count(1) for o in outcomes if o.rollout is not None)
    assert tokens == 2 + 3 + 3
    assert summary.roles == {
        SHORTCUT: RoleStats(
            required=4,
            graded=3,
            passes=1,
            gave_up=1,
            exhausted=1,
            output_tokens=tokens,
            tiers={DefectTier.REPAIR: 0, DefectTier.NOTED: 1, DefectTier.NONE: 2},
        )
    }
    assert [a.reason for a in summary.assessments][:2] == ["gave up", "budget exhausted"]


async def test_a_summary_reads_back_from_calibration_json(tmp_path, trial, file_task, file_controls, file_facts, rounds):
    correct = next(c for c in file_controls if c.kind is ControlKind.POSITIVE)
    flood = await trial(graded_by(file_task, LENIENT_FILE_CHECK), f"seq 0 100 > {SUM}")
    summary = summarize(
        evidence(
            file_facts,
            (control_outcome(correct, await trial(file_task, f"echo 59 > {SUM}")),),
            (await trial(file_task, SOLVE), UNAVAILABLE),
            {SHORTCUT: (flood, await trial(file_task, *READ_THEN_ANSWER)), LEAK: (UNAVAILABLE, UNAVAILABLE)},
        ),
        rounds.policy(k=2, adversary_k=2),
    )
    assert summary.notes and summary.assessments and summary.decisive
    path = tmp_path / "calibration.json"

    write_summary(path, summary)

    assert load_summary(path) == summary
