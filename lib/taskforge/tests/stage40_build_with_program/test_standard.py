# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard template built end to end on ShellSim, with the fake GLM writing every draft."""

import inspect
import json
from collections.abc import Mapping
from dataclasses import replace

import pytest
from pydantic import TypeAdapter
from rolloutengine.models import ModelRequest, ModelTurn
from shellbox.machine import Backend, PackageSpec
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ArtifactKind, ScriptGrader, VerifyitGrader
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import ExactSpec

from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec import decision, draft
from taskforge.spec.decision import DecisionGrader, DecisionUnavailable
from taskforge.spec.draft import VERDICT_REWARD, answer_grader, rollout_engine
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal, parse
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
)
from taskforge.stage40_build_with_program.outputs import Fixtures, Grader, Sources
from taskforge.stage40_build_with_program.run import ResearchRequirement, run_build, step_output
from taskforge.stage40_build_with_program.sdk import (
    ATTEMPTS,
    WORKDIR,
    FileDraft,
    FixturesDraft,
    GraderDraft,
    grader_package,
    machine_facts,
    packages_declared,
    solver_machine,
    task_file,
)
from taskforge.stage40_build_with_program.step_store import CacheStatus
from taskforge.stage40_build_with_program.template import standard
from taskforge.stage50_validate.lowering import lower
from tests.sandbox.fixture_images import SlowGraderFactory
from tests.spec.fake_decision import FakeDecisionClient
from tests.stage40_build_with_program.conftest import (
    FACTORIES,
    IMAGE,
    PROPOSAL,
    SAMPLING,
    WORKPIECE,
    RecordingImageBuilder,
    declaring,
    web_tool,
)

# Grader programs in sh, which ShellGraderImageFactory runs for the template's python3 grader command.
GOOD_GRADER = """last=$(printf '%s\\n' "$(cat /app/answer.txt)" | tail -n 1)
expected=$(cat /tests/key.txt)
passed=false
case "$last" in *"ANSWER = $expected") passed=true ;; esac
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "answer line", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""
WRONG_GRADER = GOOD_GRADER.replace('"ANSWER = ', '"RESULT = ')
PERCENT_GRADER = GOOD_GRADER.replace("then 1.0 else 0.0", "then 100.0 else 0.0")
LENIENT_GRADER = GOOD_GRADER.replace("then 1.0 else 0.0", "then 1.0 else 0.3")
FAILED_CHECK_GRADER = """jq -n -c '{reward: 0.0, detail: {
    checks: [{name: "reference query", passed: false, weight: 1.0}], reason: "not authorized"
}}' > /logs/verifier/reward.json
"""
NO_CHECKS_GRADER = """printf '%s' '{"reward": 1.0}' > /logs/verifier/reward.json
"""
REFERENCE_REPLY = "6 * 7 = 42\nANSWER = 42"
SHELLSIM = draft.requirements(image=None, workdir=WORKDIR)


@pytest.fixture
def proposal() -> TaskProposal:
    """The shared proposal, declaring the code verification the grader programs below implement."""
    return parse(PROPOSAL.replace("verification: answer_in_box", "verification: code"))


def file(path: str, content: str) -> dict:
    return {"path": path, "content": content, "executable": False}


def grader_call(script: str) -> dict:
    return {
        "kind": "script",
        "script": script,
        "rubric": None,
        "equivalent_variants": [],
        "private_files": [],
        "expected": "",
        "tolerance": 0.0,
        "answer_contract": "End the reply with a line `ANSWER = <n>`.",
        "reference_reply": REFERENCE_REPLY,
        "reference_files": [],
        "secret_values": ["42"],
        "rejection_ceiling": 0.0,
    }


def serve_until_the_grader(fake_glm) -> None:
    """The research run and the fixtures submission, the model calls before the grader step."""
    fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="Confirmed: 6 x 7 = 42 (https://example.org/table).")
    fake_glm.stream(
        content=json.dumps(
            {
                "agent_files": [file("workspace/question.txt", "What is 6 * 7?")],
                "private_files": [file("key.txt", "42")],
                "facts": "6 * 7 = 42",
                "packages": [],
            }
        )
    )


async def build(program, proposal, services, glm_client, tmp_path, queries):
    return await run_build(
        program,
        proposal,
        WORKPIECE,
        tmp_path / "round",
        str(tmp_path / "store"),
        glm_client,
        SAMPLING,
        replace(services, web_tools=(web_tool(queries),)),
        0,
        research=ResearchRequirement.ENFORCED,
    )


async def answer(request: ModelRequest) -> ModelTurn:
    return ModelTurn({"role": "assistant", "content": "ANSWER = 42"}, (1, 2), (3,), None, "stop")


async def test_the_template_builds_a_task_validation_can_run_and_rebuilds_from_the_store(
    proposal, services, glm_client, fake_glm, tmp_path
):
    serve_until_the_grader(fake_glm)
    # The first grader gives the reference answer no credit; the step feeds that back and asks again.
    fake_glm.stream(content=json.dumps(grader_call(WRONG_GRADER)))
    fake_glm.stream(content=json.dumps(grader_call(GOOD_GRADER)))
    # The first instruction leaks the graded answer.
    fake_glm.stream(content=json.dumps({"system": "", "instruction": "Show that 6 * 7 = 42. End with ANSWER = <n>."}))
    fake_glm.stream(content=json.dumps({"system": "", "instruction": "Read question.txt. End with `ANSWER = <n>`."}))
    queries: list[Mapping[str, object]] = []
    program = load_program_source(inspect.getsource(standard))

    draft = await build(program, proposal, services, glm_client, tmp_path, queries)
    again = await build(program, proposal, services, glm_client, tmp_path, queries)

    assert queries == [{"query": "6 times 7"}]
    assert not fake_glm.responses
    assert [outcome.status for outcome in draft.provenance.steps] == [CacheStatus.MISS] * 6
    assert {outcome.status for outcome in again.provenance.steps} == {CacheStatus.HIT}
    assert again.task == draft.task
    task = draft.task
    assert [f.path for f in task.resources.worker] == ["workspace/question.txt"]
    assert isinstance(task.grader, ScriptGrader) and task.grader.environment.docker_image == GRADER_BASE_IMAGE
    assert {f.path: resource_bytes(f).decode() for f in task.resources.verifier} == {
        "grade.py": GOOD_GRADER,
        "config.json": "{}",
        "key.txt": "42",
    }
    assert "42" not in json.dumps(task.context.model_dump(mode="json"))
    fixtures_request = fake_glm.requests[2]["messages"][-2]["content"]
    assert "### research[0]: confirm the multiplication table\nConfirmed: 6 x 7 = 42" in fixtures_request

    grader = step_output(draft, str(tmp_path / "store"), "grader", Grader)
    assert grader.reference_reply == REFERENCE_REPLY
    assert [(c.reply, c.reward) for c in grader.prototyped] == [
        (REFERENCE_REPLY, 0.0),
        (REFERENCE_REPLY, 1.0),
        ("", 0.0),
    ]

    lowered = lower(task, MachineHost.LAPTOP, FACTORIES)
    assert lowered.task_machine is not None and lowered.task_machine.backend == Backend.SHELLSIM
    assert lowered.verifier_machine is not None and lowered.verifier_machine.backend == Backend.DOCKER
    result = await rollout_engine(FACTORIES, command_timeout=None).run_task(lowered, answer)
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)


async def test_a_grader_that_rejects_its_own_reference_answer_fails_the_build(
    proposal, services, glm_client, fake_glm, tmp_path
):
    serve_until_the_grader(fake_glm)
    for _ in range(ATTEMPTS):
        fake_glm.stream(content=json.dumps(grader_call(WRONG_GRADER)))

    with pytest.raises(BuildFailure, match=r"the reference answer was graded graded reward=0\.0") as failure:
        await build(standard.PROGRAM, proposal, services, glm_client, tmp_path, [])

    assert failure.value.step == "grader"
    assert not (tmp_path / "round" / "draft").exists()


FIXTURES = Fixtures(
    agent_files=(task_file(FileDraft.model_validate(file("workspace/question.txt", "What is 6 * 7?"))),),
    private_files=(task_file(FileDraft.model_validate(file("key.txt", "42"))),),
    facts="6 * 7 = 42",
    packages=(),
)


async def test_the_grader_step_sends_back_a_numeric_answer_that_is_not_a_literal(context, proposal, fake_glm):
    for expected in ("sqrt(1764)", "42"):
        submission = {**grader_call(""), "kind": "numeric", "expected": expected, "reference_reply": "42"}
        fake_glm.stream(content=json.dumps(submission))

    grader = await standard.grader.fn(context(), proposal, replace(FIXTURES, agent_files=()), SHELLSIM)

    assert isinstance(grader.package.grader, VerifyitGrader) and grader.package.grader.mode == "numeric"
    assert "numeric value requires one finite scalar literal" in fake_glm.requests[1]["messages"][-2]["content"]


async def test_the_grader_step_sends_back_a_reward_outside_the_score_range(context, proposal, fake_glm):
    """A grader scoring 0 to 100 would read every reward as full credit; the step sends it back."""
    fake_glm.stream(content=json.dumps(grader_call(PERCENT_GRADER)))
    fake_glm.stream(content=json.dumps(grader_call(GOOD_GRADER)))

    grader = await standard.grader.fn(context(), proposal, FIXTURES, SHELLSIM)

    assert grader.rejection_ceiling == 0.0
    feedback = fake_glm.requests[1]["messages"][-2]["content"]
    assert "the reference answer got reward 100.0, outside the grader's score range [0.0, 1.0]" in feedback


async def test_the_grader_step_keeps_the_declared_rejection_ceiling(context, proposal, fake_glm):
    """The ceiling the grader declares reaches validation through the ``Grader`` output."""
    fake_glm.stream(content=json.dumps({**grader_call(LENIENT_GRADER), "rejection_ceiling": 0.3}))

    grader = await standard.grader.fn(context(), proposal, FIXTURES, SHELLSIM)

    assert grader.rejection_ceiling == 0.3
    assert "declare rejection_ceiling" in fake_glm.requests[0]["messages"][-2]["content"]


async def test_a_grader_that_times_out_twice_expires_the_build_instead_of_going_back_to_the_model(
    context, proposal, fake_glm
):
    slow = SlowGraderFactory(timeouts=2)
    fake_glm.stream(content=json.dumps(grader_call(GOOD_GRADER)))

    with pytest.raises(BuildExpired, match="the grader command timed out on the reference answer twice") as expired:
        await standard.grader.fn(
            context(factories={**FACTORIES, Backend.DOCKER.value: slow}), proposal, FIXTURES, SHELLSIM
        )

    assert expired.value.clock is ExpiredClock.GRADER
    assert "the workpiece is held for inspection" in expired.value.message
    assert len(fake_glm.requests) == 1
    assert slow.grader_runs == 2


def test_a_rejection_ceiling_at_full_credit_is_refused():
    with pytest.raises(ValueError, match=r"rejection_ceiling is a reward from 0 up to, not including, 0\.99"):
        GraderDraft.model_validate({**grader_call(GOOD_GRADER), "rejection_ceiling": 0.99})


async def test_the_grader_step_sends_back_a_grader_without_per_check_results(context, proposal, fake_glm):
    fake_glm.stream(content=json.dumps(grader_call(NO_CHECKS_GRADER)))
    fake_glm.stream(content=json.dumps(grader_call(GOOD_GRADER)))

    grader = await standard.grader.fn(context(), proposal, FIXTURES, SHELLSIM)

    assert resource_bytes(next(f for f in grader.package.resources if f.path == "grade.py")).decode() == GOOD_GRADER
    assert "no per-check results" in fake_glm.requests[1]["messages"][-2]["content"]


async def test_the_grader_step_sends_back_the_checks_its_reference_answer_failed(context, proposal, fake_glm):
    fake_glm.stream(content=json.dumps(grader_call(FAILED_CHECK_GRADER)))
    fake_glm.stream(content=json.dumps(grader_call(GOOD_GRADER)))

    await standard.grader.fn(context(), proposal, FIXTURES, SHELLSIM)

    feedback = fake_glm.requests[1]["messages"][-2]["content"]
    assert "the reference answer was graded graded reward=0.0" in feedback
    assert '"checks": [{"name": "reference query", "passed": false, "weight": 1.0}]' in feedback
    assert '"reason": "not authorized"' in feedback


async def test_a_container_task_publishes_its_image_and_grades_in_a_copy_of_it(context, proposal, fake_glm):
    container = declaring(proposal, None)
    dockerfile = "FROM python:3.12-slim\nWORKDIR /workspace\n"
    fake_glm.stream(content=json.dumps({"dockerfile": dockerfile, "context_files": []}))
    images = RecordingImageBuilder()
    fixtures = replace(FIXTURES, agent_files=())

    requirements = await standard.requirements.fn(context(images=images), container, fixtures)
    package = grader_package(GraderDraft.model_validate(grader_call("print(1.0)")), fixtures, requirements)
    assert isinstance(package, GraderPackage)

    assert requirements.docker_image == IMAGE
    assert [resource_bytes(f).decode() for f in images.published[0][0].files] == [dockerfile]
    shell = package.grader
    assert isinstance(shell, ScriptGrader)
    assert (shell.environment.docker_image, shell.reward) == (IMAGE, VERDICT_REWARD)
    assert (shell.argv, shell.cwd, shell.answer_path) == (
        ("python3", "/tests/grade.py"),
        "/workspace",
        "/app/answer.txt",
    )
    assert [(a.source, a.target, a.kind) for a in shell.artifacts] == [
        ("/workspace", "/workspace", ArtifactKind.DIRECTORY)
    ]
    assert sorted(f.path for f in package.resources) == ["grade.py", "key.txt"]


@pytest.mark.parametrize(
    ("machine", "packages", "facts"),
    [
        (SolverMachine.NONE, [], "The solver has no machine"),
        (SolverMachine.SHELLSIM, [], "The task machine is ShellSim"),
        (SolverMachine.CPYTHON, [], "The task machine is a container with CPython 3.12"),
        (SolverMachine.PACKAGES, ["numpy"], "the PyPI packages you list"),
    ],
)
async def test_the_fixtures_and_requirements_steps_build_the_declared_solver_machine(
    context, proposal, fake_glm, machine, packages, facts
):
    declared = declaring(proposal, machine)
    fake_glm.stream(content=json.dumps({"agent_files": [], "private_files": [], "facts": "f", "packages": packages}))

    fixtures = await standard.fixtures.fn(context(), declared, Sources(notes="", findings=()))
    requirements = await standard.requirements.fn(context(), declared, fixtures)

    assert facts in fake_glm.requests[0]["messages"][-2]["content"]
    assert solver_machine(requirements) is machine


async def test_cpython_fixtures_choose_the_grader_base_image(context, proposal):
    requirements = await standard.requirements.fn(context(), declaring(proposal, SolverMachine.CPYTHON), FIXTURES)
    package = grader_package(GraderDraft.model_validate(grader_call(GOOD_GRADER)), FIXTURES, requirements)
    assert isinstance(package, GraderPackage)

    assert requirements.docker_image == GRADER_BASE_IMAGE
    assert machine_facts(requirements).startswith("The task machine is a container with CPython 3.12")
    assert isinstance(package.grader, ScriptGrader) and package.grader.environment.docker_image == GRADER_BASE_IMAGE


LEDGER_LINE = "2024-03-01,ACME-SUPPLY,invoice 8812,1234.56,approved by finance"


@pytest.mark.parametrize(
    ("instruction", "problem"),
    [
        ("Reconcile the ledger. " + "Check every entry carefully. " * 130, "words outside the answer contract"),
        (f"Reconcile the ledger. The first entry is:\n{LEDGER_LINE}\n", "copies solver-visible file lines"),
        ("Reconcile workspace/ledger.csv by hand and report the total.", "asks for work by hand"),
        ("Reconcile workspace/ledger.csv; only the final line is graded.", "refers to grading ('graded')"),
    ],
    ids=["over-500-words", "inlined-fixture-line", "by-hand", "grading-language"],
)
async def test_the_instructions_step_sends_back_an_instruction_with_a_problem(
    context, proposal, fake_glm, instruction, problem
):
    contract = "End the reply with a line `ANSWER = <n>`."
    good = f"Reconcile workspace/ledger.csv and report the approved total. {contract}"
    fake_glm.stream(content=json.dumps({"system": "", "instruction": instruction}))
    fake_glm.stream(content=json.dumps({"system": "", "instruction": good}))
    ledger = task_file(FileDraft.model_validate(file("workspace/ledger.csv", LEDGER_LINE)))
    fixtures = replace(FIXTURES, agent_files=(ledger,), private_files=())
    grader = Grader(
        package=answer_grader(ExactSpec(expected=("1234.56",))),
        answer_contract=contract,
        reference_reply="ANSWER = 1234.56",
        reference_files=(),
        secret_values=(),
        rejection_ceiling=0.0,
        prototyped=(),
    )
    requirements = draft.requirements(image=GRADER_BASE_IMAGE, workdir=WORKDIR)

    text = await standard.instructions.fn(context(), proposal, fixtures, requirements, grader)

    assert text.instruction == good
    assert problem in fake_glm.requests[1]["messages"][-2]["content"]


async def test_package_fixtures_choose_a_package_machine_without_an_image(context, proposal):
    fixtures = replace(FIXTURES, packages=("numpy", "pandas==2.2.3"))
    requirements = await standard.requirements.fn(context(), declaring(proposal, SolverMachine.PACKAGES), fixtures)
    package = grader_package(GraderDraft.model_validate(grader_call(GOOD_GRADER)), fixtures, requirements)
    assert isinstance(package, GraderPackage)

    assert (requirements.docker_image, requirements.package_spec) == (
        None,
        PackageSpec(pypi_packages=("numpy", "pandas==2.2.3")),
    )
    assert "`numpy`, `pandas==2.2.3`" in machine_facts(requirements)
    assert isinstance(package.grader, ScriptGrader) and package.grader.environment.docker_image == GRADER_BASE_IMAGE


def test_fixtures_refuse_a_package_with_pip_options():
    with pytest.raises(ValueError, match="without pip options"):
        FixturesDraft.model_validate(
            {"agent_files": [], "private_files": [], "facts": "", "packages": ["--index-url=https://example.org"]}
        )


@pytest.mark.parametrize(
    ("machine", "packages", "problem"),
    [
        (SolverMachine.PACKAGES, [], "declares solver_machine: packages; list the PyPI packages"),
        (SolverMachine.SHELLSIM, ["numpy"], "declares solver_machine: shellsim, which installs no packages"),
        (None, ["numpy"], "declares a container, which installs no packages"),
    ],
    ids=["packages-without-any", "shellsim-with-some", "container-with-some"],
)
async def test_fixtures_list_packages_exactly_for_a_declared_package_machine(proposal, machine, packages, problem):
    submitted = FixturesDraft.model_validate({"agent_files": [], "private_files": [], "facts": "", "packages": packages})

    check = packages_declared(declaring(proposal, machine))

    assert problem in (await check(submitted) or "")
    matching = ["numpy"] if machine is SolverMachine.PACKAGES else []
    assert await check(submitted.model_copy(update={"packages": matching})) is None


DECISION_REFERENCE = "6 * 7 = 42, so the product is 42."
DECISION_VARIANTS = ["Multiplying six by seven gives 42.", "6 times 7 is forty-two.", "The product is 42 (6 * 7)."]
DECISION_RUBRIC = {
    "deliverable": "The product of 6 and 7.",
    "items": [
        {
            "id": "product",
            "intent": "The reply states the product.",
            "pass_criterion": "The product is 42.",
            "equivalences": ["forty-two"],
            "negation_handling": "A product named only to be rejected does not count.",
            "weight": 1.0,
            "role": "scored",
            "cap_value": None,
        }
    ],
}


def decision_call(variants: list[str]) -> dict:
    return {
        **grader_call(""),
        "kind": "decision_model",
        "rubric": DECISION_RUBRIC,
        "equivalent_variants": variants,
        "answer_contract": "Tell me the product and how you got it.",
        "reference_reply": DECISION_REFERENCE,
        "rejection_ceiling": 0.5,
    }


def decision_proposal():
    return parse(PROPOSAL.replace("verification: answer_in_box", "verification: decision_model"))


async def test_the_grader_step_calibrates_a_decision_grader_on_its_reference_and_variants(context, fake_glm):
    """A variant the rubric reads as half wrong goes back to the model; the accepted reads set the bar."""
    misread = "Six sevens: 42."
    levels = {DECISION_REFERENCE: 3.9, misread: 1.4, DECISION_VARIANTS[0]: 3.1, DECISION_VARIANTS[1]: 2.9}
    client = FakeDecisionClient({**levels, DECISION_VARIANTS[2]: 3.4})
    fake_glm.stream(content=json.dumps(decision_call([misread, *DECISION_VARIANTS[1:]])))
    fake_glm.stream(content=json.dumps(decision_call(DECISION_VARIANTS)))

    grader = await standard.grader.fn(
        context(decision=client), decision_proposal(), replace(FIXTURES, private_files=()), SHELLSIM
    )

    assert isinstance(grader.package, DecisionGrader)
    assert (grader.package.reference, grader.package.acceptance_level, grader.rejection_ceiling) == (
        DECISION_REFERENCE,
        2.9,
        0.5,
    )
    assert set(grader.package.calibration) == {"reference", "variant:1", "variant:2", "variant:3"}
    assert "known-correct candidates read below 'partially matches'" in fake_glm.requests[1]["messages"][-2]["content"]
    assert [c.reply for c in grader.prototyped[-5:]] == [DECISION_REFERENCE, *DECISION_VARIANTS, ""]
    assert "decision_model" in fake_glm.requests[0]["messages"][-2]["content"]
    # The step output keeps the decision grader type through the step store's JSON.
    stored = TypeAdapter(Grader)
    assert stored.validate_json(stored.dump_json(grader)) == grader


async def test_a_decision_grader_on_a_host_without_a_decision_model_is_the_hosts_failure(context, fake_glm):
    fake_glm.stream(content=json.dumps(decision_call(DECISION_VARIANTS)))

    with pytest.raises(BuildInfrastructureFailure, match="no_decision_model"):
        await standard.grader.fn(context(), decision_proposal(), replace(FIXTURES, private_files=()), SHELLSIM)


async def test_a_decision_model_down_twice_on_a_calibration_read_is_the_hosts_failure(context, fake_glm, monkeypatch):
    monkeypatch.setattr(decision, "DECISION_RETRY_BACKOFF", 0.0)
    client = FakeDecisionClient({}, failures=[DecisionUnavailable("503"), DecisionUnavailable("503")])
    fake_glm.stream(content=json.dumps(decision_call(DECISION_VARIANTS)))

    with pytest.raises(BuildInfrastructureFailure, match="decision_unavailable"):
        await standard.grader.fn(
            context(decision=client), decision_proposal(), replace(FIXTURES, private_files=()), SHELLSIM
        )
    assert len(client.requests) == 2
