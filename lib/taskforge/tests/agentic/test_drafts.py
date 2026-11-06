# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``draft_task`` with a scripted agent on real ShellSim machines and the real ``submit`` checks."""

import io
import json
import shlex
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, PackageSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.models import VerifyitGrader
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import JudgeSpec

from taskforge.agentic import example_build
from taskforge.agentic.drafts import DRAFT_MAX_ANSWER_SHARE, DraftConfig, DraftProposal, DraftTools, draft_task
from taskforge.agentic.grader_examples import EXAMPLES
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import VARIANT_SUFFIX, VARIANTS, reference_answer

PROPOSAL_ID = "d01.example.stock-ledger/3"
TASK_ID = "ledger--00"
TASK = f"tasks/{TASK_ID}.json"
EXAMPLE = example_build.task_spec(TASK_ID)
# Scripted solvers need only the shell and the ledger file, not the downloaded CPython runtime.
SHELL_TASK = EXAMPLE.model_copy(
    update={
        "environment_requirements": EXAMPLE.environment_requirements.model_copy(
            update={"capabilities": ("shell", "filesystem"), "package_spec": PackageSpec()}
        )
    }
)
GRADER = SHELL_TASK.grader
assert isinstance(GRADER, VerifyitGrader)
EXPECTED = str(GRADER.parameters["expected"])
# Other task ids give the expected answer followed by the text after the id's last hyphen.
BUILD = (
    f'if [ "$1" = {TASK_ID} ]; then cat task.json; '
    f'else n=${{1##*-}}; sed "s/{TASK_ID}/$1/g; '
    f's/\\"expected\\":\\"{EXPECTED}\\"/\\"expected\\":\\"{EXPECTED}$n\\"/" task.json; fi > "$2"'
)
# A program that writes the same task for every id, which submit refuses.
CONSTANT_BUILD = f'sed "s/{TASK_ID}/$1/g" task.json > "$2"'
LIMITS = RolloutLimits(
    max_turns=10,
    max_total_tokens=None,
    model_turn_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=None,
    cleanup_timeout=30,
)


@pytest.fixture(autouse=True)
def no_package_downloads(monkeypatch: pytest.MonkeyPatch):
    def refuse_download(*args, **kwargs):
        raise AssertionError("Unit tests must not download ShellSim catalog or runtime assets")

    monkeypatch.setattr(urllib.request, "urlopen", refuse_download)


def proposal(solver_machine: str = "shellsim") -> DraftProposal:
    header = {
        "id": PROPOSAL_ID,
        "environment": "shellsim",
        "solver_machine": solver_machine,
        "verification": "answer_in_box",
        "source": {"kind": "capability", "ref": "d01.example.stock-ledger", "hash": "0"},
    }
    return DraftProposal.from_json(json.dumps({"header": header, "body": "## Task\n\nRead a ledger."}))


def write_task(task_json: str, build: str) -> tuple[str, dict[str, object]]:
    command = (
        f"mkdir -p tasks; printf '%s' {shlex.quote(task_json)} > task.json; cp task.json {TASK}; "
        f"printf '%s\\n' {shlex.quote(build)} > build.sh"
    )
    return "shell", {"command": command}


def submit() -> tuple[str, dict[str, object]]:
    return "submit", {"path": TASK, "program": "build.sh"}


@dataclass
class ScriptedAgent:
    """Calls the tools of ``script`` in order, one call in each turn."""

    script: Sequence[tuple[str, dict[str, object]]]
    turns: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        name, arguments = self.script[self.turns]
        self.turns += 1
        tool_call = {
            "id": f"call-{self.turns}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(10, 1))


@dataclass(frozen=True)
class LinuxFactory:
    """Stands in for bubblewrap, which needs namespaces that a test host may not have: it accepts every
    environment and starts no machine. ``submit`` only checks a grader's machine, and never starts one."""

    backend: Backend = Backend.LOCAL

    def check_environment(self, requirements, spec) -> None:
        pass

    async def create(self, spec):
        raise AssertionError("submit starts no grader machine")


def config(agent: ScriptedAgent) -> DraftConfig:
    return DraftConfig(
        agent_model=agent,
        workspace_factory=ShellSimMachineFactory(),
        solver_factories={Backend.SHELLSIM: ShellSimMachineFactory(), Backend.LOCAL: LinuxFactory()},
        judge=JudgeConnection("http://judge.invalid/v1", "unused", default_model="deepseek"),
        agent_limits=LIMITS,
        solver_limits=LIMITS,
        command_timeout=10,
    )


def replies(result) -> dict[str, dict]:
    messages = result.rollout.conversations[-1].messages
    return {m["tool_call_id"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}


async def test_a_draft_needs_a_program_whose_ids_give_varied_tasks():
    task_json = SHELL_TASK.model_dump_json()
    agent = ScriptedAgent([write_task(task_json, CONSTANT_BUILD), submit(), write_task(task_json, BUILD), submit()])
    result = await draft_task(proposal(), config(agent))

    assert result.failure is None
    assert result.submits == 2
    [problem] = replies(result)["call-2"]["problems"]
    assert "writes the same task for another task id" in problem
    assert result.draft is not None and result.draft.task == SHELL_TASK
    with tarfile.open(fileobj=io.BytesIO(result.draft.workspace)) as archive:
        assert "build.sh" in archive.getnames()
    instances = await task_instances(result.draft, ["ledger--1", "ledger--2"], ShellSimMachineFactory(), timeout=10)
    answers = [instance.grader.parameters["expected"] for instance in instances]  # type: ignore[union-attr]
    assert answers == [f"{EXPECTED}1", f"{EXPECTED}2"]


async def test_a_draft_may_grade_with_a_judge_that_it_does_not_call():
    package = verifyit_package(JudgeSpec(references=(EXPECTED,), question="How many units are left?"))
    judged = SHELL_TASK.model_copy(update={"grader": package.grader})
    # The program changes the reference answer that the judge compares with for other ids.
    build = (
        f'if [ "$1" = {TASK_ID} ]; then cat task.json; '
        f'else n=${{1##*-}}; sed "s/{TASK_ID}/$1/g; s/\\[\\"{EXPECTED}\\"\\]/[\\"{EXPECTED}$n\\"]/" task.json; '
        'fi > "$2"'
    )
    agent = ScriptedAgent([write_task(judged.model_dump_json(), build), submit()])
    result = await draft_task(proposal(), config(agent))

    # The accepting reply ends the session, so the trace does not keep it.
    assert (result.failure, result.submits) == (None, 1)
    assert result.draft is not None and result.draft.task == judged


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_the_grader_examples_of_the_agent_pass_the_draft_checks(kind: str):
    tools = DraftTools(proposal(solver_machine="none"), config(ScriptedAgent([])))
    tasks = [EXAMPLES[kind](f"example{VARIANT_SUFFIX}{k}") for k in range(VARIANTS)]

    assert [tools.problems(task) for task in tasks] == [[]] * VARIANTS
    # The program check refuses ids whose reference answers repeat too often.
    [(_, repeats)] = Counter(reference_answer(task) for task in tasks).most_common(1)
    assert repeats <= DRAFT_MAX_ANSWER_SHARE * VARIANTS
