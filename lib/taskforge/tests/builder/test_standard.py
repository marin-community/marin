# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Mapping
from dataclasses import replace

from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, MachineSpec, ShellSimBuiltins
from taskcompendium.models import ArtifactKind, ScriptGrader, VerifyitGrader
from taskcompendium.runtime.resources import resource_bytes

from taskforge.builder.author import compile_program
from taskforge.builder.run import item_id_for, run_build
from taskforge.builder.sdk import Build
from taskforge.builder.step import StepCache
from taskforge.builder.template import standard
from taskforge.llm.agent import AgentTool
from taskforge.proposal.model import parse
from taskforge.sandbox.images import GRADER_BASE_IMAGE, DockerBuild
from taskforge.spec.controls import ControlCategory
from tests.builder.conftest import PROPOSAL

GOOD_GRADER = """import pathlib
final = pathlib.Path("/app/answer.txt").read_text().strip()
expected = pathlib.Path("/tests/key.txt").read_text().strip()
print(float(final.endswith("ANSWER = " + expected)))
"""
IMAGE = f"registry.example/taskforge-tasks/d00@sha256:{'2' * 64}"


def file(path: str, content: str) -> dict:
    return {"path": path, "content": content, "executable": False}


def grader_call(script: str) -> dict:
    return {
        "kind": "script",
        "script": script,
        "private_files": [],
        "expected": "",
        "tolerance": 0.0,
        "answer_contract": "End the reply with a line `ANSWER = <n>`.",
        "reference_reply": "6 * 7 = 42\nANSWER = 42",
        "reference_files": [],
        "secret_values": ["42"],
    }


def control(
    control_id: str, kind: str, category: str, concern: str, text: str, low: float | None, high: float | None
) -> dict:
    return {
        "id": control_id,
        "kind": kind,
        "category": category,
        "concern": concern,
        "final_reply": text,
        "files": [],
        "reward_min": low,
        "reward_max": high,
        "rationale": "r",
    }


def web_tool(queries: list[Mapping[str, object]]) -> AgentTool:
    async def search(arguments: Mapping[str, object]) -> str:
        queries.append(arguments)
        return "6 x 7 = 42 (https://example.org/table)"

    return AgentTool(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=search,
    )


async def test_template_builds_a_checked_task_and_retries_failed_checks(proposal, tmp_path, services, fake_glm):
    def submit(name: str, arguments: dict) -> None:
        fake_glm.stream(tool_calls=((name, json.dumps(arguments)),), finish="tool_calls")

    fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="Confirmed: 6 x 7 = 42 (https://example.org/table).")
    submit(
        "submit_fixtures",
        {
            "agent_files": [file("workspace/question.txt", "What is 6 * 7?")],
            "private_files": [file("key.txt", "42")],
            "facts": "6 * 7 = 42",
        },
    )
    # The first grader gives the reference answer no credit; the step feeds that back and retries.
    submit("submit_grader", grader_call(GOOD_GRADER.replace('"ANSWER = "', '"RESULT = "')))
    submit("submit_grader", grader_call(GOOD_GRADER))
    # The first instruction leaks the graded answer.
    submit("submit_instructions", {"system": "", "instruction": "Show that 6 * 7 = 42. End with ANSWER = <n>."})
    submit("submit_instructions", {"system": "", "instruction": "Read question.txt. End with `ANSWER = <n>`."})
    submit(
        "submit_controls",
        {
            "controls": [
                control("gold", "positive", "known_correct", "reference", "ANSWER = 42", 0.99, None),
                control("empty", "malformed", "empty_or_malformed", "extraction", "", None, 0.0),
                control("sum", "negative", "plausible_wrong", "acceptance", "ANSWER = 13", None, 0.0),
                control("echo", "negative", "task_specific_shortcut", "shortcut", "ANSWER = <n>", None, 0.0),
            ]
        },
    )
    queries: list[Mapping[str, object]] = []
    program = compile_program(inspect.getsource(standard), proposal.digest)

    async with services(fake_glm.base_url, web_tools=(web_tool(queries),)) as s:
        draft = await run_build(program, proposal, tmp_path / "item", tmp_path / "cache", s)

    assert queries == [{"query": "6 times 7"}]
    assert fake_glm.responses == type(fake_glm.responses)()
    task = draft.task
    assert task.environment_requirements.docker_image is None
    assert draft.lowered.runtime.task_machine is not None
    assert draft.lowered.runtime.task_machine.backend == Backend.SHELLSIM
    assert draft.lowered.runtime.verifier_machine is not None
    assert draft.lowered.runtime.verifier_machine.backend == Backend.DOCKER
    assert [f.path for f in task.resources.worker] == ["workspace/question.txt"]
    assert isinstance(task.grader, ScriptGrader) and task.grader.answer_path == "/app/answer.txt"
    assert task.grader.environment.docker_image == GRADER_BASE_IMAGE
    assert {f.path: resource_bytes(f).decode() for f in task.resources.verifier} == {
        "grade.py": GOOD_GRADER,
        "config.json": "{}",
        "key.txt": "42",
    }
    assert "42" not in json.dumps(task.context.model_dump(mode="json"))
    assert {c.category for c in draft.controls} == {
        ControlCategory.KNOWN_CORRECT,
        ControlCategory.EMPTY_OR_MALFORMED,
        ControlCategory.PLAUSIBLE_WRONG,
        ControlCategory.TASK_SPECIFIC_SHORTCUT,
    }
    assert {r.name for r in draft.provenance.resources} >= {"research/notes.md", "grader/grade.py"}


async def test_grader_step_feeds_back_a_numeric_answer_that_is_not_a_literal(proposal, tmp_path, services, fake_glm):
    def numeric(expected: str) -> None:
        arguments = {**grader_call(""), "kind": "numeric", "expected": expected, "reference_reply": "42"}
        fake_glm.stream(tool_calls=(("submit_grader", json.dumps(arguments)),), finish="tool_calls")

    numeric("sqrt(1764)")
    numeric("42")
    made = standard.Fixtures(agent_files=(), private_files=(), facts="6 * 7 = 42")
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services(fake_glm.base_url) as s:
        b = Build(proposal, cache.item_id, s, cache, tmp_path / "scratch", 0)
        machine = b.spec.requirements(image=None, workdir=standard.WORKDIR)
        graded = await standard.grader(b, made, machine, "")

    assert isinstance(graded.package.grader, VerifyitGrader) and graded.package.grader.mode == "numeric"
    retry = fake_glm.requests[1]["messages"][-1]["content"]
    assert "numeric value requires one finite scalar literal" in retry


async def test_control_files_with_shell_metacharacters_in_their_paths_are_written_verbatim():
    path = "workspace/it's $HOME/a b.txt"
    draft = standard.ControlDraft.model_validate(
        {
            **control("quoted", "positive", "known_correct", "reference", "done", 1.0, None),
            "files": [file(path, "x'y\n")],
        }
    )
    turn = standard.control_payload(draft).turns[0]
    machine = await ShellSimMachineFactory().create(MachineSpec(source=ShellSimBuiltins()))
    try:
        for call in turn.calls:
            result = await machine.run(Command(argv=("sh", "-c", call.arguments["command"])))
            assert result.exit_code == 0, result.stderr
        written = await machine.run(Command(argv=("cat", f"/{path}")))
    finally:
        await machine.close()
    assert written.stdout == b"x'y\n"


class RecordingImageBuilder:
    def __init__(self) -> None:
        self.published: list[tuple[DockerBuild, str]] = []

    async def publish(self, build: DockerBuild, repository: str) -> str:
        self.published.append((build, repository))
        return IMAGE


async def test_a_container_task_publishes_its_image_and_grades_in_a_copy_of_it(tmp_path, services, fake_glm):
    proposal = parse(PROPOSAL.replace("environment: reasoning", "environment: container"))
    dockerfile = "FROM python:3.12-slim\nWORKDIR /workspace\n"
    fake_glm.stream(
        tool_calls=(("submit_image", json.dumps({"dockerfile": dockerfile, "context_files": []})),),
        finish="tool_calls",
    )
    images = RecordingImageBuilder()
    made = standard.Fixtures(
        agent_files=(),
        private_files=(standard.task_file(standard.FileDraft.model_validate(file("key.txt", "42"))),),
        facts="",
    )
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services(fake_glm.base_url) as s:
        b = Build(proposal, cache.item_id, replace(s, images=images), cache, tmp_path / "scratch", 0)
        machine = await standard.requirements(b, made, "")
        draft = standard.GraderDraft.model_validate(grader_call("print(1.0)"))
        package = standard.grader_package(b, draft, made, machine)

    assert machine.docker_image == IMAGE
    assert [resource_bytes(f).decode() for f in images.published[0][0].files] == [dockerfile]
    shell = package.grader
    assert isinstance(shell, ScriptGrader)
    assert shell.environment.docker_image == IMAGE
    assert (shell.argv, shell.cwd, shell.answer_path) == (
        ("python3", "/tests/grade.py"),
        "/workspace",
        "/app/answer.txt",
    )
    assert [(a.source, a.target, a.kind) for a in shell.artifacts] == [
        ("/workspace", "/workspace", ArtifactKind.DIRECTORY)
    ]
    assert sorted(f.path for f in package.resources) == ["grade.py", "key.txt"]
