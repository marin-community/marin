# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from collections.abc import Mapping

from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, ShellSimBuiltins
from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import VerifierKind

from taskforge.build.author import compile_program
from taskforge.build.run import item_id_for, run_build
from taskforge.build.sdk import Build
from taskforge.build.step import StepCache
from taskforge.build.template import standard
from taskforge.llm.agent import AgentTool
from taskforge.spec.controls import ControlCategory

GOOD_GRADER = """import json, sys
messages = json.load(sys.stdin)
final = [m for m in messages if m.get("role") == "assistant"][-1].get("content") or ""
expected = open("/grader/key.txt").read().strip()
print(1.0 if final.strip().endswith("ANSWER = " + expected) else 0.0)
"""


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


def control(control_id: str, kind: str, category: str, text: str, low: float | None, high: float | None) -> dict:
    return {
        "id": control_id,
        "kind": kind,
        "category": category,
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
            "agent_files": [file("/workspace/question.txt", "What is 6 * 7?")],
            "private_files": [file("/grader/key.txt", "42")],
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
                control("gold", "positive", "known_correct", "ANSWER = 42", 0.99, None),
                control("empty", "malformed", "empty_or_malformed", "", None, 0.0),
                control("sum", "negative", "plausible_wrong", "ANSWER = 13", None, 0.0),
                control("echo", "negative", "task_specific_shortcut", "ANSWER = <n>", None, 0.0),
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
    assert task.environment.kind == EnvironmentKind.SHELLSIM
    assert [f.path for f in task.environment.files] == ["/workspace/question.txt"]
    assert task.verifier.kind == VerifierKind.SHELL
    assert {f.path: f.content.decode() for f in task.verifier.files} == {
        standard.GRADER_SCRIPT: GOOD_GRADER,
        "/grader/key.txt": "42",
    }
    assert "42" not in json.dumps(task.context.model_dump(mode="json"))
    assert {c.category for c in draft.controls} == {
        ControlCategory.KNOWN_CORRECT,
        ControlCategory.EMPTY_OR_MALFORMED,
        ControlCategory.PLAUSIBLE_WRONG,
        ControlCategory.TASK_SPECIFIC_SHORTCUT,
    }
    assert {r.name for r in draft.provenance.resources} >= {"research/notes.md", standard.GRADER_SCRIPT}


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
        machine = b.spec.environment(EnvironmentKind.SHELLSIM, workdir=standard.WORKDIR)
        graded = await standard.grader(b, made, machine, "")

    assert graded.verifier.kind == VerifierKind.NUMERIC_ANSWER
    retry = fake_glm.requests[1]["messages"][-1]["content"]
    assert "numeric value requires one finite scalar literal" in retry


async def test_control_files_with_shell_metacharacters_in_their_paths_are_written_verbatim():
    path = "/workspace/it's $HOME/a b.txt"
    draft = standard.ControlDraft.model_validate(
        {**control("quoted", "positive", "known_correct", "done", 1.0, None), "files": [file(path, "x'y\n")]}
    )
    turn = standard.control_payload(draft).turns[0]
    machine = await ShellSimMachineFactory().create(MachineSpec(source=ShellSimBuiltins()))
    try:
        for call in turn.calls:
            result = await machine.run(Command(argv=("sh", "-c", call.arguments["command"])))
            assert result.exit_code == 0, result.stderr
        written = await machine.run(Command(argv=("cat", path)))
    finally:
        await machine.close()
    assert written.stdout == b"x'y\n"
