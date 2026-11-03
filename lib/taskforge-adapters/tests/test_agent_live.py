# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of taskforge.llm.agent and taskforge_adapters.web against GLM-5.3 (interactive tier).

Each check writes ``<evidence_root>/llm/agent/<check>-<utc>.json``: the purpose, the policy,
the full ``AgentRun`` (every message, every completion with its raw stream events, every tool
result) and the check's own observations. Ledger
spans go to ``<evidence_root>/llm/agent/ledger/<check>/<workpiece>.jsonl``. ``evidence_root`` is the
fixture in ``tests/conftest.py``. The web check also needs the Parallel key file (the
``parallel_key`` fixture there).

Each check needs the live server or Parallel: the Parallel API, 20 concurrent agents over one client,
vLLM's ``tool_calls`` finish on a cut call, and the server's refusal of non-object replayed arguments.

Invariants: none (pins live GLM-5.3 and Parallel wire compatibility of the agent loop).
"""

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter
from shellbox.backends.shellsim.machine import ShellSimMachine, ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, PackageSpec
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.agent import (
    AgentRun,
    AgentStop,
    AgentTool,
    ReasoningReplay,
    ToolOutcome,
    replay_arguments,
    run_agent,
    shell_tool,
)
from taskforge.llm.client import FinishReason, ModelRequestRejected
from taskforge.llm.policy import LLMPolicy, Message, ReasoningEffort
from taskforge.llm.recording import CallLedger

from taskforge_adapters.glm_client import GLM_MODEL, GlmClient, request_kwargs
from taskforge_adapters.testing import unique_run_name
from taskforge_adapters.web import web_tools

PYPI_PACKAGE_URL = "https://pypi.org/pypi/shellsim/json"
RUNS = TypeAdapter(tuple[AgentRun, ...])
SHELL_TIMEOUT = 120.0
MALFORMED_ARGUMENTS = '{"command": "ls -la /work'
SHELL_OUTPUT_LIMIT = 64 * 1024
SHELLSIM_NOTE = (
    "The shell is ShellSim, a simulated BusyBox-like Linux with no network and no pip. python3 and a "
    "minimal pytest are installed; pytest needs explicit test file paths (pytest tests/test_x.py). "
    "Your working directory is /workspace."
)
SYSTEM = (
    "You are a capable software agent. Use the tools to act; do not claim you ran something you did not "
    "run. Say when you are finished."
)


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "llm" / "agent"


def ledger_record(evidence_dir: Path, check: str, workpiece_id: str) -> CallLedger:
    return CallLedger(
        ledger=JsonlLedger(evidence_dir / "ledger" / check), workpiece_id=workpiece_id, round=0, step=check
    )


async def shellsim() -> ShellSimMachine:
    return await ShellSimMachineFactory().create(MachineSpec(source=PackageSpec()))


def record(evidence_dir: Path, check: str, purpose: str, policy: LLMPolicy, runs: Sequence[AgentRun], **extra) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "check": check,
        "purpose": purpose,
        "policy": asdict(policy),
        **extra,
        "runs": RUNS.dump_python(tuple(runs), mode="json"),
    }
    path = evidence_dir / f"{unique_run_name(check)}.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return path


def executed(run: AgentRun, name: str) -> list[str]:
    return [
        r.output for t in run.turns for r in t.tool_results if r.call.name == name and r.outcome is ToolOutcome.EXECUTED
    ]


def last_integer(text: str) -> int | None:
    """The last integer written in ``text``, or None when it holds no digits."""
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


async def timed_agent(
    client: GlmClient, policy: LLMPolicy, messages: list[Message], tools: Sequence[AgentTool], max_turns: int, rec
) -> tuple[AgentRun, float]:
    started = time.monotonic()
    run = await run_agent(client, policy, messages, tools, max_turns, rec, ReasoningReplay.KEEP)
    return run, time.monotonic() - started


def stable_metadata(http: httpx.Client) -> tuple[str, str]:
    """shellsim's license and repository URL from PyPI's JSON API; neither changes between releases."""
    info = http.get(PYPI_PACKAGE_URL).raise_for_status().json()["info"]
    return info["license_expression"], info["project_urls"]["Repository"]


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_web_search_through_parallel(glm_source, live_provider, evidence_dir, parallel_key):
    policy = LLMPolicy()
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Find the license and the source repository URL of the shellsim package on PyPI. Start with "
                "web_search to locate sources, then fetch what you need. Cite the URLs you used."
            ),
        },
    ]

    async def go() -> tuple[AgentRun, float]:
        async with GlmClient(glm_source, live_provider) as client, httpx.AsyncClient() as http:
            tools = web_tools(http, parallel_key.value)
            return await timed_agent(client, policy, messages, tools, 20, ledger_record(evidence_dir, "web", "web"))

    # Parallel serves cached pages, so the check uses metadata that does not change between releases.
    with httpx.Client(timeout=60.0) as http:
        license_expression, repository = stable_metadata(http)
    run, wall = asyncio.run(go())
    searches = executed(run, "web_search")
    answer = str(run.messages[-1]["content"])
    record(
        evidence_dir,
        "web_parallel",
        "web search through Parallel /v1/search and /v1/extract; transcript must show search results",
        policy,
        [run],
        wall_time=wall,
        search_ids=[json.loads(s).get("search_id") for s in searches],
        extract_ids=[json.loads(f).get("extract_id") for f in executed(run, "web_fetch")],
        answer=answer,
        pypi_metadata={"license_expression": license_expression, "repository": repository},
    )
    assert run.stop is AgentStop.ANSWERED
    assert searches and all(json.loads(s)["search_id"] for s in searches)
    assert license_expression in answer, (answer, license_expression)
    assert repository.removeprefix("https://") in answer, (answer, repository)


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_twenty_concurrent_agents_share_one_client(glm_source, live_provider, evidence_dir):
    policy = LLMPolicy()
    agents = 20

    def messages(n: int) -> list[Message]:
        return [
            {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
            {
                "role": "user",
                "content": (
                    f"Write the value of {n} * {n} + 7 into /workspace/answer.txt using the shell, "
                    "show the file with cat, then reply with the number."
                ),
            },
        ]

    @dataclass(frozen=True)
    class AgentResult:
        run: AgentRun
        wall_time: float
        file: str

    async def one(client: GlmClient, n: int) -> AgentResult:
        machine = await shellsim()
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        run, wall = await timed_agent(
            client, policy, messages(n), [tool], 20, ledger_record(evidence_dir, "concurrent", f"agent-{n}")
        )
        content = await machine.run(Command(argv=("cat", "/workspace/answer.txt"), timeout=SHELL_TIMEOUT))
        await machine.close()
        return AgentResult(run, wall, content.stdout.decode().strip())

    async def go() -> tuple[list, float]:
        started = time.monotonic()
        async with GlmClient(glm_source, live_provider) as client:
            results = await asyncio.gather(*(one(client, n) for n in range(agents)), return_exceptions=True)
        return results, time.monotonic() - started

    results, total = asyncio.run(go())
    per_agent = [
        (
            {"agent": n, "error": repr(r)}
            if isinstance(r, BaseException)
            else {
                "agent": n,
                "stop": r.run.stop,
                "turns": len(r.run.turns),
                "wall_time": r.wall_time,
                "completion_tokens": r.run.usage.completion_tokens,
                "file": r.file,
                "number": last_integer(r.file),
                "expected": n * n + 7,
            }
        )
        for n, r in enumerate(results)
    ]
    runs = [r.run for r in results if not isinstance(r, BaseException)]
    record(
        evidence_dir,
        "concurrent_20",
        f"{agents} agents gathered over one GlmClient, each with its own ShellSim machine",
        policy,
        runs,
        total_wall_time=total,
        per_agent=per_agent,
    )
    assert [a for a in per_agent if "error" in a] == []
    # At temperature 1.0 an agent sometimes writes the arithmetic along with the result, as in "3 * 3 + 7 = 16".
    assert [a["agent"] for a in per_agent if a["number"] != a["expected"]] == []


def write_probe(evidence_dir: Path, check: str, purpose: str, cases: list[dict]) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{unique_run_name(check)}.json"
    path.write_text(json.dumps({"check": check, "purpose": purpose, "cases": cases}, indent=2) + "\n")
    return path


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_probe_tool_call_cut_by_max_tokens_is_reported_as_length(glm_source, live_provider, evidence_dir):
    policy = LLMPolicy(max_tokens=300, reasoning_effort=ReasoningEffort.LOW, max_continuations=0)
    messages: list[Message] = [
        {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
        {
            "role": "user",
            "content": (
                "In one shell call, create /workspace/numbers.txt holding the integers 1 to 300, one per line, "
                "written out literally in a heredoc (no seq, no loops)."
            ),
        },
    ]

    async def go() -> dict:
        machine = await shellsim()
        fields: dict[str, object] = {
            "tools": [shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT).definition()]
        }
        await machine.close()
        async with GlmClient(glm_source, live_provider) as client:
            started = time.monotonic()
            completion = await client.complete(messages, policy, fields)
            wall = time.monotonic() - started
        body = request_kwargs(GLM_MODEL, messages, policy.max_tokens, policy, fields)
        final = completion.attempts[-1]
        raw = [json.loads(e) for e in final.events]
        case = {
            "request": body,
            "wall_time": wall,
            "finish_reason": completion.finish_reason,
            "raw_finish_reasons": [
                c["finish_reason"] for e in raw for c in e.get("choices", []) if c.get("finish_reason")
            ],
            "usage": asdict(completion.usage),
            "tool_calls": [asdict(c) for c in completion.tool_calls],
            "events": list(final.events),
        }
        return case

    case = asyncio.run(go())
    write_probe(
        evidence_dir,
        "probe_tool_call_cut",
        "a heredoc tool call cut by max_tokens=300: vLLM reports finish_reason tool_calls, GlmClient reports length",
        [case],
    )
    assert case["usage"]["completion_tokens"] >= policy.max_tokens
    assert case["tool_calls"], "the model spent the budget before starting a tool call"
    assert case["raw_finish_reasons"] == ["tool_calls"]
    assert case["finish_reason"] == FinishReason.LENGTH


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_probe_replayed_tool_call_arguments_must_be_a_json_object(glm_source, live_provider, evidence_dir):
    policy = LLMPolicy(max_tokens=2000, reasoning_effort=ReasoningEffort.LOW)
    tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the task workspace. Files persist between commands.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
    fields: dict[str, object] = {"tools": [tool]}

    def conversation(arguments: str) -> list[Message]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Create /workspace/hello.txt containing the word hello."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-0", "type": "function", "function": {"name": "shell", "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-0", "content": "error: tool arguments are not valid JSON"},
        ]

    async def go() -> list[dict]:
        cases = []
        async with GlmClient(glm_source, live_provider) as client:
            for raw in (MALFORMED_ARGUMENTS, "[1]"):
                for replayed in (raw, replay_arguments(raw)):
                    messages = conversation(replayed)
                    case: dict = {
                        "replayed_arguments": replayed,
                        "request": request_kwargs(GLM_MODEL, messages, policy.max_tokens, policy, fields),
                    }
                    started = time.monotonic()
                    try:
                        completion = await client.complete(messages, policy, fields)
                    except ModelRequestRejected as error:
                        case.update(accepted=False, http_status=error.status, body=error.body)
                    else:
                        case.update(accepted=True, finish_reason=completion.finish_reason, content=completion.content)
                    case["wall_time"] = time.monotonic() - started
                    cases.append(case)
        return cases

    cases = asyncio.run(go())
    write_probe(
        evidence_dir,
        "probe_replay_arguments",
        "replaying malformed or non-object tool-call arguments raw versus wrapped by replay_arguments",
        cases,
    )
    assert [(c["accepted"], c.get("http_status")) for c in cases] == [(False, 400), (True, None)] * 2
