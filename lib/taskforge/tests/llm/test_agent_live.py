# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of taskforge.llm.agent and taskforge.llm.web against GLM-5.3 (interactive tier).

Each check writes ``lib/taskforge/.evidence/llm/agent/<check>-<utc>.json``: the purpose, the policy,
the full ``AgentRun`` (every message, every completion with its raw stream events, every tool
result), a summary (turns, tool calls, tokens, wall time) and the check's own observations. Ledger
spans go to ``.evidence/llm/agent/ledger/<check>/<item>.jsonl``. The web check also needs the
Parallel key file (the ``parallel_key`` fixture in ``tests/conftest.py``).
"""

import asyncio
import json
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter
from shellbox.backends.shellsim.machine import ShellSimMachine, ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, ShellSimBuiltins

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.agent import (
    AgentLedger,
    AgentRun,
    AgentStop,
    AgentTool,
    ToolOutcome,
    replay_arguments,
    run_agent,
    shell_tool,
)
from taskforge.llm.client import FinishReason, GlmClient, GlmEndpoint, GlmRequestRejected, Pool, request_body
from taskforge.llm.policy import LLMPolicy, Message, ReasoningEffort
from taskforge.llm.web import web_tools

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / ".evidence" / "llm" / "agent"
PYPI_PACKAGE_URL = "https://pypi.org/pypi/shellsim/json"
VERSION_PATTERN = re.compile(r"\b\d+\.\d+\.\d+\b")
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


def endpoint(glm_settings, base_url: str | None = None) -> GlmEndpoint:
    return GlmEndpoint(base_url=base_url or glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)


def ledger_record(check: str, item_id: str) -> AgentLedger:
    return AgentLedger(ledger=JsonlLedger(EVIDENCE_DIR / "ledger" / check), item_id=item_id, round=0, step=check)


async def shellsim() -> ShellSimMachine:
    return await ShellSimMachineFactory().create(MachineSpec(source=ShellSimBuiltins()))


def summary(run: AgentRun, wall_time: float) -> dict[str, object]:
    return {
        "stop": run.stop,
        "turns": len(run.turns),
        "wall_time": wall_time,
        "usage": asdict(run.usage),
        "per_turn": [
            {
                "finish_reason": turn.completion.finish_reason,
                "completion_tokens": turn.completion.usage.completion_tokens,
                "cached_tokens": turn.completion.usage.cached_tokens,
                "wall_time": turn.completion.wall_time,
                "attempts": [(a.segment, a.outcome, a.max_tokens, a.http_status) for a in turn.completion.attempts],
                "tool_calls": [
                    {
                        "name": r.call.name,
                        "arguments": r.call.arguments[:500],
                        "outcome": r.outcome,
                        "output": r.output[:500],
                        "wall_time": r.wall_time,
                    }
                    for r in turn.tool_results
                ],
            }
            for turn in run.turns
        ],
    }


def record(check: str, purpose: str, policy: LLMPolicy, runs: Sequence[AgentRun], **extra) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {
        "check": check,
        "purpose": purpose,
        "policy": asdict(policy),
        **extra,
        "runs": RUNS.dump_python(tuple(runs), mode="json"),
    }
    path = EVIDENCE_DIR / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    return path


def executed(run: AgentRun, name: str) -> list[str]:
    return [
        r.output for t in run.turns for r in t.tool_results if r.call.name == name and r.outcome is ToolOutcome.EXECUTED
    ]


async def timed_agent(
    client: GlmClient, policy: LLMPolicy, messages: list[Message], tools: Sequence[AgentTool], max_turns: int, rec
) -> tuple[AgentRun, float]:
    started = time.monotonic()
    run = await run_agent(client, policy, messages, tools, max_turns, rec)
    return run, time.monotonic() - started


T1_TASK = (
    "In /workspace, write a small Python package `textstats` with two modules: `tokens.py` with a function "
    "`tokenize(text)` that splits text into lowercase word tokens, ignoring punctuation; and `stats.py` with "
    "`frequencies(tokens)` returning word counts and `top_k(tokens, k)` returning the k most common words, ties "
    "broken alphabetically. Write tests/test_textstats.py with at least 8 tests covering both modules, including "
    "edge cases. Run the tests and fix failures (in code or tests) until they all pass. Report the final pytest "
    "output."
)


@pytest.mark.live_glm
@pytest.mark.timeout(3600)
def test_t1_package_and_tests_in_shellsim(glm_settings):
    policy = LLMPolicy()
    messages: list[Message] = [
        {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
        {"role": "user", "content": T1_TASK},
    ]

    async def go() -> tuple[AgentRun, float, dict]:
        machine = await shellsim()
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        async with GlmClient(endpoint(glm_settings)) as client:
            run, wall = await timed_agent(client, policy, messages, [tool], 60, ledger_record("t1_shellsim", "t1"))
        check = await machine.run(Command(argv=("sh", "-c", "pytest -v tests/test_textstats.py"), timeout=SHELL_TIMEOUT))
        await machine.close()
        return (
            run,
            wall,
            {"exit_code": check.exit_code, "stdout": check.stdout.decode(), "stderr": check.stderr.decode()},
        )

    run, wall, rerun = asyncio.run(go())
    pytest_outputs = [json.loads(o) for o in executed(run, "shell")]
    failing_runs = sum(1 for o in pytest_outputs if "FAILED" in o["stdout"] or "Error" in o["stderr"])
    record(
        "t1_shellsim",
        "T1: package plus tests in a ShellSim machine, fix until green; independent pytest rerun afterwards",
        policy,
        [run],
        summary=summary(run, wall),
        independent_rerun=rerun,
        shell_outputs_with_failures=failing_runs,
    )
    assert run.stop is AgentStop.ANSWERED
    assert rerun["exit_code"] == 0
    assert rerun["stdout"].count("PASSED") >= 8


BUGGY_TEXTSTATS = """mkdir -p textstats tests
cat > textstats/__init__.py <<'EOF'
EOF
cat > textstats/tokens.py <<'EOF'
def tokenize(text):
    return [word.strip(".,!?;:") for word in text.split(" ")]
EOF
cat > textstats/stats.py <<'EOF'
def frequencies(tokens):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def top_k(tokens, k):
    counts = frequencies(tokens)
    return sorted(counts, key=lambda word: counts[word], reverse=True)[:k]
EOF
cat > tests/test_textstats.py <<'EOF'
from textstats.stats import frequencies, top_k
from textstats.tokens import tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! hello") == ["hello", "world", "hello"]


def test_tokenize_collapses_whitespace():
    assert tokenize("  a\\tb\\n c ") == ["a", "b", "c"]


def test_top_k_breaks_ties_alphabetically():
    assert top_k(["b", "a", "c", "a", "b"], 2) == ["a", "b"]


def test_frequencies():
    assert frequencies(["x", "y", "x"]) == {"x": 2, "y": 1}
EOF
"""
T1B_TASK = (
    "/workspace holds a Python package `textstats` and tests/test_textstats.py. Some tests fail. Run the tests, "
    "fix the package (not the existing tests) until every test passes, then add at least 4 more tests of your "
    "own for edge cases and make sure the whole file passes. Report the final pytest output."
)


@pytest.mark.live_glm
@pytest.mark.timeout(3600)
def test_t1b_fix_loop_in_shellsim(glm_settings):
    policy = LLMPolicy()
    messages: list[Message] = [
        {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
        {"role": "user", "content": T1B_TASK},
    ]
    pytest_command = Command(argv=("sh", "-c", "pytest -v tests/test_textstats.py"), timeout=SHELL_TIMEOUT)

    async def go() -> tuple[AgentRun, float, dict, str]:
        machine = await shellsim()
        seeded = await machine.run(Command(argv=("sh", "-c", BUGGY_TEXTSTATS), timeout=SHELL_TIMEOUT))
        assert seeded.exit_code == 0, seeded
        before = await machine.run(pytest_command)
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        async with GlmClient(endpoint(glm_settings)) as client:
            run, wall = await timed_agent(client, policy, messages, [tool], 60, ledger_record("t1b_fix_loop", "t1b"))
        check = await machine.run(pytest_command)
        await machine.close()
        rerun = {"exit_code": check.exit_code, "stdout": check.stdout.decode(), "stderr": check.stderr.decode()}
        return run, wall, rerun, before.stdout.decode()

    run, wall, rerun, before = asyncio.run(go())
    record(
        "t1b_fix_loop",
        "T1b: seeded failing package in ShellSim; the agent must run, fix, extend and rerun the tests",
        policy,
        [run],
        summary=summary(run, wall),
        seeded_pytest=before,
        independent_rerun=rerun,
        shell_calls=len(executed(run, "shell")),
    )
    assert before.count("FAILED") >= 2
    assert run.stop is AgentStop.ANSWERED
    assert rerun["exit_code"] == 0
    assert rerun["stdout"].count("PASSED") >= 8


def latest_releases(http: httpx.Client, count: int) -> list[str]:
    """The ``count`` most recently uploaded shellsim versions, from PyPI's JSON API."""
    releases = http.get(PYPI_PACKAGE_URL).raise_for_status().json()["releases"]
    uploaded = {version: max(f["upload_time_iso_8601"] for f in files) for version, files in releases.items() if files}
    return sorted(uploaded, key=uploaded.__getitem__, reverse=True)[:count]


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_web_search_through_parallel(glm_settings, parallel_key):
    policy = LLMPolicy()
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Find the three most recent releases of the shellsim package on PyPI and the date of "
                "each. Start with web_search to locate sources, then fetch what you need. Cite the URLs you used."
            ),
        },
    ]

    async def go() -> tuple[AgentRun, float]:
        async with GlmClient(endpoint(glm_settings)) as client, httpx.AsyncClient() as http:
            tools = web_tools(http, parallel_key.value)
            return await timed_agent(client, policy, messages, tools, 20, ledger_record("web", "web"))

    # Releases can land while the agent runs, so its answer may match PyPI before or after the run.
    with httpx.Client(timeout=60.0) as http:
        before = latest_releases(http, 3)
        run, wall = asyncio.run(go())
        after = latest_releases(http, 3)
    searches = executed(run, "web_search")
    answer = str(run.messages[-1]["content"])
    answered = set(VERSION_PATTERN.findall(answer))
    record(
        "web_parallel",
        "web search through Parallel /v1/search and /v1/extract; transcript must show search results",
        policy,
        [run],
        summary=summary(run, wall),
        search_ids=[json.loads(s).get("search_id") for s in searches],
        extract_ids=[json.loads(f).get("extract_id") for f in executed(run, "web_fetch")],
        answer=answer,
        pypi_latest_three={"before_run": before, "after_run": after},
    )
    assert run.stop is AgentStop.ANSWERED
    assert searches and all(json.loads(s)["search_id"] for s in searches)
    assert set(before) <= answered or set(after) <= answered, (answered, before, after)


@pytest.fixture
def corrupting_proxy(glm_settings) -> Iterator[dict]:
    """A relay to the real router that truncates the arguments of the first tool call it sees."""
    state: dict = {"original_first_fragment": None, "done": False, "requests": 0}
    upstream = glm_settings.base_url

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"] += 1
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as http:
                with http.stream(
                    "POST",
                    f"{upstream}/chat/completions",
                    content=body,
                    headers={"Authorization": self.headers["Authorization"], "Content-Type": "application/json"},
                ) as response:
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", response.headers.get("content-type", "text/event-stream"))
                    self.end_headers()
                    for line in response.iter_lines():
                        self.wfile.write((self._rewrite(line) + "\n").encode())
                        self.wfile.flush()

        def _rewrite(self, line: str) -> str:
            data = line.removeprefix("data:").strip()
            if not line.startswith("data:") or data == "[DONE]" or state["done"]:
                return line
            event = json.loads(data)
            for choice in event.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls") or []:
                    function = call.get("function", {})
                    if "arguments" not in function:
                        continue
                    if state["original_first_fragment"] is None:
                        state["original_first_fragment"] = function["arguments"]
                        function["arguments"] = MALFORMED_ARGUMENTS
                    else:
                        function["arguments"] = ""
            if state["original_first_fragment"] is not None and any(
                c.get("finish_reason") for c in event.get("choices", [])
            ):
                state["done"] = True
            return f"data: {json.dumps(event)}"

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
    server.server_close()


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_malformed_tool_call_is_returned_to_the_model_and_recovered(glm_settings, corrupting_proxy):
    policy = LLMPolicy()
    messages: list[Message] = [
        {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
        {"role": "user", "content": "Create /workspace/hello.txt containing the word hello, then show its contents."},
    ]

    async def go() -> tuple[AgentRun, float, str]:
        machine = await shellsim()
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        async with GlmClient(endpoint(glm_settings, corrupting_proxy["base_url"])) as client:
            run, wall = await timed_agent(client, policy, messages, [tool], 20, ledger_record("malformed", "malformed"))
        content = await machine.run(Command(argv=("cat", "/workspace/hello.txt"), timeout=SHELL_TIMEOUT))
        await machine.close()
        return run, wall, content.stdout.decode()

    run, wall, content = asyncio.run(go())
    first = run.turns[0].tool_results[0]
    record(
        "malformed_tool_call",
        "a relay truncates the first tool call's arguments; the loop returns an error result, the replay is "
        "accepted by the real server, and the model retries",
        policy,
        [run],
        summary=summary(run, wall),
        proxy={k: v for k, v in corrupting_proxy.items() if k != "base_url"},
        hello_txt=content,
    )
    assert first.outcome is ToolOutcome.INVALID_ARGUMENTS
    assert run.stop is AgentStop.ANSWERED
    assert content.strip() == "hello"


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_twenty_concurrent_agents_share_one_client(glm_settings):
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

    async def one(client: GlmClient, n: int) -> tuple[AgentRun, float, str]:
        machine = await shellsim()
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        run, wall = await timed_agent(client, policy, messages(n), [tool], 20, ledger_record("concurrent", f"agent-{n}"))
        content = await machine.run(Command(argv=("cat", "/workspace/answer.txt"), timeout=SHELL_TIMEOUT))
        await machine.close()
        return run, wall, content.stdout.decode().strip()

    async def go() -> tuple[list, float]:
        started = time.monotonic()
        async with GlmClient(endpoint(glm_settings)) as client:
            results = await asyncio.gather(*(one(client, n) for n in range(agents)), return_exceptions=True)
        return results, time.monotonic() - started

    results, total = asyncio.run(go())
    per_agent = [
        (
            {"agent": n, "error": repr(r)}
            if isinstance(r, BaseException)
            else {
                "agent": n,
                "stop": r[0].stop,
                "turns": len(r[0].turns),
                "wall_time": r[1],
                "completion_tokens": r[0].usage.completion_tokens,
                "file": r[2],
                "expected": str(n * n + 7),
            }
        )
        for n, r in enumerate(results)
    ]
    runs = [r[0] for r in results if not isinstance(r, BaseException)]
    record(
        "concurrent_20",
        f"{agents} agents gathered over one GlmClient, each with its own ShellSim machine",
        policy,
        runs,
        total_wall_time=total,
        per_agent=per_agent,
    )
    assert [a for a in per_agent if "error" in a] == []
    assert [a["agent"] for a in per_agent if a["file"] != a["expected"]] == []


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_length_cut_inside_a_tool_call_is_returned_and_retried(glm_settings):
    policy = LLMPolicy(max_tokens=400, reasoning_effort=ReasoningEffort.LOW)
    messages: list[Message] = [
        {"role": "system", "content": f"{SYSTEM}\n{SHELLSIM_NOTE}"},
        {
            "role": "user",
            "content": (
                "Create /workspace/numbers.txt holding the integers 1 to 300, one per line, written out "
                "literally in a heredoc (no seq, no loops). Then run wc -l on it."
            ),
        },
    ]

    async def go() -> tuple[AgentRun, float, str]:
        machine = await shellsim()
        tool = shell_tool(machine, timeout=SHELL_TIMEOUT, output_limit_bytes=SHELL_OUTPUT_LIMIT)
        async with GlmClient(endpoint(glm_settings)) as client:
            run, wall = await timed_agent(client, policy, messages, [tool], 30, ledger_record("length_cut", "cut"))
        lines = await machine.run(Command(argv=("sh", "-c", "wc -l < /workspace/numbers.txt"), timeout=SHELL_TIMEOUT))
        await machine.close()
        return run, wall, lines.stdout.decode().strip()

    run, wall, lines = asyncio.run(go())
    outcomes = [r.outcome for t in run.turns for r in t.tool_results]
    record(
        "length_cut_tool_call",
        "max_tokens=400 cuts a large heredoc tool call; the cut call is returned as an error and the model "
        "re-issues the work in smaller calls",
        policy,
        [run],
        summary=summary(run, wall),
        outcomes=outcomes,
        numbers_txt_lines=lines,
    )
    assert ToolOutcome.TRUNCATED_CALL in outcomes
    assert outcomes.index(ToolOutcome.TRUNCATED_CALL) < len(outcomes) - 1
    assert run.stop is AgentStop.ANSWERED
    assert lines == "300"


def write_probe(check: str, purpose: str, cases: list[dict]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{check}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps({"check": check, "purpose": purpose, "cases": cases}, indent=2) + "\n")
    return path


@pytest.mark.live_glm
@pytest.mark.timeout(1800)
def test_probe_tool_call_cut_by_max_tokens_is_reported_as_length(glm_settings):
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
        async with GlmClient(endpoint(glm_settings)) as client:
            started = time.monotonic()
            completion = await client.complete(messages, policy, fields)
            wall = time.monotonic() - started
        body = request_body(client.endpoint.model, messages, policy.max_tokens, policy, fields)
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
def test_probe_replayed_tool_call_arguments_must_be_a_json_object(glm_settings):
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
        async with GlmClient(endpoint(glm_settings)) as client:
            for raw in (MALFORMED_ARGUMENTS, "[1]"):
                for replayed in (raw, replay_arguments(raw)):
                    messages = conversation(replayed)
                    case: dict = {
                        "replayed_arguments": replayed,
                        "request": request_body(client.endpoint.model, messages, policy.max_tokens, policy, fields),
                    }
                    started = time.monotonic()
                    try:
                        completion = await client.complete(messages, policy, fields)
                    except GlmRequestRejected as error:
                        case.update(accepted=False, http_status=error.status, body=error.body)
                    else:
                        case.update(accepted=True, finish_reason=completion.finish_reason, content=completion.content)
                    case["wall_time"] = time.monotonic() - started
                    cases.append(case)
        return cases

    cases = asyncio.run(go())
    write_probe(
        "probe_replay_arguments",
        "replaying malformed or non-object tool-call arguments raw versus wrapped by replay_arguments",
        cases,
    )
    assert [(c["accepted"], c.get("http_status")) for c in cases] == [(False, 400), (True, None)] * 2
