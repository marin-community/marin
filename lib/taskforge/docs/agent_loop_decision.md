# Agent loop decision

Decided 2026-10-05. Four candidates were run live against GLM-5.3 on the interactive tier
(router `vllm-0.28.0-tp8-f8f644d5`). This document ranks them against the goals in
`DESIGN.md` and picks one. Each candidate's full report is in `docs/agent_loop/`. Raw evidence is under
`lib/taskforge/.evidence/agent_loop/`, which is gitignored and exists only on the machine that ran
the evaluation. The evidence links below are relative to this file.

**Decision: a Taskforge-owned loop (`taskforge.llm.agent`) built on the existing
`taskforge.llm.client.GlmClient`. Second choice: vanilla pi (`@earendil-works/pi-coding-agent`).**

## What was checked, and how far to trust it

These claims were checked directly against the raw evidence, not taken from the reports:

- **Web search ran in all four candidates.** Every transcript has tool results with real Parallel
  `search_id`/`extract_id` payloads:
  - pi: `mcp__parallel__web_search` then `mcp__parallel__web_fetch`, with the correct argument
    names on the first call.
  - omp: `write xd://mcp__parallel_web_search`.
  - LangChain: `web_search`/`web_fetch` in 5 of 6 runs.
  - In-house: 4 of 6 runs. Two in-house runs (`T2_rep2`, `T2_rep3`) never searched. They used
    `curl` on the PyPI JSON API through the shell, and the answers were still correct. One
    LangChain run (`T2_rep2`) mixed shell and search.
- **The request bodies show the output budget.**
  - pi sent `max_tokens: 131072` in all 37 logged requests.
  - LangChain sent `max_completion_tokens: 131072` in 32 requests.
  - omp sent `max_completion_tokens: 64000` in the 4 requests made before
    `clampOutputToModelMax`, and 131072 after.
  - The in-house runs record `max_tokens: 131072` in each `summary.json`.
- **T3 output sizes.** pi produced 69,004 tokens (`stop`/`stop`), omp 69,078, LangChain 46,767
  and in-house 40,173. Each was one response, and none was cut. None of the four outputs is valid
  JSON. That is a model defect and does not count against any harness.
- **Continuation was weaker than the reports say.** The in-house forced-length run has the spliced
  text `},\n{"{"id":219` (`inhouse/T3_forced_length_8192/transcript.json`), and every
  continuation came back under `reasoning`. The user-turn continuation in `GlmClient` is also
  lossy at the splice. Its own evidence (`.evidence/llm/c_continue_on_length.json`) joins
  `...decimal number systemeleven — ...` and `...around the world\nnineteen`: the boundary
  punctuation and newline were dropped at 2 of 3 splices.
- **pi Machine redirect.** `pi/T4-bridge/bridge_calls.jsonl` has 18 calls: `run`, `read`, `write`,
  `dump`. The machine dump holds only `/workspace/inventory/items.csv` with `fig,9`.
- **In-house and LangChain ran their shell on the host.** Both used `LocalMachine` from
  `scratch/common/tasks.py`. It implements `shellbox.machine.Machine.run` but runs a host
  subprocess. The Machine seam is the same one a sandbox uses, but neither ran inside a sandbox.
  pi and omp ran in ShellSim through a localhost bridge. No candidate ran against Silo.
- **pi process footprint, measured for this decision.** Four idle `pi --mode rpc` processes each
  had about 135 MB RSS (`ps`, 6 s after start, no model call).

## Comparison

"Machine.run" means shell execution routed into a `shellbox` Machine instead of the host.
"Concurrency" means being driven from Python at hundreds-wide concurrency. No candidate was load
tested; that column shows the process model only.

### The six properties

| Candidate | max_tokens 131072 | Continue on `length` | Shell into Machine.run | Transcript and usage | Pinned install | Python, hundreds wide |
|---|---|---|---|---|---|---|
| **In-house** (httpx loop; becomes `llm.agent` over `GlmClient`) | Yes, sent as `max_tokens` and honored | Built and run live, but not lossless: the forced run corrupted a splice, and continued text arrives in `reasoning`. `GlmClient`'s user-turn variant also drops boundary characters. | Native. The handler is `Callable[[Command], Awaitable[Result]]`. Run live only with a host-subprocess Machine. | Full: messages with replayed `reasoning_content`, plus per-call usage, `cached_tokens` and `finish_reason` | `httpx==0.28.1`, already in the repo `uv.lock`; no new packages | asyncio coroutines in one process. `GlmClient` pools 512 connections. Not load tested. |
| **pi** 1.0.3 | Yes by default, clamped to the remaining context | No. It keeps the output and ends the run. A `turn_end` handler or a driver follow-up would be needed (not built). | Yes, through documented `operations` hooks for bash, read, write and edit, plus a localhost bridge, path mapping and a system-prompt rewrite. Run live on ShellSim. | Yes, through `--mode json`/`rpc` events and session JSONL. Raw bodies need the `before_provider_request` hook. | npm exact version plus a lockfile of 396 packages with sha512. Needs Node 22.19 or later; `node_modules` is 370 MB. | One Node process per agent at about 135 MB idle RSS (measured), plus the RPC driver. Not load tested. |
| **LangChain** `create_agent` 1.4.3 | Yes, sent as `max_completion_tokens` | No. It returns silently with truncated text. Needs middleware (not built). | Yes, through `StructuredTool` over an injected `Machine.run`. Run live only with a host-subprocess Machine. | Partial: GLM `reasoning` is dropped unless two private `ChatOpenAI` methods are overridden | 63 packages hash-pinned, none in the repo `uv.lock`. Needs `--no-config` inside the repo. | asyncio in-process. Not load tested. |
| **omp** 18.1.14 | Only with `compat.clampOutputToModelMax: true`; otherwise silently 64,000 | No. It keeps the output and ends the run. | Bash only. The shim throws for `operations` on edit, write and grep. Its 14.7k-token system prompt describes the host. | Yes through events. The raw `finish_reason` is mapped away, so a proxy was needed. | Release binary pinned by sha256 (135 MB) or npm with Bun 1.3.14 or later. Releases move fast. | One process per agent with about 28k context per turn. Not load tested. |

### Live task results

| Candidate | T1 shell, package and tests | T2 web search | T3 long output |
|---|---|---|---|
| In-house | Works: 3/3 green (14, 10, 8 tests). No fix loop was needed, because the first pytest run passed each time. [T1](../.evidence/agent_loop/inhouse/T1/) | Works: 6/6 correct. Parallel REST search and fetch in 4 runs; 2 runs used `curl` via the shell instead. [T2](../.evidence/agent_loop/inhouse/T2/), [webonly](../.evidence/agent_loop/inhouse/T2_webonly_rep1/) | Works: 40,173 tokens, `stop`. The forced 8k run continued 4 times, corrupted 1 splice, and stopped at the cap. [T3](../.evidence/agent_loop/inhouse/T3/), [forced](../.evidence/agent_loop/inhouse/T3_forced_length_8192/) |
| pi | Works: T1 green after one fix. T1b gave 58 tests after a fix loop. [T1](../.evidence/agent_loop/pi/T1/), [T1b](../.evidence/agent_loop/pi/T1b/), [replay fix](../.evidence/agent_loop/pi/T1-replayfix/). T4 in ShellSim worked: [T4](../.evidence/agent_loop/pi/T4-machine/), [bridge](../.evidence/agent_loop/pi/T4-bridge/). | Works: built-in MCP client, no plugin, correct. [T2](../.evidence/agent_loop/pi/T2/) | Works: 69,004 tokens, `stop`. A forced 4,096 run ended with no continuation. [T3](../.evidence/agent_loop/pi/T3/), [probe](../.evidence/agent_loop/pi/T3-length-probe/) |
| LangChain | Works: 4/4 green; run 1 fixed 2 failing expectations. [T1](../.evidence/agent_loop/langchain/T1/), [ChatGLM subclass](../.evidence/agent_loop/langchain/T1_glm_subclass/) | Works: 6/6 correct through `langchain-mcp-adapters`. [T2](../.evidence/agent_loop/langchain/T2/), [webonly](../.evidence/agent_loop/langchain/T2_webonly_rep1/) | Works: 46,767 tokens, non-streamed (257 s with no bytes). A forced 8k run returned silently with no continuation. [T3](../.evidence/agent_loop/langchain/T3/), [forced](../.evidence/agent_loop/langchain/T3_forced_length_8192/) |
| omp | Works: T1 green on the first run. T1b gave 53 tests after a fix loop. [T1](../.evidence/agent_loop/omp/T1/), [T1b](../.evidence/agent_loop/omp/T1b/). T4 worked for bash only: [T4](../.evidence/agent_loop/omp/T4-machine/). | Works through MCP as `xd://` virtual files. One bad-argument call recovered in the 64k-clamp run. [T2](../.evidence/agent_loop/omp/T2/), [64k](../.evidence/agent_loop/omp/T2-default-64k-clamp/) | Works: 69,078 tokens. A forced 4,096 run ended with no continuation. [T3](../.evidence/agent_loop/omp/T3/), [probe](../.evidence/agent_loop/omp/T3-length-probe/) |

## Answer to the stated concern

The user asked whether a library loop would be fragile or lack web search, and whether LangChain
`create_agent` or vanilla pi with a web-search plugin would reach the goals without added
complexity.

- **Web search does not decide this.** All four candidates searched correctly on GLM-5.3:
  - Vanilla pi needs no plugin at all. Version 1.0.3 has a built-in MCP client, and the Parallel
    server with `exposure: direct` gave plain function tools that GLM called correctly on the
    first try.
  - LangChain needs about 10 lines with `langchain-mcp-adapters`.
  - The in-house loop needs about 20 lines against Parallel's REST API.
  - MCP is also available without new dependencies: `mcp` 2.2.0 is already in the repo
    `uv.lock`.
- **None of the libraries removed the GLM-specific fragility. Each one hid a different piece of
  it behind a default:**
  - pi replays reasoning under a field name GLM's template ignores.
  - LangChain drops reasoning entirely, and ends the run silently on a length cut or a malformed
    tool call.
  - omp silently caps output at 64,000 tokens.
  - None of them continues on `length`.

  Each fix is a hook, a subclass of private methods, or a config flag in someone else's code. The
  in-house loop handles the same quirks in code we test.
- **Neither pi nor LangChain reaches the goals without added complexity.**
  - pi needs a TypeScript extension (reasoning rename, Machine tools, prompt rewrite, length
    continuation), a localhost bridge, a Python RPC driver and a Node runtime: about 350 lines
    plus 370 MB.
  - LangChain needs 63 new packages, overrides of private `ChatOpenAI` methods, and two untested
    middlewares.

## Ranking

1. **In-house loop over `GlmClient`** (`taskforge.llm.agent`).
   - It meets four of the six properties outright: the full output budget, Machine.run natively,
     full transcript and usage, and a pinned install with no new dependency. It is in-process
     asyncio, which is the cheapest shape for hundreds-wide concurrency.
   - Most of the hard part already exists and is live-validated in `taskforge.llm.client`:
     streaming, stall timeout, retries and pool holds on `rigging.timing`, a 512-connection pool,
     continuation, typed `Completion`/`Attempt`/`Usage`, and raw SSE events for the ledger. The
     agent loop adds only a turn loop, tool dispatch and reasoning replay, about 150 lines.
   - Agent calls share the same retry, hold and ledger semantics as every other Taskforge model
     call. With a subprocess agent, the agent's model calls would bypass them.
   - Every GLM quirk is in our code and covered by our tests: `reasoning_content` replay,
     index-keyed tool-call deltas, mid-stream error events, malformed arguments returned to the
     model as an error, and continuation.
2. **pi.** This is the best library candidate. It worked on every task, it redirected all four
   tools into ShellSim, and it searches with no plugin. Its costs are listed in the next section.
3. **LangChain `create_agent`.** It works, but stock `ChatOpenAI` loses GLM reasoning and its
   cross-turn prefix-cache benefit. Fixing that means overriding private methods, and three more
   for streaming. It ends the run silently on `length` and on invalid tool calls. It adds a
   63-package closure outside the repo lock, and it is non-streaming by default, which makes the
   read timeout a wall-clock cap. Nothing about it is easier than ranks 1 or 2.
4. **omp.** It silently caps output at 64k unless a compat flag is set. Only bash can be redirected,
   and its native file tools always touch the host. Its 14.7k-token system prompt describes the
   host OS, and it misled the model inside the Machine. It is a 135 MB binary on a fast release
   cadence whose defaults change between versions. It is the weakest fit for the sandbox contract.

## What the second choice (pi) would cost

- **Runtime:** Node 22.19 or later in every worker image, plus a `package.json` and
  `package-lock.json` (396 packages, 370 MB `node_modules`) maintained alongside `uv.lock`.
- **Code:** about 350 lines across two languages:
  - A TypeScript extension of about 100 lines: Machine-backed bash, read, write and edit;
    `before_provider_request` renaming `reasoning` to `reasoning_content`; a `before_agent_start`
    rewrite of the host-cwd system prompt; and a length-continuation handler, which is not built
    or tested.
  - A Python localhost bridge serving `Machine.run`, `upload` and `download` (about 85 lines).
  - A Python driver over `pi --mode rpc` (about 160 lines).
- **Concurrency:** about 135 MB idle RSS per agent process, so roughly 40 GB at 300 concurrent
  agents before any work. There is one subprocess and one bridge route per agent. pi's HTTP layer
  has its own timeout (`httpIdleTimeoutMs`) and retries, so agent calls would not go through
  `GlmClient`'s holds, retries or attempt records. Ledger capture would come from pi events and a
  request hook, not from `Attempt.events`.
- **Upgrades:** the package was renamed once already (`@mariozechner` to `@earendil-works`, May
  2026). The reasoning replay depends on internal behavior that only a hook works around.
- **What it gives in return:** ready-made read, write and edit tools, which helped fix loops in
  T1b, and MCP support with no code. The winner can add both cheaply: a `write_file` tool over
  `Machine.upload`, and `mcp` from the existing lock if MCP is ever needed.

## Integration shape for the winner

**Module.** `taskforge/llm/agent.py` owns the loop. It sits in `llm` and not in `build`, because
`build/author.py` (the builder agent), `validate/solver.py` and `validate/adversary.py` all need
it, and stage packages never import each other. `taskforge/llm/web.py` holds the Parallel tools.
Neither module imports `taskforge.sandbox`. The shell tool depends only on the external
`shellbox.machine.Machine` protocol, so Silo, ShellSim and the fake are interchangeable.

```python
# taskforge/llm/agent.py
@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: Mapping[str, object]               # JSON schema
    handler: Callable[[Mapping[str, object]], Awaitable[str]]

class AgentStop(StrEnum):
    ANSWERED = "answered"          # finish_reason stop, no tool calls
    MAX_TURNS = "max_turns"
    LENGTH = "length"              # continuations exhausted, partial output kept

@dataclass(frozen=True)
class AgentTurn:
    completion: Completion                         # usage, finish_reason, reasoning, Attempt.events
    tool_results: tuple[ToolResult, ...]           # call id, name, arguments, output, error flag, wall time

@dataclass(frozen=True)
class AgentRun:
    messages: tuple[Message, ...]                  # as replayed, including reasoning_content
    turns: tuple[AgentTurn, ...]
    stop: AgentStop
    usage: Usage

async def run_agent(client: GlmClient, policy: LLMPolicy, messages: Sequence[Message],
                    tools: Sequence[AgentTool], max_turns: int) -> AgentRun: ...

def shell_tool(machine: Machine, *, timeout: float, output_limit_bytes: int) -> AgentTool: ...
# taskforge/llm/web.py
def web_tools(api_key: str) -> tuple[AgentTool, AgentTool]: ...   # Parallel /v1/search, /v1/extract
```

**Hooks and seams it uses:**

- `GlmClient.complete(messages, policy, request_fields={"tools": ...})` for each turn. This brings
  streaming, the stall timeout, retries, pool holds, `max_tokens` at 131,072 clamped to the
  remaining context, and continuation on `length`.
- Assistant messages are replayed with `reasoning_content`. GLM's template ignores `reasoning`,
  and replaying it raised turn-2 cached tokens from 704 to 768–1,344 in
  `scratch/probe_prefix_cache_reasoning_replay.json`.
- Tool-argument JSON that fails to parse goes back to the model as an error tool result; it does
  not end the run. Exceptions raised by a tool handler propagate. The `validate` classifier
  turns them into a typed `Cause`.
- `shellbox.machine.Machine.run(Command)` is the only execution path, and `Machine.upload` backs a
  file-write tool if one is added. The system prompt names `/workspace` in the Machine, never a
  host path.
- The ledger records each `AgentTurn`: `Completion.attempts` with raw SSE events, usage and
  finish reasons, and tool calls with their outputs. These go through `taskforge.ledger`; the loop
  itself does no I/O.
- Concurrency comes from the caller's `asyncio` gather over one shared `GlmClient`, with 512
  pooled connections. There is no per-agent process.

**Work needed before relying on it:**

1. Make continuation lossless. `GlmClient`'s user-turn continuation drops boundary characters
   (`.evidence/llm/c_continue_on_length.json`), and `continue_final_message` corrupted a splice
   and returned text under `reasoning`. Pick one method, test it on a 40k-token structured
   output, and validate the joined output before it is used. Structured outputs already need a
   parse-and-repair step, because all four T3 outputs were invalid JSON.
2. Decide what happens on a `length` cut inside a tool call. `GlmClient` stops continuing when
   the stream carries tool names, so truncated arguments currently reach the loop as malformed
   JSON.
3. Run the loop live on the paths not yet exercised: a T1b-style fix loop, a run inside a real
   Machine (ShellSim now, Silo when a broker is up), and a concurrency run of at least 100 agents.

## Not achieved by any candidate

- **Lossless continue-on-length.** pi, LangChain and omp do not continue at all. The in-house
  `continue_final_message` corrupted 1 of 4 splices. `GlmClient`'s user-turn continuation dropped
  characters at 2 of 3 splices on a 150-token toy. No candidate continued a truncated tool call.
- **Hundreds-wide concurrency.** Every run was a single agent. No candidate was load tested. The
  only data point is pi's idle RSS of about 135 MB per process.
- **Execution in a production sandbox.** No candidate ran against Silo (no broker is running;
  see `STATUS.md`). In-house and LangChain ran their shell tool only through a host-subprocess
  `Machine`.
- **Malformed tool calls from GLM.** None were observed in any candidate, so every candidate's
  handling of them is untested against the real model. LangChain's silent stop was reproduced
  only with a fake model.
- **A real stream stall.** No candidate's stall timeout was triggered.
- **Valid long structured output.** All four T3 outputs had 400 items and were not cut, but none
  parsed as JSON. That is a model defect, but the pipeline has to validate and repair whichever
  loop it uses.
