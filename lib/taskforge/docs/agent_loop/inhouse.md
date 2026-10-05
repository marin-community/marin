# Agent loop candidate D: minimal in-house loop

Evaluated 2026-10-05 against GLM-5.3 on the interactive tier (router `vllm-0.28.0-tp8-f8f644d5`).
Raw evidence: `lib/taskforge/.evidence/agent_loop/inhouse/` (per run: `summary.json` with per-call
usage, finish reason and wall time; `transcript.json` with every message including replayed
reasoning; `machine_commands.jsonl` with every command the shell tool executed; the agent's
`workspace/`). Loop: `.evidence/agent_loop/scratch/inhouse/loop.py` (184 lines), driver `run.py`;
shared prompts and `LocalMachine` in `scratch/common/tasks.py`.

## Result

All three tasks completed, with the same outcomes as LangChain at similar wall times. The loop
streams over httpx, keeps GLM reasoning and replays it, returns malformed tool-call arguments to the
model as an error instead of stopping, and continues on `finish_reason == "length"`. Continuation
is mechanically sound and nearly fully prefix-cached, but forced truncation showed two GLM/vLLM
problems that any continuing loop must handle: continued text comes back in `reasoning`, and one
splice duplicated a token boundary (`{"{"id":219`). Only dependency: `httpx` (already in the repo
lock) plus `shellbox.machine`.

## Reusing `shellbox.agent.BashAgent`

Not reusable outside Harbor. `shellbox/agent.py` imports `harbor.agents.base`, and `harbor` is in
neither shellbox's dependencies nor the root environment
(`uv run --with-editable lib/shellbox python -c "import shellbox.agent"` gives
`ModuleNotFoundError: No module named 'harbor'`). It also drives a persistent
`BashSessionProvider.open_bash_session()` shell, not `Machine.run`. It is non-streaming, sends no
`max_tokens`, has no retries, crashes on malformed tool arguments (`json.loads`), and drops
reasoning. The loop here copies its shape (messages list, `tool_calls` dispatch, usage tally) and
nothing else.

## Tasks

| Task | Runs | Result | Wall time | Notes |
|---|---|---|---|---|
| T1 shell, package + tests | 3 | works, 3/3 green (14, 10, 8 tests), verified by rerunning `uv run pytest` | 7.8–8.3 s | 2–3 turns, 1–2 shell calls, no failures to fix. |
| T2 web search (shell + web tools) | 3 | works, 3/3 correct | 3.1–6.5 s | Run 1 called `web_fetch` and `web_search`. Runs 2 and 3 never searched: they ran `curl` on the PyPI JSON API through the shell. |
| T2 web search (web tools only) | 3 | works, 3/3 correct | 5.2–6.7 s | Every run called `web_search` then `web_fetch`; tool results are real Parallel `search_id`/`extract_id` responses. |
| T3 long output | 1 | works (harness), model JSON has one escaping error | 204 s | 40,173 output tokens (960 reasoning), `finish_reason: stop`, 400 items, no cut. |
| T3 forced `max_tokens=8192` | 1 | partial | 214 s | 1 call plus 4 continuations, each `length`; stopped at the continuation cap with 339 items; one splice corrupted (below). |

Ground truth for T2 (PyPI JSON, `scratch/shellsim_ground_truth.txt`): 0.1.25 on 2026-10-01,
0.1.22 and 0.1.21 on 2026-09-29. Every answer matched.

T3 JSON validity: does not parse (`Expecting ',' delimiter` at char 77,022, an unescaped `"` inside
a `command` string). Same class of model error as the LangChain run.

## Fragility handled, exactly

- **Tool-call parsing.** Streamed `tool_calls` arrive as deltas keyed by `index`; `id` and `name`
  come once, `arguments` arrive in fragments. Accumulated per index. Bad argument JSON is returned to
  the model as `error: tool arguments are not valid JSON (...)`. GLM produced none in these runs.
- **Reasoning field.** Streamed as `delta.reasoning`. It must be replayed as `reasoning_content`:
  the GLM template ignores `reasoning` (prompt tokens 176 with `reasoning`, 176 with none, 193 with
  `reasoning_content`; `scratch/probe_replay_and_continue.json`). Replaying it raises the turn-2
  cache hit from 704 to 768–1,344 tokens (`scratch/probe_prefix_cache_reasoning_replay.json`).
- **Mid-stream errors.** vLLM can send an `error` event with HTTP 200; the loop raises on it. Not
  triggered in these runs.
- **Retries.** Transport errors and 429/5xx retried with exponential backoff, 4 attempts. Not
  triggered. In Taskforge this should be `rigging.timing.retry_with_backoff`.
- **Timeouts.** httpx `read=600` is a per-chunk stall timeout on the stream, not a wall-clock cap;
  T3 streamed for 204 s without hitting it. A real stall was not tested.
- **Truncation.** On `length` with no tool calls, the loop resends with the partial assistant
  message, `continue_final_message: true`, `add_generation_prompt: false`. Two findings:
  1. vLLM's GLM reasoning parser puts the continued text in `reasoning`, with `content: null`
     (`continued_in_reasoning: true` on all 4 continuations). The loop has to read both.
  2. One of 4 splices was corrupted: the joined text has `},\n{"{"id":219`. The token boundary was
     not continued cleanly. Continuations 2–4 hit 99% of the prefix cache (16,064 of 16,174
     tokens, and so on); continuation 1 missed entirely.
  So continuation keeps the output, as DESIGN.md requires, but a continued structured document
  still needs validation, and continuation should not be trusted to be lossless.

## Capability questions

1. **max_tokens 131072.** Sent as `max_tokens`; honored (T3 reached 40,173 and stopped naturally;
   the forced run stopped at exactly 8,192 per call). The server accepts 200,000 and rejects
   262,144 as exceeding context.
2. **Shell redirection.** Native: the shell handler is a function of an injected
   `Callable[[Command], Awaitable[Result]]`, so `machine.run` is passed directly. About 10 lines;
   `machine_commands.jsonl` records every command.
3. **Transcript, usage, finish reasons.** Full: the message list (with reasoning) plus one record
   per HTTP call with usage, `finish_reason`, wall time, and continuation flags.
4. **Pinning.** Nothing to pin beyond `httpx==0.28.1`, already hash-locked in the repo `uv.lock`.
   Web search uses the Parallel REST API (`/v1/search`, `/v1/extract`) directly. `/v1/search`
   rejects `max_results`. MCP would need about 40 more lines of JSON-RPC, or the `mcp` package.
5. **GLM quirks.** As above; plus `reasoning_effort` is accepted and thinking stays on at every
   level. Turn-2 prefix-cache misses after large tool results appear in both candidates
   (`scratch/prefix_cache_table.txt`), so they look like router placement.
6. **Taskforge integration size.** About 250 lines in `llm/` or `loop/`: this loop (184) moved onto
   `rigging.timing` retries, typed turn and transcript records in place of dicts, continuation that
   validates splices, plus the two web tools. No third-party additions.

## Verdict

Works for all three tasks, with web search included, in 184 lines on one existing dependency. The
GLM fragilities (reasoning replay, streamed tool-call assembly, continuation landing in
`reasoning`, splice corruption) are visible and testable in our own code instead of behind
private-method overrides. Continuation is the one part that needs more work before it is trusted.
