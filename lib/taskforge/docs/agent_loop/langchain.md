# Agent loop candidate C: LangChain `create_agent`

Evaluated 2026-10-05 against GLM-5.3 on the interactive tier (router `vllm-0.28.0-tp8-f8f644d5`).
Raw evidence: `lib/taskforge/.evidence/agent_loop/langchain/` (per run: `summary.json`,
`transcript.json`, `http.jsonl` with every chat-completions request and response, auth stripped,
`machine_commands.jsonl` with every command the shell tool executed, and the agent's `workspace/`).
Harness: `.evidence/agent_loop/scratch/lc/lc_agent.py`, `chat_glm.py`; shared prompts and the
`LocalMachine` in `scratch/common/tasks.py`.

## Result

All three tasks completed. The loop works, the shell tool redirects cleanly through an injected
`Machine.run`, and Parallel web search via `langchain-mcp-adapters` ran and grounded correct
answers. The costs are GLM-specific: stock `ChatOpenAI` discards GLM's `reasoning` field and never
replays it, `finish_reason == "length"` ends the run silently with no continuation, and a tool call
with malformed JSON arguments ends the run silently. Fixing the first takes overrides of two
private `ChatOpenAI` methods; the other two take middleware. The dependency closure is 63 packages.

## Setup

- `langchain==1.4.3`, `langchain-openai==1.6.7`, `langchain-mcp-adapters==0.3.2` (pulls
  `langgraph==1.2.12`, `langchain-core==1.6.6`, `openai==3.24.0`, `mcp==1.30.0`).
- `ChatOpenAI(model="glm-5.3", base_url=..., max_tokens=131072, reasoning_effort="high")`.
- Tools: `shell` (a `StructuredTool` whose coroutine calls an injected
  `Callable[[shellbox.machine.Command], Awaitable[Result]]`), plus `web_search` and `web_fetch`
  from the Parallel MCP server (`https://search.parallel.ai/mcp?mode=advanced`, streamable HTTP).

## Tasks

| Task | Runs | Result | Wall time | Notes |
|---|---|---|---|---|
| T1 shell, package + tests | 4 | works, 4/4 green (13, 12, 10, 10 tests), verified by rerunning `uv run pytest` | 7.8–10.6 s | 2–6 turns, 1–5 shell calls, 0 tool errors. Run 1 fixed 2 failing test expectations. |
| T2 web search (shell + web tools) | 3 | works, 3/3 correct | 5.2–6.8 s | 2 runs called `web_search`/`web_fetch`; 1 run used `curl` on the PyPI JSON API through the shell plus one `web_search`. |
| T2 web search (web tools only) | 3 | works, 3/3 correct | 6.1–9.7 s | Every run called `web_search` and `web_fetch`; tool results are real Parallel responses with the release dates in the excerpts. |
| T3 long output | 1 | works (harness), model JSON has one escaping error | 257 s | 46,767 output tokens (768 reasoning), `finish_reason: stop`, 400 items, no cut. |
| T3 forced `max_tokens=8192` | 1 | partial | 40 s | `finish_reason: length`; agent returned normally with the truncated text and no continuation. |

Ground truth for T2 (PyPI JSON, `scratch/shellsim_ground_truth.txt`): 0.1.25 on 2026-10-01,
0.1.22 and 0.1.21 on 2026-09-29. Every answer matched.

T3 JSON validity: the text does not parse (`Expecting ',' delimiter` at char 18,758, an unescaped
quote inside a `command` string). That is the model, not the harness; the in-house loop saw the same
class of error.

## Capability questions

1. **max_tokens 131072.** Settable and honored. `ChatOpenAI` sends it as `max_completion_tokens`
   (see `http.jsonl`); vLLM honors that name (`scratch/probe_replay_and_continue.json`:
   `max_completion_tokens=40` returns 40 tokens, `finish_reason: length`). The server does not
   clamp at 131,072: 200,000 was accepted; 262,144 was rejected as exceeding context
   (`scratch/probe_endpoint.json`, `probe_reasoning_effort.json`).
2. **Shell redirection.** Yes. The tool is a `StructuredTool.from_function(coroutine=...)` over an
   injected `Machine.run`; about 15 lines. `machine_commands.jsonl` shows every T1 command went
   through it. No framework hook is needed.
3. **Transcript, usage, finish reasons.** `ainvoke` returns the message list;
   `AIMessage.usage_metadata` and `response_metadata["finish_reason"]` are populated. The GLM
   `reasoning` text is not: `ChatOpenAI` documents that non-standard fields are dropped
   (`langchain_openai/chat_models/base.py` header). Raw bodies need an httpx event hook on
   `http_async_client` (used here).
4. **Pinning.** Yes: `uv pip compile --generate-hashes` produced 63 pinned packages with hashes
   (`scratch/lc/requirements.lock`) that install under `--require-hashes`. Inside this repo it needs
   `--no-config`; otherwise the root `pyproject.toml` injects `tfp-nightly>=0.1.dev0`, which fails
   hash mode. None of these packages are in the repo's `uv.lock` today.
5. **GLM quirks.**
   - Reasoning dropped and not replayed. With a 26-line `ChatGLM` subclass overriding
     `_create_chat_result` and `_get_request_payload` (both private) to keep `reasoning` and replay
     it as `reasoning_content`, the replay works (`T1_glm_subclass/http.jsonl`). The GLM template
     renders `reasoning_content` and ignores `reasoning`. Replaying it raises the prefix-cache hit
     because the generated thinking is reused (`scratch/probe_prefix_cache_reasoning_replay.json`:
     turn-2 cached tokens 768–1,344 with replay against 704 without). Streaming mode would need a
     third override for deltas.
   - Default calls are non-streaming, so the 1,800 s httpx read timeout here is effectively a
     wall-clock cap on T3, not a stall timeout. Streaming plus the delta override fixes that.
   - Malformed tool-call JSON: `create_agent` routes an `AIMessage` holding only
     `invalid_tool_calls` to the end node, so the run stops silently. Reproduced with a fake model
     (`invalid_toolcall_probe.txt`). Not seen with GLM in these runs.
   - Tool exceptions other than argument-validation errors propagate and abort the run unless
     `handle_tool_errors` is configured (`langgraph/prebuilt/tool_node.py::_default_handle_tool_errors`).
   - Prefix cache: turn-2 misses (`cached_tokens: 0`) after a large tool result show up in both this
     and the in-house loop (`scratch/prefix_cache_table.txt`), so they look like router placement,
     not harness behavior.
   - `reasoning_effort` is accepted; thinking stays on at every level tested.
6. **Taskforge integration size.** About 150 lines: the model subclass (about 40 with streaming),
   the shell tool (15), MCP client setup (10), middleware for length continuation and invalid tool
   calls (about 50, not written or tested), transcript extraction (20). Plus 63 third-party
   packages and a dependency on private `ChatOpenAI` internals that can change in any release.

## Verdict

Works for all three tasks with GLM-5.3. Web search via MCP is not a weakness: it was the
least-effort part. Using it in Taskforge means taking 63 packages and overriding private methods
to keep reasoning, then adding middleware for continuation and malformed tool calls. Those are the
same three problems the in-house loop solves directly in fewer lines.
