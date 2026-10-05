# Agent loop candidate B: vanilla pi (`@earendil-works/pi-coding-agent`)

Evaluated 2026-10-05 against GLM-5.3 on the interactive tier (router `vllm-0.28.0-tp8-f8f644d5`).
Raw evidence: `lib/taskforge/.evidence/agent_loop/pi/` (per run: `events.jsonl` with the full
`--mode json` event stream, `requests.jsonl` with every chat-completions request body as pi sent
it, `session/*.jsonl` with the persisted transcript, `timing.json`, `stderr.txt`, `prompt.txt`).
Harness: `.evidence/agent_loop/scratch/run_pi.sh`, config in `scratch/pi/agent/`
(`models.json`, `mcp.json`, `settings.json`), extensions in `scratch/pi/ext/`, the Machine bridge
in `scratch/machine_bridge.py`, the Python RPC driver probe in `scratch/pi_rpc_probe.py`.

## Result

All three tasks completed, plus a fourth run with every file and shell tool redirected into a
`shellbox` Machine. pi has the two seams Taskforge needs as documented, supported APIs: tool
`operations` (`BashOperations.exec`, `ReadOperations`, `WriteOperations`, `EditOperations`) to
run tools somewhere other than the host, and a `before_provider_request` hook that can rewrite the
request body. Web search works through pi's built-in MCP client with no extension. Two GLM costs:
pi replays reasoning under the field it received (`reasoning`), which GLM's chat template drops,
so a 10-line hook is needed to rename it to `reasoning_content`; and on `finish_reason ==
"length"` pi keeps the output and ends the run without continuing.

The npm name in the task, `@mariozechner/pi-coding-agent`, is deprecated at 0.73.1 (2026-05-07,
"please use @earendil-works/pi-coding-agent instead"). Everything here uses the successor,
`@earendil-works/pi-coding-agent@1.0.3`.

## Setup

- Install: `npm install --save-exact @earendil-works/pi-coding-agent@1.0.3` in
  `scratch/pi/` (Node v22.23.2; the package requires Node >= 22.19.0). `package-lock.json` pins
  396 packages by sha512; copied to `.evidence/agent_loop/pi/package-lock.json`. Root package
  integrity `sha512-t2lb0dw4y/jr5a2PRo6eTHGTZOPB3/YAMVyhhYFC1W3Hl5xE+462I/gMWjF4gCLuhGipNEfuNqONFmdqLFz4SQ==`.
  `pi-mcp-adapter@5.0.0` and `pi-web-access@0.36.0` were installed for inspection but not used;
  pi 1.0.3 has MCP built in.
- `PI_CODING_AGENT_DIR=scratch/pi/agent` isolates all config from `~/.pi`.
- `models.json`: provider `glm-orion`, `api: openai-completions`, `baseUrl:
  http://127.0.0.1:18000/v1`, `apiKey: "!cut -d= -f2- .../glm_api_token.txt"` (command, so no
  token on disk), `contextWindow: 262144`, `maxTokens: 131072`, `reasoning: true`, compat
  `thinkingFormat: openai`, `supportsReasoningEffort: true`, `maxTokensField: max_tokens`.
- `mcp.json`: `parallel` at `https://search.parallel.ai/mcp?mode=advanced`, header
  `Authorization: Bearer ${PARALLEL_API_KEY}` (env interpolation), `exposure: direct`.
  `pi mcp list` reports `connected, 2 tools (direct): web_search, web_fetch`.
- CLI: `pi --mode json --model glm-orion/glm-5.3 --thinking high --no-context-files --no-skills
  --no-prompt-templates -na -e ext/glm_evidence.ts --session-dir <ev>/session "<prompt>"`.
  `--no-context-files` matters: the workspace sits under the Marin checkout, and pi would otherwise
  load the repo's `AGENTS.md`/`CLAUDE.md` into the system prompt.

## Tasks

| Task | Result | Wall | Turns | Tool calls (errors) | Notes |
|---|---|---|---|---|---|
| T1 package + tests | works | 8.6 s | 6 | 8 (1) | 13 tests green; first `pytest` failed on the model's own wrong expected count, fixed with one `edit`. Rerunning `uv run pytest -q` in the workspace: `13 passed`. |
| T1 with reasoning-replay fix | works | 7.7 s | 5 | 8 (0) | 11 tests green first run; rerun confirms `11 passed`. |
| T1b harder variant (SemVer 2.0 + npm caret ranges, >= 20 tests) | works | 33.4 s | 7 | 12 (1) | One failing run, two `edit`s, green. Rerun: `58 passed`. |
| T2 web search | works, correct | 6.5 s | 3 | 2 (0) | `mcp__parallel__web_search` then `mcp__parallel__web_fetch`; results carry `search_id`/`extract_id` from Parallel. Answer 0.1.25 (2026-10-01), 0.1.22 (2026-09-29), 0.1.21 (2026-09-29) with PyPI URLs, matching `pypi.org/pypi/shellsim/json`. |
| T3 40k-token JSON | works (harness); model JSON invalid | 247.9 s | 1 | 0 | 69,004 output tokens (507 reasoning), `finish_reason: stop`, one response, 400 ids reached, nothing cut. The JSON does not parse: case 9 contains `"A".repeat(500)` (9 such JS expressions in total). Model defect, not harness. |
| T3 forced `maxTokens: 4096` | partial | 16.6 s | 1 | 0 | `stopReason: length`, `rawStopReason: length`, 4,096 output tokens kept, run ended (`agent_end` once), exit 0, no continuation. |
| T4 tools redirected into a Machine | works | 15.7 s | 15 | 14 (1) | See below. |

Token totals (sum over assistant turns, from the `usage` pi records): T1 input 1,510 + cache read
24,704 (94% prefix-cache hit); T1b 2,493 + 60,288 (96%); T2 12,255 + 12,544 (50%, the search
result is new input). The no-tools system prompt is 692 tokens (T3 `input`).

## The six questions

1. **max_tokens = 131072.** Yes. pi sends `model.maxTokens` by default, clamped to
   `contextWindow - estimated input` (`pi-ai/dist/api/simple-options.js:clampMaxTokensToContext`).
   Every logged request carries `"max_tokens": 131072` (`requests.jsonl`), the server accepted it,
   and T3 reached 69,004 output tokens in one response.
2. **Shell redirect.** Yes, supported API. `createBashToolDefinition(cwd, { operations: { exec } })`
   replaces where commands run; read/write/edit take `ReadOperations`, `WriteOperations`,
   `EditOperations` the same way (the shipped `examples/extensions/ssh.ts` does exactly this over
   SSH). Glue written and run: `scratch/pi/ext/machine_tools.ts` (59 lines) registers all four
   tools against a localhost bridge; `scratch/machine_bridge.py` (85 lines) serves
   `shellbox.backends.shellsim.machine.ShellSimMachine` (`run`, `upload`, `download`). Two
   details matter: tools pass host-absolute paths, so the extension maps the session cwd prefix to
   `/workspace`, and the system prompt names the host cwd, so a `before_agent_start` hook rewrites
   it. In T4 the host workspace stayed empty, `bridge_calls.jsonl` shows every `run`/`read`/`write`,
   and `POST /dump` returned only `/workspace/inventory/items.csv` with the edited content
   (`fig,9`). pi has no RPC-level "client executes tools" callback, so a Python-owned Machine
   needs this localhost bridge (or the tool must run in-process via the TypeScript SDK).
3. **Transcript, usage, finish reasons.** Yes, three ways. `--mode json` streams every event
   (`message_end` carries the full message with `usage {input, output, cacheRead, reasoning}`,
   `stopReason`, `rawStopReason`, `responseId`). The session JSONL holds the same messages.
   `--mode rpc` adds `get_messages`, `get_session_stats`, `steer`, `follow_up`, `abort`; the
   Python probe (`pi/rpc-python-probe/`) ran two prompts in one session over stdin/stdout and read
   back stats (4 assistant messages, stop reasons `toolUse, stop, toolUse, stop`). The raw request
   bodies come from `before_provider_request`; pi does not log them by default.
4. **Pinnable with a hash.** Yes: npm exact version plus `package-lock.json` sha512 integrity, and
   `npm ci`. Install footprint is 370 MB of `node_modules` (94 MB of it `@earendil-works/*`), and
   a Node >= 22.19 runtime.
5. **GLM quirks.**
   - Reasoning replay: pi stores the field name it parsed (`reasoning`) as the block signature and
     replays under that name (`openai-completions.js` ~line 1000; only the `opencode-go` provider
     is special-cased to `reasoning_content`). There is no compat flag for this. Vanilla T1
     requests show assistant messages with a `reasoning` key; GLM's template renders only
     `reasoning_content`, so prior reasoning is silently dropped. `scratch/pi/ext/glm_evidence.ts`
     renames it in `before_provider_request`; T1-replayfix requests carry `reasoning_content`.
     At `--thinking high` reasoning was short (32-2,385 tokens per run), so the measured cache hit
     was the same (94%) with and without the fix; the loss grows with reasoning length.
   - Tool calls: no malformed calls in any run. MCP tools with `exposure: direct` are plain
     function tools; GLM used the right argument names first time.
   - `finish_reason: length` ends the run with output kept; continuing needs either a
     `turn_end`/`agent_before_settle` handler returning `continue: true` or a driver-side
     follow-up prompt. Not built here.
   - Defaults that would bite: `httpIdleTimeoutMs` 300,000 (set to 600,000 here); no
     payload-level priority or cache key unless configured.
6. **Lines of Taskforge integration.** About 350: the TypeScript extension (machine tools,
   reasoning rename, length continuation, ~100), the Python Machine bridge (~85), and a Python
   driver that writes the per-run agent dir, spawns `pi --mode rpc`, and maps events to typed
   outcomes (~160). Plus a pinned `package.json`/`package-lock.json` and Node 22 in the image.

## Observed environment bug

ShellSim's `sort` ignores every `-k` key after the first: `sort -t, -k2,2nr -k1,1` returned
`apple,3` as the maximum (T4 `bridge_calls.jsonl`). Both pi and omp spent several turns isolating
it. This is a ShellSim defect, not an agent one.
