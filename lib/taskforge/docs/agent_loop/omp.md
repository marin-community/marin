# Agent loop candidate A: omp (oh-my-pi) 18.1.14

Evaluated 2026-10-05 against GLM-5.3 on the interactive tier (router `vllm-0.28.0-tp8-f8f644d5`).
Raw evidence: `lib/taskforge/.evidence/agent_loop/omp/` (per run: `events.jsonl` from
`omp -p --mode json`, `http/NNNN.json` with every chat-completions request body and the raw SSE
response captured by a logging proxy (request headers, including Authorization, are never
written), `session/` with omp's own transcript, `timing.json`, `stderr.txt`, `prompt.txt`).
Harness: `.evidence/agent_loop/scratch/run_omp.sh`, `scratch/logproxy.py`, the overlay agent dir
`scratch/omp/agent/` (copied `models.yml` + minimal `config.yml`; `~/.omp/agent/models.yml` was
not edited), `scratch/omp/mcp.json`, `scratch/omp/overlay_routeA.yml`, and the Machine
extension `scratch/omp/ext/machine_bash.ts`. The omp source used for reading is the npm package
`@oh-my-pi/pi-coding-agent@18.1.14` unpacked in `scratch/omp/pkg/`.

## Result

All three tasks completed, and bash redirected into a `shellbox` Machine worked. Two findings
change how omp must be configured. First, omp silently caps output at 64,000 tokens:
`maxTokens: 131072` in `models.yml` produced `"max_completion_tokens": 64000` on the wire until
`compat.clampOutputToModelMax: true` was added. Second, only `bash` can be redirected: omp's
pi-compatibility shim honors `BashOperations` but throws for `operations` on `edit`, `write` and
`grep`, so a Machine-backed run must drop omp's file tools (`--tools bash`) or reimplement them.
GLM reasoning replay is handled correctly by the existing `reasoningContentField` compat. On
`finish_reason == "length"` omp keeps the output and ends the run without continuing.

## Setup

- Binary: `/Users/k3sc0re/.local/bin/omp`, `omp v18.1.14`, Bun-compiled Mach-O arm64, 135 MB,
  sha256 `66c09cc5ffc8e080335649579c7cc32212d7e2a13d486a14f45d00f3088ea6d2`.
- `PI_CODING_AGENT_DIR=<per-run copy of scratch/omp/agent>`; its `models.yml` is the
  `glm-orion` block from `~/.omp/agent/models.yml` with `maxTokens: 131072`,
  `clampOutputToModelMax: true`, and `baseUrl` pointed at the logging proxy (which forwards to
  `127.0.0.1:18000`). The API key stays a `!cut ... glm_api_token.txt` command.
- Web search, route A from `build_envs/stage0/run_area.sh`: `<workspace>/.omp/mcp.json` with the
  Parallel MCP server (`Authorization: Bearer ${PARALLEL_API_KEY}`, interpolated from the
  environment, so no key on disk) and an overlay that disables omp's built-in `web_search`,
  `fetch` and `browser`, so MCP is the only web path.
- CLI: `omp -p --mode json --config overlay_routeA.yml --model glm-orion/glm-5.3 --thinking high
  --auto-approve --no-lsp --no-rules --no-skills --no-title --cwd <ws> --session-dir <ev>/session
  "<prompt>"`.

## Tasks

| Task | Result | Wall | Turns | Tool calls (errors) | Notes |
|---|---|---|---|---|---|
| T1 package + tests | works | 9 s | 3 | 5 (0) | 9 tests green first run; rerunning `uv run pytest -q` in the workspace: `9 passed`. No fix loop needed. |
| T1b harder variant (SemVer 2.0 + npm caret ranges, >= 20 tests) | works | 46 s | 8 | 12 (1) | One failing run, `read` + two `edit`s, green. Rerun: `53 passed`. |
| T2 web search | works, correct | 12 s | 4 | 4 (0) | `read xd://mcp__parallel_web_search` (schema), then `write xd://mcp__parallel_web_search` and `write xd://mcp__parallel_web_fetch`; results carry Parallel `search_id`/`extract_id`. Answer 0.1.25 (2026-10-01), 0.1.22 (2026-09-29), 0.1.21 (2026-09-29) with PyPI URLs, matching `pypi.org/pypi/shellsim/json`. |
| T2, before the clamp fix | works, correct | 13 s | 4 | 3 (1) | First MCP call failed validation (`objective`, `search_queries` required; GLM sent `queries`), then recovered. Kept as `omp/T2-default-64k-clamp/`; its requests show `max_completion_tokens: 64000`. |
| T3 40k-token JSON | works (harness); model JSON invalid | 290 s | 1 | 0 | 69,078 output tokens (738 reasoning), raw `finish_reason: stop`, one response, ids through 400, nothing cut. JSON does not parse at char 23,808 (a `steps` array closed with `}`). Model defect, not harness. |
| T3 forced `maxTokens: 4096` | partial | 23 s | 1 | 0 | `stopReason: length`, 4,096 output tokens kept (omp reports all 4,096 as `reasoningTokens`), run ended, exit 0, no continuation. |
| T4 bash redirected into a Machine | works | 25 s | 13 | 12 (2) | See below. |

Token totals (sum over assistant turns): T1 input 932 + cache read 85,632 (98% prefix-cache hit);
T1b 2,717 + 294,144 (99%); T2 17,098 + 121,856 (87%). omp's own system prompt is large: T3, with
`--no-tools`, had 14,740 prompt tokens, against 692 for vanilla pi; each T1 turn carried about
28k tokens of context.

## The six questions

1. **max_tokens = 131072.** Yes, but not by setting `maxTokens` alone. The binary defines
   `QF = 64000` and computes the request value as
   `min(requested, model.maxTokens, providerOutputClamp ?? 64000)`, where `providerOutputClamp`
   is set only for `cline-pass`, the z.ai dialect, or `compat.clampOutputToModelMax: true`. With
   that flag every request carries `"max_completion_tokens": 131072` (`omp/T1/http/0001.json`)
   and T3 reached 69,078 output tokens. Without it the server never sees more than 64,000.
2. **Shell redirect.** Bash only. Loading a pi-style extension (`-e`) that registers
   `createBashTool(cwd, { operations: { exec } })` works through omp's legacy-pi shim, which
   aliases `@mariozechner/*` and `@earendil-works/*` imports
   (`src/extensibility/legacy-pi-coding-agent-shim.ts:486`). The same shim throws "operations is
   not supported" for `grep`, `edit` and `write` (lines 556, 715, 732): omp's native file tools
   always touch the host filesystem. The run used `--tools bash` plus
   `scratch/omp/ext/machine_bash.ts` (28 lines) against `scratch/machine_bridge.py` (85 lines,
   ShellSim backend). Host workspace stayed empty apart from `.omp/mcp.json`; the machine dump
   holds `/workspace/inventory/items.csv` with `fig,9`; all 13 commands are in
   `omp/T4-bridge/bridge_calls.jsonl`. The model edited with `sed -i` because no edit tool was
   available, and it reasoned from omp's system prompt that the host was macOS ("BSD sort"), which
   was wrong for the Machine: omp's prompt describes the host, and there is no supported hook to
   rewrite that section short of `--system-prompt`.
3. **Transcript, usage, finish reasons.** Yes. `--mode json` emits every event; `message_end`
   carries `usage {input, output, cacheRead, reasoningTokens}`, `stopReason`, `responseId`, `ttft`,
   `duration`. omp maps the raw finish reason and does not keep it as a separate field, so the raw
   value came from the proxy logs. omp also has `--mode rpc` and an ACP server (`omp acp`).
4. **Pinnable with a hash.** Yes in two ways: the release binary by sha256 (above), or the npm
   package `@oh-my-pi/pi-coding-agent@18.1.14` (integrity
   `sha512-CgDPLMV0Lz8D+UlyNMVvW/X+2nc++l46M0xUZmbw1Fkjje1bbHO4h3rconaPbzuFQy2T9VDl77RMRzmVip6ONw==`),
   which runs from TypeScript source and needs Bun >= 1.3.14. The current npm release is 18.6.1;
   omp moves fast, and its catalog and defaults (the 64k clamp among them) change with it.
5. **GLM quirks.**
   - Reasoning replay: correct. `compat.reasoningContentField: reasoning_content` replays every
     prior assistant turn under `reasoning_content` (visible in every `http/*.json` request).
   - Output cap: the 64,000 default above.
   - Tool calls: with default tools, omp exposes MCP servers as virtual files: the model `read`s
     `xd://mcp__parallel_web_search` for the schema and `write`s JSON arguments to it. GLM got the
     arguments wrong once (`queries` instead of `objective`/`search_queries`) and recovered on the
     next turn. With `--tools bash`, MCP tools were declared as ordinary functions instead.
   - Prefix cache: 87-99% hit, helped by omp's stable prompt, but the 14.7k-token system prompt
     makes every turn about 5x the context pi sends.
   - Request shape: `max_completion_tokens` (not `max_tokens`), `store: false`,
     `reasoning_effort: high`, `stream_options.include_usage`.
6. **Lines of Taskforge integration.** About 450: config generation for `models.yml`, `mcp.json`
   and the overlay (~60), the bash extension (~30), replacements for read/write/edit as custom
   tools if the model should keep them (~120), the Python Machine bridge (~85), and a Python driver
   over `--mode json`/`rpc` (~160). Plus a 135 MB binary or a Bun runtime in the image.
