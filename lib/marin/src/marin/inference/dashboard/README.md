# Marin serve dashboard

Vue app behind the `marin-serve` browser dashboard. `rsbuild` builds it into a
single self-contained HTML file (scripts and styles inlined, no CDN), committed
as the sibling `../serve_dashboard.html` and served by
`dashboard_server.py` via `importlib.resources`. The single-file artifact
is what lets the dashboard ship unchanged through both serve paths — the
bundled workspace and the PyPI `marin-core` wheel.

```bash
npm install
npm run build          # rebuild ../serve_dashboard.html (commit the result)
npm run build:check    # vue-tsc, then build
npm run dev            # local dev server (relative /v1, /info, /health fetches
                       # need a Marin serve dashboard server behind it)
```

After editing anything under `src/`, rerun `npm run build` and commit the
regenerated artifact alongside the source change.

## Chat template controls

The Chat view's **More settings** panel exposes two arguments to the active
model template. **Thinking** sends `enable_thinking=true`, sends
`enable_thinking=false`, or omits the argument for the model default. Marin's
current template renders those boolean values as `/think` and `/nothink`.
**Custom template instructions** sends `custom_instructions`; Marin's current
template places it in an auxiliary system header. Other templates may interpret
or ignore these arguments. This differs from **System prompt**, which adds a
normal `role: "system"` message to the conversation transcript.

The dashboard does not expose raw `xml_tools`, `python_tools`, or `tools`
template arguments. Functions entered in **Python tools** are converted to the
standard OpenAI `tools` request field so the served model's active template
formats them.

Enable **Raw chat** above the Chat composer to replace the rendered messages
with one plain-text transcript of the entire chat. The transcript includes the
system and user messages, unparsed assistant `content` and reasoning fields,
structured tool calls, tool results, and errors. The OpenAI-compatible response
does not include numeric token IDs, so the dashboard cannot display those.

## Custom Python tools

Open **Python tools** above the Chat composer to define functions for the active
conversation. The source is stored with the conversation in browser storage.
The two Python-tool cards on an empty chat populate the editor with working
examples and send a prompt that asks the model to use them.

Tool source may contain only top-level function definitions. Every parameter
and return value must be typed; positional-only parameters, `*args`, `**kwargs`,
keyword-only parameters, decorators, and asynchronous functions are rejected.
Annotations may use `bool`, `float`, `int`, `str`, `object`, `None`, lists,
string-keyed dictionaries, and `|` unions. Defaults must be JSON literals.

```python
def lookup_weather(city: str, units: str = "celsius") -> dict[str, object]:
    """Return the current weather for a city."""
    return {"city": city, "units": units, "temperature": 21}
```

Before each user turn's first model request, the dashboard converts each
function signature and docstring to an OpenAI-compatible JSON tool definition.
It sends those definitions in the request's `tools` field, so the tokenizer's
active tool-aware chat template controls the model prompt. The serving process
also reports that template's reasoning and tool-call delimiters to the browser.
For example, the Datakit template emits:

```xml
<tool_call>{"name":"lookup_weather","arguments":{"city":"Paris"}}</tool_call>
```

The Delphi template instead uses `<|tool_call|>` and
`<|tool_call_end|>`, while Llama-style templates emit a bare JSON object with
`name` and `parameters`. The UI parses the format selected from the served
model's template, runs each call in the Iris service, and adds the call and
result to the next request as structured `tool_calls` and `role: "tool"`
messages. The same template then renders the result in its trained format.
The server derives schemas from the function syntax without executing the
source. Pydantic validates arguments before each call and the return value
afterward. Each call runs in a fresh ShellSim environment that cannot access
the host filesystem, processes, network, environment variables, or clock. The
simulation is limited to 10 million CPU ticks, 64 MiB of memory, 4 MiB of disk,
and 1 MiB of output. An outer worker also enforces a 10-second wall timeout,
and source is limited to 64 KiB. ShellSim implements a source-compatible subset
of Python rather than full CPython, so unsupported modules and language features
fail the tool call.

One model response counts as one round, including a response with multiple
calls. The UI executes calls from the eighth round, stops before another model
request, and displays a limit error.

## ShellSim agent workspaces

Open **Shell workspace** above the Chat composer to give the model a
conventional `bash` tool backed by an isolated ShellSim filesystem. Initial
files are a JSON object whose keys are relative paths and whose values are UTF-8
text. Enabling a workspace adds the standard function definition to the
request's `tools` field, so the served model's active chat template controls how
shell calls and results are rendered just like custom Python tools. The three
agent cards on an empty chat demonstrate repairing code, investigating a log,
and updating a configuration.

ShellSim starts in `/work` with the initial files committed as a fresh Git
baseline. The browser stores each executed command with the conversation. For
the next call, the service reconstructs the filesystem and replays those
commands before executing the new command, which preserves file edits and Git
state without keeping a host process or directory alive. Editing the initial
files or choosing **Reset commands** clears that replay history.

To start from an existing project, enter a public GitHub repository URL and
choose **Load public repo**. The dashboard service downloads the default
branch's archive outside ShellSim, extracts bounded UTF-8 regular files, and
passes only that file map into the simulation. It does not forward credentials,
clone remote Git history, load submodules, or support private repositories.
ShellSim itself has no real network access, and `git clone` is unavailable; the
simulated repository receives a new baseline commit after import.

Workspace inputs are limited to 500 files, 256 KiB per file, and 4 MiB total.
Repository downloads are limited to 4 MiB compressed and 10,000 archive
entries. Binary files and common generated directories such as `node_modules`,
`.venv`, `target`, and `__pycache__` are skipped. A conversation may replay at
most 32 commands totaling 128 KiB; each command is limited to 16 KiB. Each
simulation is limited to 50 million CPU ticks, 64 MiB of memory, 16 MiB of disk,
and 2 MiB of output, with the same 10-second outer worker timeout used for
custom Python tools.

After the endpoint becomes ready, `marin-serve iris` prints a capability URL.
Possession of that URL authorizes inference and simulated Python tool calls.
It also authorizes ShellSim commands and bounded public GitHub downloads. Share
the URL only with trusted users and treat it as a credential.
