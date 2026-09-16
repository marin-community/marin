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

## Custom Python tools

`marin-serve iris` accepts a repeated `--tool MODULE:FUNCTION` option. The
module must be importable when the command runs. For the example below, put
`my_tools.py` at the checkout root and run the command from that root; Iris syncs
the checkout into the service job. Define each tool as a top-level function with
typed parameters. The function docstring is used as the model-facing
description, and parameter defaults remain optional in the generated JSON
schema.

```python
# my_tools.py
def lookup_weather(city: str, units: str = "celsius") -> dict[str, object]:
    """Return the current weather for a city."""
    return {"city": city, "units": units, "temperature": 21}
```

```bash
marin-serve iris Qwen/Qwen3-0.6B \
  --tool my_tools:lookup_weather \
  --tool-call-parser hermes
```

The vLLM backend requires `--tool-call-parser`. Select the parser that matches
the model's tool format, such as `hermes`, `qwen3_coder`, or `mistral`.
For models in `experiments/evaluation/serve/models/`, use the YAML file's
`serve.tool_call_parser` value. Levanter does not use this option; its chat
template must render the `tools` argument and tool messages.

The Chat UI sends the generated OpenAI tool definitions with each model
request. It accepts structured OpenAI tool calls, raw `<tool_call>` tags, and
the bare `{"name": ..., "parameters": ...}` JSON emitted by Llama 3 templates.
It runs each function in the Iris service, appends its JSON result as a tool
message, and asks the model to continue. The tagged inline form is
`<tool_call>{"name":"lookup_weather","arguments":{"city":"Paris"}}</tool_call>`.
One model response counts as one round, including a response with multiple tool
calls. The UI executes calls from the eighth round, stops before another model
request, and displays a limit error.

Functions can be synchronous or asynchronous. Positional-only parameters,
`*args`, and `**kwargs` are rejected. Every parameter needs a type annotation,
and the return value must match its return annotation when one is present. Add
tool dependencies to a package in the synced checkout. Repeat `--extra NAME`
for any optional dependency groups that the tool imports. Argument validation,
function exceptions, and return-value validation failures are returned to the
model as tool error messages.

After the endpoint becomes ready, `marin-serve iris` prints a capability URL.
Possession of that URL authorizes both inference and tool execution. Only
expose trusted functions, and treat the URL as a credential.
