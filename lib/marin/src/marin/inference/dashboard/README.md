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

Open **Python tools** above the Chat composer to define functions for the active
conversation. The source is stored with the conversation in browser storage.
The two Python-tool cards on an empty chat populate the editor with working
examples and send a prompt that asks the model to use them.

Tool source may contain only top-level function definitions. Every parameter
and return value must be typed; positional-only parameters, `*args`, `**kwargs`,
and decorators are rejected. Functions can be synchronous or asynchronous.

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
Argument and return annotations are validated with Pydantic. Schema generation
and each call run in a fresh subprocess with a 10-second timeout, and source is
limited to 64 KiB. One model response counts as one round, including a response
with multiple calls. The UI executes calls from the eighth round, stops before
another model request, and displays a limit error.

After the endpoint becomes ready, `marin-serve iris` prints a capability URL.
Possession of that URL authorizes both inference and arbitrary Python execution
through the tool editor. The subprocess is a process boundary, not a security
sandbox: tool code has the same environment and credentials as the Iris
service. Share the URL only with trusted users and treat it as a credential.
