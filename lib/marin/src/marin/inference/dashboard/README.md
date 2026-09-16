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

For each model request, the UI adds the source to the system message inside a
`<python_tools>` XML element. The model calls a function with this form:

```xml
<tool_call>{"name":"lookup_weather","arguments":{"city":"Paris"}}</tool_call>
```

The UI runs each tagged call in the Iris service, returns its JSON result in a
`<tool_result>` XML element, and asks the model to continue. Argument and return
annotations are validated with Pydantic. Each call runs in a fresh subprocess
with a 10-second timeout, and source is limited to 64 KiB. One model response
counts as one round, including a response with multiple calls. The UI executes
calls from the eighth round, stops before another model request, and displays a
limit error.

After the endpoint becomes ready, `marin-serve iris` prints a capability URL.
Possession of that URL authorizes both inference and arbitrary Python execution
through the tool editor. The subprocess is a process boundary, not a security
sandbox: tool code has the same environment and credentials as the Iris
service. Share the URL only with trusted users and treat it as a credential.
