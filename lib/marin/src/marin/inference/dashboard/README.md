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
