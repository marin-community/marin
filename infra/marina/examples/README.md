# App examples

Choose an example for the frontend and backend you need. If you expect to change
the app again, keep its source under `infra/marina/apps/<name>/`. Otherwise,
release it as an ad hoc app. See [Building Marina apps](../../../docs/references/marina.md).

- `static-html-app/` is a one-page app with checked-in `dist/` files and no
  build step.
- `vue-app/` builds a Vue single-page app with Vite. Its relative asset base
  works below stable and revision URLs.
- `problem-set-app/` adds a Python backend and an app-owned Postgres table.

Each frontend loads JavaScript from packaged files. Marina rejects executable
inline scripts because its content security policy sets `script-src 'self'`.
