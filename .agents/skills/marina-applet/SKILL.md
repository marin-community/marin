---
name: marina-applet
description: Build, validate, publish, update, inspect, query, roll back, or archive a dynamic Marina applet. Use for Artifact+-style web apps that should run behind Marina authentication, carry small bundled assets or data, use an applet-owned Postgres schema, or add a trusted Python ASGI backend without a Marina image deployment.
---

# Work with Marina applets

Read the `Publishing an applet` section of `infra/marina/README.md` before
changing an applet or publishing one. Use
`infra/marina/examples/problem-set-applet/` as the minimal working package and
backend reference. Treat those files and `marina --help` as authoritative when
this skill disagrees with the checkout.

## Choose the applet shape

Ensure the built package contains:

```text
my-applet/
  applet.toml
  dist/index.html
```

Add browser assets below `dist/`. Use only relative asset, API, and query URLs
so both `/a/<uuid>/` and `/a/<uuid>/v/<revision>/` work.

The source directory may omit `dist/index.html` only when `applet.toml` has a
`build_command` that creates it before packaging.

Use plain HTML and JavaScript for a small single-view applet. Use Vue with Vite
when the applet needs multiple views, reusable components, or substantial
client state. For Vue applets:

- Set Vite's `base` to `"./"` so built asset URLs remain relative.
- Use `createWebHashHistory()` when the app has client-side routes. This avoids
  hard-coding an applet UUID or revision into the router base.
- Use relative backend calls such as `fetch("api/results")` and
  `fetch("query", ...)`; do not start applet-owned URLs with `/`.
- Bundle dependencies into `dist/`. Marina's content security policy does not
  allow scripts loaded from a third-party CDN.

Keep the source `index.html`, `package.json`, the Vite configuration, and `src/`
beside `applet.toml`. They are local build inputs and are not uploaded. A
typical manifest uses `build_command = "npm ci && npm run build"`. Marina runs
that command before every publish by default. Use `--no-build` only when the
existing `dist/` has already been built and validated.

Use the applet's implicit Postgres schema for mutable or queryable data. Do not
create or attach a separate database. Load simple scalar tables with the CLI,
or add `server/` when the applet needs parameterized business logic, custom
responses, or server-side integration.

An optional backend has this shape:

```text
my-applet/
  server/__init__.py
  server/app.py
```

Declare it in `applet.toml`:

```toml
title = "Problem sets"
description = "Browse generated math problem sets."
python_entrypoint = "server.app:create_api"
```

Export `create_api(services) -> ASGIApp`. Use `services.engine()` for an engine
configured with the applet role and schema. Optionally export
`migrate(connection) -> None`; it runs inside the publish transaction before
the new revision becomes current. Keep schema changes compatible with retained
revisions.

Use `get_verified_identity()` from `rigging.server_auth` when a backend needs
the authenticated caller. Import only packages already declared in
`infra/marina/pyproject.toml`.

Treat Python applets as trusted plugins. They run inside Marina with its
filesystem, network, credentials, and process identity. Do not publish code
from an untrusted source or depend on lifespan events, background workers, or
local filesystem persistence.

## Build and validate

Inspect existing Marina and repository helpers before introducing a framework
or dependency. Build frontend output locally; Marina does not install applet
dependencies. `marina publish` runs a declared `build_command` before packaging
by default, and the command may create `dist/index.html`. Pass `--no-build` to
reuse an existing validated `dist/`.

Validate the exact package without changing server state:

```bash
uv run marina publish my-applet --dry-run
```

Respect the current package limits: 25 MiB total, 8 MiB per file, and 2,000
regular files. Keep large data out of the package until Marina gains an object
store path.

Run the complete local stack with:

```bash
uv run marina publish my-applet --local
```

This starts disposable Postgres in Docker, installs Marina's privileged applet
provisioning, migrates the registry, starts Marina on a free loopback port,
publishes the applet with its migration, and prints its immutable revision URL.
It requires a running Docker daemon, binds only to loopback, and does not enable
IAP authentication. It stays in the foreground. After validation, send Ctrl-C
and verify that both Marina and the container stop. If `uv` cannot write its
cache in a restricted checkout, use the current checkout's `.venv/bin/marina`
executable instead of installing or synchronizing dependencies.

Open the printed immutable revision URL. Exercise the frontend, every backend
route, caller identity when used, and at least one schema read/write path. For
Marina code changes, run the focused tests and then:

```bash
cd infra/marina && uv run pytest
```

## Use applet data

The applet role owns `applet_<uuid-without-hyphens>` and inherits read access to
checked-in and other applet schemas. It can write only its own schema. Prefer
the CLI for small datasets:

```bash
uv run marina applets table load <uuid> observations rows.parquet --replace
uv run marina applets sql <uuid> 'SELECT * FROM observations'
```

Table loading accepts JSON arrays, JSONL, CSV, and Parquet with scalar columns.
Browser code may post one SQLAlchemy text statement to relative URL `query`.
Query results are limited in Postgres to 10,000 rows and 5 MiB with a 10-second
timeout. Transaction and role-control commands, data-modifying `WITH`, and
mutation statements with `RETURNING` are rejected.

Prefer a Python backend over browser-submitted SQL when inputs need validation,
authorization, stable response types, or multiple database operations.

## Publish and operate

Treat a request to build or update a Marina applet as authorization to publish
it and perform the server writes needed for the requested behavior, including
schema migrations and requested data loading. Validate the package, then publish
without asking for separate confirmation. Honor requests for local-only work,
drafts, or review before publishing. Rollback, archive, and unrelated data
mutations still need authorization within the user's request. The default target is
`https://marina.oa.dev`, and the CLI uses cached `iris login` credentials or
ambient Google credentials through IAP.

First publish:

```bash
uv run marina publish my-applet --json
```

Record the returned UUID, revision, and immutable revision URL. A successful
publish runs its migration and atomically selects the new revision as current.
Fetch the stable current URL with:

```bash
uv run marina applets versions <uuid> --json
```

Update the same applet with optimistic concurrency:

```bash
uv run marina publish my-applet --update <uuid> --base-version <current>
```

Inspect before mutating lifecycle state:

```bash
uv run marina applets list
uv run marina applets versions <uuid>
```

Roll back only to a retained revision:

```bash
uv run marina applets rollback <uuid> <revision>
```

Archive an applet with:

```bash
uv run marina applets archive <uuid>
```

Archival hides the applet but preserves its schema and retained revisions.
Marina retains the current revision plus four other recent revisions.

Report the applet UUID, stable URL, immutable revision URL, validation or test
results, and any trusted-code or schema-compatibility caveat that affects use.
