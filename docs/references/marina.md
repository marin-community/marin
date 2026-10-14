# Building Marina apps

Marina hosts small Marin web apps with authentication, an app directory,
Postgres schemas, static assets, Python APIs, and browser journeys. Checked-in
apps can also expose MCP tools and the Loom page-agent panel. Applets publish
without rebuilding the Marina image.

## Choose the app type

| | Checked-in app | Dynamic applet |
| --- | --- | --- |
| Source | `infra/marina/apps/<name>/` | A directory containing `applet.toml` and `dist/` |
| Delivery | PR and Marina image deployment | `marina publish`; retained, numbered revisions |
| URL | `/<name>/` on Marina | `/a/<uuid>/` on an isolated applet origin |
| Backend factory | `create_api(services)` returns `registered_api(api)` | `create_api(services)` returns an ASGI app |
| Migration | `migrate(engine)` at deploy | `migrate(connection)` inside the publish transaction |
| Refresh work | Manifest-declared scheduled jobs | Explicit loader or bounded request-driven refresh; no background workers |

Use an applet for a small report, explorer, or tool that needs independent
publishing. Use a checked-in app when it needs scheduled ingestion, new server
dependencies, or Marina's MCP and page-agent integrations. Applet backends use
packages already installed in Marina and run as trusted plugins, not sandboxes.

## Build and try it

For an applet, start with the
[Vue starter](https://github.com/marin-community/marin/tree/main/infra/marina/examples/vue-applet)
or the [Postgres backend example](https://github.com/marin-community/marin/tree/main/infra/marina/examples/problem-set-applet).
Give `applet.toml` a readable title and one or two sentences describing its use.
Use plain HTML/JavaScript for a single view, or Vue/Vite for substantial UI state.
Keep Vite's base at `"./"`, use hash routing, and use relative asset and API URLs.
Bundle scripts locally; executable inline scripts and third-party CDNs are blocked.
Commit `package-lock.json` when the build uses `npm ci`.

```bash
uv run marina validate my-applet
uv run marina publish my-applet --local  # disposable Postgres and loopback server
```

Exercise the page, API, and a database read/write while the local process runs.
Ctrl-C stops it and removes its database. `validate` alone checks packaging and
backend imports; it does not run migrations, the factory, or browser interactions.
When ready to publish to Marina:

```bash
uv run marina publish my-applet --json
uv run marina publish my-applet --update <uuid> --base-version <current>
```

For a checked-in app, add `app.toml`, `web/`, and optional `app.py`. Follow the
[adding-an-app reference](https://github.com/marin-community/marin/blob/main/infra/marina/README.md#adding-an-app)
for bundler aliases, registered APIs, and job declarations. Set
`MARINA_DATABASE_URL` to local Postgres when the app uses a database, then run:

```bash
uv run marina build --only <name>
uv run marina migrate
uv run marina dev
uv run marina journey <name>
```

Use the [shared frontend kit](https://github.com/marin-community/marin/blob/main/infra/marina/web/README.md):
`Shell` supplies the title, searchable app drawer, navigation slot, and optional
agent panel for checked-in apps. `AppBar` supplies navigation without kernel
fetches for isolated Vue applets. Pass its `home` URL explicitly; do not expose
the private directory API to public applets.

## Load and cache data

Use `services.engine()` to create one SQLAlchemy engine in the backend factory
for the app's existing Postgres schema. Use that engine in request handlers and
transactions; do not provision a separate database.

| Data | Storage and serving approach |
| --- | --- |
| User edits, votes, annotations, plans | Postgres is authoritative. Read and write it through the API. |
| Catalogs, filterable metadata, summaries reused across instances | Load into Postgres once per source revision. Serve indexed queries or a bounded in-memory snapshot read from Postgres. |
| Expensive object listings or remote API results | Cache on demand in Postgres when reuse justifies it. Record source version, fetch time, expiry, and refresh status. |
| Small immutable report or evidence snapshot | Bundle with the applet; no second database copy needed. |
| Large archives, media, Parquet payloads | Keep in object storage; index metadata in Postgres and fetch bounded previews or selected rows. |
| Live telemetry | Query its source or use a short TTL with visible age. A startup snapshot becomes stale. |

For small scalar tables, load a local JSON array, JSONL, CSV, or Parquet file:

```bash
uv run marina applets table load <uuid> observations rows.parquet --url <origin>
uv run marina applets sql <uuid> 'SELECT * FROM observations LIMIT 20' --url <origin>
```

Use the printed local origin and UUID while `publish --local` runs, or
`https://marina.oa.dev` for production. Omitting `--url` targets production.

Repeated loads append rows. `--replace` drops and recreates the table, then
inserts batches in separate requests; it is not an atomic live-catalog refresh.
Use it for disposable or offline tables. For a live catalog, stage a new load,
validate it, and select it with a current-version pointer in one transaction.
See the [Atlas loader](https://github.com/marin-community/marin/blob/main/infra/marina/applets/rl_data_catalog/load.py).
Retained applet revisions share the same mutable schema, so migrations and
refreshes must remain compatible with them.

Keep ingestion separate from serving:

- Run remote scans and parsing in an explicit loader or a checked-in app's
  scheduled job. Use migrations for schema changes and small seeds.
- At startup, connect to Postgres and read a bounded committed snapshot if useful.
  Do not scan buckets or rebuild a catalog in `create_api`; checked-in factories
  affect the whole service's startup, and applet factories run on first use.
- Key cached data by source identity/version and query parameters. Invalidate
  process-local snapshots using a database generation or bounded TTL.
- Serialize competing refreshes in Postgres. Keep the last complete snapshot
  available on refresh failure and expose its age and error. Do not report an
  unavailable source as a successful empty refresh.

[EvalDash](https://github.com/marin-community/marin/blob/main/infra/marina/apps/evaldash/README.md)
shows scheduled ingestion, atomic catalog generations, and serving-instance
refreshes. In-memory caches are per worker and disappear on restart; applets
cannot rely on lifespan events, background workers, or local disk persistence.

## Access and limits

Applets are private by default. Public mode exposes frontend files and GET/HEAD
backend routes without login; it does not expose browser SQL. Serve public data
through a read-only backend. Private browser SQL supports one parameterized
statement, bounded to 10,000 rows, 5 MiB, and 10 seconds. Prefer a backend for
authorization, validation, or multi-statement transactions.

Applet packages are limited to 25 MiB total, 8 MiB per file, and 2,000 files.
See the [Marina reference](https://github.com/marin-community/marin/blob/main/infra/marina/README.md)
for authentication, deployment, publishing, rollback, and schema ownership.
