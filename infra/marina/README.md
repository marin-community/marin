# Marina

One Cloud Run service that hosts Marin's small internal web apps. Checked-in
apps live under `apps/`; dynamic applets are uploaded with `marina publish` and
stored in Postgres. Both use the same IAP login, database, app directory, and
browser test harness.

## Layout

```
infra/marina/
  src/marina/      the kernel: manifests, auth, the FastAPI process, db, journeys, the CLI
  web/             the shared Vue kit apps import as @marina (tokens, Shell, Prose)
  apps/<name>/     one app: app.toml, web/, dist/ after a build, and for Python
                   apps an app.py with create_api() and migrate(), tests/, journeys/
  examples/        applet packages that can be published to a running Marina
  .data/<app>/     the local data root (gitignored); gs://marin-marina in production
  Dockerfile       node stage builds every apps/*/web; python stage runs `marina serve`
  __main__.py      Pulumi: database, bucket, the service, and manifest-declared job runners
```

`web/README.md` describes the kit and the frontend contract.

## What the kernel gives an app

- `/<name>/`: files from `dist/`, `index.html` for every other path so client
  routes survive a reload. A `x.json.gz` beside `x.json` is served compressed.
- `/<name>/data/<path>`: files from `<data root>/<name>/`. Large or changing
  data lives there rather than in the image or the repository.
- `/<name>/api/`: the ASGI app a Python app's `create_api(services)` returns,
  mounted behind the same authentication. Handlers read the caller with
  `rigging.server_auth.get_verified_identity()`.
- `services.engine()`: a SQLAlchemy engine on the app's own Postgres schema
  (`search_path = <name>, public`). `migrate(engine)` runs at deploy, before
  the new image takes traffic, and must be idempotent.
- `/api/marina/apps` and `/api/marina/me`: the directory and the caller, which
  the Shell uses for its app switcher and identity chip. `/healthz` is public.

## Adding an app

1. Create `apps/<name>/app.toml`:

   ```toml
   title = "TaskTrove"
   description = "Browse the TaskTrove task collection."
   connect_src = []          # extra origins the page may fetch; 'self' is implied
   build_command = "cd web && npm ci && npm run build"

   [[jobs]]                  # optional non-serving work
   name = "refresh"
   runner = "hourly"
   schedule = "0 * * * *"
   command = ["python", "-m", "tasktrove.refresh"]
   timeout = 1800
   cpu = 1
   memory_gib = 1
   secrets = []              # environment names registered by this stack
   ```

   Unknown keys are rejected. Jobs on one runner must declare the same schedule. A runner
   migrates every Python app, then runs its jobs in stable app/name order with each job's
   timeout and secrets. The runner takes the largest CPU and memory declaration among its
   jobs. Ordinary job failures do not prevent later jobs from running.
2. Put a Vue + rsbuild app in `apps/<name>/web/`, aliasing `@marina` to
   `../../../web`, emitting to `../dist` with asset prefix `/<name>/`, and
   rendering inside `<Shell app="<name>">`. Copy `apps/tasktrove/web` as the
   starting point.
3. For an API, add `apps/<name>/__init__.py` and `app.py` with
   `create_api(services)` and, if it has tables, `migrate(engine)`. Copy
   `apps/echo` for the shape.
4. Write a journey in `apps/<name>/journeys/test_*.py` (below).
5. Run it locally:

   ```bash
   uv run marina check                 # parse manifests, report build state
   uv run marina build --only <name>   # run the app's build_command
   uv run marina dev                   # http://127.0.0.1:8080/<name>/
   ```

   Nothing in Pulumi or the Dockerfile changes for a new app.

## Database

Production is the `marina` database on the `marin-metadata` Cloud SQL instance,
reached through the connector as the service account (`CLOUDSQL_CONNECTION`,
`PGDATABASE`, `PGUSER`). Locally, set `MARINA_DATABASE_URL` to any Postgres
with the `vector` and `pg_trgm` extensions, then `uv run marina migrate`.
Tests get one from `marina.testing.test_database()`, a throwaway
`pgvector/pgvector` container, or from `MARINA_TEST_DATABASE_URL`.

## Journeys

A journey is a browser walk through an app that is also a test. Specs live in
`apps/<name>/journeys/test_*.py` and take a `journey` fixture:

```python
def test_sources_filter(journey):
    journey.visit("/").sees("Sources").fill("Filter sources", "nl2bash").sees("nl2bash-tasks").shoot("filtered")
```

`visit`, `sees`, `absent`, `click`, `fill`, `select` speak in what a person
sees; `offers()` lists every control on the page, `reads()` the visible text,
`api()` fetches a same-origin JSON endpoint; `shoot()` and `widths()` keep
screenshots. Every uncaught page error and failed API call during the walk
fails the test at the end.

```bash
uv run marina journey                  # every app
uv run marina journey echo --video     # one app, with a video per spec
uv run marina journey -k filter --headed
```

Screenshots and videos land in `journeys-out/<app>/`. The kernel under test
serves the real apps directory, the local data root, and the database in
`MARINA_DATABASE_URL`, so journeys that need rows seed them first. Plain
`pytest` skips journeys; `marina journey` passes `--journeys`.

## Serving

`marina serve` reads `MARINA_APPS_DIR`, `MARINA_DATA_ROOT`,
`MARINA_IAP_AUDIENCE`, the database variables above, and `MARINA_HOST_APPS`
(`echo.oa.dev=echo,...`: a host that used to be one app's own origin). A request
to `/api` on a legacy host is routed directly to that app's API so existing
clients keep their authorization header; page requests redirect into the app's
prefix on the canonical origin. The auth chain is IAP's signed assertion header
when an audience is configured, then loopback; on Cloud Run a missing audience
is a startup error rather than an open service.

`MARINA_APPLET_ORIGIN` names the separate origin that serves `/a/*`. Requests
for applet pages on Marina's main host redirect there. The applet host returns
404 for checked-in apps and Marina control routes. `MARINA_APPLET_OPERATORS` is
a comma-separated set of user IDs allowed to update, roll back, or archive any
applet; otherwise only the recorded owner may do so.

## Deploying

```bash
cd infra/marina && uv run pulumi up
```

The image builds from the repository root (`Dockerfile.dockerignore` allowlists
what it copies). `pulumi up` builds the image, creates one Cloud Run job and
Scheduler trigger per manifest runner, executes the longest-timeout runner in
migration-only mode, and then updates the service. The service receives each
runner's resource name as `<APP>_<JOB>_JOB`, with hyphens converted to
underscores, so an app can trigger its own declared job. Scheduled runners also
apply migrations before app work, so a scheduler tick cannot run new code
against an old schema. One PostgreSQL advisory lock serializes migrations
across every runner. Reader grants run only in the deployment migration.

The service keeps one warm instance for Echo search latency but uses
request-based billing: CPU is throttled outside requests because periodic work
runs in Cloud Run jobs. The `hourly` runner refreshes every configured Echo
repository at `0 * * * *` UTC. EvalDash uses its own 1-vCPU, 1-GiB runner every
ten minutes.

IAM for the service account, the deploy account, and IAP viewers is declared in
the `marin` stack by `infra/pulumi/src/iac/gcp/marina.py`. Its bindings attach
to the repository, service, and bucket this stack creates, so a first deployment
runs this stack as a privileged operator and then applies the `marin` stack;
after that, apply `marin` whenever the grants change.

Data under the bucket is uploaded by hand when an app's files change:

```bash
gsutil -m rsync -r infra/marina/.data/tasktrove gs://marin-marina/tasktrove
```

## Publishing an applet

An applet is a small application uploaded without rebuilding Marina. Its UUID
is stable; each publish creates a numbered revision. Every applet gets a
Postgres schema and role named `applet_<uuid>` on its first publish. Production
URLs use `https://applets.marina.oa.dev/a/<uuid>/` so uploaded JavaScript does
not share an origin with checked-in apps or Marina's control API.

```text
my-applet/
  applet.toml
  dist/index.html
  server/__init__.py       # optional
  server/app.py            # optional
```

The manifest names the applet and, when present, its Python entry point:

```toml
title = "Problem sets"
description = "Browse generated math problem sets."
python_entrypoint = "server.app:create_api"
```

### Vue and other built frontends

Marina serves any frontend that builds into `dist/`. Plain HTML and JavaScript
fit a small single-view applet. Vue with Vite fits applets with multiple views,
reusable components, or substantial client state. Keep frontend sources beside
the files Marina packages:

```text
my-applet/
  applet.toml
  package.json
  package-lock.json
  index.html                # Vite source entry
  vite.config.ts
  src/
  dist/                     # generated and uploaded
  server/                   # optional and uploaded
```

Configure Vite to emit relative asset URLs:

```ts
import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [vue()],
});
```

Use `createWebHashHistory()` from `vue-router` for client-side routes. Hash
routes keep the server path at the applet revision root and avoid a router base
that contains the generated UUID and revision. Use relative backend URLs such
as `fetch("api/results")` and `fetch("query", ...)`; a leading `/` escapes the
applet prefix. Bundle dependencies into `dist/` because the applet content
security policy permits scripts from the applet origin only.

The manifest may define its local build:

```toml
build_command = "npm ci && npm run build"
```

`marina publish` runs a declared `build_command` before every publish by
default, including when `dist/index.html` already exists. Pass `--no-build`
only to reuse an existing bundle that has already been built and validated:

```bash
uv run marina publish .
uv run marina publish . --no-build
```

Publish a built directory:

```bash
uv run marina publish infra/marina/examples/problem-set-applet
```

The production target defaults to `https://marina.oa.dev`. The client uses the
same cached `iris login` credentials or ambient Google service-account
credentials as the other Marina CLIs; the caller needs IAP access to Marina.
On the first publish the server generates the stable UUID and records the IAP
identity's `user_id` as owner. `MARINA_APPLET_OPERATORS` is a comma-separated
list of those exact user IDs (normally email addresses in production).

The command prints the immutable revision URL under
`https://applets.marina.oa.dev/a/<uuid>/v/<revision>/`. A successful publish
runs the migration and then atomically changes the stable current URL,
`https://applets.marina.oa.dev/a/<uuid>/`, to that revision. To update the same
applet, pass both the UUID and the current revision:

```bash
uv run marina publish my-applet --update <uuid> --base-version 1
```

`--dry-run` validates and lists the package without sending it. Packages are
limited to 25 MiB, 8 MiB per file, and 2,000 regular files. Web files and Python
source are stored as inline blobs in Postgres and deduplicated by digest and
media type. Marina retains the current revision plus four other recent
revisions and removes blobs that no retained revision references.

Frontend assets must use relative URLs so they work below both the stable and
revision prefixes. From a revision page, `fetch("api/problems")` calls that
revision's Python backend and `fetch("query", {method: "POST", ...})` runs a
query for the applet. Marina falls back to `index.html` only for paths accepted
as HTML; missing asset paths remain 404.

Every applet has `POST /a/<uuid>/query`, which accepts one SQLAlchemy text
statement and an optional parameter object. Statements run as the applet's
generated Postgres role with its schema first on `search_path`. The role owns
that schema and inherits `marina_reader`, which can select from checked-in and
other applet schemas. It cannot write outside its own schema. Transaction and
role-control commands, data-modifying `WITH`, and mutation statements with
`RETURNING` are rejected. Postgres bounds query results before sending them to
the Marina process. Responses are limited to 10,000 rows and 5 MiB, with a
10-second statement timeout.

Agents can inspect applets, query them, and load small tables without writing an
API backend:

```bash
uv run marina applets list
uv run marina applets versions <uuid>
uv run marina applets sql <uuid> 'SELECT * FROM problems'
uv run marina applets table load <uuid> observations rows.parquet --replace
uv run marina applets rollback <uuid> 2
uv run marina applets archive <uuid>
```

Table loading accepts JSON arrays, JSONL, CSV, and Parquet with scalar columns.
The `marina.applet_catalog` view records active applets and the tables and
columns visible in their schemas.

A Python applet exports `create_api(services) -> ASGIApp` and may export
`migrate(connection) -> None`. `services` is an `AppletServices`; its
`engine()` method returns a SQLAlchemy `Engine` configured for the applet role
and schema. The migration receives a SQLAlchemy `Connection` inside the publish
transaction, so an exception leaves the old revision current and rolls back its
database changes. Before publishing, Marina imports the module in a
short-lived subprocess with environment credentials removed and verifies that
the declared factory and migration are callable. The server imports the
revision again under a unique module name, runs its migration as the applet
database role before activation, and serves its ASGI application at the
revision and current `/api/` paths. Applets may import only dependencies already
installed by `infra/marina/pyproject.toml`; publishing does not install packages.
A worker caches a failed runtime load and returns 503 for that revision. Publish
a corrected revision or roll back to a retained revision to bypass that cache;
retrying the same broken revision requires replacing or restarting the worker.

`services.engine()` assumes the applet role and schema. Python applets still run
inside the Marina process with its installed dependencies, service credentials,
filesystem, and network access. They are trusted plugins. They must not require
lifespan events, background workers, or local filesystem persistence.
Migrations must remain compatible with retained backend revisions because every
revision observes the same mutable schema.

For an end-to-end local run:

```bash
uv run marina publish infra/marina/examples/problem-set-applet --local
```

`--local` starts a disposable `pgvector/pgvector` Docker container, installs the
applet provisioning function and Marina registry, starts Marina on a free
loopback port, runs the applet migration through a normal publish, and prints
the immutable revision URL. It remains in the foreground so the URL stays live.
It requires a running Docker daemon, binds only to loopback, and does not enable
IAP authentication. Ctrl-C stops Marina and removes the container. `--local`
cannot be combined with `--dry-run`, `--update`, or `--base-version` because
every run starts with an empty registry.

Open the printed revision URL to exercise the exact published frontend and
backend. Use `marina publish ... --url http://127.0.0.1:8080` instead when
attaching to a separately managed local Marina server. Run
`marina applets versions <uuid> --url http://127.0.0.1:8080` before choosing a
rollback target on that persistent server.

On local Postgres, `marina migrate` installs the provisioning function when the
configured database user has `CREATEROLE`. In production the Marina Pulumi
stack runs `infra/marina/database_grants.py` as the Cloud SQL administrator,
then starts the migration job. The service account can execute that function
but does not need unrestricted role creation.

## Tests

```bash
cd infra/marina && uv run pytest        # kernel tests and every app's tests
```
