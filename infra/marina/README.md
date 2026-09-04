# Marina

One Cloud Run service that hosts Marin's small internal web apps. An app is a
directory under `apps/` with an `app.toml` and a `web/` frontend, and optionally
a Python package with an API. The kernel in `src/marina` discovers every app at
startup and serves each one under `/<name>/` on a single IAP-gated origin. What
an app would otherwise rebuild for itself lives in the kernel and the shared kit
in `web/`: login, a database, a data bucket, deployment, a top bar, and a
browser test harness.

## Layout

```
infra/marina/
  src/marina/      the kernel: manifests, auth, the FastAPI process, db, journeys, the CLI
  web/             the shared Vue kit apps import as @marina (tokens, Shell, Prose)
  apps/<name>/     one app: app.toml, web/, dist/ after a build, and for Python
                   apps an app.py with create_api() and migrate(), tests/, journeys/
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

## Tests

```bash
cd infra/marina && uv run pytest        # kernel tests and every app's tests
```
