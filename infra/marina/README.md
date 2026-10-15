# Marina

Marina serves independently released applications behind IAP. Each application has a
stable UUID, immutable frontend/Python packages, a current revision in Postgres, and
an owned database schema. `marina release` is the application release command.

A Cloud Run rollout updates Marina and its installed dependencies. It does not
release application code, run application migrations, load datasets, or change
current application revisions. The production image contains no application source
or frontend bundles. A restart loads the same selected packages from Postgres.

See [Building Marina apps](../../docs/references/marina.md) for product guidance and
[RL Data Atlas](../../docs/references/rl-data-atlas.md) for Atlas's separate data loader.

## Source and package layout

Maintained sources live in `apps/<name>/` or `applets/<name>/`. Both directories use
the same manifest, builder, registry and release command. The directory name does
not select a deployment mechanism.

```text
my-app/
  applet.toml       # package metadata, optional stable UUID and shared source inputs
  dist/index.html   # built frontend
  server/app.py     # optional backend; root app.py and backend/app.py also work
  web/              # optional frontend source
```

A minimal manifest is:

```toml
title = "My app"
description = "Browse a small report."
python_entrypoint = "server.app:create_api"  # omit for a static application
build_command = "cd web && npm ci && npm run build"  # omit for tracked static files
connect_src = []
```

The archive contains `applet.toml`, files below `dist/`, and application Python
modules. Python files below tests, journeys, web, dependencies and caches are
excluded. Only `dist/` is browser-readable. Limits are 25 MiB total, 8 MiB per file,
and 2,000 regular files; large payloads belong in object storage.

Most applets use `/a/<uuid>/` on the isolated authenticated applet origin. A manifest
with `name = "echo"` registers the stable authenticated route `/echo/`. Initial named
registration requires an operator, and names cannot change on later releases.
Echo, EvalDash, TaskTrove and Zephyr keep their existing names, UUID
bindings in `applet.toml`, data locations and database schemas.

Named applications can declare an allowed `data_url`, scheduled jobs and an agent
panel. These capabilities are checked by the server against `runtime.toml`.
Ordinary unnamed applets cannot request them. Named applications remain private.

## Release an application

Add a maintained application's stable identity to its `applet.toml`:

```toml
# applet.toml (alongside title, description and other package fields)
applet_id = "01234567-89ab-cdef-0123-456789abcdef"
source_paths = ["infra/marina/web"]  # shared inputs, relative to the repository root
```

From a clean checkout at the current remote main:

```bash
uv run marina release-status my-app --json
uv run marina applets versions 01234567-89ab-cdef-0123-456789abcdef --json
uv run marina release my-app --base-version CURRENT_VERSION --json
```

Use `--base-version 0` for the first release of the declared UUID. It succeeds only
if that UUID is absent. Updates require the exact inspected current revision and
owner/operator authorization; concurrent changes return a conflict. A successful
release builds, validates imports, uploads the package, runs its migration and
atomically selects the new revision. A failed migration rolls back its database
changes and leaves the previous revision current.

Frontend builds start with empty `dist/`. The builder restores existing local
output after packaging, including on failure. Production releases always build;
there is no `--no-build` release path. Backend files and unbuilt static files must
be tracked by default. When `applet_id` is set, the default release checks clean
current main before and after the build and records its commit, source path and
hashes of the app and declared shared inputs in `dist/marina-release.json`.
The metadata contains no publisher identity and is
public when the applet is public.

Use `--allow-dirty` to explicitly release from a branch or uncommitted checkout.
It bypasses the Git and tracked-file checks and omits verified source metadata;
`release-status` reports `unrecorded` from a clean checkout. Package validation,
expected-revision checks and server authorization still apply. Use `--local`
for a disposable preview that leaves production unchanged.

`release-status` compares the published source with the checkout. It reports
`current`, `different`, `modified`, or `unrecorded` for packages without provenance.
Local edits take precedence: a dirty checkout reports `modified` even if the
released package has no source metadata.
Fetch main before assessing freshness. Declare repository-relative shared build inputs in `source_paths`; the four named
apps include `infra/marina/web`. Runtime dependencies can also affect behavior.

An ad hoc applet can use the same command without a Git repository or declared `applet_id`:

```bash
uv run marina release my-report --base-version 0 --json
uv run marina release my-report --id RETURNED_UUID --base-version CURRENT_VERSION
```

The first command allocates a UUID. Record it before updating. Maintained source
should declare `applet_id` in `applet.toml` so ordinary releases cannot
accidentally select another application. A declared UUID must match the upload
target, including direct API uploads. Source checks are client safeguards; server
owner/operator and version checks still apply to direct uploads and older clients.

Merging source does not release it. Changing shared browser components requires
releasing the applications that consume them. Dataset loading is a separate
operation; an application release must not scan storage or rebuild a catalog.

## Validate and preview

```bash
uv run marina validate my-app
uv run marina release my-app --dry-run --json
uv run marina release my-app --local --json
```

Validation builds the package and imports its backend and job modules in Marina's
installed environment. It does not call the backend factory, run migrations,
connect to Postgres, or drive a browser. Keep module imports free of operational
side effects. Inline executable scripts are rejected by the package validator;
load JavaScript from relative packaged files.

`release --local` starts disposable Postgres and a loopback kernel, releases the
package through the registry, and prints its revision URL. Stop it with Ctrl-C;
Marina and the container are removed. For another edit, rerun the preview. Omitting
`--url` from data commands targets production even while a preview runs.

To preview all maintained named source apps together:

```bash
uv run marina build
uv run marina dev
uv run marina journey evaldash
```

`dev` uses disposable Postgres and explicitly installs built source packages.
Production `serve` has no source-directory option and never installs packages.
Journeys explicitly install packages into their local test database as well.

## Frontends

Use relative asset/API URLs for UUID applets so both the stable URL and
`/a/<uuid>/v/<revision>/` work. Vite applications use `base: "./"` and hash routing.
Named applications can retain their existing `/<name>/` router base and API paths.
Their stable frontend routes use revalidation; immutable revision URLs retain
immutable caching. Missing named frontend assets are searched in retained revisions
so an open page can fetch an older hashed chunk after a release.

Dependencies must be bundled into `dist/`; third-party script CDNs and executable
inline scripts are blocked. Set `connect_src` for required browser fetch origins.
Use the shared [Marina browser components](web/README.md) at build time. The
application package includes their compiled output; a runtime rollout does not
replace it.

## Python backends and migrations

Declare an importable module and factory, for example
`python_entrypoint = "server.app:create_api"`. Relative imports stay within the
revision's private package namespace. Do not import another application's source
package by its global name. Shared helpers belong in installed `lib/*` packages.
Application releases cannot install dependencies; dependency changes require a
base-runtime rollout.

`create_api(services)` returns an ASGI application. To expose MCP tools, return
`registered_api(api)` with a FastAPI API:

```python
from fastapi import FastAPI
from marina.apps import registered_api
from marina.mcp import OperationRisk, operation_extension


def create_api(services):
    api = FastAPI()

    @api.get(
        "/summary",
        operation_id="summary",
        description="Read the current report summary.",
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def summary() -> dict[str, str]:
        return {"application": services.name}

    return registered_api(api)
```

Services provide `applet_id`, `version`, `name`, `data_url`, and `engine()`. The
engine assumes the application's role and searches its schema. Read the caller
through `rigging.server_auth.get_verified_identity()`.

An optional `migrate(connection)` runs on the release transaction's SQLAlchemy
Connection, before the revision becomes current. It must not commit independently,
start workers, install extensions, or perform remote scans. Keep schema changes
compatible with retained revisions and requests/jobs already running old code.
`marina migrate` upgrades only Marina's registry and base database provisioning.

Named applications adopt their existing schema in place on first release. Tables
and their dependent sequences move to the applet role; data is neither copied nor
reloaded. Existing reader grants on the four maintained schemas are retained.
Ordinary applets own `applet_<uuid-without-hyphens>`. Every application role inherits
`marina_reader` for cross-application reads and writes only its own schema.

Backends are trusted Python plugins sharing Marina's process, installed packages,
network and service identity. They are not security or failure sandboxes. The
backend factory must not depend on lifespan events or persistent local files.
Factories that fail are retried on the next request. Shared model instances belong
in `marina.process_cache.process_cache`. Applications own construction and choose
a namespace and key containing all resource settings. The cache serializes
initialization, retries failed factories and retains values until worker exit.
Echo uses it for ONNX models shared by retained revisions.

```python
from functools import partial
from marina.process_cache import process_cache

model = process_cache.get_or_create(
    "my-app.embedding", (model_name, threads), partial(load_model, model_name, threads)
)
```

Retained revisions using the same namespace and settings receive the same object.
Include a format version in the key if a new revision changes the resource's type
or construction contract. Resource construction functions passed to
`get_or_create` must not call the process cache.

## MCP and agent panels

`/api/marina/mcp/` exposes authenticated Streamable HTTP MCP. Clients discover
operations with `find_tool` and execute them through `call_tool`. Only explicitly
annotated operations are visible; operation IDs must match `[a-z][a-z0-9_]*` and
have descriptions. Tools retain application namespaces such as `echo_search`.
`/api/marina/mcp/read/` exposes only `OperationRisk.READ` operations.

Discovery and tool lookup resolve the current Postgres revisions. An app release
changes its tools without restarting Marina. HTTP and MCP calls carry the verified
Marina identity. Ordinary ASGI backends can omit the MCP projection.

A named application's released manifest can enable the Loom panel:

```toml
[agent]
profile = "marina"
repository = "marin-community/marin"
starters = ["What is on this page?"]
```

For an application named `my-app`, `/api/marina/agent/config?app=my-app` reads
its current released metadata. The base runtime supplies `MARINA_AGENT_ORIGIN`;
the page supplies bounded, versioned context through the shared `AgentPanel`
component. `MARINA_AGENT_ORIGIN` is added to that app's CSP.

## Scheduled jobs

`runtime.toml` declares Cloud Run runners, schedules, capacity, available secrets,
per-application permissions and existing job-trigger environment names. Pulumi
builds those base-image runners independently of application source. A package
can bind a module to a permitted runner:

```toml
[[jobs]]
name = "sync"
runner = "hourly"
module = "sync.main"
timeout = 7200
secrets = ["MARINMIRROR_TOKEN"]
```

The module exports `main()` and invokes it under an `if __name__ == "__main__"`
guard. Use `marina.jobs.job_engine()` for the selected package's schema and role.
The server rejects unknown runners, excess timeouts, and secrets outside both
the runner's and application's configured permissions. New capacity or credential
access requires an infrastructure change.

`marina run hourly` leases the runner, snapshots current packages in one database
statement, materializes them, and executes their modules in child processes.
Each child receives only its declared runner secrets. Running jobs keep their
snapshot through releases and revision pruning; the next execution selects the
new revision. Jobs do not apply application migrations.
Children retain Marina's working directory so shared libraries can find runtime
configuration. Locate files bundled in the package relative to `__file__`.

Echo's `sync.main` runs hourly; EvalDash's `ingest` runs every ten minutes.
`ECHO_SYNC_JOB` and `EVALDASH_INGEST_JOB` still name their existing Cloud Run resources.

## Data and lifecycle operations

Use an application's Postgres schema for mutable state, serving catalogs and
reusable indexes. Keep large payloads in object storage. Load remote data separately
and select complete snapshots atomically; preserve the last good snapshot on failure.
For small scalar tables:

```bash
uv run marina applets table load UUID observations rows.parquet --url ORIGIN
uv run marina applets sql UUID 'SELECT * FROM observations LIMIT 20' --url ORIGIN
uv run marina applets versions UUID --json
uv run marina applets rollback UUID RETAINED_REVISION
uv run marina applets archive UUID
```

Table loading accepts JSON arrays, JSONL, CSV and Parquet scalar columns. Repeated
loads append. `--replace` drops the table before separate insert requests; use a
staged load and atomic pointer change for a live catalog. Query responses are
limited to 10,000 rows and 5 MiB with a ten-second timeout. Transaction/role control,
data-modifying `WITH`, and mutations with `RETURNING` are rejected.

The registry retains the current revision and four other recent revisions.
Rollback explicitly selects a retained revision; it does not reverse database
migrations. Archival hides an application while preserving its schema and revisions.

Applets default to private. Use `release --mode public` or `applets mode UUID public`
only for data and trusted GET handlers intended for the internet. Public applets
use `https://public.applets.marina.oa.dev`; that service admits only GET and HEAD
and exposes no SQL, publication, MCP or named-app routes. Marina cannot prove that
a GET handler is free of side effects.

## Named applet hosts

Named applet hosts are stored in `marina.applet_hosts` in Postgres. Every request
resolves the hostname there, so rebinding a host reaches all serving Cloud Run
revisions without a deployment. Each bound host serves only its assigned applet
behind the same authentication. Its root redirects to `/v/<current>/`; assets,
queries, and Python APIs remain pinned to that applet revision. Existing URLs on
`MARINA_APPLET_ORIGIN` remain valid.
Named-host files require cache revalidation because a different applet can reuse
the same `/v/<revision>/` paths after a binding changes.

Pulumi's `marin-marina:applet_domains` lists only the hostnames whose DNS records
and Cloud Run domain mappings it provisions. Provision a new hostname through
Pulumi, then bind it with an identity listed in `MARINA_APPLET_OPERATORS`:

```bash
uv run marina applets hosts list
uv run marina applets hosts bind filer.marina.oa.dev 7f536ce3-0562-40f1-b209-e6110d8ab4e8
uv run marina applets hosts unbind filer.marina.oa.dev
```

The operator list is configured with `marin-marina:applet_operators`. An empty
list disables host management through the API. The commands use
`/api/marina/applet-hosts` on the canonical Marina origin. Rebinding and unbinding
do not change DNS. Unbinding retains a disabled hostname record that returns 404,
so the hostname cannot expose Marina's control routes. Marina's canonical, shared
applet, public applet, and legacy app hosts cannot be rebound through this API.

Pulumi passes the provisioned hostname list as `MARINA_APPLET_DOMAINS`, without
applet UUIDs. A listed hostname returns 404 until it has an enabled database
binding. Other hosts bypass the registry lookup. Binding an unprovisioned
hostname is rejected; unbinding an unknown binding returns 404.

The deployment migration imports the existing Filer binding once, before the
new service takes traffic. It leaves subsequent rebindings and deletions alone.
Fresh databases without the production Filer applet start with no bindings.

## Base runtime deployment and first cutover

The `marin-marina` Pulumi stack owns the authenticated service, public reader,
Cloud SQL resources, bucket and runner infrastructure. IAM stays in the global
`marin` stack. The production source guard requires clean current main, and CI
checks out main when a runtime deployment job starts. App source, examples and
shared browser code select Marina tests without selecting a runtime rollout.

The image carries Marina, installed shared libraries, `runtime.toml` and model disk
caches. Deployment runs `marina migrate --require-apps`; authenticated Cloud Run
startup also checks that the required named UUIDs are registered. Neither step
publishes packages. A first unattended rollout stops at this gate and leaves the
previous serving revision in traffic.

Perform the initial handover before the first runtime cutover. Prepare these
prerequisites while the existing service and workers still run:

- The operator needs `roles/iam.serviceAccountTokenCreator` on
  `marina@hai-gcp-models.iam.gserviceaccount.com`, plus its established Cloud SQL
  access. The Marina stack does not grant impersonation. Arrange an approved
  binding through the global `marin` IAM workflow before attempting bootstrap.
- Copy the `user` value from `/api/marina/me` on the deployed authenticated
  service for `OWNER_ID`. The default operator list is empty: only that exact
  publisher ID can update or archive the four named apps after bootstrap.
  To share those operations, commit the intended verified user IDs in the stack's
  `applet_operators` configuration and apply that configuration during cutover.
- Inspect schema and relation ownership through an administrative SQL connection.
  The following query must return no rows; resolve any other owners explicitly
  before release:

```sql
WITH application_schemas AS (
    SELECT oid, nspname, nspowner FROM pg_namespace
    WHERE nspname IN ('echo', 'evaldash', 'tasktrove', 'zephyr')
)
SELECT nspname AS schema, 'schema' AS kind, nspname AS object,
       pg_get_userbyid(nspowner) AS owner
FROM application_schemas
WHERE pg_get_userbyid(nspowner) <> 'marina@hai-gcp-models.iam'
UNION ALL
SELECT n.nspname, c.relkind::text, c.relname, pg_get_userbyid(c.relowner)
FROM pg_class c JOIN application_schemas n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
  AND pg_get_userbyid(c.relowner) <> 'marina@hai-gcp-models.iam';
```

Also inspect custom functions, enum types and domains in these schemas. Automatic
adoption covers schemas, tables, views and sequences; other application objects
need an explicit ownership plan. A failed adoption rolls back the registration.

Pause both scheduler triggers and wait for their running executions to finish.
Do not cancel running work. Keep them paused through the handover and rollout:

```bash
gcloud scheduler jobs pause marina-hourly-trigger --location us-central1 --project hai-gcp-models
gcloud scheduler jobs pause marina-evaldash-trigger --location us-central1 --project hai-gcp-models
gcloud run jobs executions list --job marina-hourly --region us-central1 --project hai-gcp-models
gcloud run jobs executions list --job marina-evaldash --region us-central1 --project hai-gcp-models
```

Run the following commands from the repository root in a clean current-main
checkout. The new kernel runs locally as the service account against the existing
database. Replace the literal `OWNER_ID` with the verified value collected above:

```bash
gcloud auth application-default login \
  --impersonate-service-account=marina@hai-gcp-models.iam.gserviceaccount.com
export CLOUDSQL_CONNECTION=hai-gcp-models:us-central1:marin-metadata
export PGDATABASE=marina
export PGUSER=marina@hai-gcp-models.iam
export MARINA_DATA_ROOT=gs://marin-marina
uv run marina migrate
uv run marina serve --host 127.0.0.1 --local-publisher OWNER_ID
```

The Postgres connection must use that service role, which owns the existing schemas;
the operator's personal database login cannot adopt them. Do not run the bootstrap
with `K_SERVICE` or `MARINA_IAP_AUDIENCE` set. `--local-publisher` is rejected with
IAP, in Cloud Run, or on a non-loopback bind. The local kernel admits loopback
callers only; the owner ID is the operator's explicit assignment.
Ownership changes wait at most five seconds for database locks. On lock contention,
leave the old service serving, identify the holder and retry after it finishes.

Leave that server running. In a second terminal at the same clean checkout's root,
release each named app through the local kernel. A separate checkout is unnecessary.
Inspect versions first if a previous attempt completed any releases:

```bash
uv run marina release infra/marina/apps/echo --base-version 0 --url http://127.0.0.1:8080
uv run marina release infra/marina/apps/evaldash --base-version 0 --url http://127.0.0.1:8080
uv run marina release infra/marina/apps/tasktrove --base-version 0 --url http://127.0.0.1:8080
uv run marina release infra/marina/apps/zephyr --base-version 0 --url http://127.0.0.1:8080
```

Verify existing records, named APIs, MCP discovery and job-module imports. In the
first terminal, stop the local kernel with Ctrl-C, then check registrations using
its existing database environment and dispatch the main-branch runtime workflow:

```bash
uv run marina migrate --require-apps
gh workflow run ops-pulumi-rollout.yaml --ref main -f service=marina
```

Verify the deployed named routes and scheduled runners. Resume both triggers
after successful cutover, or if aborting the handover and keeping the old runtime:

```bash
gcloud scheduler jobs resume marina-hourly-trigger --location us-central1 --project hai-gcp-models
gcloud scheduler jobs resume marina-evaldash-trigger --location us-central1 --project hai-gcp-models
```

Atlas and Filer keep their existing pointers throughout. No dataset load is part
of this handover.

Keep the previous service serving until the handover and runtime rollout succeed.
Application migrations must support that overlap. Prefer an explicit app rollback
for an application fault. A runtime downgrade must remain compatible with the
active package format, services and installed dependencies; it must not select
application revisions as a side effect.

If Pulumi reports a lock, identify its owning rollout/process first. Once that
owner has stopped, `uv run pulumi cancel --stack marin-marina` from `infra/marina/`
clears its lock; then dispatch the main rollout. Never cancel a live update to
make room. See the [Marina incident record](https://marina.oa.dev/echo/wiki/734).

## Verification

```bash
uv run pytest infra/marina
./infra/pre-commit.py --changed-files --fix
```

The HTTP/MCP, package registry, schema-adoption, job-snapshot and source-app suites
use disposable databases. Browser journeys run explicitly with `marina journey`.
