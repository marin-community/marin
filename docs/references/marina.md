# Building Marina apps

Marina hosts independently released application packages behind authentication.
Frontends, Python backends and current revisions live in Postgres. Every application
uses `marina release`; restarting or deploying Marina updates the base runtime and
preserves application selections. Merging source does not release it or load its data.

## Choose the application shape

Use a static package for a report, a Python backend for validated APIs, and a named
application when it needs stable named routes, scheduled jobs or the Loom panel for
asking an agent to work on the application. These
are capabilities of the same release mechanism. Maintained source under
`infra/marina/apps/` and `infra/marina/applets/` follows the same workflow.

Ordinary applets have stable `/a/<uuid>/` URLs on an isolated origin. Named apps such
as Echo and EvalDash retain `/echo/` and `/evaldash/` on Marina. Initial named registration
requires an operator. External storage URLs and job credentials need explicit base-runtime
configuration; uploaded packages cannot allocate infrastructure or grant themselves
new credentials.

Python backends are trusted plugins sharing Marina's dependencies, process,
resources and service identity. Applications needing separate dependencies,
credentials or failure isolation need their own service project under the
[infrastructure conventions](https://github.com/marin-community/marin/blob/main/infra/pulumi.md).

## Build and release

Start with `infra/marina/examples/problem-set-applet/` for Python/Postgres or
`infra/marina/examples/vue-applet/` for Vue. A package contains `applet.toml`,
`dist/index.html`, browser assets, and optional Python modules. Maintained source
declares its stable `applet_id` UUID and optional repository-relative
`source_paths` for shared build inputs in the same `applet.toml`.

```bash
uv run marina validate my-app
uv run marina release my-app --dry-run --json
uv run marina release my-app --local --json
```

Preview opens a disposable Postgres-backed local stack. Exercise the UI and API at
the printed revision URL, then stop the stack with Ctrl-C. Production releases from
maintained source default to a clean checkout at the current remote main:

```bash
uv run marina release-status my-app --json
uv run marina applets versions UUID --json
uv run marina release my-app --base-version CURRENT_VERSION --json
```

Add `--allow-dirty` to release branch or uncommitted source to the selected
server. This bypasses the Git checks and omits verified source metadata, so
`release-status` reports `unrecorded` from a clean checkout. Use `--local` for a disposable preview.

The first release uses `--base-version 0`. Each later release requires the inspected
current revision. Release builds from empty output, validates Python imports, runs
the application's migration and atomically selects the new package. Old frontend
assets and backend revisions remain available within the retention window. Failed
migrations roll back their database changes and leave the prior revision selected.
The stable UUID and existing access mode are preserved.

Ad hoc directories without a Git repository use the same command. Their first
`release --base-version 0` returns a UUID; later releases supply `--id UUID` and
`--base-version CURRENT_VERSION`. Declare `applet_id` in `applet.toml` when maintaining the source.

`release-status` compares recorded source trees, distinguishes local edits, and
reports `unrecorded` for packages without provenance. Declared shared build inputs
are included in the comparison; installed runtime dependencies also affect behavior. Source checks are operator
safeguards; server owner/operator permissions and version checks remain authoritative.

For source layouts, manifests, auth, runtime configuration and the first production
handover, see the [Marina reference](https://github.com/marin-community/marin/blob/main/infra/marina/README.md).

## Frontend and API

Use relative asset/API URLs in UUID applets. Vue/Vite packages use `base: "./"` and
hash routing so stable and versioned URLs both work. Bundle dependencies into
`dist/`; executable inline scripts and third-party script CDNs are blocked.

Named applications keep their existing router and API prefixes. They can consume
[shared browser components](https://github.com/marin-community/marin/blob/main/infra/marina/web/README.md) at build time, including
navigation and the agent panel. Updating those components requires releasing the
consuming app; the base runtime does not ship frontend code.

A Python factory returns an ASGI application, or `marina.apps.registered_api(api)`
for explicitly annotated MCP operations. The factory's `services` argument supplies the selected applet UUID,
revision, name, data URL and a role-scoped `engine()`. Read authenticated identity
through `rigging.server_auth.get_verified_identity()`.

An optional `migrate(connection)` receives the release transaction's SQLAlchemy
Connection. It must not commit independently. Keep migrations compatible with
retained revisions and running old requests/jobs. Shared extensions belong to base
provisioning. A Marina runtime restart never reruns application migrations.

## Load and cache data

Choose the serving data path before building the backend:

- Keep mutable state, reusable catalogs, filter indexes and summaries in the
  application's Postgres schema. A process cache disappears on restart.
- Load remote datasets separately from serving. Read a bounded committed snapshot
  when useful; never scan buckets or rebuild a catalog in a backend factory.
- Run refreshes in an explicit loader or a released scheduled-job module. Base
  configuration controls runner capacity, schedule and allowed credentials.
- Key caches by source version and query, record freshness, and serialize
  competing refreshes in Postgres. Preserve the last complete snapshot on failure.
- Keep large payloads in object storage, small immutable reports in the package,
  and live telemetry fresh through source queries or short TTLs.

For small scalar datasets:

```bash
uv run marina applets table load UUID observations rows.parquet --url ORIGIN
uv run marina applets sql UUID 'SELECT * FROM observations LIMIT 20' --url ORIGIN
```

Omitting `--url` targets production. Use the printed origin/UUID while a local
preview runs. Loading without `--replace` appends. `--replace` drops the table before
separate inserts; a live catalog needs staged loading and an atomic selection.
See [Atlas](rl-data-atlas.md) for a code release plus explicit catalog-load workflow.

## Operate

MCP discovery, named routes, directory entries and agent configuration follow the
selected Postgres revision without a restart. Jobs snapshot and materialize their
released packages before execution; an in-flight job keeps its snapshot while the
next execution selects newly released code.

```bash
uv run marina applets versions UUID --json
uv run marina applets rollback UUID RETAINED_REVISION
uv run marina applets archive UUID
```

Rollback explicitly selects a retained package; database migrations are not reversed.
The registry retains the current revision and four other recent revisions. A runtime
rollback must still support the active package APIs and dependency requirements.

Applets default to private. Public mode exposes GET/HEAD only on the public applet
origin and must contain only data and trusted handlers intended for unrestricted
access. Named apps stay private. See the [reference](https://github.com/marin-community/marin/blob/main/infra/marina/README.md)
for mode changes, query limits, deployment gates and recovery.
