---
name: marina-applet
description: Build, validate, release, inspect, query, roll back, or archive a Marina application. Maintained source apps and uploaded applets use Postgres packages and marina release, independently of Marina runtime deployment.
---

# Work with Marina applications

Read [Building Marina apps](../../../docs/references/marina.md) and the relevant
sections of [the Marina reference](../../../infra/marina/README.md). Treat source,
CLI help and those docs as authoritative. Use the existing examples as package
references.

Every application uses `marina release`. Source under `infra/marina/apps/` and
`infra/marina/applets/` has the same lifecycle. Marina image deployment and startup
never publish packages, select application revisions, or run application migrations.
Do not restart Marina to release an application.

## Source and package

Use `applet.toml`, `dist/index.html`, optional browser assets, and optional Python
modules. A declared `build_command` can generate `dist/`. Maintained sources declare
the stable `applet_id` and optional shared build inputs in `source_paths` within
the same `applet.toml`, as the named apps do for `infra/marina/web`. Keep existing UUIDs,
names, database schemas and data locations when updating an application.

Prefer plain HTML/JavaScript for small reports and Vue/Vite for substantial client
state. UUID applets need relative asset/API URLs, Vite `base: "./"`, and hash routing.
Named applications may retain their existing route bases. Bundle dependencies;
executable inline scripts and third-party script CDNs are blocked.

An optional `python_entrypoint = "server.app:create_api"` names a factory returning
an ASGI app. Root modules and `backend/` modules also work. Return
`registered_api(api)` to expose annotated FastAPI operations through MCP. Use relative
intra-application imports and installed shared libraries. Dependencies must already
exist in Marina's base environment.

Use `services.engine()` for role-scoped database access. An optional
`migrate(connection)` runs inside the release transaction; it must not commit,
install extensions, perform remote scans or start workers. Keep schema changes
compatible with retained revisions and in-flight requests/jobs.

Python packages are trusted plugins sharing Marina's process and service identity.
They have no security sandbox, reliable lifespan hooks or persistent local disk.
Separate dependencies, credentials or failure isolation require a separate service.

## Capabilities

A named app declares `name` in its manifest and uses `/<name>/`. First named
registration requires an operator. Data URLs, scheduled jobs and Loom agent metadata
are available to named apps; the server checks data prefixes, runner capacity and
per-application secret permissions against infrastructure-owned `runtime.toml`.
Never expand those permissions merely to make a test or release pass.

Job modules export `main()` and execute it under the usual `__main__` guard. Use
`marina.jobs.job_engine()` for the selected package's role/schema. Runners snapshot
and materialize packages before starting children, so a release does not replace
running job code. New runner capacity or secret access is an infrastructure change.

## Validate and preview

```bash
uv run marina validate my-app
uv run marina release my-app --dry-run --json
uv run marina release my-app --local --json
```

Release builds from empty frontend output and restores previous local output after
packaging, including on failure. Validation imports backend and job modules but
never invokes factories/migrations, connects to Postgres or drives the browser.
Exercise those paths in the local stack. Stop it with Ctrl-C and verify cleanup.

`release --local` starts disposable Postgres, explicitly releases the package and
prints a revision URL. Omitting `--url` from data commands still targets production.
Use the printed UUID/origin for local data loading and queries.

For source-app or kernel changes, run focused tests and the affected safe suite.
Browser journeys use an explicitly installed Postgres-backed preview:

```bash
uv run marina build
uv run marina dev
uv run marina journey evaldash
uv run pytest infra/marina
```

## Release

Treat a request to build or update an application as authorization to release it
and perform the requested schema/data writes. Honor requests for PR-only work,
local previews, drafts or review before release. Rollback, archive and unrelated
data mutations still require authorization within the request.

For maintained source, use a clean checkout at current remote main by default:

```bash
uv run marina release-status my-app --json
uv run marina applets versions UUID --json
uv run marina release my-app --base-version CURRENT_VERSION --json
```

Use `--allow-dirty` to explicitly release branch or uncommitted source without
verified Git provenance; `release-status` then reports `unrecorded` from a clean checkout. Use `--local`
for a disposable preview.

Use zero for a new declared UUID. Version conflicts mean the live selection changed;
inspect the new revision before retrying. Do not blindly reread the version and
retry stale source. Release preserves the access mode unless explicitly changed.
The default origin is `https://marina.oa.dev`, authenticated through cached Marin
login or ambient IAP credentials.

An ad hoc applet without a declared `applet_id` or Git repository uses the same command:

```bash
uv run marina release my-report --base-version 0 --json
uv run marina release my-report --id RETURNED_UUID --base-version CURRENT_VERSION
```

Record the first result's UUID and immutable revision URL. Maintained sources should
declare `applet_id` in `applet.toml`. A merge or runtime restart does not release source or load
application datasets. `release-status` compares recorded app/shared-input trees;
runtime dependencies also matter.

Atlas's code release and catalog load are separate. Follow
[its loader guide](../../../docs/references/rl-data-atlas.md). Never replace a serving
catalog with partial data, and never load remote data in a factory or migration.

## Data and lifecycle

Use Postgres for mutable state and reusable serving catalogs; keep large payloads
in object storage. Key caches by source version/query and record freshness. Load
remote data separately and select complete snapshots atomically.

```bash
uv run marina applets table load UUID observations rows.parquet --url ORIGIN
uv run marina applets sql UUID 'SELECT * FROM observations LIMIT 20' --url ORIGIN
uv run marina applets versions UUID --json
uv run marina applets rollback UUID RETAINED_REVISION
uv run marina applets archive UUID
```

Repeated table loads append. `--replace` drops before separate insert requests;
use staged loading for live catalogs. SQL/query response limits and role-control
restrictions are documented in the reference. Rollback selects a retained package
and does not reverse schema migrations. The registry retains the current revision
and four other recent revisions.

Unnamed applets default to private. Public mode is only for trusted GET/HEAD
handlers and data intended for unrestricted disclosure; Marina cannot prove GET
handlers have no side effects. Named apps stay private.

The first source-app cutover has an explicit bootstrap and readiness gate; follow
the reference rather than introducing filesystem fallback or automatic publication
in a runtime deployment. Report the UUID, stable/revision URLs, validation evidence
and any schema/runtime compatibility limits relevant to the released change.
