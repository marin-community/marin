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

## Python playground

The trusted Python playground lives in `infra/marina/applets/python_playground/`.
Start its disposable local preview with:

```bash
uv run marina release infra/marina/applets/python_playground --local --json
```

Open the printed `url`. Stop the preview with Ctrl-C to remove its
disposable server and Postgres container, including all local projects and results.
To update a running preview, use its origin and inspected current package version:

```bash
uv run marina release infra/marina/applets/python_playground \
  --url LOCAL_ORIGIN --base-version CURRENT_VERSION --allow-dirty --json
```

This preserves its database. Starting another `--local` preview creates a new database.

Edit text files, upload binary or text inputs, select a `.py` entrypoint and list
one Python package requirement per line. Packages install through `uv run --with`
when execution starts. The host needs uv and access to the requested package
indexes. Run time includes package setup and is limited to 120 seconds.

Programs run in a fresh project directory under full CPython with the host's
privileges, network and inherited environment. Use this app only with trusted
users. A subprocess is not a security sandbox. The app does not use Shellsim.

Write `output/index.md` for the report, with relative image links such as
`![Posterior](posterior.png)`. Collected images under `output/` also appear in the image
gallery; other generated files are downloadable. `OUTPUT_DIR` contains the absolute
output path. The app creates `output/` before each run and removes the workspace
after collecting results. It does not mount `/output` at the host root.
Failed and timed-out programs retain collected files written before they stopped. The Logs
tab shows combined stdout and stderr, including Python tracebacks.

The index lists shared projects with name search, ascending/descending sorting by
name, creation time, last edit or last run, and pagination. Create a blank project
or import a project JSON file. Existing projects open from their names in the list.
The editor can load either example into the current project. Each editor has a
stable hash URL; the logo and title return to the project index. Edits autosave to Postgres; the Save button retries a failed save. Run
waits for pending saves before executing. Conflicting edits from another tab return
an error and keep the local editor open for export.

The applet's Postgres schema contains `projects`, `input_files`, `runs`,
`output_files` and `blobs`. Project and run metadata use typed columns; file contents
use binary `BYTEA` blobs. JSON is used for HTTP requests and project exports only.
Unchanged input blobs are reused when saving metadata edits. The app keeps the latest
completed run per project, including logs and downloadable outputs. Saving a new run's
results atomically replaces the previous output blobs, including when the new program
failed or timed out. Previous results stay in Postgres until that replacement commits.
Deleting a project removes all its files and results.
Results from an earlier source revision remain visible with a notice until the next run.
There is no total app storage or project-count cap.

Projects and results survive browser reloads and application releases. The local
preview uses disposable Postgres, so stopping it removes its database. Export source
projects and download any outputs you want to keep beyond that preview. Project exports
include source, uploaded binary/text inputs, entrypoint and package requirements; they
exclude run results. JSON import
creates a new project. The index can import a browser draft saved by the initial
prototype on the same origin. There is no durable run queue.

The Bayesian example computes a beta-binomial posterior and a failure budget from
editable `config.json`: the largest number of failures at the given trial count
whose posterior probability of exceeding the target success rate still meets the
required confidence. Its conclusions assume independent Bernoulli trials and
the selected prior. The Seaborn example generates a report from seeded synthetic
data without downloading a dataset.

Each saved project accepts at most 100 input files, including source and uploaded
inputs, totaling 5 MiB decoded. Results
include at most 100 files and 10 MiB of output, plus 256 KiB of combined stdout and
stderr. Oversized artifacts are skipped with a warning; logs are truncated. These
bound each project's saved inputs and latest output set. They do not limit a trusted
program's disk or memory use. Input saves above the limit fail without replacing
the existing files.

### Upload projects from agents and the CLI

Upload a directory from the repository root:

```bash
uv run -m infra.marina.applets.python_playground.cli upload ./my-analysis \
  --title "My analysis"
```

The command creates a new project and prints its ID, source revision and URL as
JSON. It reuses Marin login credentials. Add `--url http://127.0.0.1:PORT` for a
local preview. `--entrypoint` defaults to `main.py`; repeated `--package` options
override the root `requirements.txt`. Hidden paths, Python caches, `node_modules`,
`venv` and `output/` are excluded. Symlinks are rejected.

The raw upload endpoint is
`POST /a/95b59a50-a74b-4aa6-870c-b89d05f501ba/api/projects/import?title=NAME`.
Send a ZIP body with `Content-Type: application/zip` and project files at the archive
root. Optional `entrypoint` and repeated `packages` query parameters select the
entrypoint and override requirements. The compressed upload limit is 6 MiB; the
decoded project limit remains 5 MiB across 100 files. Imports create new projects.
Use `PUT /api/projects/{project_id}` with the expected source revision to update
an existing project through the JSON API.

The backend publishes `list_projects`, `get_project`, `create_project`,
`save_project`, `run_project` and `delete_project` through Marina's MCP discovery.
Find them through `/api/marina/mcp/` using `find_tool` with “Python playground”.
Read-only discovery exposes listing and reading. Execution is an external-write
operation because trusted Python can access the host's network and credentials.
The binary ZIP endpoint is HTTP-only; MCP creation accepts the same source/input
file records as the editor.

### Storage example and default runtime packages

Programs can import Marina's installed libraries, including `marin-rigging`,
PyArrow and FineStore, without adding them to the extra package list. uv resolves
extra requirements into its run environment first; Marina's installed packages
are appended to Python's import path. Execution uses Marina's Python version so
its installed native extensions remain compatible.

Choose “Object storage preview” under “Create a new example”. It reads a small
Parquet object through `rigging.fsutil.parquet.parquet_lines`, which uses the same
bucket routing as `filesystem_for()`. Edit `config.json` to change the `gs://` or
`s3://` object and preview row count. The report includes its schema and bounded
row preview, plus a downloadable text file.

The subprocess inherits the host environment, including `CW_KEY_ID` /
`CW_KEY_SECRET`, `R2_KEY_ID` / `R2_KEY_SECRET`, standard AWS variables and GCS
application default credentials when configured. Secrets are not added to project
files, database rows or browser requests. The example reports only whether each
key pair is available. A backend whose credentials are absent fails with the
storage library's credential error; this app does not grant storage access.

The “Ask Marina” button opens a Loom conversation beside the workspace, or as a
drawer on smaller screens. Conversations and drafts are scoped to each project.
The agent receives project identifiers, saved revision numbers, the selected file
name, and whether edits are unsaved. It reads saved code and results through
Marina's read-only MCP API; save editor changes before asking about them. The
panel automatically selects Loom's `marina` profile, which permits inspection
and advice.

Connection is best effort. Marina's runtime environment variable
`MARINA_AGENT_ORIGIN` supplies the Loom URL; environments without it show an
unavailable state. Browser connections require a Loom sign-in and Loom's
`browser.allowed_origins` setting to permit the private applet origin
(`https://applets.marina.oa.dev` in production). The production policy permits
both the main Marina origin and its private applet origin. Operators manage it in
`infra/loom/Pulumi.marin-loom.yaml`. The playground's `applet.toml` permits
connections to `https://loom.oa.dev` through `connect_src`; a different Loom URL
also requires updating that list.
