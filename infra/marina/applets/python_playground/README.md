# Python playground

Edit and run trusted Python projects, upload input files, and view generated
Markdown reports, images and downloads. Projects and the latest results are saved
in Postgres.

Open the [production playground](https://applets.marina.oa.dev/a/95b59a50-a74b-4aa6-870c-b89d05f501ba/).
Bookmark this stable URL to follow new releases. URLs containing `/v/NUMBER/`
pin a specific application revision; remove that segment to open the current
version while keeping the `#/projects/PROJECT_ID` fragment for the same project.
Run the commands below from the repository root.

## Local preview

Start a disposable local preview:

```bash
uv run marina release infra/marina/applets/python_playground --local --json
```

Open the printed `url`. Stop the preview with Ctrl-C to remove its
disposable server and Postgres container, including all local projects and results.
The origin is the URL's scheme, host and port, such as `http://127.0.0.1:35993`;
the printed `version` is the current package version.
To update a running preview, use its origin and inspected current package version:

```bash
uv run marina release infra/marina/applets/python_playground \
  --url LOCAL_ORIGIN --base-version CURRENT_VERSION --allow-dirty --json
```

This preserves its database. Starting another `--local` preview creates a new database.
Inspect revisions before a later update with:

```bash
uv run marina applets versions 95b59a50-a74b-4aa6-870c-b89d05f501ba --url LOCAL_ORIGIN --json
```

## Editing and execution

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

## Projects and persistence

The index lists shared projects with name search, ascending/descending sorting by
name, creation time, last edit or last run, and pagination. Create a blank project
or import a project JSON file. Existing projects open from their names in the list.
Use “Create a new example” on the index to create a separate project. The editor
can also replace the current project with an example after confirmation. Each editor
has a stable hash URL; the logo and title return to the project index. Edits autosave
to Postgres; the Save button retries a failed save. Run
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

## Examples

The Bayesian example computes a beta-binomial posterior and a failure budget from
editable `config.json`: the largest number of failures at the given trial count
whose posterior probability of exceeding the target success rate still meets the
required confidence. Its conclusions assume independent Bernoulli trials and
the selected prior. The Seaborn example generates a report from seeded synthetic
data without downloading a dataset.

## Storage limits

Each saved project accepts at most 100 input files, including source and uploaded
inputs, totaling 5 MiB decoded. Results
include at most 100 files and 10 MiB of output, plus 256 KiB of combined stdout and
stderr. Oversized artifacts are skipped with a warning; logs are truncated. These
bound each project's saved inputs and latest output set. They do not limit a trusted
program's disk or memory use. Input saves above the limit fail without replacing
the existing files.

## Upload projects from agents and the CLI

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
Production HTTP requests go to `https://applets.marina.oa.dev`. The upload CLI
handles authentication; Python clients can use `marina.client.bearer_token()`
after Marin login:

```python
from pathlib import Path

import requests
from marina.client import bearer_token

response = requests.post(
    "https://applets.marina.oa.dev/a/95b59a50-a74b-4aa6-870c-b89d05f501ba/api/projects/import",
    params={"title": "My analysis"},
    headers={"Authorization": f"Bearer {bearer_token()}", "Content-Type": "application/zip"},
    data=Path("project.zip").read_bytes(),
    timeout=120,
)
response.raise_for_status()
project = response.json()
```

To update a project, GET `/a/95b59a50-a74b-4aa6-870c-b89d05f501ba/api/projects/{project_id}`
for its current `revision` and `document`, then PUT to the same path with
`{"revision": current_revision, "document": edited_document}`. The document contains
`title`, `entrypoint`, a newline-separated `packages` string, and a `files` array.
Each file has `path`, `content` and `encoding`: use `utf-8` for text or `base64` for
base64-encoded binary contents. PUT replaces the full document and rejects a stale
revision. These are also the document fields accepted by MCP creation.

The backend publishes `list_projects`, `get_project`, `create_project`,
`save_project`, `run_project` and `delete_project` through Marina's MCP discovery.
Find them through `https://marina.oa.dev/api/marina/mcp/` using `find_tool` with
“Python playground”. See [Marina MCP](../../README.md#mcp-and-agent-panels) for
the shared discovery and operation conventions.
Read-only discovery exposes listing and reading. Execution is an external-write
operation because trusted Python can access the host's network and credentials.
The binary ZIP endpoint is HTTP-only; MCP creation accepts the same source/input
file records as the editor.

## Storage example and default runtime packages

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

## Agent pane

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
