# atlas-release — the scripted Atlas release pipeline (R40 gate)

One entry point builds an Atlas release end to end. Every stage is a maintained
module in this package; the graph, the plan document and the guard live here too.
Nothing outside this repository is read or executed: no per-release scripts, no
`sed`ed workers, nothing under `~/Documents/experiments` (this README and the
sealed receipts the pipeline itself writes are the only things that ever appear
there). See `DESIGN.md` for the design note and `INVENTORY.md` (written to the
retired-scripts directory by `inventory.py`) for the retirement mapping.

## Operating credentials and repository-relative pins

The preflight and run modes read the release prefix on object storage, so the
operating environment must carry the storage credentials the repository's
storage layer resolves (the operator's secrets environment: `CW_KEY_ID` /
`CW_KEY_SECRET` for the CoreWeave buckets). The preflight type-check launches
`uvx`, which needs a writable uv tool directory (`UV_TOOL_DIR`) and cache
(`UV_CACHE_DIR`) — in a sandboxed environment point both at writable paths.
Every stage emits its alerts directly to the watchdog feed, which defaults to
the operator's `~/Documents/experiments/active/job-watchdog/alerts.jsonl` and
is overridden with `ATLAS_WATCHDOG_FEED` — a sandboxed environment points it at
a writable path (that variable changes where alerts are WRITTEN; it never makes
anything under `~/Documents/experiments` an input). Both job-wiring variables
have first-class flags on every mode: `--watchdog-feed <path>` sets the alert
feed and `--landing-log <path>` sets the atlas-push landing log the guard reads
(`ATLAS_PUSH_LOG`), so a sandboxed run states its wiring on the command line
instead of exporting environment variables.
Credentials are infrastructure access, not pipeline configuration — nothing
outside the repository supplies any pipeline input.

Repository-relative pins (for example the registry index under `plans/`) resolve
from the checkout root: run `atlas-release` from the checkout.

## The release plan document

Everything release-specific lives in one typed JSON document — the release plan.
Release plans are reviewed configuration: they live in this package at
`plans/<release>.json`, landed via atlas-push like any other config, and the
pipeline seals a write-once copy under the release prefix the first time it runs
(a changed plan means a new release prefix and a bumped `artifact_version`).
Nothing under `~/Documents/experiments` is ever an input. It is the only place a
release id, load ID or snapshot name enters the system; stage sources carry none
of them (`guard.py` refuses any that appear as literals). Pass the plan path to
every invocation — e.g. `plans/m3-dryrun.json` for the current dry-run cut.

Schema (see `ReleasePlan` in `graph.py` for the authoritative field list):

```json
{
  "kind": "atlas-release-plan",
  "release_id": "<release id, e.g. the version this cut is named after>",
  "source_revision": "<40-hex landed atlas-env-audit sha this release cuts from>",
  "release_root": "s3://<region-bucket>/marin/rl-data/atlas/preview-release/<cut-id>",
  "catalog_root": "s3://<region-bucket>/marin/rl-data/atlas/catalog/production",
  "artifact_version": "YYYY.MM.DD(.N)",
  "prior": {
    "release_id": "<prior release>", "load_id": "<32-hex prior catalog load>",
    "snapshot_uri": "s3://…/prior-snapshot", "snapshot_sha256": "<64-hex>",
    "catalog_uri": "s3://…/prior-catalog"
  },
  "prior_snapshot_id": "<prior assembly snapshot id>",
  "snapshot_id": "<this release's assembly snapshot id>",
  "targets": ["<newly admitted source ids>"],
  "expected_source_count": 0, "expected_browse_source_count": 0,
  "expected_contract_count": 0, "expected_contract_tasks": 0,
  "expected_selected_panels": 0, "expected_prior_sources": 0,
  "expected_panels": 0, "expected_population_tasks": 0,
  "region_prefix": "s3://<region-bucket>/marin/rl-data/atlas/",
  "populations": {"<source id>": "<frozen population artifact dir>"},
  "scan_regions": {"<region label>": "s3://<region-bucket>/…/"},
  "pins": {
    "prior_declarations": {"uri": "s3://…", "sha256": "<64-hex>"},
    "prior_identity": {"uri": "s3://…", "sha256": "<64-hex>"},
    "prior_backend": {"uri": "s3://…", "sha256": "<64-hex>"},
    "prior_release": {"uri": "s3://…", "sha256": "<64-hex>"},
    "prior_coverage": {"uri": "s3://…", "sha256": "<64-hex>"},
    "prior_manifest": {"uri": "s3://…", "sha256": "<64-hex>"},
    "backend_survey": {"uri": "s3://…", "sha256": "<64-hex>"},
    "backend_validation_receipt": {"uri": "s3://…", "sha256": "<64-hex>"},
    "registry_index": {"uri": "s3://…", "sha256": "<64-hex>"},
    "fresh_qc_index": {"uri": "s3://…", "sha256": "<64-hex>"},
    "oracle_inventory": {"uri": "s3://…", "sha256": "<64-hex>"},
    "source_counts": {"uri": "s3://…", "sha256": "<64-hex>"},
    "exposure_input_check": {"uri": "s3://…", "sha256": "<64-hex>"},
    "parity_receipt": {"uri": "s3://…", "sha256": "<64-hex>"}
  },
  "service": {
    "service_url": "https://marina.oa.dev",
    "applet_id": "<preview applet uuid>",
    "applet_origin": "https://applets.marina.oa.dev/a/<applet-id>",
    "public_root": "https://atlas.<host>",
    "production_version": 0,
    "preview_base_version": 0
  },
  "measured": {
    "module": "<backend survey module path>", "revision": "<its revision>",
    "sha256": "<64-hex module digest>", "schema_sha256": "<64-hex schema digest>"
  },
  "catalog_target_load_id": "<32-hex, pre-allocated>",
  "curriculum_target_load_id": "<32-hex, pre-allocated>",
  "prior_curriculum_load_id": "<32-hex>",
  "embed_cache_root": "s3://…/embedding-cache",
  "package_output_root": "<local dir OUTSIDE the repo checkout>",
  "gpu_runtime": null,
  "previously_available": ["<source ids Available in the prior selected load>"],
  "qc_exclusions": [], "migrations": [],
  "rebound_source": "", "rebound_tasks": 0,
  "strict_audit_local": false,
  "slice_output_root": ""
}
```

Field notes:

- `prior_manifest` may be `null` for the first release published from this pipeline.
- `parity_receipt` + `gpu_runtime` decide GPU vs CPU embedding: both set → CUDA
  (a valid parity receipt for the serving runtime is required, BRIEF requirement 4);
  both null → CPU.
- `rebound_source`/`rebound_tasks` are typed passthroughs. The maintained
  additions-mode compose refuses a declared rebound (the projection pass is not
  wired into the compose yet) — resolve the rebound before composing.
- `slice_output_root` (optional) redirects preflight slice artifacts; the default
  is `<release_root>/preflight-slices`.

## Modes

All modes validate `--version` against the plan (`plan.release_id`) first.

### `plan` — the lowered graph, I/O-free

```bash
uv run atlas-release --version <release-id> --plan <plan.json> plan
```

Prints every node of the lazy artifact graph (name, dependencies, fingerprint) and
lists the full dry-run schedule. Nothing is executed, nothing is written. Use this
to review the build shape before anything touches a cluster or a store.

### `preflight` — per-stage type-check + small slice on real inputs

```bash
uv run atlas-release --version <release-id> --plan <plan.json> preflight \
  [--stage <node>]... [--source <source id>] [--max-files N]
```

For each requested node: a `pyrefly` type-check of its module plus a small-slice
run of the stage's real code path on a bounded slice of real inputs (one census
source, one corpus preparation, the real prior-load SQL reads, the real registry
compose verification — never synthetic fixtures). Each slice seals a write-once
`PreflightReceipt` under `<release_root>/preflight/<node>/`, binding the landed
repo sha and the slice-input digests.

Receipts are write-once and bind the landed source sha: a receipt whose repo sha
or slice inputs no longer match is STALE, and a stale receipt is never
overwritten. The documented rerun procedure is a FRESH RELEASE PREFIX — update
the plan's `release_root` (bump the numeric suffix), which moves the lazy-graph
identities and the receipt root together, then rerun. This is the same
fix-bump-retry discipline the release artifacts themselves follow
(RELEASE.md: each attempt gets a new artifacts directory).

Slices available before any write-mode upstream receipt exists: `declarations`,
`census_inputs`, `census`, `corpus_assembly`, `embed`, `admission_check`,
`history_staging`, `identity_backend_compose`, `qc_browse_plan`, `package_preview`.

Every other node is reported `BLOCKED` with its reason — its real inputs are
write-mode receipts of upstream stages, so its preflight runs after those stages
complete (see the list in `preflight_slices.BLOCK_REASONS`). Blocked nodes do not
fail the mode; a failed slice does (exit 1).

### `run` — the submission path

```bash
uv run atlas-release --version <release-id> --plan <plan.json> run [--stage <node>]...
```

The brief's one-command shape also works: a leading `<stage|all>` token is the run
dispatch for that selection — `python -m experiments.post_training.task_curation.release_stages
census --plan <plan.json> --version <release-id>` runs the census stage, and `all`
(or no selection with an explicit mode) runs the whole release. A bare invocation
with neither a mode nor a selection is a usage error, never a whole-release run.

The submission guard (`guard.py`) refuses — with no bypass — unless the checkout
HEAD is landed on `atlas-env-audit` and no stage source carries release-identifying
literals. Then every stage gates on its own fresh preflight receipt immediately
before it executes (a runnable slice runs automatically if the receipt is missing
or stale; a blocked node without a fresh receipt refuses the stage), and the lazy
graph builds the release: driver-local stages run in the entry-point process,
cluster stages submit through `marin.execution.remote` with resources from the
measured sizing curves. A failed stage stops its downstream stages; fix and use
`resume`.

### Per-source parallelism and sizing

The heavy stages fan out across sources inside one job; the worker counts ride the
typed stage inputs and derive from the measured sizing curves (`sizing.py`), never
from silent defaults:

- **census** — `measure_source` runs across a spawn `ProcessPoolExecutor`, one
  source per dispatch, with the worker count from `census_sizing(...).cpu`. Each
  worker builds the source registry once (the run's revision is frozen) and pins
  pyarrow to one core. The results reassemble in the declared row order and every
  per-source seal stays write-once at its source-keyed path, so a parallel run
  seals exactly the rows a serial run seals.
- **corpus_assembly** — `prepare_sources` runs the per-source preparation across
  workers sized by `corpus_sizing(...).workers`.
- **qc_browse** — the panel copy fans out across paced SQL clients (one client per
  copy thread, the shared retry budget halting peers on a failure), with the worker
  count from `qc_browse_sizing(...).workers`; panels are disjoint (one pipeline key
  each) and the results collect in the plan's declared order.
- **history_staging** — already parallel (see `history_staging_sizing`).

Per-stage receipts are byte-identical between a serial and a parallel run modulo
the output root; the census's identical-rows property is pinned by tests and was
verified in-region against the sealed R40 cut-v22 census.

### `status` — what has run

```bash
uv run atlas-release --version <release-id> --plan <plan.json> status [--json]
```

Per node: the lazy-artifact record state under the storage prefix and the preflight
receipt state (`fresh`, `stale`, `failed`, or `missing`).

### `resume` — failed stages restart alone

```bash
uv run atlas-release --version <release-id> --plan <plan.json> resume [--stage <node>]...
```

A rerun with `force_run_failed`: stages whose artifact records exist with matching
fingerprints are skipped (pruned by the runner); failed stages restart alone; a
changed plan fingerprint is a hard refusal (bump `artifact_version` and use a fresh
release prefix instead of mixing recipes).

## The main-source sync stage

`main_source_sync` keeps the release aware of the training sources origin/main
declares. The node is wired into the graph for every plan (visible in `plan` and
`status`), but it is strictly plan-gated: a plan opts in by declaring

```json
"main_source_sync": {"main_source_revision": "<40-hex origin/main sha>", "queue_path": null}
```

(`queue_path` relocates the ingest queue; the operator default follows the
watchdog-feed pattern and `ATLAS_MAIN_SYNC_QUEUE` overrides it.) A plan without
the opt-in never executes the node — it hangs outside the terminal's dependency
edges, so opting in or out moves no other stage's fingerprint — and an
unrestricted `run` executes it on every build whose plan opts in.

Each run evaluates main's `all_sources()` at the pinned revision (the checkout
must contain it — `git fetch origin main` first), canonicalizes IDs through the
alias table, and diffs the training sources against the build's declared set plus
the Atlas's standing decisions (pending intake, retired admissions, license bans,
metadata exclusions). It seals a typed report write-once under
`<release_root>/main-source-sync/` (bound into the release's graph receipt) and
appends each new source to the ingest queue as one unclaimed `P1` row under the
queue's `.candidates.lock`. It never auto-admits, never removes sources and never
edits a declaration; the append is idempotent (a source the table already names is
skipped), so repeat runs and later builds make no duplicate rows. Its preflight is
the upstream-witness receipt over the sealed declarations, so it prefights after
`declarations` has run.

## Where artifacts land

- Release documents and stage receipts: `<release_root>/<stage>/…`, sealed
  write-once with byte-identical readback.
- Lazy-artifact records (provenance, config, typed results): the storage prefix,
  `atlas-release-<release-id>-<node>/<artifact_version>`.
- Preflight receipts: `<release_root>/preflight/<node>/preflight-receipt.json`.
- Stage alerts: every stage emits start/finish/fail events directly to the ops
  feed / job-watchdog `alerts.jsonl`; each stage submission proposes its own
  watchdog registry entry (the registry stays operator-owned).

## Acceptance

A fresh cold reader must be able to dry-run the whole build from a clean checkout
with this README and the entry point alone: `plan` lists the full graph I/O-free,
`preflight` exercises every runnable slice on real inputs and reports the blocked
ones with reasons, `status` reports state, and `run` refuses anything unqualified
through the guard — all without reading or executing anything outside the
repository.

## Release-bound difficulty seals

`difficulty_snapshots.json` binds a named future cut to a verified immutable
Difficulty manifest URI and SHA-256. Loading that cut's plan incorporates the
binding into `ReleasePlan.difficulty_snapshot` and catalog-load identity; a
conflicting explicit plan binding fails closed. Other release IDs retain their
existing inputs.

Catalog-load preflight verifies all sealed bytes. Write mode uses the maintained
`load_difficulty_seal` loader, reads its records back, and binds the seal to the
new load without overwriting a prior binding. Atlas source columns and aggregate
coverage use only cells matching the loaded declaration's population. Partial
cells keep their attempt counts while difficulty and pass rate remain null.
The column retains the immutable manifest receipt for inspection.
