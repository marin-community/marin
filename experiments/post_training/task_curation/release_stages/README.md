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

For private GCS references such as the difficulty seal, submit forwards the
operator's application default credentials as an Iris file secret. It reads
`GOOGLE_APPLICATION_CREDENTIALS` when configured, otherwise the gcloud ADC file
under `CLOUDSDK_CONFIG` or `~/.config/gcloud`. The root and ordinary stage children
configure Google's client libraries to use the mounted file. Credential contents
are never included in the workspace bundle or command arguments.
Use `submit --job-name <name>` when the root must be registered before submission;
otherwise submit generates a release-scoped name.

Repository-relative pins (for example the registry index under `plans/`) resolve
from the checkout root: run `atlas-release` from the checkout.

## Verify curriculum input reuse

`public_input_reuse` checks every task belonging to its named target sources in a
fresh census against the archived curriculum through `curriculum_projection.task_rebinding`. Its typed inputs
name the release, frozen source revision, SHA-bound `archive_release` and
`census_receipt`, target source IDs and an output root. The census must bind the
same release and revision, and its public preparation must match its measured
declarations and task stream. Changed public content or ambiguous correspondence
refuses reuse.

Run the comparison in the data's region:

```bash
uv run python -m experiments.post_training.task_curation.release_stages.public_input_reuse \
  --inputs /workspace/reuse-inputs.json --mode dry-run
uv run python -m experiments.post_training.task_curation.release_stages.public_input_reuse \
  --inputs /workspace/reuse-inputs.json --mode write
```

The dry run checks every target population without publishing outputs. Write mode
seals task-identity mappings and `reuse-receipt.json`. It computes no embeddings
or alignment scores. The receipt establishes eligibility for curriculum semantic
alignment reuse; grader execution and reward correctness retain their separate
admission requirements. The composition stage still verifies its selected
replacement manifests before applying a rebind.

## The release plan document

The graph's regional `contract_census` stage consumes the frozen population census.
It reuses a registered contract only when its actor binding, population path and
scanned TaskSpec digest match. Changed populations receive a new complete contract
census; completed receipts resume without rescanning. The stage does not execute
graders or call models.

Contract task totals flow from the frozen target actor bindings through the
contract-census proof, snapshot publication, catalog staging and catalog loading.
A historical plan's registry-derived count cannot replace that measured total.
Resuming the same sealed plan preserves upstream census, corpus and embedding
artifacts. New plan derivations also count the frozen target bindings rather than
older registry receipts.

Retry an individual stage and its dependencies in the configured region through
the maintained submitter:

```bash
uv run atlas-release --version <release-id> --plan <sealed-plan-uri> \
  submit --cluster <regional-cluster> --stage snapshot_publication
```

The selected root reuses completed dependency artifacts. A standalone workstation
`run` has no ambient Iris client, so use `submit` for a stage that needs new regional
CPU or GPU work.

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
- The singleton `rebound_source`/`rebound_tasks` plan fields are not a proof of
  reusable alignment. Composition accepts per-source proofs through
  `public_input_rebindings` in its composition-input manifest, as described below.
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
measured sizing curves. A failed stage stops its downstream stages; fix it and run
the same command again.

An unrestricted run in the submitter-attested regional bundle builds through
`strict_audit_before`, including composition, scans and the serving loads. It prints
`REGIONAL_DATA_READY` after that audit succeeds. Packaging, prepublication browser
checks and app publication require a clean landed workstation checkout with Git,
Docker and Chromium, so the regional bundle does not execute those stages.
Continue the same sealed plan from that checkout:

```bash
MARIN_PREFIX=<regional-artifact-prefix> uv run atlas-release \
  --version <release-id> --plan <sealed-plan-uri> run
```

Use the regional root's artifact prefix (for example, the regional bucket's
`marin` directory) so the completed upstream artifacts remain cached. This is
the same maintained graph and plan; no stage inputs are reconstructed by hand.

Snapshot publication requires identity fingerprints for every declared population.
An `Excluded` inventory entry without an `actor_census_input` may have no
fingerprint. All other missing fingerprints, unknown source IDs, backend coverage
gaps and survey publication-binding mismatches refuse publication.

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
- **catalog_load QC replay** — the fresh-panel replay computes its record total
  before it starts (every panel's tables at their pins, from the loader's own row
  groups) and logs `PROGRESS replay done/total (percent, records/s, eta)` at a
  fixed cadence. The replay itself splits every panel's tables into page-range
  segments (at most `_REPLAY_SEGMENT_ROWS` each) and fans the segments of all
  panels over paced workers sized by `catalog_replay_sizing(...).workers`, so the
  five largest panels cannot bound the makespan. Each panel's validation and its
  cohort summaries plus pointer publish stay ordered around its segments, and the
  shared retry budget halts the peers on a failure. The parallel path stores
  exactly what the serial path stores (pinned byte-identical by test).
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
- Preflight receipts: `<release_root>/preflight/<node>/preflight-receipt/<context-sha>/<receipt-sha>.json`.
  The context binds the landed revision and input digests. Each attempt is immutable;
  changed upstreams or code create a new context, and retries append a new receipt.
  The gate reads the latest attempt in its exact context. Prior witnesses use a
  separate `prior-witness-receipt` directory and cannot satisfy the run-mode gate.
  Regional preflights attest the submitted code bundle's revision; the plan keeps
  its population-freeze revision separately.
  Each stage is gated before downstream handles capture it, so selecting a final
  node also gates its uncached dependencies. Upstream witnesses hash each
  dependency's `.artifact.json` record, not its output directory marker.
  Once upstream records exist, `preflight --stage <node>` checks those current
  records without re-executing the stage; before then it reports a blocked slice
  or checks prior-release state where a prior witness is available.
- Catalog readers remove null struct columns absent from the current typed
  schema, which Arrow can fill when historical and current records share a table.
  Unknown non-null fields still fail model validation; map entries and known
  nullable fields retain their values.
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

## Calendar release names and population replacements

New catalog releases use `atlas-YYYY.MM.DD`, with `.N` for another cut on the
same date. The config filename, `release_id`, and difficulty snapshot registry
key use that same name. Marina's `/v/NN` identifies an applet revision. Historical
release names retain their original spelling.

Derivation selects new source IDs and existing IDs whose fingerprint differs
from either the prior declarations or the retained curriculum inventory. This
also detects a curriculum projection that lagged behind an earlier catalog cut.
Every target passes the existing census, corpus preparation and fresh alignment
stages. Composition removes replaced populations' archived assignments, source
rows and coverage vectors before merging the fresh alignment. Assignments and
statistics for unaffected sources retain their existing values. The handoff
records replacement source IDs and scopes its unchanged-value evidence to the
retained sources.

Resuming the same frozen plan compares all derivation receipt fields except the
generation timestamp. If they match, the original receipt stays unchanged;
changed measured inputs refuse resume.

QC browse staging preserves only prior panels whose source inputs and selected
evidence still match the frozen declaration. Derivation omits changed panels
from that copy and counts copied panels separately from the final selected set.
The fresh-QC index combines copied SQL evidence with digest-bound review artifacts
selected by the frozen declarations through the maintained content-review loader.
Each review keeps the task IDs recorded in its artifact; its current-pin flag is
computed against the frozen declaration. Reviews drawn from another pin remain stale. Empty browse
copies are valid when all prior panels were omitted. Explicit policy exclusions
stay excluded. No new reviews are requested.

Catalog loading reuses panels whose receipt identities and row counts match the
sealed browse-copy completion and current QC index. It checks their selected
evidence and recorded current-pin flag against the current declarations: that
flag must agree with whether the reviewed population matches the declared
population. It then checks the target SQL presentation pointers in bounded
batches. A changed pointer or failed binding check stops catalog loading. Only new or replaced panels download and replay staged review payloads;
copied task, reviewer and QA identities remain unchanged.
The parallel loader commits every reviewer-batch segment and compares all
expected rows and fields against the sealed staged panel before dispatching
chunks and task links, which reference those batches through SQL foreign keys.
Segments run in parallel within each phase. The SQL pointer selecting the served
review panel updates only after both phases succeed.
The partial-QC resume check compares extra assessment headers with the
corresponding digest-verified staged QC panels. Every stored header field must
match, and the retained prior-header count must equal the staging receipt. An
unknown panel or changed header stops the resume.

To refresh an already staged index after a declaration changes, supply its
digest-bound `prior_index` reference to `fresh_qc_index` and put replacement
review references in `fresh_panels`. Use a new output root and current frozen
declarations. A different target load may reuse staged panels when the prior
index's browse completion and the new target's browse completion name the same
release and source load. Each completion must bind its respective target load.
For a retained panel copied to the new target, its stored review receipt must
match that panel's original receipt in the new browse completion. The stage
verifies every retained compressed panel's digest,
counts, receipt identity, input binding and restriction, preserving its existing
artifact reference. Changed task inputs, selected review or restriction require
a fresh panel for that source. This path makes no
SQL requests and records the prior index in the new index's provenance. Dry run
performs the same verification without writing. The resulting index still needs
the normal catalog-stage and catalog-load checks before release.

When a source is deferred from the release, list it in `removed_sources` and
omit its declaration and fresh panel. The index drops only explicitly removed
panels and records those identities in provenance. A missing declaration without
an explicit removal still refuses. Retained panel artifacts remain unchanged;
the stage does not delete historical evidence.

The content-review loader validates the selected artifact and completes panel
row checks before publishing its QC presentation pointer. A changed selection
updates the pointer for that load/source pair to reference the validated panel.
Historical review rows and catalog declarations remain unchanged by the
content-review loader.

## Compose verified public-input rewrites with fresh alignment

The composition-input manifest can contain `public_input_rebindings`, an array
of `{source_id, source_manifest, public_inputs}` records. Both manifest fields
are `{uri, sha256}` references to complete frozen public-input preparation.
`current_target_declarations` covers the union of these sources and the fresh
alignment inventory. A source must appear in only one group.

Aligned declarations may adopt updated license receipts without recomputing
alignment. The current receipt must pass the catalog's exact-pin permission and
source-ban checks. Task-input fingerprints, actor census bindings and task counts
must still match the aligned inventory.

The graph seals selected derived-contract receipts in `target-contracts.json`
and includes its digest-bound `target_contracts` reference in the composition
manifest. A declaration may adopt newly measured contract fields after alignment
only when its receipt binds the same source, complete actor census input and
scanned TaskSpec digest. The catalog's contract projection must reproduce every
current contract field. Missing, stale or contradictory evidence refuses
composition; retained alignment scores and unrelated declaration fields remain
subject to the existing checks.

When an actor census embeds only a preview of its integrity groups, plan
derivation resolves the full integrity artifact. Its population path, group
count, embedded preview and total task coverage must match the actor census.
Backend validation receives the complete group list and retains its
whole-population coverage check.

Identity/backend composition takes an explicit `snapshot_id` from the release
plan. Both output surveys and their receipts bind that publication snapshot;
input survey references and measurement code revisions remain unchanged. A
metadata-only snapshot can reuse verified survey inputs. Registry fingerprints,
runtime settings and the current backend schema are still checked.
Snapshot publication refuses either survey if its catalog root or snapshot ID
differs from the destination, before sealing or moving the selected pointer.

Each rebind proves every current public input against its archived population.
Stable IDs must preserve their public inputs. Changed IDs require unique public
content or verified immutable upstream rows. A smaller current population may
retain an injective subset; all omitted archived IDs are recorded in the identity
mapping with null new IDs. The entire archived source must have uniform readiness; otherwise the subset
requires fresh alignment, since source-level counts cannot establish the retained split.
Changed public inputs or ambiguous archived matches cannot reuse scores.

Composition applies these proofs in one transformation, verifies assignment
publication by readback, removes obsolete rows, and adds fresh alignment for changed public inputs. It rebuilds
statistics, confidence and review samples. Unaffected sources and complete
rebindings preserve their exact statistics; reduced populations must have the
new denominator and componentwise subset bounds. The loader checks each pinned
identity mapping against the selected source's task count, unique old/new IDs,
removal count and changed-ID count before accepting the scoped handoff.
