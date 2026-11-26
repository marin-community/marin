# Atlas preview release procedure

This is the canonical release checklist for the maintained Atlas app. Use the
repository checker below for each new preview revision. Release-specific scripts
and old deployment records are historical evidence, not a substitute for these
gates.

The target is app
`c9742a15-fb46-4d56-bc88-aa916351ebd3`, public on `atlas.oa.dev` since revision 35.
Pass `--expected-mode public` for its release checks. The checker asserts the
selected mode and does not change it. The archived prior app
`fb11c931-5861-4878-8bb5-a964d652b45f` must remain at revision 76. This procedure never publishes production or changes access mode,
datasets, infrastructure or service configuration.

Run the commands below from the worktree root. Install the coordinator-provided
`atlas-push` command on this machine; it is the temporary lock/landing tool in
[Current admission policy](../../../../docs/references/task-curation.md#current-admission-policy).
Use the [Marina development environment](../../../../infra/marina/README.md#validate-and-preview),
authenticate with `uv run iris login`, and install Chromium with
`uv run playwright install chromium`. Journeys start disposable Postgres through
Docker, so a working Docker daemon is required. Alternatively,
`MARINA_TEST_DATABASE_URL` may point to a test server where the test user can
create/drop throwaway databases; never set it to a serving database.
`release_check` owns one disposable Postgres server for its local API tests and
browser journeys. It passes the server URL to child processes through
`MARINA_TEST_DATABASE_URL`, then removes the container and anonymous volume
when those commands finish or fail. `marina journey` also keeps ownership in its
parent process, so an abrupt pytest exit cannot skip cleanup. SIGTERM sent to
the owning process runs the same cleanup; removal failures fail the run.
A server supplied through `MARINA_TEST_DATABASE_URL` is borrowed, never stopped.
Cleanup uses the exact container name created by the run, not a prefix sweep.
The `marina.local-postgres.owner-pid` label is diagnostic ownership metadata.

Sync this checkout's own environment with
`uv sync --frozen --package marin-marina --package marin-dupekit --group test`;
the deduplication package is needed by the repository-wide type check. Do not reuse editable
workspace libraries from another worktree.

## Prepare a landed cut

Work in a dedicated worktree under `marin-worktrees/`, never the main checkout.
Complete the required advisory review and resolve its findings before landing.
Review findings require judgment; retain the advisory artifact with the release
record. Follow [manage-pr](../../../../.agents/skills/manage-pr/SKILL.md) for the
review command and findings workflow. `--who` takes this worker's exact Rhizome
session ID. While CI is suspended, workers share `atlas-env-audit` as the
integration branch and land committed source with the locked workflow:

```bash
atlas-push --test "uv run --no-sync pytest infra/marina/tests/test_rl_data_catalog.py infra/marina/apps/rl_data_catalog/tests -q" \
  --who "<session id>" --what "<change summary>"
```

Keep the checkout clean at that landed cut. The checker fetches the integration
branch and proves the checked commit is landed. A later unrelated landing does
not require rebuilding this cut.

Choose a fresh artifact directory outside the checkout. Set `PACKAGE`, `COUNTS`,
`INVENTORY`, `ARTIFACTS` and `PRODUCTION_HOLD` explicitly. `COUNTS` is the qualified
current-input count manifest; `INVENTORY` is the full-population oracle inventory
ledger. Their exact bytes and the package's source/model provenance are checked
and recorded. Do not infer admission from a source's converted count, QC or
backend fit. Preparing a package does not authorize a new catalog load.

The curation/evidence owner supplies the pinned count manifest and inventory
paths for the intended catalog selection; keep their upstream audit receipts.
The count manifest binds retained populations and unique static exclusions to
exact source inputs. The inventory scans each source's full retained population
for original oracle solutions. See [preview packaging](../../../../docs/references/rl-data-atlas.md#preview-packaging-before-the-shared-runtime-is-updated).
Obtain `PRODUCTION_HOLD` from the coordinator's release directive, never by
silently accepting whatever production currently serves. Example path setup:

```bash
ARTIFACTS=/absolute/path/to/new-release-artifacts
PACKAGE="$ARTIFACTS/package"
COUNTS=/absolute/path/to/qualified-source-counts.json
INVENTORY=/absolute/path/to/full-population-oracle-inventory.json
PRODUCTION_HOLD=...  # Replace with the coordinator's held revision number.
mkdir -p "$ARTIFACTS"
```

`PACKAGE` must not exist; `prepare_preview` creates it. `ARTIFACTS` may exist,
but each receipt name and its matching artifact subdirectory must be new.

```bash
uv run python -m infra.marina.apps.rl_data_catalog.prepare_preview \
  --checkout "$PWD" --output "$PACKAGE" \
  --source-counts "$COUNTS" --oracle-inventory "$INVENTORY"

uv run python -m infra.marina.apps.rl_data_catalog.release_check \
  --dist "$PACKAGE" --source-counts "$COUNTS" --oracle-inventory "$INVENTORY" \
  --production-version "$PRODUCTION_HOLD" --expected-mode public --receipt "$ARTIFACTS/preflight.json"
```

`--dist` takes the package directory containing `applet.toml`, not its `dist/`
subdirectory. Use a synced Marin environment with Chromium installed for
Playwright and disposable Postgres available for the installed-app journeys.

## Automated gates

Preflight and hosted invocations run these blocking gates and exit nonzero on failure:

| Gate | What it proves |
| --- | --- |
| Landed cut and preview manifest | Source is clean, committed and present on freshly fetched `origin/atlas-env-audit`; the package targets the private preview UUID. |
| Package provenance and validation | Bundled model/source bytes match the cut; count observations and oracle inventory match explicit inputs; Marina package/import validation passes. |
| Lint, API/count tests and installed journeys | Repository mechanical checks and the app's behavior tests pass; journeys exercise the app through disposable Postgres and the kernel. |
| Runtime task accounting | Per-source populations partition into admission, exclusions and pending/unmeasured tasks; admitted tasks split into TRAIN and EVAL without conflating EVAL with TRAIN selection. |
| Main table at 1440/390, light/dark | Every displayed Examples value matches its bound observation. Headers, cells and chips fit; no chip subtitles or page overflow; filters and Show everything work and persist. |
| Header reconciliation | Shown + hidden QUICK/BAD + filtered + folded equals total tasks, and the same admitted subsets equal total admission. Quality totals use task populations. |
| Curriculum API invariants | Every selected/previous release × TRAIN/EVAL/all × exported confidence level × domain reconciles outcomes to its denominator and aggregate counts to source vectors. Display projections agree with the recorded cutoffs; every taxonomy node, including empty nodes, is present and capability labels differ from their subjects. Parents include their children plus remainder. |
| Browser pass | Preflight: representative (below). Hosted: the full sweep. Post-publish: the families whose content changed since the hosted sweep, in full. Representative: one selected release, Use and measured domain per layout loads the confidence control and sampled-task list. One Evidence/Backends presentation and each authored page family pass geometry and readability checks, with screenshots. |
| Status keyboard access | Every reason summary and evidence link stays visible when reached with Tab, including deep rows and opened explanations, at all release widths in both themes. |
| Catalog and archive preservation | Code-only releases retain selected catalog/curriculum pointers and source declarations. Every previously retained assessment and packet content checksum remains unchanged by immutable pipeline/task identity, even if an intended catalog refresh copies history to a new load ID. |
| Production hold | Production's actual revision matches the explicit hold both before and after checking. |
| Hosted revision | The post-release API, package provenance and oracle inventory match the preflight baseline; the same browser and preservation gates pass against the immutable revision URL. |

Local real-data views serve the package's browser assets using responses from the
selected live catalog, with the package's input-bound count projection. They do
not load or mutate the live catalog. The post-release run verifies the actual
hosted API. Within each run, read-only API responses are cached across layouts;
catalog pointer checks bracket the sweep to reject a mixed selection.

The version-2 JSON receipt contains the checked cut, input hashes, expected source
projection, before/after catalog evidence, command results/log paths, rendered
headers, task partitions, API invariant coverage/timing, browser results and response hashes. It records
`passed: false`, the failing gate and a nonzero exit on failure. Logs and screenshots
live beside it in a directory named after the receipt without `.json`. Existing
receipts and artifact directories are never overwritten; choose a new name for
each attempt. No release-specific source, domain or task totals are compiled
into the checker.

## Publish and verify

The source Evidence browser gate requires the source title and core evidence
within 15 seconds. Auxiliary ledgers have independent deadlines and visible
errors; they must not block the core page. Archive storage and preservation
checks remain required even though there is no Review Archive UI.

Eyeball the preflight screenshots in all four layouts. Automated geometry cannot
judge typography, readability or whether a label communicates the intended
policy. Resolve any unexpected or missing evidence before publication.

Use the preflight receipt's `preview_base_version` as `BASE_VERSION`; do not
silently replace it after a conflict. Publish only the checked package:

```bash
uv run marina release "$PACKAGE" --base-version "$BASE_VERSION" --allow-dirty \
  --json > "$ARTIFACTS/release.json"

uv run python -m infra.marina.apps.rl_data_catalog.release_check \
  --revision "$REVISION" --baseline "$ARTIFACTS/preflight.json" \
  --production-version "$PRODUCTION_HOLD" --expected-mode public --receipt "$ARTIFACTS/hosted.json"
```

Set `REVISION` from `release.json`'s `version`; its `url` is the immutable URL and
its `id` must equal the preview UUID above. A revision conflict requires
inspection and a new preflight; never retry stale source against a new base.
Do not set mode/public flags. Do not restart Marina. The checker performs no
release, rollback or catalog load itself.

For an independently authorized catalog refresh between preflight and hosted
verification, pass `--refresh-snapshot <exact intended catalog snapshot>` to the
hosted check. This permits the explicit new selected catalog and curriculum
pointers, and derives count expectations from that selection using the pinned
input manifest. It never permits mutation or removal of old assessments, packets
or retained curriculum manifests. Follow the separate staged catalog-loader
procedure in [Updating Atlas](../../../../docs/references/rl-data-atlas.md#updating-atlas);
never use this option to excuse unexplained drift.

Catalog selection retains earlier loads, including their source declarations,
backend and overlap evidence, assessments and packet bytes. The staged loader
does not garbage-collect them after switching the current pointer. Retained app
revisions and release receipts can therefore still refer to the original load.

Eyeball the hosted screenshots, then report the immutable URL, revision, landed
SHA, package hash, changelog, exact rendered headers and both checker receipt paths
to the coordinator. Include any remaining unmeasured evidence. The coordinator
syncs the release state in the reference docs. Preserve all prior artifacts and
the source/count/evidence pins used by each release.

The local real-data curriculum sweep uses the candidate code's shared read-only
SQL and response projections for area breakdowns and sampled task queries. Hosted
checks call the published API directly. The sweep verifies that every taxonomy
node, including zero-count capabilities, appears in its domain, that source
breakdowns sum to the area count, and that labels distinguish taxonomy wrappers
from their subjects. Source geometry is checked after column resizing as well as
at automatic widths. Package verification compares every tracked packaged input
against the landed cut, applying only the preview preparation transformations.

Preflight and hosted verification check history before and after browser work
as blocking gates. Retained packet checksums execute in bounded groups of pipeline keys, preserving all
distinct task/content variants across loads. Failure receipts identify whether
the baseline or final readback failed and retain the failing SQL and parameter
bindings. Database timeouts fail the release check, including those
after page checks finish.

Code-only history checks also checksum stored source fingerprints, duplicate
groups, identity survey records and pair reviews. Recomputing display groups from
those records must preserve their archived bytes. A filtered-out canonical source
leaves its eligible group members unfolded; the checker rejects sibling folding
without the canonical representative.

The non-blocking full evidence sweep selects a source for each observed combination of
Quality, Result, provisional status, filtration state, actor safety, Use and
backend measurement basis, review waiver/progress and difficulty measurement states. It opens both Evidence and Backends at all four
layouts, checking field/table bounds and the Daytona verdict and operations link.
Hosted verification also compares recomputed duplicate groups and directional
notes against the input-bound projection and checks that stored groups remain
accessible as archived evidence.

Design invariants: every main-table source cell contains only the source-name
button and optional fold toggle. Source subtitles, dataset links, overlap notes
and all other secondary content belong in Evidence/detail views. Enum chips
carry their label without secondary text underneath. The browser gates enforce
both invariants at 1440/390 in light/dark themes before every publication.
Table header labels stay on one line, including unsortable headers. The geometry
gate rejects text that fits its cell only by stacking letters vertically. Use
labels also remain on one line.

## Full sweep, post-publish diff and nightly audit

The hosted check sweeps the new immutable revision in full: every
release/Use/domain/confidence browser case, every Evidence/Backends presentation
and every readability page family. The sweep runs as independent per-layout units
(domains and evidence cases split into chunks) on a bounded pool of parallel
pages (`SWEEP_WORKERS`, 16), each on its own browser context. A hosted check at the
preflight's own checkout records `code_gates_from` instead of rerunning lint and
the unit/journey commands, which that passing preflight receipt already covers.

The post-publish check (`--postpublish --hosted-receipt <hosted.json>`) compares
against that hosted sweep: it re-reads every API response the hosted sweep
recorded, re-sweeps in full only the page families whose served content changed,
and runs the representative sweep for the rest. It is a blocking phase; the hosted
check starts no detached audit.

To repeat the audit after publication or from a nightly scheduler, use the same
clean landed cut, the successful version-2 preflight baseline and the revision's
hosted receipt:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.release_check \
  --revision "$REVISION" --baseline "$ARTIFACTS/preflight.json" \
  --production-version "$PRODUCTION_HOLD" --expected-mode public \
  --postpublish --hosted-receipt "$ARTIFACTS/hosted.json" \
  --receipt "$ARTIFACTS/nightly.json"
```

API and preservation checks still run, and production remains held. There is no
new persistent service or scheduler configuration.

## Intended measurement refreshes

Stage measurement changes before the preflight baseline, from the landed cut.
Keep source declarations, assessment packets and archived measurements intact.
Each loader creates only its additive measurement tables. Never replace a catalog
just to refresh an independent measurement. Retained app revisions continue to
read their original overlap tables.

For an explicitly approved overlap survey, pin both its URI and SHA-256:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.load_identity \
  --survey "$SURVEY_URI" "$SURVEY_SHA256" --app-id "$PREVIEW_APP_ID" \
  --receipt "$ARTIFACTS/identity-load.json"
```

The loader compares immutable fingerprint inputs, population counts, sample
hashes and seeds against the frozen catalog. It excludes reviews touching
current empty populations, retains the full survey reference and records that
exclusion. Any other input mismatch aborts selection. Repeating an identical
load is safe; conflicting stored reviews fail. Pass
`--identity-measurement "$SURVEY_SHA256"` to the preflight checker to require that
exact measurement. Its selected pointer and every archived review checksum are
then covered by the hosted baseline.

Load derived-contract manifests with explicit byte hashes:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.load_derived_contract \
  --input "$GRADER_MANIFEST" "$GRADER_SHA256" --app-id "$PREVIEW_APP_ID" \
  --receipt "$ARTIFACTS/grader-load.json"
```

Grader profiles bind to source ID, dataset ID/revision, converter version/revision,
TaskSpec digest and population size. Unbound profiles show Pending. Mixed is one
chip; the complete type counts, judge profile and receipt are in Evidence.

For a new confidence export, run the curriculum loader in the export's storage
region. Supply the exact release, coverage and hashed confidence sidecar:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.load_curriculum \
  --release "$CURRICULUM_RELEASE_URI" --coverage "$COVERAGE_URI" \
  --confidence "$CONFIDENCE_URI" "$CONFIDENCE_SHA256" --app-id "$PREVIEW_APP_ID"
```

The sidecar must bind the same immutable release/coverage bytes and sample union.
Every confidence level must reconcile outcomes to the population; every aggregate
must equal its per-source vectors; children cannot exceed their parent. Old loads
are retained. Confidence selection reads precomputed vectors and filters cached
scored samples in the browser. Source is optional: examples are pooled across
sources, each with its Source field. Lists contain stored samples, not a complete
instance index. A stricter subject-only view filters the full subject sample pool,
so samples are not lost merely because they were capability matches at a broader
level. The blocking API check exercises all exported levels and task partitions, including
the original cutoffs and historical populations. The representative browser pass
checks the confidence control; the full browser walk runs after publication.

## Readability release gate

The installed `test_rendered_contrast_readability_and_reflow` journey and the
input-bound `browser_checks` both run the representative readability pass. Any
violation in that pass blocks release. It covers the source table, curriculum, source evidence, backends, assessments,
and guide, plus representative source/evidence states, filters and folded groups, confidence levels and one measured curriculum domain, public instructions from a stored task example in each layout, source breakdowns, empty/error states, hover,
active and keyboard focus on representative controls. The full sweep retains
every filter value, keyboard target and grouped pointer style. Layouts are 1440, 390 and 320 CSS px in both themes;
page and dialog checks also apply 200% text size and WCAG text-spacing overrides.

The [style standard and measured palette](README.md#contrast-and-readability)
require 7:1 for all text, 3:1 for meaningful boundaries/graphics, visible 2 px or
larger focus rings, underlined prose links, 24 px targets (44 px mobile primary
controls), 14/16 px body text and 320 px reflow. Keep tables in their own
horizontal scrollers. Review screenshots for clipped text and overlap, including
the guide's linear diagram and public-instruction dialogs. Native select popups,
title tooltips and visited-link paint require visual review because their paint
is not exposed by DOM computed styles.

Run the focused gate while editing:

```bash
uv run marina journey rl_data_catalog -k 'rendered_contrast or checker_measures' --shots /tmp/atlas-readability
```

The focused command uses the installed app and disposable Postgres fixtures.
Run the installed journeys with `-k 'not full_browser'` for the blocking pass.
The release checker repeats
the gate against its pinned catalog projection and selected curriculum release
through the existing `CatalogReader`; see the preflight commands above for
`--source-counts`, `--oracle-inventory` and `--baseline`. Retain `readability/readability.json`,
its incremental `states.jsonl`, and the adjacent screenshots with the release
receipt. For the focused command these live under
`/tmp/atlas-readability/rl_data_catalog/readability/`. Release runs store them under
the receipt artifacts directory: `journeys/rl_data_catalog/readability/` for the
fixture walk and `screenshots/readability/` for the frozen-input walk. Failed runs retain their completed states for diagnosis. Do not waive a
failure by removing a layout, lowering a threshold or filtering a selector.

Pending private candidates may appear on a source's Evidence page with a SAMPLE
receipt bound to the candidate's repository and revision. Adding a card must
preserve the selected catalog release and its By source Examples and Quality.
Remove the card when the catalog selects that candidate with a qualified
current-pin admission receipt; preserve its earlier package and receipts as history.
The full post-publish Evidence and readability sweeps read source IDs from the released
app's candidate manifest and check every card, including sources with the same
Quality, Result or admission state as another source. Check the displayed
candidate pin and filtered task count against its linked receipt. A private
publication alone does not qualify Atlas admission.

## Review and rollout evidence

Review/rollout browsing loads evidence independently of source admission. Resolve
panels from each frozen declaration's selected receipt, not the latest registry
entry. Keep the original input pin visible for archived lineage. Never replace
an old verdict with a retry because a mutable batch pointer advanced. The loader
checks each task/runtime contract and typed observation against the hashed records
Parquet and seals the original batch bytes. Provenance distinguishes a surviving
publisher-pointer hash from a hash captured by the staging readback. For FULL
assemblies, `map_original_batches` follows their recorded batch/overlay origins and
requires exact task-ID and complete typed-judgment equality within the recorded
origin graph. A unique original batch may carry older input hashes: show both
its original contract and the consolidation's hashes as original-reviewer lineage,
not current-pin QA. Do not rebind admission or QA from this browsing join. Multiple matches and
missing matches stay “not mapped” with a reason. Keep the immutable mapping URI
and SHA; never choose a later judgment to fill a missing original.

Stage bulk content in the artifact's S3 region with
`python -m infra.marina.apps.rl_data_catalog.stage_browse_evidence` and an explicit
manifest URI/SHA, shard and output root. The manifest pins frozen declarations,
review artifacts or complete difficulty observations and the producer code hashes.
Keep the generated shard indexes and readback hashes. PRIVATE_ALWAYS content is
refused before artifact reads. Review and rollout imports also reject
license-excluded content.

For a single selected panel or observation, the maintained loader commands are:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.load_reviews \
  --artifact "$REVIEW_URI" "$REVIEW_SHA256" --source-id "$SOURCE_ID" \
  --app-id "$PREVIEW_APP_ID" --receipt "$ARTIFACTS/review-load.json"
uv run python -m infra.marina.apps.rl_data_catalog.load_rollouts \
  --measurement-id "$MEASUREMENT_ID" --app-id "$PREVIEW_APP_ID" \
  --receipt "$ARTIFACTS/rollout-load.json"
```

For FULL assemblies, pass the sealed mapping with `--mapping URI SHA256`.
Bulk imports use `python -m infra.marina.apps.rl_data_catalog.load_browse_evidence`
with each shard's `--index URI SHA256`, the frozen `--load-id`, service URL,
`--auth-token-secret` (an Iris-delivered secret file name), worker count and an
immutable receipt URI. The bulk loader is restricted to the private preview app.

Large imports run regionally. Review panels use an additive selection table;
stored declarations, old assessment keys and packet bytes remain intact. Rollout
references come from each measurement's hashed aggregate, including ancestor
attempt receipts; never derive opaque trial paths from source names. Only named
transcript and verifier files are staged from trial archives. Trial configuration
and environment values are not served. Compressed SQL content is size/hash checked
when read, and runtimes need no new object-store credentials. Store each original
reviewer batch once with per-task links, so whole-source exports and serving rows
do not duplicate the same batch prompt for every reviewed task. The release gate
checks those links and refuses missing shared batches.

The blocking journeys exercise Quality chip → list → raw review and solver →
attempt → rollout, both themes at 1440/390. Readability also visits the raw review,
defect legend, attempt list and rollout. The blocking API coverage check requires every open FULL/SAMPLE source panel,
matching task/raw-row counts, every recorded difficulty observation with attempts,
and reconciled defect totals. Sources marked content-withheld by the API require
no content staging. Original reviews without a unique batch mapping remain
visible with that reason in the raw-review viewer.
Candidate checks execute these API routes
through the existing read-only SQL gateway before publication; hosted checks use
the published routes. Verify cold and warm timings for the new pages and retain
source/Status warm performance. Keep all old evidence and production on HOLD.
