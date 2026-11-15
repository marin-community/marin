# Atlas preview release procedure

This is the canonical release checklist for the maintained Atlas app. Use the
repository checker below for each new preview revision. Release-specific scripts
and old deployment records are historical evidence, not a substitute for these
gates.

The target is the private preview app
`c9742a15-fb46-4d56-bc88-aa916351ebd3`. Production
`fb11c931-5861-4878-8bb5-a964d652b45f` must remain at the coordinator's explicit
hold revision. This procedure never publishes production or changes access mode,
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
  --production-version "$PRODUCTION_HOLD" --receipt "$ARTIFACTS/preflight.json"
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
| Representative browser pass | One selected release, Use and measured domain per layout loads the confidence control and sampled-task list. One Evidence/Backends presentation and each authored page family pass geometry and readability checks, with screenshots. |
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
  --production-version "$PRODUCTION_HOLD" --receipt "$ARTIFACTS/hosted.json"
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

## Full post-publish or nightly browser audit

After a successful hosted check, the checker starts the unchanged full curriculum,
Evidence/Backends and readability walks in a detached local process. The hosted
receipt records `postpublish_job` with its PID, command, log and receipt paths.
This job does not delay release completion. Its receipt has `phase: postpublish`
and `blocking: false`; a finding gives that job a nonzero exit without changing
the successful hosted receipt. A launch failure is recorded in `launch_error`.
Report any launch failure or audit finding to the coordinator, with the receipt
and screenshots, and track the fix for the next release.

To repeat the full audit after publication or from a nightly scheduler, use the
same clean landed cut and successful version-2 preflight baseline:

```bash
uv run python -m infra.marina.apps.rl_data_catalog.release_check \
  --revision "$REVISION" --baseline "$ARTIFACTS/preflight.json" \
  --production-version "$PRODUCTION_HOLD" --full-browser-sweep \
  --receipt "$ARTIFACTS/nightly-full.json"
```

The full audit retains every release/Use/domain/confidence browser case and every
Evidence/Backends and readability presentation from the earlier blocking sweep.
It skips repeated lint and unit/journey commands. API and preservation checks
still run, and production remains held. Keep the clean worktree and credentials
available until the recorded job finishes; inspect its receipt after completion.
There is no new persistent service or scheduler configuration.

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
archive and guide, plus representative source/evidence states, filters and folded groups, confidence levels and one measured curriculum domain, public instructions from a stored task example in each layout, source breakdowns, archive records, empty/error states, hover,
active and keyboard focus on representative controls. The full audit retains
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
