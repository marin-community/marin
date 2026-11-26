# RL Data Atlas style

Follow the canonical [release procedure](RELEASE.md) and its `release_check`
entry point for every preview release. Complete curriculum count/taxonomy checks
run once through the API for every release, Use, confidence level and domain.
A representative browser/readability pass and screenshots block publication;
the hosted check sweeps the published revision in full on parallel pages, and the
post-publish check re-sweeps only the page families whose content changed; see
[full sweep and post-publish commands](RELEASE.md#full-sweep-post-publish-diff-and-nightly-audit).

Synced to preview revision 21 (2026-11-16).

Every page renders the same navigation from `dist/common.js`: Status, By source,
By curriculum, Difficulty and Getting started. `source.html`,
`assessment.html`, `review.html`, `defects.html` and `backends.html` keep By source active.
`attempts.html` and `rollout.html` keep Difficulty active. The installed browser
journey in `journeys/test_status.py` checks every page's labels, destinations and
current view.

FULL and SAMPLE Quality chips link directly to a source's paged task-review list.
The Evidence page links to the same list. Each row shows the verdict, typed defect
chips, reviewer identity and confidence; verdict and defect filters select the
stored reviews. A raw-review link opens the recorded reviewer requests, responses
and their provenance. Defect chips link to the catalog-wide defect legend and
counts. Archived panels display their original input pins and never assert a new
current-pin qualification.

The main table’s Difficulty cells, Difficulty page and source Evidence solver rows link to paged attempts and
individual rollout viewers. Attempts retain score, solver, trial identity,
duration and exception metadata. Transcripts include tool calls and verifier
output; Filer links retain the raw receipt/archive references. Outcomes without
a recorded attempt artifact remain explicitly unavailable. PRIVATE_ALWAYS and
license-excluded sources show a marker; their task, review and rollout content
is withheld. The release gate does not require their withheld content to be staged.

Source Evidence renders its core API response without waiting for auxiliary ledgers.
Each supplementary request has an 8-second deadline and its own visible error;
the core request has a 15-second deadline. The embedded assessment report loads
only when expanded; QC chips still open the paged review list directly. The
Review Archive page and navigation entry are removed; archived data and APIs
remain intact. Failed required scripts on the Source Evidence page show an error
and a **Reload page** action that preserves the source selection. The initial HTML
contains the fallback, so it remains visible if the bootstrap script cannot execute.

Catalog loading stores the counts and grades from the `atlas_cohort_summary`
SQL view in `atlas_cohort_summaries` before making the new catalog snapshot
current. App migration backfills that current snapshot before serving requests.
Source and assessment requests read the stored summaries, avoiding repeated
aggregation over whole-population FULL reviews. Increment
`COHORT_SUMMARY_VERSION` when changing the view's counting or grading rules;
loading and migration rebuild summaries with a missing or older version.
Archived reviews and packets stay unchanged. The Status page requests only
`api/source-statuses`; that endpoint reads source metadata without task,
difficulty or backend evidence.

The By source top-area lookup reads `curriculum_source_areas`, populated before
making each imported curriculum load current. App migration backfills the
currently loaded release; rerunning the loader with the same release also repairs
missing or obsolete summaries. Increment
`SOURCE_AREA_VERSION` when changing subject selection or tie ordering. The
stored statistics, samples and release pointers remain unchanged.

The By source page requests `api/sources?projection=table`. It returns the values
used by the table, tooltips and source dialog. Historical difficulty records,
raw review findings and license documents remain in the complete `api/sources`
response and the source Evidence endpoints. The release checker compares the
table response against the complete input-bound projection before release.

Each backend process caches one payload per source projection. Every request
checks committed row versions of the inventory's inputs; inserts, updates,
deletions and a new selected load invalidate the payload. A response built
during an input change is not cached. Headers and authorization remain per
request. Add new inventory query dependencies to `INVENTORY_KEY_SQL` so changed
evidence cannot reuse an older payload.

The By source query selects fingerprint version, seed, task count, duplicate
count, sample hash and lineage in PostgreSQL. It leaves fingerprint samples and
hash arrays in storage; `api/source-fingerprint` returns the full evidence on
request. Keep the list query compact when adding new fingerprint fields.

The main table's source cell contains only its source-name button and, where
needed, the fold toggle. No origin/build subtitle, dataset link, overlap note or
other secondary text belongs in that cell. Put those details in Evidence or the
detail pane. The release checker enforces this at all four layouts.

Enum chips carry the entire state label. Never put text below a Quality, Result,
Contamination or Filtration chip. Put numeric issue rates and lexical overlap counts
in their own columns, using an em dash for missing measurements. Keep review
progress, reviewed/retained counts and lower bounds on the source Evidence page.

A Result includes `provisional` only while its required content review is
incomplete, including unreviewed or uncertain observations. A completed SAMPLE
review is final for its required sample. An explicitly accepted whole-source
waiver also makes its Result final and Quality FULL, while Evidence shows false
strict coverage, unreviewed counts and a pinned receipt with every exact reason.
The loader computes this state from review
coverage and conclusive reviewed/required counts; never infer it from the Quality
label. Rerunning the loader refreshes older stored completeness metadata once,
even when the snapshot ID is unchanged. SAMPLE requires a complete review of a selected sample; FULL requires a complete review of
the whole retained population, with explicit whole-source waivers as described
above. Execution qualification remains separate.
The source table legend and Getting started guide define that variant and all five
Contamination states. Keep the API, table, source details and guide consistent.
The preview loader displays sources awaiting an exact-input SAMPLE Result as
QUICK with “SAMPLE review in progress” in details and the Quality tooltip.
It preserves the declaration and leaves Result, issue rate and completeness
unmeasured. Production loads and catalog publication retain the strict Result
requirement; preview mode still requires complete admission and grading evidence.
Matching aggregate and independent SAMPLE-panel counts display SAMPLE even when
declared execution oversight is unchanged. One pure projection serves the loader,
source list and details; existing catalogs need no display-metadata backfill.
Use the shared chip colors, with light and dark theme rules; a remediation warning
also has a strong border. Phone layouts keep all columns reachable through
contained horizontal scrolling, with the table header aligned to its body.
Do not rely on color alone to convey a state.

Drag a source-table header edge to resize a column. Focus its separator and use
arrow keys for keyboard resizing; Shift increases the step, Home uses the minimum,
and End uses the maximum. Widths persist in viewer-local storage when available.
Use **Reset column widths** to return to automatic sizing.

The main table's **Backends** link and tooltip carry the actor Daytona verdict,
snapshot count and cap, failed execution requirements, measurement provenance and
operations link. Daytona measurements remain in the catalog; they have no separate
main-table column.

Curriculum nodes include empty capabilities and clickable subject-only remainders.
Each node opens its exact instance count, complete source breakdown and sampled
examples with the existing task evidence. Sort sibling areas by name or task count.
Taxonomy groups that repeat their subject's name display as “Capability groups
for …”; archived IDs and labels remain unchanged.

Run the installed-app browser regressions with:

```bash
uv run pytest infra/marina/apps/rl_data_catalog/journeys --journeys -n 0
```

The journeys install only this app into disposable local Postgres. See the
[Atlas reference](../../../../docs/references/rl-data-atlas.md) for packaging,
loading and preview release procedures.

Filtration displays EXCLUDED and Examples 0 only for complete, typed private
filtration evidence at the exact current pin with zero retained tasks and explicit
Excluded status. This path preserves excluded sources in the catalog; it never
admits them. All positive-output admission checks remain in force. Sources without
qualified current admission proof show IN PROGRESS and Examples Pending, with
pre-filtration counts and receipt hashes in Evidence.

Evidence can also show a separately pinned filtered private candidate, labeled
“pending qualification,” with its oracle-filtered count and receipt hashes.
This candidate is not selected or admitted. Its missing output-pin qualification
is listed explicitly; the catalog pin, Examples and Quality stay unchanged.

Curriculum parents include an Other/unlisted row for subject-only matches and
children hidden by search. Parent counts equal displayed children plus that
remainder; the bounded task example sample does not reduce population counts.

Every header count is in tasks: total population, admission (TRAIN plus EVAL),
pending oracle filtration, exclusions and QUICK/SAMPLE/FULL quality populations.
Releases with zero retained tasks contribute their canonical filtration receipt's
original excluded input count; other releases contribute the current deduplicated
population. Counts are source-specific task instances: the same contract in two
sources can count twice. The default view hides Excluded and Pending rows
alongside QUICK-quality/BAD-result ones. Shown, hidden excluded, hidden
pending, hidden QUICK-quality/BAD-result, filtered and folded-duplicate tasks
form a mutually exclusive partition of the total. Each
omitted group also reports its admitted subset. **Show everything** clears all
filters, includes QUICK-quality/BAD-result sources and unfolds every duplicate
group. **All Difficulty** is the default; selecting a model orders sources by
that model's solve rate, highest first. Returning to All restores source-name order.

Preview source counts can be supplied to `prepare_preview.py --source-counts` as
version-1 observations. Each record binds the exact actor-census input and Use,
the oracle inventory, the unique current static-blocker count and its audit,
and policy ruling SHA `028c7dfb…e2f3953`. The API withholds counts on any input or
Use change. No-oracle Examples equal retained minus the measured blocker union;
EVAL tasks enter total Atlas admission as evaluation data and remain excluded
from TRAIN selection. Oracle
sources stay Pending oracle filtration without final evidence. Missing static
measurements remain explicit and do not become zero exclusions.

Count observations are bundled as immutable Python data with a provenance hash;
releasing them does not alter the selected catalog, archived packets or training
selection. The production package's default observation set is empty.

Actorless count observations also cite ruling `b02c3b35…22da118`. Their bound
audit removes agentic-only exclusions using the actual TaskSpec actor/harness
contract, while preserving grading and other applicable checks. Both policy
references remain visible in source Evidence.

Curriculum confidence is a viewer choice over precomputed count vectors and
scored sample examples. Changing it performs no classification work. Examples
are pooled across sources with an explicit Source field; source selection is an
optional filter. Complete per-source counts remain available below the samples.
The confidence sidecar and overlap/Grader measurement loaders are documented in
[Release procedure](RELEASE.md#intended-measurement-refreshes).

Backend fit preserves Supported, Partial and Unsupported survey states. Partial
names the conditional capabilities and their affected-task counts. Partial on a
production backend can satisfy the backend admission gate; other admission checks
still apply. Missing measurements, validation
failures and incompatible environment kinds retain their typed reasons.
N/A is reserved for an absent role. Counts from agent
and grader roles can overlap and must not be added into a unique-task union.

## Contrast and readability

Atlas follows [WCAG 2.2 AA text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
(4.5:1, or 3:1 for large text). Its house rule requires **7:1 for all text**,
including instructions, muted text, table cells, badges, placeholders and SVG
labels, following [enhanced contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-enhanced.html).
Disabled controls may be exempt but must remain visibly disabled. State labels
must convey their meaning in words; links in prose are underlined.

Use the semantic custom properties in `dist/style.css` for every page. The system
color-scheme preference selects the palette; there is no separate theme toggle.
Do not add page-local color literals or change counts to address presentation.
These ratios use the WCAG relative-luminance formula without rounding thresholds:

| Token pair | Light colors | Ratio | Dark colors | Ratio |
| --- | --- | ---: | --- | ---: |
| `ink` / `surface` | `#15291d` / `#ffffff` | 15.36 | `#ecf5ef` / `#17231b` | 14.60 |
| `ink` / `soft` (instructions) | `#15291d` / `#eef4ef` | 13.77 | `#ecf5ef` / `#213326` | 12.05 |
| `muted` / `soft` | `#34483a` / `#eef4ef` | 8.82 | `#c4d5c9` / `#213326` | 8.75 |
| `green` / `soft` | `#064b32` / `#eef4ef` | 9.14 | `#b4ebca` / `#213326` | 10.01 |
| `good-text` / `good-fill` | `#064b32` / `#e2f0e5` | 8.65 | `#b4ebca` / `#1b3525` | 9.91 |
| `warning-text` / `warning-fill` | `#593b06` / `#fff5d9` | 9.41 | `#f5dfa9` / `#392e14` | 10.17 |
| `bad-text` / `bad-fill` | `#722a20` / `#ffede8` | 8.96 | `#ffd0c6` / `#3d211b` | 10.54 |
| `ink` / darkest heat cell | `#15291d` / `#91c4a2` | 7.77 | `#ecf5ef` / `#2b5035` | 8.20 |
| `line` / darkest heat cell | `#405d48` / `#91c4a2` | 3.69 | `#91ac9a` / `#2b5035` | 3.72 |

The five heat intensities use theme-specific fills and visible numeric counts.
Control boundaries, resize handles, chart marks and focus indicators need
[3:1 non-text contrast](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html).
Every keyboard target needs a visible ring at least 2 CSS px thick, contrasting
with both its fill and its surroundings, with no obscuring overlay. Atlas uses
3 px rings and follows the [focus appearance](https://www.w3.org/WAI/WCAG22/Understanding/focus-appearance.html)
house rule. Targets need [24 × 24 px or the spacing exception](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html);
primary mobile controls need 44 × 44 px.

Text must resize to 200% and [reflow at 320 CSS px](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html).
Only data tables may scroll horizontally within their own containers. Keep
match scores on one line; the task dialog uses a table wide enough for its
headers and decimal values, with scrolling inside the dialog. The guide
uses the diagram's same labels and live counts in a linear reading order when
its diagram cannot fit or text spacing increases. Allow the WCAG
[text-spacing overrides](https://www.w3.org/WAI/WCAG22/Understanding/text-spacing.html):
1.5 line height, 2em paragraph spacing, .12em letter spacing and .16em word
spacing. Body/prose needs at least 14 px on desktop, 16 px at widths ≤480 px,
line height ≥1.4, and roughly 80 characters per line. No text may be below 12 px;
the shared palette currently uses a stricter 14/16 px floor throughout.

`journeys/test_readability.py` and the release check use `readability.py` /
`readability.js`. They composite computed colors and opacity over actual ancestor
surfaces, including SVG node fills; unsupported background paint fails closed.
Reports include each selector, rule, measured value and threshold, grouped by
page, state, viewport and theme. Keep native menus/tooltips and browser-private
visited rendering in the visual review; the browser does not expose their paint
through DOM computed styles. Hover and active styles are checked through Chromium
pseudo states; keyboard focus is checked by tabbing. Both installed fixtures and
the input-bound release data must pass, followed by screenshot review.
