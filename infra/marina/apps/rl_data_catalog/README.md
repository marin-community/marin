# RL Data Atlas style

Follow the canonical [release procedure](RELEASE.md) and its `release_check`
entry point for every preview release.

Enum chips carry the entire state label. Never put text below a Quality, Result,
Contamination, Filtration or Daytona chip. Put numeric issue rates and lexical overlap counts
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

Curriculum parents include an Other/unlisted row for subject-only matches and
children hidden by search. Parent counts equal displayed children plus that
remainder; the bounded task example sample does not reduce population counts.

Every header count is in tasks: total population, admission (TRAIN plus EVAL),
pending oracle filtration, exclusions and QUICK/SAMPLE/FULL quality populations.
Releases with zero retained tasks contribute their canonical filtration receipt's
original excluded input count; other releases contribute the current deduplicated
population. Counts are source-specific task instances: the same contract in two
sources can count twice. Shown, hidden QUICK-quality/BAD-result, filtered and
folded-duplicate tasks form a mutually exclusive partition of the total. Each
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
