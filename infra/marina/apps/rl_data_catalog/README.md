# RL Data Atlas style

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

Header totals cover all live sources. Admitted examples show the final known
admitted sum and source counts still pending filtration or excluded. Quality
shows catalog-wide QUICK/SAMPLE/FULL totals; both cards label displayed counts
separately as Shown after filters and duplicate folding.
