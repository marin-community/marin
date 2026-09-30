# Review criteria fixed before launch

Apply the same general dimensions as the bedtools reconciliation test: dataset
identity, source support, correct unit/data links, tool-role consistency,
scientific coverage preservation, accurate inspection status, and honest
readiness recommendation. Worker receives no reviewer findings.

Specific probes: merge raw PBMC3k and its processed derivative as linked assets,
while keeping PBMC10k and PBMC68k separate. Preserve each asset and the ingest
reference/query distinction after relinking units. Preserve pancreas study
lineage without inventing per-study file URLs or unverified partitions of the
combined H5AD. Check unit boundaries for ingest and BBKNN and retain trajectory
and Pearson-residual coverage. Preserve actual AnnData prerequisites, processing
stages, and source-specific caveats. Do not claim that newly inspected pinned
repository files retroactively pin the input's mutable rendered sources.

The input was produced by an earlier discovery prompt; the supplied contract is
the current general find-units prompt. This tests reconciliation transfer on a
saved inventory, not an end-to-end run of the latest discovery recipe.

Additional probes after reconciliation 01: inspect each split API unit's own
dependencies, locator and evidence; copied ingest requirements/evidence do not
support BBKNN. Preserve processed-asset metadata after merging PBMC3k and
remove stale claims that newly read fields remain unknown. For Paul15, inspect
the adjacent informative-gene intersection and the comment about removing ten
corrupted names before treating 3461/3451 as an unexplained contradiction.
Distinguish comments from documented returned dimensions and actual execution.
