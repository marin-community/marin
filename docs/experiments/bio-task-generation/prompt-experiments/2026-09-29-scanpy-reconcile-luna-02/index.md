# Scanpy reconciliation with replacement-record audits

[Experiment index](../index.md)

The worker returned seven units and five datasets. It retained the PBMC identity
repair and produced more focused workflow records, but copied dependencies
remain and the inspection map contains removed unit IDs. The audit table claims
those replacements contain only appropriate prerequisites; the JSON disagrees.
The generic prompt adds a per-replacement evidence audit for
split and merged records and a check of processing stage and adjacent source
context before reporting contradictions. The comparison is
[Scanpy reconciliation 01](../2026-09-29-scanpy-reconcile-luna-01/index.md).

The original input files, target inventory requirements, source-access setup,
fresh Luna context, and ten-minute budget are unchanged. The worker receives
no prior reconciliation output, parent findings, or Scanpy-specific repair hints.

- [Run configuration and input hashes](run.json)
- [Generic prompt](template.md) and [resolved prompt](resolved-prompt.md)
- [Launch message](launch-message.txt) and [source access](source-access.txt)
- Frozen [units](inputs/units.jsonl), [datasets](inputs/datasets.jsonl),
  [inspection](inputs/inspection.md), and [requirements](inputs/discovery-prompt.md)
- [Review criteria](review-criteria.md)
- Revised [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl),
  [inspection](outputs/inspection.md), and [reconciliation report](outputs/reconciliation.md)
- [Structural, retention, and source-map checks](structural-review.json)
- [Worker final response](worker-final.txt)

JSON fields, IDs, and references pass. Three original unit IDs remain, and the
two combined tutorial/API records map to four replacements. The output merges
raw/processed PBMC3k while retaining PBMC10k, PBMC68k, pancreas, and Paul15 as
separate records. All distinct original asset URLs remain somewhere in the
inventory; that check alone does not establish preservation of each asset's
lineage or metadata.

The workflow records now separate PBMC ingest from pancreas BBKNN. Their
scientific-use and input/output descriptions are more focused, and the API
evidence distinguishes label/embedding transfer from graph construction.
However, all four replacements retain copied dependency text: ingest still
lists BBKNN, and BBKNN retains ingest/reference prerequisites. The audit table's
claim of method-appropriate dependencies therefore fails on the final artifact.
PBMC BBKNN and pancreas reference mapping, present in the original combined
record, also lose their explicit dataset-to-operation connections without a
documented deferral, despite the assertion that no scientific use was removed.

The main source map and relationship section still refer to removed IDs
`scanpy.tutorial.ingest-bbknn` and `scanpy.api.ingest-bbknn`. A later appended
section lists the new IDs, but does not repair the earlier handoff. An additional
check, added after this result, detects those unresolved map references. Earlier
structural checks covered JSON references only.

The worker no longer reports Paul15's number difference as a blocker, but it
also does not explain the adjacent filtering code. Thus this run does not
demonstrate that the new source-context instruction resolved the earlier false
alarm. Its inspected-source list omits the ingest/BBKNN API files cited as
inspected evidence elsewhere; a full tool trace would be needed to settle that
provenance discrepancy.

The worker recommends `ready_for_authoring_review`. Parent review disagrees
because the copied dependencies and stale map references are specific inventory
defects. The added audit table did not reliably enforce its own claims. No
further prompt revision was made after this result.

The two-stage recipe improved data identity in all four reconciliation trials,
but unit-boundary changes remain inconsistent. The initial follow-up proposal
was to limit reconciliation to data identity, links, roles, and consistency
while holding unit boundaries fixed. That variant was not written or tested.
The [2026-09-30 design decision](../index.md) supersedes this proposal: retain
one discovery worker and fold useful checks into find-units. These results do
not justify unattended inventory acceptance.
