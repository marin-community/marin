# Kang IFN-beta task-design review

## Status

`unreviewed_do_not_publish`. The 25-task output is a candidate review set, not an
accepted educational release.

## Design sequence

1. Selected an independent donor-paired perturbation study rather than expanding
   the Stephenson TRAV1-2 table with more threshold variants.
2. Pinned the 123,669,441-byte TheisLab AnnData convenience artifact by SHA-256.
3. Extracted donor-condition-cell-type pseudobulks for a predefined 10-gene IFN
   panel and two housekeeping comparators.
4. Defined CPM normalization, a 1 CPM pseudocount, donor pairing, support filters,
   aggregation, tie policy, rounding, categorical conclusions, and limitations.
5. Enumerated 87 formal candidates across five families.
6. Selected 25 by interleaving families and deterministically diversifying each
   family's parameter bindings.
7. Ran all 25 through Taskforge structural authoring and objective controls.
8. Normalized and exported all 25 through TaskCompendium and TaskTrove with zero
   rejects.

## Candidate composition

```text
analysis_unit                       6
cell_type_response_comparison       6
ifn_panel_gene_rank                 6
paired_gene_response                6
ifn_signature_cell_type_rank        1
```

The 87 formal candidates are not 87 automatically acceptable lessons. Many share
the same computation with a different gene or cell type. The 25-task set is sized
for review and calibration, not as a quota.

## Demo disposition

For the upcoming demo, all 25 structurally accepted candidates proceed without
item-level human keep/revise/reject review. This is an explicit speed/observability
decision: weak, ambiguous, trivial, or repetitive tasks are useful demo evidence
and must remain visible rather than being silently filtered. The waiver applies
only to the demo. It does not change `unreviewed_do_not_publish`, complete the
rights audit, or qualify the set as a curated release.

## Human acceptance checks for a later release

For each candidate, mark keep, revise, or reject after checking:

1. Every required value and biological rule is solver-visible.
2. Donors, not cells, are treated as biological replicates.
3. Missing donor-condition-cell-type groups are not converted to zeros.
4. The prompt applies the biology card rather than merely decorating arithmetic.
5. The task requires more than copying one table cell.
6. The reference answer follows the frozen normalization and pairing contract.
7. The categorical interpretation follows from reported quantitative evidence.
8. Transcript evidence is not inflated into protein, mechanism, or clinical claims.
9. The task is not a near-duplicate of another retained candidate.
10. A model failure would reveal a meaningful scientific-analysis weakness.

## Calibration

Taskforge currently records reference reward 1.0 and severe-negative reward 0.0.
The demo may proceed directly to solver and adversary trials without prior human
filtering. Preserve per-task solve rate,
model/version, rollout count, token/runtime cost, malformed-answer rate, leakage
findings, and qualitative failure category. Do not turn those dimensions into one
opaque effectiveness score.
