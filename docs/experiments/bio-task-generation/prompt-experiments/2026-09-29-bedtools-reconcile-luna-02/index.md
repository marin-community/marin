# bedtools reconciliation with consistency and readiness checks

[Experiment index](../index.md)

The worker returned 17 units and 12 data records. It reproduced the data-identity,
link, and role repairs and corrected the manual count. It kept the broad tutorial
boundary, so it did not reproduce the first reconciliation's workflow extraction.
The generic prompt now requires checking all fields after a
record changes, deriving counts from final files, and distinguishing actual
inventory defects from authoring limitations. It responds to residual failures
in [reconciliation 01](../2026-09-29-bedtools-reconcile-luna-01/index.md).

The worker receives the same original bedtools 06 input bytes, source-access
instructions, fresh Luna context, and ten-minute budget. It does not receive
the previous reconciliation output or the parent's findings.

- [Run configuration and input hashes](run.json)
- [Generic prompt](template.md) and [resolved prompt](resolved-prompt.md)
- [Launch message](launch-message.txt) and [source access](source-access.txt)
- Frozen [units](inputs/units.jsonl), [datasets](inputs/datasets.jsonl),
  [inspection](inputs/inspection.md), and [requirements](inputs/discovery-prompt.md)
- [Review criteria](review-criteria.md)
- Revised [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl),
  [inspection](outputs/inspection.md), and [reconciliation report](outputs/reconciliation.md)
- [Structural and retention checks](structural-review.json)
- [Worker final response](worker-final.txt)

Required fields, unique IDs, and references pass. All 17 original unit IDs and
12 input asset URLs survive. The six tutorial products and six packaged fixture
candidates are separated. The two original mixed-role units become tool use.
The inspection report correctly counts 41 manuals as 17 inspected and 24 pending.
The worker recommends `ready_for_authoring_review` with explicit data and
discovery limitations, consistent with the revised readiness definition.

The broad tutorial record still does not expose the DNase matrix and plots as
an independently selectable workflow. Its scientific-use text asks the author
to choose individual questions, while its outputs remain general interval
comparisons. This misses the workflow-boundary reviewer probe that trial 01
recovered. The v2 result also does not test correction of trial 01's new ChromHMM
answer-key contradiction: it did not create that separate record.

The two output reports disagree about source inspection. `inspection.md` says
the fixed-revision tree was checked; `reconciliation.md` says tree metadata was
carried forward and the discrepancy awaits fresh enumeration. The count itself
is correct, but this provenance inconsistency remains unresolved without a full
tool trace. Some split fixture prose still describes the original collection.

The revised prompt improves the reported count and readiness rationale in this
sample; it does not establish a consistent overall gain. The next comparison
uses the same template on a saved Scanpy inventory to test merging processing
stages and preserving distinct biological datasets.
