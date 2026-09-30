# Scanpy reconciliation transfer

[Experiment index](../index.md)

The worker returned six units and five datasets. It merged the PBMC3k processing
stages, retained independent PBMC collections, and split two API operations.
The split copied operation-specific evidence into the wrong record, and a
reported source contradiction overlooks adjacent filtering code.
A fresh Luna worker applied the exact generic template from
[bedtools reconciliation 02](../2026-09-29-bedtools-reconcile-luna-02/index.md)
to the saved [Scanpy 01 inventory](../2026-09-29-scanpy-luna-01/index.md).
The test includes a processing-stage duplicate, independent PBMC collections,
and a multi-study pancreas asset. Reviewer findings are not supplied to Luna.

The input came from an earlier discovery prompt. Current inventory requirements
are supplied as the target contract. The runner provides an immutable revision
for new source reads, while the input's original mutable sources remain marked
as unresolved. This tests reconciliation transfer on saved output, not an
end-to-end run of the current discovery recipe.

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

Structural checks pass. Four original unit IDs remain and the combined
ingest/BBKNN API record maps to two replacements. All distinct original asset
URLs remain. PBMC3k raw and processed assets share one record; PBMC10k and
PBMC68k remain separate. The pancreas record preserves the four named source
studies without inventing per-study download locations. Newly inspected pinned
sources are distinguished from carried-forward mutable documentation.

The replacement API records still share copied dependencies and evidence.
The ingest record lists BBKNN as a dependency. The BBKNN record cites ingest
mapping sections and describes reference-trained PCA, label transfer, and
projection as its evidence. Those statements support ingest rather than
batch-balanced graph construction. This is a split-record consistency failure.
The PBMC68k record also retains a limitation saying its cell count is unknown
after adding the documented 700-cell dimension elsewhere.

The worker's sole stated readiness blocker is Paul15's 3461/3451 difference.
The pinned [loader source](https://github.com/scverse/scanpy/blob/8c1463d5d97272d5811ad3f4efb57483e23b4c7e/src/scanpy/datasets/_datasets.py#L252)
documents a returned shape of 2730 by 3451. The 3461 value is in a code comment,
not the docstring as the worker says. Immediately before selection, another
comment says ten corrupted gene names are removed by an intersection. That
adjacent filtering step offers an explanation the review missed. The actual
asset was not inspected or executed; the evidence does not justify treating
the numeric difference alone as an unexplained inventory defect.

The final response reports stopping after checks, but `inspection.md` invokes
the practical time limit. Completion was observed more than five minutes before
the deadline. The time-limit account is unsupported. Full tool traces remain
unavailable.

The [next trial](../2026-09-29-scanpy-reconcile-luna-02/index.md) uses the same
original input and adds an audit for each split/merged replacement, plus a
check of entity, processing stage, and adjacent code before declaring a source
contradiction. These additions remain general across repositories.
