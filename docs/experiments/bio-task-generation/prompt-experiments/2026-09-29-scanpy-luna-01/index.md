# Scanpy transfer trial

[Experiment index](../index.md)

Status: completed and reviewed. This trial used the exact generic prompt from
[bedtools trial 03](../2026-09-29-bedtools-luna-03/index.md), changing only the
repository substitutions. A fresh Luna worker has the same ten-minute budget
and lightweight source-inspection envelope. The reviewer expects no fixed unit
count and supplies no repository-specific hints.

- [Run configuration](run.json)
- [Generic prompt](template.md) and [resolved prompt](resolved-prompt.md)
- [Exact launch message](launch-message.txt)
- [Review criteria fixed before launch](review-criteria.md)
- Original outputs: [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Independent structural checks and output hashes](structural-review.json)
- [Worker's final message](worker-final.txt)

This checks transfer to API and notebook sources. It is not a controlled
before/after comparison on the same repository. Effective reasoning effort,
model version, context/output/compaction limits, usage, and cost remain
unreported by the runner.

## Results and interpretation

The five units and six data records pass the independent structural checks.
The inventory covers ingest/BBKNN integration, Paul15 trajectory analysis, and
Pearson-residual preprocessing, with narrower API records for some workflows.
It describes AnnData input state and changes to labels, embeddings, graphs, and
expression values, and classifies all units as tool use. It preserves coverage
gaps and does not claim scientific execution or task validation.

The raw PBMC3k matrix and its processed version have separate dataset IDs even
though the processed record identifies itself as a derivative of the same
observations. They should be assets under one dataset record. This differs from
the earlier bedtools failure: the worker now keeps independent references
separate, but does not reconcile repeated observations across sources.
The pinned [dataset loader](https://github.com/scverse/scanpy/blob/8c1463d5d97272d5811ad3f4efb57483e23b4c7e/src/scanpy/datasets/_datasets.py#L474)
also documents the processed derivative.

The primary clustering tutorial returned HTTP 429. The source map makes that
gap visible, but the worker did not recover the underlying notebook. The parent
could read the [repository notebook](https://github.com/scverse/scanpy/blob/8c1463d5d97272d5811ad3f4efb57483e23b4c7e/docs/tutorials/basics/clustering.ipynb)
through GitHub's contents API. Source retrieval can therefore be improved
without changing the scientific unit definition. Other units extend into later
tutorial sections that the worker says it identified only from outlines;
those output claims need further source inspection before authoring.

The worker reports an abbreviated September 3 commit from its GitHub view.
The parent had resolved a different September 29 HEAD before this trial.
Neither identifies an immutable revision for all inspected stable documentation,
so the review preserves the worker's revision gap.

This is evidence that the general recipe can produce useful records for another
repository type. It does not establish comprehensive coverage or consistent
data grouping. The bedtools tool-role error did not recur in this sample.

## Next change

The next revision adds a final reconciliation of dataset records across source
collections, suggests official source files when important rendered pages are
blocked or incomplete, and defines tool use, tool creation, and mixed work.
The [second Scanpy trial](../2026-09-29-scanpy-luna-02/index.md) tests this revision
with the same review criteria and execution envelope.
