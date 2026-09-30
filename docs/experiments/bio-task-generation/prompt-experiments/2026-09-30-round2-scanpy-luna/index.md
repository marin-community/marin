# scanpy: second cross-repository round

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The shared round 2 prompt used the same repository pin, model, source access,
and instructed ten-minute maximum as round 1. The output has 11 units and
8 data records. Structural checks pass; scientific execution and authoring
success were not tested.

The worker separates PBMC3k from PBMC68k and groups PBMC3k raw/processed
states together, while retaining the curated multi-study pancreas product.
It also separates PBMC and pancreas mapping/integration uses and includes a
Pearson-residual workflow. These are useful data-identity and coverage changes.

The preprocessing unit falsely says the modern clustering notebook does not
preserve raw counts. Parent inspection of the pinned notebook confirms cell 22
assigns `adata.layers["counts"] = adata.X.copy()` before normalization. Its
marker unit also calls AnnData gene-by-cell in one field, inconsistent with the
cell-by-gene convention elsewhere. The input-state self-check has not prevented
factual errors in records that look detailed and source-backed.

The final inspection report gives 11 units/8 data records at the top but later
says all eight units and seven datasets. The source-map rewrite instruction
did not eliminate stale counts. It also marks the experimental section
uninspected in an overview row while documenting Pearson-residual inspection
in detail later; partial states need coherent descriptions.

The final clock check is 14:45:17 against a 14:37:52 deadline, an overrun of
7 minutes 25 seconds. The parent's execution gap prevented timely observation;
the runner supplied a deadline but did not enforce a process timeout. Wider
coverage cannot be credited as an equal-budget gain. This is a budget-adherence
failure, not a validated scientific result or a model-cost measurement.

Markdown trailing spaces were normalized for required lint. The [exact original](original-inspection.json) retains the worker text and SHA-256; no semantic edits were made.

The parent checked count preservation in the [pinned clustering notebook](https://github.com/scverse/scanpy/blob/8c1463d5d97272d5811ad3f4efb57483e23b4c7e/docs/tutorials/basics/clustering.ipynb), cell 22.
