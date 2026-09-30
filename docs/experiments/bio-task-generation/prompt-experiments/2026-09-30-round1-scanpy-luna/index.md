# scanpy: first cross-repository round

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [frozen review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The unchanged prompt from `37f8a601d2` ran in a fresh Luna context with a ten-minute
maximum. JSON structure and inventory references pass. Counts are 5 units
and 6 data records; counts alone do not measure quality.

Five units cover modern clustering and annotation, PAGA trajectories,
reference mapping and BBKNN integration, and cell-cycle scoring. The worker
separates the cell-cycle gene reference from the mouse expression data despite
shared packaging. It records ingest's reference state and common-variable
requirements, and separates gene-expression inputs from the broader multiome
assay. Parent inspection of the pinned cell-cycle notebook supports the cited
matrix source, reference-list origin, and distinct before/after PCA objects.

The `scanpy-pbmc3k-pbmc68k-example-assets` record nevertheless combines two
reference/query products while explicitly saying their shared observations
are not established. This violates the existing identity rule. The four-study
pancreas H5AD, in contrast, is an identifiable curated combined product; its
multi-study provenance alone is not an error or reason to split the product.

The source map claims BBKNN is mentioned only in a catalog despite the inspected
integration tutorial and unit describing it. It also calls the continuation
queue partial. The final stopping report gives 29 seconds remaining, a plausible
finalization interval; this is not grouped with the much earlier UCSC and DESeq2
stops. An intermediate inspection file had a different stop explanation, but
review uses the worker's final saved artifact.

The integration unit's broad boundary is provisional and is not counted as a
verified error. The worker preserves both PBMC and pancreas uses, though an
author may choose separate assignments. Scientific execution, ortholog mapping
in the packaged data, and task feasibility were not verified.

The required formatter removed trailing spaces from the Markdown inspection
copy. The [exact original](original-inspection.json) preserves its text and
SHA-256; the JSONL records are unchanged. No semantic correction was applied.
