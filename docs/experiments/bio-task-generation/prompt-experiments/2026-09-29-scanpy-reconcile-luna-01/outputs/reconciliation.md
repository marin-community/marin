# Scanpy inventory reconciliation

Input inventory: `docs/experiments/bio-task-generation/prompt-experiments/2026-09-29-scanpy-reconcile-luna-01/inputs`. Repository: scverse/scanpy. Pinned revision supplied by runner: `8c1463d5d97272d5811ad3f4efb57483e23b4c7e`.

## Sources inspected

- At the supplied revision, `src/scanpy/datasets/_datasets.py`, targeted definitions for `paul15`, `pbmc68k_reduced`, `pbmc3k`, and `pbmc3k_processed`.
- At the supplied revision, `docs/tutorials/basics/integrating-data-using-ingest.ipynb`, targeted markdown and code cells for PBMC mapping, BBKNN, pancreas batch integration, pancreas reference mapping, and annotation consistency.
- All original inventory files and requirements. PAGA and Pearson-residual source descriptions were carried forward; not newly retrieved during this pass.

## Corrections and record mapping

- Merged dataset `scanpy-pbmc3k-processed` into `scanpy-pbmc3k-10x-v1`, retaining the processed loader as a linked asset. The pinned loader calls it processed PBMC3k, says it follows the basic clustering tutorial, and documents shapes of 2638×1838 (processed) and 2700×32738 (raw). Mapping: `scanpy-pbmc3k-processed` → `scanpy-pbmc3k-10x-v1`.
- Kept `scanpy-pbmc68k-reduced` separate. The pinned loader identifies it as subsampled/processed PBMC68k and documents 700×765; it is not the PBMC3k observations.
- Split `scanpy.api.ingest-bbknn` into `scanpy.api.ingest` and `scanpy.api.bbknn`. The pinned tutorial describes reference-based label/embedding transfer separately from batch-balanced neighbor graph construction. Mapping: `scanpy.api.ingest-bbknn` → `scanpy.api.ingest`, `scanpy.api.bbknn`.
- Updated unit links after the dataset merge. The distinct four-study pancreas collection remains a separate record. The tutorial's multiple PBMC and pancreas procedures remain represented by the broad tutorial unit and narrower API records.
- The Paul15 loader's docstring states “3461 informative genes,” while its returned example states 2730×3451 and the code filters using the informative-gene name set. This inconsistency is preserved as unresolved.

## Unresolved claims and stopped checks

Paul15's 3461/3451 discrepancy needs authoritative data/source verification. The underlying Paul15 asset URL and license, PBMC68k original accession, pancreas asset availability/terms and detailed per-study metadata remain unknown. The rest of the PAGA and Pearson-residual tutorials, Scanpy API implementations/tests, task execution, graders, Harbor compatibility and release eligibility were not checked in this review. No data were downloaded or executed. Carried-forward mutable stable-doc claims are explicitly not attributed to the pinned revision.

JSONL parsing, required-field presence, unique identifiers, and all dataset and related-unit links were checked after edits. Final files contain 6 units and 5 dataset records. This check does not establish scientific validation or data suitability.

Recommendation: `needs_inventory_repair` — the Paul15 gene-count inconsistency is a specific unresolved inventory defect. The pass stopped after the targeted pinned-source checks identified the identity corrections and this unresolved defect; resolving it requires additional authoritative source or asset inspection.
