# Scanpy inventory review

Repository: scverse/scanpy. Review revision: `8c1463d5d97272d5811ad3f4efb57483e23b4c7e`. Inputs: supplied inventory and requirements under this experiment's `inputs/` directory. This is a reconciliation pass, not repository discovery, data execution, or task validation.

## Source map and status

| Collection/source | Status | Records / remaining leads |
|---|---|---|
| Scanpy integration tutorial notebook (`docs/tutorials/basics/integrating-data-using-ingest.ipynb`) | Inspected in targeted cells at `8c1463d5d97272d5811ad3f4efb57483e23b4c7e`: ingest explanation, PBMC mapping, PBMC BBKNN, pancreas BBKNN, iterative ingest and consistency checks. | `scanpy.tutorial.ingest-bbknn`, `scanpy.api.ingest`, `scanpy.api.bbknn`; other tutorial sections/figures and cited BBKNN paper not inspected. |
| Dataset loader source (`src/scanpy/datasets/_datasets.py`) | Inspected targeted definitions at `8c1463d5d97272d5811ad3f4efb57483e23b4c7e` for `paul15`, `pbmc68k_reduced`, `pbmc3k`, `pbmc3k_processed`. | PBMC identity and loader dimensions corrected. Other loader definitions and underlying asset files not inspected. |
| PAGA tutorial and Pearson residual notebook | Carried forward from input inspection only; not newly retrieved in this pass. | `scanpy.tutorial.paga-paul15`, `scanpy.tutorial.pearson-residuals`, and `scanpy.api.pearson-residuals`; later tutorial sections remain partial. |
| Other APIs, tests, external assets and terms | Not inspected in this reconciliation. | Need source-specific follow-up before task specification or release. |

## Changes and evidence

- Merged `scanpy-pbmc3k-processed` into `scanpy-pbmc3k-10x-v1`. The pinned loader documents PBMC3k as 10x healthy-donor data and `pbmc3k_processed()` as processed using the basic clustering tutorial (2638×1838 versus raw 2700×32738). The processed asset remains as a linked representation. Mapping: `scanpy-pbmc3k-processed` → `scanpy-pbmc3k-10x-v1`.
- Kept `scanpy-pbmc68k-reduced` separate. The pinned loader identifies it as subsampled/processed PBMC68k (700×765), not a representation of PBMC3k.
- Split the combined API unit `scanpy.api.ingest-bbknn` into `scanpy.api.ingest` and `scanpy.api.bbknn`. The pinned tutorial supports distinct reference label/embedding transfer and batch-balanced graph operations. Mapping: `scanpy.api.ingest-bbknn` → both new records. The broader tutorial remains a useful composition record.
- Updated unit data links to the records supported by their inputs. PBMC raw and processed forms now resolve through one observation record; pancreas remains separate multi-study data.
- Kept existing scientific coverage for PAGA and Pearson-residual workflows. Their supporting source detail is carried forward from the input inspection and was not reverified here.
- The Paul15 loader reveals a documentation discrepancy: its prose says 3461 informative genes, while the documented returned shape is 3451 genes and code selects from the informative-gene name intersection. Preserve as unresolved, not silently “corrected.”

## Unresolved claims and stopping reason

No biological assets were downloaded, previewed, or executed. Dataset license/redistribution terms, the Dropbox pancreas asset, PBMC68k original accession, Paul15 raw source asset and study-level metadata remain unknown. The complete later PAGA and Pearson-residual sections, Scanpy API implementations/tests, task graders, runtime, Harbor compatibility and data release suitability were outside this bounded reconciliation. Input inspection statements remain labeled as carried-forward where not freshly checked. These unknowns alone do not fail inventory reconciliation. The Paul15 gene-count inconsistency is a specific unresolved inventory defect requiring checking the source data or an authoritative corrected loader description.

## Summary and queue

Final counts: 6 unit records; 5 dataset records. Units: scanpy.tutorial.ingest-bbknn, scanpy.tutorial.paga-paul15, scanpy.tutorial.pearson-residuals, scanpy.api.ingest, scanpy.api.bbknn, scanpy.api.pearson-residuals. Datasets: scanpy-pbmc3k-10x-v1, scanpy-pbmc10k-10x-v3, scanpy-pbmc68k-reduced, scanpy-pancreas-4studies, scanpy-paul15.

Pending queue: (1) resolve Paul15 3461/3451 discrepancy using authoritative source asset; (2) inspect remaining PAGA and Pearson residual sections; (3) verify PBMC and pancreas asset metadata/terms. This queue is carried forward or newly identified; no completion is implied.

Recommendation: `needs_inventory_repair` — the Paul15 gene-count claim conflicts within the inspected loader documentation and its reported returned shape; authoritative source evidence is needed.

The pass stopped when the assigned review's bounded time window reached its practical limit. Only lightweight source retrieval and inventory edits were performed; no execution, data download or scientific analysis occurred.
