# Scanpy source inventory

Retrieved 2026-09-29 from the public Scanpy repository and its live stable documentation. GitHub's `main` history page showed latest commit abbreviated as `ec37402` (“ci: fix autofix workflow”, Sep 3, 2026); the full SHA was not exposed in the inspected view. Documentation pages identify themselves as stable but do not expose immutable source revisions consistently. This is a revision gap; tutorial rendered package snapshots are recorded per unit where useful. No repository clone, data download, or analysis execution was performed.

Scanpy is a Python toolkit for single-cell gene-expression preprocessing, visualization, clustering, differential expression and trajectory analysis. The inspected material supports three distinct scientific workflows: reference label/embedding mapping and batch-aware integration; mouse hematopoietic trajectory reconstruction; and UMI count QC plus Pearson-residual variable-gene selection. The revised inventory contains 7 unit records and 5 dataset records. Two combined ingest/BBKNN records were split by scientific operation, and the processed PBMC3k record was merged into its raw PBMC3k record as a linked stage asset.

## Source map and coverage

| Collection | Location | Status and resulting unit IDs | Uninspected leads |
|---|---|---|---|
| Repository overview and structure | https://github.com/scverse/scanpy | Inspected README summary, top-level folders (`docs`, `src`, `tests`, `benchmarks`), public API statement, and latest main history commit. | Source implementation, tests, benchmark suite, release notes and license text not reviewed. |
| Basic workflow tutorials | https://scanpy.readthedocs.io/en/stable/tutorials/index.html | Index inspected. Ingest/BBKNN tutorial covered as `scanpy.tutorial.ingest-bbknn`. | Primary preprocessing/clustering tutorial was inaccessible on direct fetch due HTTP 429; clustering, DE markers, QC and UMAP workflow remain leads. Legacy PBMC tutorial not fully reviewed. |
| Batch integration tutorial | https://scanpy.readthedocs.io/en/stable/tutorials/basics/integrating-data-using-ingest.html | Inspected ingest conceptual distinctions, PBMC reference/query sequence, BBKNN calls, and pancreas data context. IDs: `scanpy.tutorial.ingest-bbknn`, `scanpy.api.ingest-bbknn`. | Remaining pancreas tutorial analysis/evaluation not fully inspected; linked BBKNN documentation and cited studies not followed. |
| Trajectory tutorial | https://scanpy.readthedocs.io/en/stable/tutorials/trajectories/paga-paul15.html | Inspected setup/data dimensions, Zheng17 preprocessing, PCA/neighbors/draw_graph and optional diffusion denoising. ID: `scanpy.tutorial.paga-paul15`. | Later PAGA, root/pseudotime, selected path and gene trend sections need targeted inspection; source link to PAGA repo and Paul study not followed. |
| Experimental preprocessing tutorial | https://scanpy.readthedocs.io/en/stable/tutorials/experimental/pearson_residuals.html | Inspected conceptual rationale, 10x URLs/checksums, QC thresholds and variable gene selection setup. ID: `scanpy.tutorial.pearson-residuals`. | Later optional settings/wrapper sections only noted from outline; no source implementation/test inspection. |
| Public API index | https://scanpy.readthedocs.io/en/stable/api.html | API categories and signatures/roles surveyed from index; focused records `scanpy.api.ingest-bbknn`, `scanpy.api.pearson-residuals`. | Large preprocessing/tools/plotting, external API, IO, spatial and query APIs remain leads; index alone was not treated as proof of a scientific unit. |
| Dataset loaders and external assets | Scanpy dataset API, PBMC loader page, tutorial download links | Dataset metadata captured for PBMC3k raw, PBMC10k raw, processed/reduced PBMC loaders, pancreas collection, and Paul15. | No asset downloaded or previewed; access terms and redistribution eligibility largely unknown. See dataset limitations. |

## Relationships and handoff

`scanpy.tutorial.ingest-bbknn` depends on Scanpy API operations represented by `scanpy.api.ingest-bbknn`; they share PBMC reference/query data and the four-study pancreas collection. `scanpy.tutorial.pearson-residuals` uses the raw 3k and 10k PBMC matrices and marker genes from the PBMC clustering tutorial, which makes it a potential bridge to a fuller PBMC workflow after that tutorial is inspected. `scanpy.tutorial.paga-paul15` uses the Paul15 dataset and composes preprocessing, graph construction, PAGA and gene trends; the PAGA/DPT API components remain linked leads rather than separate detailed units. The processed PBMC3k derivative and PBMC68k reduced dataset are linked to PBMC3k only where the docs identify them as PBMC data/use; their exact observational relationships remain explicit unknowns.

Source-backed facts include tutorial operations, named assets, documented dimensions, stated study citations and public URLs. Hypotheses for task authoring include grading marker recovery, reference label transfer, batch mixing, or a PAGA topology/path result. The inspected pages do not establish suitable ground truth, executable deterministic graders, or Harbor compatibility.

## Stop reason and remaining work

This pass stopped after a bounded source/documentation survey within the assigned ten-minute window. A direct basics tutorial fetch returned HTTP 429; the primary coverage achieved through other rendered pages and repository overview was sufficient to save provisional records. The next pass should inspect the primary PBMC preprocessing/clustering tutorial and remaining sections of PAGA and Pearson residuals, obtain the full repository revision and per-file revisions, inspect implementation/tests for selected APIs, and verify data metadata, provenance, access and terms. No execution claim is made.


## Reconciliation pass (2026-09-29)

Pinned repository revision supplied by runner: `8c1463d5d97272d5811ad3f4efb57483e23b4c7e`. Read pinned GitHub tree; read bounded source for `docs/tutorials/basics/integrating-data-using-ingest.ipynb`, `docs/tutorials/trajectories/paga-paul15.ipynb`, `docs/tutorials/experimental/pearson_residuals.ipynb`, `src/scanpy/datasets/_datasets.py`, and `src/scanpy/experimental/pp/_highly_variable_genes.py`. The notebook tutorial pages were inspected selectively for stated purpose, calls, inputs and output sections; outputs embedded in notebooks are prior notebook content, not executions in this pass. Pinned API source confirms Pearson residual HVG expects raw count input and requires `n_top_genes`; loader source confirms pbmc3k_processed returns a processed AnnData with graph/embedding fields. No external asset was fetched.

The initial inventory cited mutable stable docs and abbreviated `ec37402`; replacements cite the runner-pinned SHA only for GitHub files actually read. Carried-forward stable-doc observations remain dated/live documentation evidence and are not retroactively pinned.

## Recomputed coverage and queue

Final counts: **7 units**, **5 datasets**. Inspected unit IDs: scanpy.tutorial.ingest-pbmc, scanpy.tutorial.bbknn-pancreas, scanpy.tutorial.paga-paul15, scanpy.tutorial.pearson-residuals, scanpy.api.pearson-residuals, scanpy.api.ingest, scanpy.api.bbknn. Dataset links are scoped: ingest uses PBMC3k and PBMC68k; BBKNN uses the four-study pancreas collection; PAGA uses Paul15; Pearson residual tutorial/API use the two separately recorded 10x PBMC matrices.

Uninspected leads carried forward: primary PBMC clustering tutorial (previous retrieval returned HTTP 429; this pass did not retry rendered docs), remaining API categories, implementation/tests beyond selected methods, original study/accession and terms for Paul15, PBMC68k and pancreas; full later tutorial sections and actual asset preview. Existing reported cell counts and asset facts are source-doc statements, not newly executed results. Discovery remains partial.
