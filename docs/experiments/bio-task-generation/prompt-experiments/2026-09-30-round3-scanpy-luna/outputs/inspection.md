# Scanpy source inventory

## Repository and revision

This pass inspected `scverse/scanpy` at revision `8c1463d5d97272d5811ad3f4efb57483e23b4c7e`, resolved by the runner before launch. GitHub's recursive tree response reported `truncated: false`. GitHub API source files and selected notebook source cells were retrieved at that exact revision on 2026-09-30. No repository was cloned, and no data files were downloaded or executed.

The pinned README describes Scanpy as a Python toolkit for scalable single-cell gene-expression analysis, with preprocessing, visualization, clustering, trajectory inference and differential-expression testing; it notes experimental Dask support and directs users to a documented public API. The inspected APIs and tutorials support those categories. Other repository code includes data I/O, metrics, queries, experimental functions, plotting and external-package integrations.

## Inventory summary

`units.jsonl` contains 7 inspected units; `datasets.jsonl` contains 7 identifiable study-data or processed-data records. The records span:

- multi-sample QC, clustering and marker-based annotation on human bone-marrow multiome gene-expression counts;
- asymmetric reference/query mapping with `tl.ingest` and batch-aware graph integration with BBKNN on PBMC and pancreas data;
- graph abstraction and root-dependent pseudotime for mouse myeloid progenitors;
- analytic Pearson-residual feature selection and downstream clustering for two PBMC count assays;
- cell-cycle scoring and regression in murine hematopoietic progenitors;
- exact versus approximate neighbor graph backends on Paul15;
- global graph autocorrelation with Moran's I on a processed PBMC68k example.

All 7 units classify as `tool_use`: the sources compose or apply existing Scanpy and ecosystem APIs. The inventory does not characterize API implementation work as tool creation. The named datasets distinguish raw matrices from processed representations and preserve unknown accessions, license terms, sizes and data lineage. Public URLs, repository presence and source hashes do not establish redistribution eligibility. None of the candidate workflows has been executed or validated as a task.

## Source map and inspection queue

| Collection / location | Inspection status | Resulting unit IDs | Specific pending leads |
|---|---|---|---|
| Repository README (`README.md`) | Inspected: toolkit scope, public API policy, documentation pointers and citations | — | Follow README's API/doc links only when needed for a selected lead. |
| Tutorial index (`docs/tutorials/index.md`) | Inspected: basic workflows, visualization, trajectories and experimental sections | — | External scverse tutorial collection at `https://scverse.org/learn/` was not followed. |
| Basic tutorial index (`docs/tutorials/basics/index.md`) | Inspected: enumerated `clustering`, `clustering-2017`, and `integrating-data-using-ingest` | — | `clustering-2017.ipynb` remains pending; it is the source of processed PBMC3k and may add details about preprocessing/markers. |
| Clustering tutorial (`docs/tutorials/basics/clustering.ipynb`) | Inspected narrative and setup/workflow source cells across QC, feature selection, embeddings, clustering and annotation | `scanpy:clustering-cell-annotation-multiome` | Notebook has more plotting/output cells than reviewed; no execution. OpenProblems DOI landing metadata and individual files remain uninspected. |
| Ingest/integration tutorial (`docs/tutorials/basics/integrating-data-using-ingest.ipynb`) | Inspected narrative, selected setup/mapping/comparison cells; PBMC reference/query and pancreas sections | `scanpy:ingest-bbknn-integration` | Linked BBKNN pancreas curation notebook and FTP archive remain uninspected; source-to-batch mappings and Dropbox H5AD contents are open. |
| Trajectory index (`docs/tutorials/trajectories/index.md`) | Inspected: lists PAGA Paul15 tutorial | — | Listed tutorial below inspected. |
| PAGA tutorial (`docs/tutorials/trajectories/paga-paul15.ipynb`) | Inspected headings and selected cells for recipe, graph, clustering, PAGA, DPT and gene paths | `scanpy:paga-dpt-paul15-hematopoiesis` | Full plot/output material not reviewed; Paul2015 paper/data lineage and reuse terms require follow-up. |
| Experimental tutorial index (`docs/tutorials/experimental/index.md`) | Inspected: lists Pearson residuals and Dask tutorials | — | Listed tutorials below: Pearson workflow inspected; Dask has only a title/purpose preview. |
| Pearson-residual tutorial (`docs/tutorials/experimental/pearson_residuals.ipynb`) | Inspected narrative and selected setup/workflow source cells for PBMC 3k/10k, QC, residual HVGs, PCA and clustering | `scanpy:pbmc-pearson-residual-workflow` | Complete optional-argument/wrapper sections and linked 10x metadata pages remain pending. Dense residual memory requirements need author-specific sizing. |
| Dask tutorial (`docs/tutorials/experimental/dask.ipynb`) | Partial: opening title and purpose only; says tutorial illustrates Dask in a simple Scanpy analysis | — | Inspect notebook sections, data and computation examples before identifying any unit. |
| How-to index (`docs/how-to/index.md`) | Inspected: lists cell-cycle, KNN transformers and Marsilea plotting notebooks | — | Marsilea plotting notebook remains uninspected. |
| Cell-cycle how-to (`docs/how-to/cell-cycle.ipynb`) | Inspected narrative/setup and workflow outlines for Nestorowa matrix, gene lists, scoring and regression | `scanpy:cell-cycle-scoring-regression-nestorowa` | Inspect ZIP member and the two cited publications/data versions if task author needs data/schema/licensing evidence. |
| KNN transformer how-to (`docs/how-to/knn-transformers.ipynb`) | Inspected headings and code outlines for exact, PyNNDescent and Annoy neighbors plus Leiden/UMAP comparison | `scanpy:approximate-knn-transformer-comparison` | Do not use tutorial timings as portable performance claims; full environment pinning is open. |
| API dataset index (`docs/api/datasets.md`) and dataset getters (`src/scanpy/datasets/_datasets.py`, `registry.yaml`) | Dataset index entries inspected; getter source selectively inspected for PBMC3k, PBMC68k, Paul15 and linked data; registry selectively checked for PBMC3k and Paul15 | Dataset records linked by units: PBMC3k, PBMC68k, Paul15 | Remaining getter entries (`blobs`, `ebi_expression_atlas`, `krumsiek11`, `moignard15`, `toggleswitch`, `visium_sge`) are uninspected. `src/scanpy/datasets/10x_pbmc68k_reduced.zarr.zip` was identified by tree metadata only; archive contents not read. |
| Core preprocessing API index (`docs/api/preprocessing.md`) | Inspected category headings and enumerated listed function symbols: basic preprocessing, recipes, data integration, doublet detection, demultiplexing and neighbors | Workflow units above exercise selected functions | Individual docs/source for most entries remain pending; useful leads include `calculate_qc_metrics`, `scrublet`, `normalize_total`, `highly_variable_genes`, `pca`, `combat`, `harmony_integrate`, `hashsolo`, and `neighbors`. |
| Core tools API index (`docs/api/tools.md`) | Inspected categories and symbols for embeddings, clustering/trajectory, ingest, markers, gene scores/cell cycle and simulation | Workflow units above exercise selected functions | Individual docs/source for most entries remain pending; useful leads include `rank_genes_groups`, `filter_rank_genes_groups`, `score_genes_cell_cycle`, `sim`, and embedding/trajectory tools. |
| Metrics API (`docs/api/metrics.md`, `src/scanpy/metrics/_morans_i.py`) | Metrics index inspected; Moran's I documentation and implementation semantics inspected | `scanpy:morans-i-on-cell-graph` | `gearys_c`, `confusion_matrix` and `modularity` are listed but not inspected in detail. |
| Other API indexes (`docs/api/classes.md`, `experimental.md`, `get.md`, `io.md`, `queries.md`, `plotting.md`, `settings.md`) | Inspected section titles and symbol listings; not individual operations | — | Potential distinct leads: AnnData extraction/aggregation (`get.aggregate`), count-matrix input (`io.read_10x_mtx`), gene coordinate/annotation queries, experimental Pearson wrappers, and broader visualization API. |
| External integrations (`docs/external/preprocessing.md`, `docs/external/tools.md`) | Inspected listed headings and symbols only | — | BBKNN/Scanorama/MNN/MAGIC and external trajectory/embedding tools merit source-context review if those ecosystems are in scope. |
| Legacy/broad source collections (`docs/tutorials/basics/clustering-2017.ipynb`, plotting tutorials, API implementation modules, tests) | Not inspected beyond tree inventory or cross-reference | — | The large 2017 clustering and plotting notebooks could add example/data context; source module implementations and tests can refine interfaces, edge cases and grader contracts. |

## Data records and relationships

`scanpy-data:pbmc3k-raw-and-processed` groups the raw 10x PBMC3k matrix, Scanpy-hosted raw H5AD and processed H5AD because the inspected getter explicitly ties those representations to the same study observations. Their stages and dimensions remain distinct. It supports ingest's processed reference and the residual tutorial's raw counts.

The two 10x PBMC assays used by the Pearson-residual tutorial are separate records because they are distinct 3k v1 and 10k v3 datasets. OpenProblems multiome samples are represented together as a registry-backed pair of sample assets. The pancreas H5AD and paper FTP ZIP remain separate assets under a combined-product record because the source did not prove they are the same object. Paul15 and Nestorowa are distinct mouse progenitor study assets. PBMC68k is a processed 700-cell subset linked to its 10x source; the bundled ZIP is known from repository tree metadata, not content inspection.

## Access issues and external sources

The GitHub CLI tree and contents requests succeeded for public pinned source. An attempt to open the Figshare DOI, current 10x PBMC pages, and Paul2015 DOI through the web reader returned “URL … is not accessible via this tool.” Those external pages were not treated as inspected evidence. The inventory retains the source URLs and marks their metadata and terms unresolved. Dropbox and Sanger FTP links were observed in the integration notebook but not fetched. No authenticated credentials were inspected.

## Coverage, hypotheses, and stopping point

Source-backed findings are the documented interfaces, data descriptions and operations recorded in each unit. Possible tasks—reproducing clustering, comparing integration, evaluating trajectory structure, scoring cell cycle, comparing approximate graphs, or ranking feature autocorrelation—are hypotheses for authoring, not validated tasks. Human label annotation, reference choice, pseudotime root selection, package versions, data licensing and deterministic grading remain open decisions.

This is a partial discovery pass, not a claim that Scanpy's repository or API was exhaustively inspected. At 15:21 UTC the parent agent reminded me that the supplied cutoff (2026-09-30T15:19:22.091677+00:00) had passed and directed me to stop inspection. The clock was checked at 2026-09-30 15:22:15 UTC; I then performed only the required file-integrity check. The remaining queue is explicit above: clustering-2017, Dask, plotting tutorials, external-tool details, remaining API symbols, tests and external data pages remain leads for continuation. No execution, Harbor compatibility, redistribution eligibility or scientific validity is claimed.
