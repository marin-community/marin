# Scanpy source inventory inspection

Repository: [scverse/scanpy](https://github.com/scverse/scanpy)
Inspected revision: `8c1463d5d97272d5811ad3f4efb57483e23b4c7e` (resolved by runner through GitHub commits API)
Retrieval date: 2026-09-30 UTC. External landing-page probes were attempted on 2026-09-30.

## Repository map and inspected scope

Scanpy is a Python toolkit for single-cell gene-expression analysis. The README identifies preprocessing, visualization, clustering, trajectory inference and differential-expression testing as core activities, with AnnData as the data container. Source tree also has external-method adapters, metrics, plotting, data accessors, and dataset loaders. This inventory prioritizes two tutorial workflows with biological context and linked datasets; it is a partial discovery pass, not coverage of all public APIs.

| Collection and location | Status | Units / queue |
|---|---|---|
| `README.md` | Inspected for project scope and API declaration | Repository is a scalable single-cell analysis toolkit; directs readers to API docs |
| `docs/tutorials/index.md` | Inspected; enumerates Basic workflows, Visualization, Trajectory inference, Experimental; notes old Scanpy spatial tutorials are deprecated | Continue through plotting and experimental indexes |
| `docs/tutorials/basics/index.md` | Inspected; enumerates `clustering`, `clustering-2017`, `integrating-data-using-ingest` | `clustering.ipynb` inspected (units `scanpy-tutorial-bm-qc-preprocess-cluster-8c1463d`, `scanpy-tutorial-bm-cell-annotation-markers-8c1463d`); ingest inspected (unit `scanpy-tutorial-ingest-bbknn-integration-8c1463d`); clustering-2017 pending |
| `docs/tutorials/basics/clustering.ipynb` | Inspected source cells and relevant prose from load through marker analysis | QC/preprocessing/clustering and marker annotation/DE recorded as separate related units; later notebook sections after marker analysis not inspected |
| `docs/tutorials/basics/integrating-data-using-ingest.ipynb` | Inspected PBMC mapping and pancreas integration, BBKNN, iterative reference mapping and label consistency sections | Unit `scanpy-tutorial-ingest-bbknn-integration-8c1463d`; dataset records for PBMC inputs and four-study pancreas data |
| `docs/tutorials/trajectories/index.md` | Inspected; enumerates only `paga-paul15` | `paga-paul15.ipynb` inspected across setup, preprocessing, graph denoising, PAGA, embedding, pseudotime and path summaries; unit `scanpy-tutorial-paul15-paga-dpt-8c1463d` |
| `docs/tutorials/trajectories/paga-paul15.ipynb` | Inspected source cells/prose and code for trajectory workflow | PAGA and diffusion pseudotime unit recorded; linked Paul15 data record |
| `docs/how-to/cell-cycle.ipynb` | Inspected source cells for the full scoring/regression analysis | Unit `scanpy-howto-cell-cycle-score-regress-8c1463d`; Nestorowa observed data and distinct Tirosh cell-cycle gene reference records; terms and full sample metadata pending |
| `docs/api/preprocessing.md` | Inspected full short API catalog | Catalog entries listed below; tutorial-used operations have usage evidence, but remaining operations pending |
| `docs/api/tools.md` | Inspected full short API catalog | Catalog entries listed below; tutorial-used operations have usage evidence, but remaining operations pending |
| `docs/api/metrics.md` | Inspected short catalog only | `modularity`, `confusion_matrix`, `gearys_c`, `morans_i` added as 4 pending operation entries; source semantics and use examples remain uninspected |
| `docs/api/datasets.md` | Inspected full short API catalog | Dataset loader entries listed below; only `paul15` examined in depth |
| `src/scanpy/datasets/registry.yaml` | Inspected beginning containing `burczynski06`, `moignard15`, `paul15`, `pbmc3k`, `pbmc3k_processed` | Public URLs and checksums visible for these; later registry entries not inspected |
| `src/scanpy/datasets/_datasets.py` | Inspected relevant definitions: `paul15`, `pbmc3k`, `pbmc3k_processed`, `pbmc68k_reduced`, `krumsiek11` | Paul15 details used for data record; other loaders are leads only |

### API catalogs requiring continuation

Preprocessing catalog in `docs/api/preprocessing.md`: `calculate_qc_metrics`, `filter_cells`, `filter_genes`, `highly_variable_genes`, `log1p`, `pca`, `normalize_total`, `regress_out`, `scale`, `sample`, `downsample_counts`; recipes `recipe_zheng17`, `recipe_weinreb17`, `recipe_seurat`; integration `combat`, `harmony_integrate`, `bbknn`, `neighbors`; doublets `scrublet`, `scrublet_simulate_doublets`; demultiplexing `hashsolo`. Tutorial-backed use was inspected for QC metrics, cell/gene filtering, Scrublet, normalization, HVGs and Zheng recipe, PCA, neighbors, and BBKNN is mentioned only by catalog; detailed individual API semantics remain pending.

Tools catalog in `docs/api/tools.md`: `pca`, `tsne`, `umap`, `draw_graph`, `diffmap`, `embedding_density`, `leiden`, `dendrogram`, `dpt`, `paga`, `ingest`, `rank_genes_groups`, `filter_rank_genes_groups`, `marker_gene_overlap`, `score_genes`, `score_genes_cell_cycle`, `sim`. Tutorial-backed use was inspected for PCA, UMAP, Leiden, DPT, PAGA, graph drawing, rank genes; other operations and API docstrings remain pending.

Dataset catalog in `docs/api/datasets.md`: `blobs`, `ebi_expression_atlas`, `krumsiek11`, `moignard15`, `pbmc3k`, `pbmc3k_processed`, `pbmc68k_reduced`, `paul15`, `toggleswitch`, `visium_sge`. Only Paul15 was followed into the loader in depth. PBMC loaders, embryo qRT-PCR, Visium examples, simulated fixtures and EBI query remain pending.

Queue count from the four inspected API catalogs: 21 preprocessing entries, 17 tools entries, 10 dataset loaders, and 4 metrics entries (52 catalog entries total; categories can refer to overlapping operations). These are still pending as API-reference entries even where tutorial use supplied operational semantics. One basics notebook (`clustering-2017`) is explicitly pending; plotting and experimental child-page contents are not enumerated yet and need listing before their queue can be counted.

## Findings and inventory organization

Five source-backed units are recorded in `units.jsonl`. The two units over the Open Problems human bone-marrow samples are deliberately separated: one performs QC through graph clustering; the next interprets clusters with marker sets and differential expression, and links to the first for its upstream AnnData/neighbor graph. The Paul15 analysis is a distinct mouse hematopoiesis trajectory workflow using PAGA and diffusion pseudotime. Dataset records separately represent the two-sample human benchmark input, Paul et al. mouse progenitors, PBMC3K/PBMC68K reference/query products, compiled human pancreas data, Nestorowa mouse expression data, and the Tirosh human cell-cycle reference list. Tutorial prose alone does not establish data redistribution rights.

Source-backed facts include the described assay/sample origins, input filenames, tutorial calls and loader fields. Hypotheses for task authoring include grading an AnnData output, graph/cluster labels, ranked markers, or lineage path summaries; the source does not establish a deterministic grading contract or Harbor compatibility.

## External sources, access, and failed retrievals

The basic tutorial declares Figshare DOI `10.6084/m9.figshare.22716739.v1` through a `pooch` DOI registry and maps two HDF5 feature-barcode files to sample IDs. The DOI landing page and Figshare API were inaccessible through the available browser; no asset was downloaded. Record sizes, checksums, license and exact per-file metadata are therefore unknown.

The Scanpy dataset registry maps `paul15` to its hosted `paul15.h5`, includes SHA-256 `6161984f758dd464992edc23f1a8ab89b2081600130c6aa93a1ab12b3ada5bfb`, and gives `https://falexwolf.de/data/paul15.h5` as fallback. The loader says the data was supplied by email from Amit Lab and links to `theislab/scAnalysisTutorial` for R loading. The actual binary was not fetched or checked. `https://doi.org/10.6084/m9.figshare.22716739.v1` and `https://api.figshare.com/v2/articles/22716739` browser opens both returned inaccessible errors; `https://falexwolf.de/data/paul15.h5` was likewise inaccessible to browser. These are source-access limitations, not evidence that the files are unavailable through other supported routes.

A bounded `gh api`/Python extraction attempt for selected dataset definitions failed due to a malformed regular expression (Python `PatternError: missing ), unterminated subpattern`). It was retried with a simpler extractor and succeeded; no repository file was changed by the failed attempt.

## Deduplication and continuation leads

The two sample HDF5 files in the Figshare deposit are grouped into one dataset record because the tutorial retrieves and concatenates them as samples in one study dataset; the sample assets remain explicit. Paul15's hosted file and fallback URL are grouped as access routes for one loader-defined dataset, not separate observations. The tutorial's analysis transformations are described as processing stages rather than additional data records.

Next useful inspections: `clustering-2017.ipynb` for the legacy PBMC workflow; `docs/api/metrics.md` entries and implementations for graph modularity, confusion matrices, Geary C and Moran I; experimental Pearson residuals and Dask tutorials; remaining dataset loader docs and registry entries; external integrations and metrics API sources. The modern clustering tutorial points to a cited external marker resource and suggests Scanorama/scvi-tools for integration, but those methods were not followed here.

## Stopping point and coverage

This is a partial pass. Five distinct, source-inspected units and six data records are saved; the API catalogs and other tutorials listed above still contain useful leads. The pass stopped at 2026-09-30 14:14:45 UTC with 29 seconds remaining before the assigned maximum deadline (14:15:14.879376 UTC); that remaining interval was too short to inspect another queued operation and reconcile an additional record responsibly. This is a partial pass, not a claim that the deadline had already elapsed. Source inspection establishes documented uses only; no scientific tools, notebooks, Harbor jobs or model calls were executed.
