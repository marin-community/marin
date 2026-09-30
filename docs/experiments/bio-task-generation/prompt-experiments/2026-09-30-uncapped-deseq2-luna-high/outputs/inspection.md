# DESeq2 source inventory inspection

## Scope and revision

- Repository: [thelovelab/DESeq2](https://github.com/thelovelab/DESeq2), pinned Git revision `c62c60c6ff83fd84ce115cacd1c49827533f85a7`.
- The recursive GitHub tree response at that revision was not truncated. It showed an R package with `DESCRIPTION`, `NAMESPACE`, `NEWS`, R implementation modules, manual pages, one long R Markdown vignette, extdata, scripts, C++ sources, and testthat files.
- `DESCRIPTION` was read. It reports package version 1.53.5, negative-binomial high-throughput assay modeling, core dependencies including SummarizedExperiment/GenomicRanges/S4Vectors, and optional packages including apeglm, ashr, tximport, tximeta, airway and glmGamPoi.
- No README, documentation site configuration, or documentation index appeared in the pinned tree. The primary operation map is the package manual and `vignettes/DESeq2.Rmd`.
- Source inspection only: no R execution, model fitting, full biological-data download, test, or Harbor validation was performed.

## Inventory organization

The 23 unit records capture distinct input routes, model analyses, diagnostics, transformations, and selected manual operations. Unit relationships link the Pasilla matrix setup to standard and multifactor analyses; fitted results to shrinkage, thresholding, and outlier work; and size/dispersion estimation to model fitting and transformations.

Five dataset/reference records are used:

- `pasilla-rnaseq-gse18508`: observed Drosophila Pasilla RNAi study; embedded count and annotation assets are kept at their consumed stages and linked to the external package/study.
- `airway-rnaseq-gse52778`: separate human airway cell-line RNA-seq study consumed as a packaged RangedSummarizedExperiment.
- `geuvadis-tximportdata-salmon-demo`: observed transcript quantification files used as a DESeq2 import fixture. The demo's A/B labels are artificial.
- `tximportdata-gencode-v27-map`: separate transcript-to-gene reference mapping used with the quantification files.
- `deseq2-unmix-manual-toy`: explicitly fabricated inline matrices for the unmix manual example, not observed tissue data.

Pasilla count matrix and sample annotations are grouped because the vignette states they were copied from the same named pasilla package and consumes them jointly. The embedded copy is not assumed byte-identical to any current external package release. The tximport transcript-to-gene map remains separate from the GEUVADIS observations because it is an independently sourced reference. The airway study is independent.

## Vignette source map and queue

Headings and R chunk names were enumerated from the entire pinned `vignettes/DESeq2.Rmd`. Status describes actual reads this turn; pending/partial entries are remaining leads, not evidence of absent behavior.

| Location | Status |
|---|---|
| Lines 1-43 setup | Pending; chunk name `setup` mapped, setup libraries not fully read |
| Lines 44-94 standard workflow and quick start | Pending; headings/chunk `quickStart` mapped |
| Lines 95-190 help, acknowledgments, funding | Pending; informational sections mapped |
| Lines 191-255 input counts and DESeqDataSet | Inspected; unnormalized-count rule and object/design semantics |
| Lines 256-360 tximport/tximeta | Inspected tximport path through `txi2dds`; lines 361-399 tximeta metadata examples remain pending |
| Lines 400-499 count matrix input | Inspected; Pasilla loading, sample alignment, constructor, feature metadata |
| Lines 500-546 HTSeq input | Inspected; chunks `htseqDirI`/`htseqDirII` explicitly eval=FALSE |
| Lines 547-566 SummarizedExperiment input | Inspected; airway object and constructor |
| Lines 567-653 prefilter, factor levels, replicate collapsing, Pasilla context | Inspected |
| Lines 654-832 differential workflow, shrink entry, parallel note, summaries and IHW | Inspected; IHW chunk explicitly eval=FALSE |
| Lines 833-945 MA plots and shrinkage methods | Inspected in returned bounded source; method comparison and examples mapped |
| Lines 946-1082 plotCounts, result metadata, reporting and export | Inspected; export chunk eval=FALSE; external reporting package instructions were not followed |
| Lines 1083-1190 multifactor design | Inspected; Pasilla type adjustment and contrast |
| Lines 1191-1441 transformations, clustering and PCA | Inspected |
| Lines 1442-1486 standard component steps and controlGenes | Inspected; parallel to DESeq composite pipeline |
| Lines 1487-1623 contrasts/interactions | Inspected; generic design examples, many chunks eval=FALSE or display-only |
| Lines 1624-1684 time-series/LRT | Inspected; LRT code chunks eval=FALSE |
| Lines 1685-1836 extended shrinkage | Partly inspected in bounded tool output; remaining portions should be re-read from source before relying on detailed method comparisons or every example |
| Lines 1837-1893 single-cell recommendations | Inspected; source cites external benchmarks/packages, not independently reviewed |
| Lines 1894-1971 outlier handling | Inspected |
| Lines 1972-2020 dispersion alternatives | Inspected |
| Lines 2021-2109 independent filtering and threshold testing | Inspected |
| Lines 2110-2191 calculated-value access | Inspected |
| Lines 2192-2237 normalization factors | Inspected; supplied examples eval=FALSE |
| Lines 2238-2427 rank deficiency, nested individuals, absent factor combinations | Inspected; synthetic design examples |
| Lines 2428-2691 model theory, DESeq changes, methods, outlier theory, contrasts and independent-filtering theory | Pending; headings mapped, explanatory theory not needed to characterize the recorded interfaces but remains useful for deeper authoring |
| Lines 2692-2967 FAQ | Pending; headings mapped, including NA p-values, VST/PCA, paired samples, multi-group comparisons, no replicates, continuous covariates, LRT interpretation and workflow steps |
| Lines 2968-end session info and references | Pending; bibliography and render session not reviewed |

## Manual and implementation source map

Manual pages are a bounded collection enumerated by the pinned tree. Pages marked with a unit ID were read for the listed interface/meaning; “partial” means only the listed portion was read. Other pages remain mapped and pending.

| Manual source | Status |
|---|---|
| `DESeq.Rd` | Inspected; U `deseq2-pasilla-standard-wald`, LRT and negative-binomial model interface |
| `DESeq2-package.Rd` | Pending package-level description |
| `DESeqDataSet.Rd` | Inspected; constructor semantics used by U `deseq2-airway-summarizedexperiment-input` and input units |
| `DESeqResults.Rd` | Pending class description |
| `DESeqTransform.Rd` | Pending class description |
| `coef.Rd` | Pending accessor manual |
| `collapseReplicates.Rd` | Inspected; U `deseq2-collapse-technical-replicates` |
| `counts.Rd` | Pending accessor manual |
| `design.Rd` | Pending accessor manual |
| `dispersionFunction.Rd` | Pending accessor manual; its custom-fit use is covered in the vignette |
| `dispersions.Rd` | Pending accessor manual |
| `estimateBetaPriorVar.Rd` | Pending specialized fitting helper |
| `estimateDispersions.Rd` | Inspected; U `deseq2-dispersion-estimation` and U `deseq2-pasilla-dispersion-fit` |
| `estimateDispersionsGeneEst.Rd` | Pending lower-level fitting helper |
| `estimateSizeFactors.Rd` | Inspected; U `deseq2-size-factor-estimation` |
| `estimateSizeFactorsForMatrix.Rd` | Pending matrix-level helper |
| `fpkm.Rd`, `fpm.Rd` | Inspected; U `deseq2-expression-fpm-fpkm` |
| `lfcShrink.Rd` | Inspected; U `deseq2-pasilla-lfc-shrink` |
| `makeExampleDESeqDataSet.Rd` | Inspected; confirms a simulated NB fixture generator |
| `nbinomLRT.Rd` | Inspected; U `deseq2-lrt-model-comparison` |
| `nbinomWaldTest.Rd` | Inspected interface and summary; core details beyond the first 85 lines remain pending |
| `normTransform.Rd` | Inspected interface/description only; covered as a comparison in the transformation vignette |
| `normalizationFactors.Rd` | Inspected; U `deseq2-gene-sample-normalization` |
| `normalizeGeneLength.Rd` | Inspected; deprecated and moved to tximport, skipped as a standalone operation |
| `plotCounts.Rd` | Inspected; U `deseq2-pasilla-gene-count-plot` |
| `plotDispEsts.Rd` | Pending plotting helper |
| `plotMA.Rd` | Inspected interface and meaning; uses represented in shrinkage/result sections |
| `plotPCA.Rd` | Inspected; covered in U `deseq2-pasilla-transformation-qc` |
| `plotSparsity.Rd` | Inspected; U `deseq2-count-sparsity-diagnostic` |
| `priorInfo.Rd` | Pending accessor manual |
| `replaceOutliers.Rd` | Inspected; U `deseq2-pasilla-outlier-handling` |
| `results.Rd` | Inspected interface through line 110; later format/details/examples remain pending; U `deseq2-pasilla-standard-wald`, `deseq2-results-threshold-filter`, `deseq2-contrasts-interactions` |
| `rlog.Rd` | Inspected; U `deseq2-pasilla-transformation-qc` |
| `show.Rd` | Pending display method |
| `sizeFactors.Rd` | Pending accessor manual |
| `summary.Rd` | Pending summary method; examples are represented in the standard workflow |
| `unmix.Rd` | Inspected; U `deseq2-unmix-reference-mixtures` |
| `varianceStabilizingTransformation.Rd` | Inspected; U `deseq2-pasilla-transformation-qc` |
| `vst.Rd` | Inspected; U `deseq2-pasilla-transformation-qc` |

The R implementation files were enumerated, but not read as implementation sources: `R/AllClasses.R`, `R/AllGenerics.R`, `R/RcppExports.R`, `R/core.R`, `R/expanded.R`, `R/fitNbinomGLMs.R`, `R/helper.R`, `R/lfcShrink.R`, `R/methods.R`, `R/parallel.R`, `R/plots.R`, `R/results.R`, `R/rlog.R`, `R/vst.R`, and `R/wrappers.R`. The units classify use of existing tools from documentation; they make no claims about implementation internals.

## Other mapped collections

- Root package metadata: `DESCRIPTION` inspected; `NAMESPACE`, `NEWS`, `inst/CITATION`, `.gitignore`, `CODE_OF_CONDUCT.md`, and `CONTRIBUTING.md` pending or outside scientific operation scope.
- `inst/extdata/pasilla_sample_annotation.csv`: fully read (527 bytes); linked in U `deseq2-pasilla-matrix-input`.
- `inst/extdata/pasilla_gene_counts.tsv.gz`: only path and 167,602-byte tree metadata inspected; no data download/content preview.
- `inst/script/icobra_benchmarks.R`, `icobra_pkg_versions.txt`, `makeSim.R`, `runScripts.R`, `testsuite.Rmd`, `vst.nb`, `vst.pdf`, and `icobra.png`: enumerated, not read. Script/tutorial contents and PDF are remaining leads.
- `src/DESeq2.cpp`, `src/RcppExports.cpp`, `src/Makevars`, `src/Makevars.win`: enumerated, not inspected as implementation.
- Test collection enumerated; no tests were run or read: `tests/testthat.R`, `test_DESeq.R`, `test_LRT.R`, `test_QR.R`, `test_addMLE.R`, `test_betaFitting.R`, `test_collapse.R`, `test_construction_errors.R`, `test_design_matrix.R`, `test_dispersions.R`, `test_edge_case.R`, `test_factors.R`, `test_fpkm.R`, `test_interactions.R`, `test_lfcShrink.R`, `test_linear_mu.R`, `test_methods.R`, `test_model_matrix.R`, `test_nbinomWald.R`, `test_optim.R`, `test_outlier.R`, `test_parallel.R`, `test_plots.R`, `test_results.R`, `test_size_factor.R`, `test_txi.R`, `test_unmix.R`, `test_weights.R`, and `test_zero_zero.R`.

## External sources, retrieval notes, and failed access

- Pasilla: read the Bioconductor 3.23 package page (package version 1.40.0) and a 3.21 preparation-vignette search result. The page identifies per-gene/per-exon counts over selected genes, RNAi knockdown, and GSE18508. The DESeq2 sample CSV has seven rows while the package page enumerates six GEO sample accessions; because the embedded file's source release is unknown, exact release/sample-file linkage remains unresolved rather than treated as a same-version contradiction. No raw data were downloaded.
- Airway: read the Bioconductor 3.21 package page (version 1.28.0), identifying four human airway smooth muscle cell lines, dexamethasone treatment and GSE52778. The DESeq2 vignette does not pin the airway package version.
- tximportData: a bounded archived Bioconductor 3.11 vignette read supplied GEUVADIS sample/run IDs, Salmon directory layout, Salmon 0.8.2 invocation and Gencode v27 reference use. The DESeq2 source does not pin tximportData; the archived documentation is a version gap, and quantification files/map rows were not fetched.
- Public GitHub repository metadata probes for guessed repositories `thelovelab/tximportData`, `Bioconductor/airway`, and `Bioconductor/pasilla` returned HTTP 404. Data package materials were instead discovered through Bioconductor pages. No credentials were inspected.
- Web retrieval of the current tximportData package page and one Pasilla preparation-vignette page returned cache-miss errors. The archived tximportData mirror, Pasilla release page, DESeq2 source and search-result metadata provided the recorded facts. Failed fetches do not establish missing documentation.

## Boundaries, limitations, and stopping point

- Separate units are retained for matrix construction, HTSeq import, airway object conversion and Salmon/tximport because they consume distinct upstream representations and dependencies. These input routes are not represented as interchangeable assets.
- DESeq plus results is one composed inferential unit; shrinkage, thresholds, contrasts, multifactor adjustment and outlier workflows are separate where they add a distinct inferential or scientific purpose.
- VST/rlog are grouped with sample distance/PCA QA because the vignette uses transformed values for that downstream purpose and explicitly says transformations are not DE testing input.
- The tximport demonstration's artificial A/B factor is retained as a caveat, not a biological design. The toy unmix matrices remain labeled synthetic.
- Source-reported claims are separated from file preview and execution: only the small Pasilla sample annotation was previewed; no scientific operation or full-data asset was executed.
- Stopping reason: this pass mapped and recorded the distinct documented operation/input routes identified from the pinned tree and vignette, after checking additional API manuals and related data metadata. Remaining pending entries are mostly class/accessor references, tests, implementation internals, the vignette theory/FAQ sections and supplementary scripts; they may support deeper task-specific follow-up but do not introduce an unrecorded distinct workflow based on the source map inspected here. No time cutoff, resource limit, or access blocker caused the stop. Elapsed time was not instrumented at turn start; the end-to-end work interval is an approximate observation (about 20 minutes).
