# DESeq2 scientific source inventory

Scope: thelovelab/DESeq2 at c62c60c6ff83fd84ce115cacd1c49827533f85a7 (DESCRIPTION version 1.53.5). The complete GitHub tree reports `truncated=false`. The package has one main R Markdown vignette, generated Rd API references, an observed-study test-suite notebook, simulation/benchmark scripts, a Mathematica VST derivation, and synthetic regression tests. No notebook execution, dependency installation, cloning, biological-data download, cloud work or other agents occurred.

Investigation began 2026-09-30 at approximately 17:08:08 UTC (first measured timestamp; launch read immediately preceded it). Final elapsed time is recorded below. Sources were read using elevated normal `gh api` and public web reads. Only the launch, resolved prompt, source-access instructions and discovered external repository/source material were read. Only the three assigned output files were written.

Inventory organization: 90 units and 19 data records, computed by parsing the final JSONL files. There are 86 tool-use, three mixed implementation/use, and one tool-creation units. IDs reflect scientific operations; the DESeq2 repository/revision fields identify the inventory target, while external units retain their separately pinned source revisions in `sources`. Units have provisional boundaries and do not establish executable tasks or grading readiness.

## Boundaries and state decisions

- Constructors, normalization, dispersion, fitting, results extraction and transformation are separate operations linked to compositions. Repeated API/vignette/test presentations were consolidated.
- Observed-study analyses remain separate when study design, selection or upstream processing changes the question. The original airway aligned-count object, two-sample Salmon import demonstration, full v29 gene-quantification product, and described v49 re-quantification are linked stages of one study, not interchangeable matrices.
- GENCODE, Ensembl, PomBase and org.Hs.eg.db are independent reference/annotation products. A shared package does not merge human GEUVADIS and fly SRR1197474 observations. Missing Bottomly parameter products remain distinct from legacy ReCount until actual lineage is verified.
- Simulated example labels, hidden interaction illustrations, heterogeneous-group simulation and dropout fixture are explicit. The human tximportData A/B labels are fabricated. The nested-design vignette supplies metadata, not a corresponding count experiment.
- Reading R implementation does not change tool-use classification. Only explicit IRLS/posterior/filter callback implementations and the symbolic VST derivation received mixed/tool-creation roles.
- General-purpose plotting/report frontends were not expanded into their own repositories: the main vignette gives package/API leads, while the scientific plots and result artifacts are mapped here. Package installation/support pages, bibliographic works, generic quantifier tutorials, slides and forum discussions were not recursively inventoried when they repeat operations or are outside this package-centered scientific scope.
- Source-reported timing/size and stored outputs are not execution evidence. No checksums, download eligibility, data rights or Harbor runtime have been established.

## Entry-level repository queue

All paths below are at the pinned revision. “Inspected” means source text read; “skipped” is deliberate exclusion with reason. No listed pending operation remains in the accessible core scope. Unresolved provenance leads are detailed separately.

| Entry | Status / resolving units |
|---|---|
| `.gitignore` | Skipped repository governance/configuration; no scientific operation. |
| `CODE_OF_CONDUCT.md` | Skipped repository governance/configuration; no scientific operation. |
| `CONTRIBUTING.md` | Skipped repository governance/configuration; no scientific operation. |
| `DESCRIPTION` | Inspected package version, dependency and LGPL>=3 metadata. |
| `NAMESPACE` | Inspected complete exports; public scientific API covered by Rd queue. |
| `NEWS` | Skipped historical changelog/citation metadata; pinned current interface inspected. |
| `R/AllClasses.R` | Partly inspected matrix/HTSeq/tximport constructor implementations, result/transform constructors and processTximeta. Remaining class validity/constructor preamble skipped as validation infrastructure already documented in Rd/tests. |
| `R/AllGenerics.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/RcppExports.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/core.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/expanded.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/fitNbinomGLMs.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/helper.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/lfcShrink.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/methods.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/parallel.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/plots.R` | Inspected all plot implementations and documentation; `deseq2-pca`, `deseq2-ma`, `deseq2-gene-counts`, `deseq2-dispersion-plot`, `deseq2-sparsity`. |
| `R/results.R` | Partly inspected coefficient/contrast assembly and threshold branch including LRT stop/UPSHOT formulas; remaining implementation skipped duplicate public API/test coverage. |
| `R/rlog.R` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `R/vst.R` | Partly inspected vst implementation; remaining VST implementation covered by full Rd and derivation, skipped duplicate. |
| `R/wrappers.R` | Inspected complete native-fitter wrappers, scales and NA checks; implementation context for ridge/dispersion-reference; not new tool-use units. |
| `inst/CITATION` | Skipped historical changelog/citation metadata; pinned current interface inspected. |
| `inst/extdata/pasilla_gene_counts.tsv.gz` | Skipped biological binary download; tree size/path and loading lineage inspected; pasilla record. |
| `inst/extdata/pasilla_sample_annotation.csv` | Inspected complete 7-row metadata; pasilla record. |
| `inst/script/icobra.png` | Skipped stored benchmark image; no claim that benchmark was executed or image reproduced. |
| `inst/script/icobra_benchmarks.R` | Inspected entire algos/grid/generation/evaluation/export/disabled parameter derivation; `deseq2-benchmark`. |
| `inst/script/icobra_pkg_versions.txt` | Partly inspected header and relevant method-version rows; remaining unrelated package rows skipped as environment metadata, not scientific operations. |
| `inst/script/makeSim.R` | Inspected complete generator; `deseq2-benchmark-generate`. |
| `inst/script/runScripts.R` | Inspected all ten wrapper functions (DESeq2/LRT,edgeR/robust,DSS/BH/FDR,voom,SAMseq/BH/FDR,EBSeq); six method-family units `deseq2-wrapper-deseq`, `deseq2-wrapper-edger`, `deseq2-wrapper-dss`, `deseq2-wrapper-voom`, `deseq2-wrapper-samseq`, `deseq2-wrapper-ebseq`. |
| `inst/script/testsuite.Rmd` | Inspected all function definitions, run/plot chunks and session metadata; `deseq2-airway-analysis`, `deseq2-pasilla-suite`, `deseq2-hammer`, `deseq2-bottomly`, `deseq2-parathyroid`, `deseq2-fission`. |
| `inst/script/vst.nb` | Inspected parametric definition/integral/scaling/limits/template and local-fit note; stored GraphicsBox payload not decoded; `deseq2-vst-derivation`. |
| `inst/script/vst.pdf` | Skipped duplicate rendered derivation; Mathematica source inspected. |
| `man/DESeq.Rd` | Inspected API; `deseq2-wald-workflow`, `deseq2-lrt`, `deseq2-outliers`, `deseq2-single-cell` |
| `man/DESeq2-package.Rd` | Inspected API; `deseq2-wald-workflow` |
| `man/DESeqDataSet.Rd` | Inspected API; `deseq2-matrix-import`, `deseq2-tximport`, `deseq2-tximeta`, `deseq2-htseq-import`, `deseq2-airway-import` |
| `man/DESeqResults.Rd` | Inspected API; `deseq2-contrasts`, `deseq2-inspect-fit` |
| `man/DESeqTransform.Rd` | Inspected API; `deseq2-vst`, `deseq2-rlog`, `deseq2-norm-transform` |
| `man/coef.Rd` | Inspected API; `deseq2-inspect-fit`, `deseq2-fission-profiles` |
| `man/collapseReplicates.Rd` | Inspected API; `deseq2-collapse` |
| `man/counts.Rd` | Inspected API; `deseq2-size-factors`, `deseq2-outliers`, `deseq2-inspect-fit` |
| `man/design.Rd` | Inspected API; `deseq2-multifactor`, `deseq2-rank-deficiency` |
| `man/dispersionFunction.Rd` | Inspected API; `deseq2-custom-dispersion`, `deseq2-frozen-transform` |
| `man/dispersions.Rd` | Inspected API; `deseq2-dispersions`, `deseq2-inspect-fit` |
| `man/estimateBetaPriorVar.Rd` | Inspected API; `deseq2-beta-prior` |
| `man/estimateDispersions.Rd` | Inspected API; `deseq2-dispersions` |
| `man/estimateDispersionsGeneEst.Rd` | Inspected API; `deseq2-dispersions`, `deseq2-custom-dispersion` |
| `man/estimateSizeFactors.Rd` | Inspected API; `deseq2-size-factors` |
| `man/estimateSizeFactorsForMatrix.Rd` | Inspected API; `deseq2-size-factors` |
| `man/fpkm.Rd` | Inspected API; `deseq2-fpkm` |
| `man/fpm.Rd` | Inspected API; `deseq2-fpm` |
| `man/lfcShrink.Rd` | Inspected API; `deseq2-shrink`, `deseq2-shrink-fsos` |
| `man/makeExampleDESeqDataSet.Rd` | Inspected API; `deseq2-simulate` |
| `man/nbinomLRT.Rd` | Inspected API; `deseq2-lrt` |
| `man/nbinomWaldTest.Rd` | Inspected API; `deseq2-wald-workflow`, `deseq2-t-wald` |
| `man/normTransform.Rd` | Inspected API; `deseq2-norm-transform` |
| `man/normalizationFactors.Rd` | Inspected API; `deseq2-normalization-matrix` |
| `man/normalizeGeneLength.Rd` | Inspected; normalizeGeneLength is deprecated/moved to tximport, consolidated under deseq2-tximport; no runnable new operation. |
| `man/plotCounts.Rd` | Inspected API; `deseq2-gene-counts` |
| `man/plotDispEsts.Rd` | Inspected API; `deseq2-dispersion-plot` |
| `man/plotMA.Rd` | Inspected API; `deseq2-ma` |
| `man/plotPCA.Rd` | Inspected API; `deseq2-pca` |
| `man/plotSparsity.Rd` | Inspected API; `deseq2-sparsity` |
| `man/priorInfo.Rd` | Inspected API; `deseq2-inspect-fit` |
| `man/replaceOutliers.Rd` | Inspected API; `deseq2-outliers` |
| `man/results.Rd` | Inspected API; `deseq2-contrasts`, `deseq2-threshold-tests`, `deseq2-independent-filter` |
| `man/rlog.Rd` | Inspected API; `deseq2-rlog`, `deseq2-frozen-transform` |
| `man/show.Rd` | Inspected API; `deseq2-inspect-fit` |
| `man/sizeFactors.Rd` | Inspected API; `deseq2-size-factors` |
| `man/summary.Rd` | Inspected API; `deseq2-export` |
| `man/unmix.Rd` | Inspected API; `deseq2-unmix` |
| `man/varianceStabilizingTransformation.Rd` | Inspected API; `deseq2-vst`, `deseq2-frozen-transform` |
| `man/vst.Rd` | Inspected API; `deseq2-vst` |
| `src/DESeq2.cpp` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `src/Makevars` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `src/Makevars.win` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `src/RcppExports.cpp` | Skipped internal implementation/generated bindings/build files: scientific operations covered by exported APIs and inspected tests; no requested underlying-tool modification. |
| `tests/testthat.R` | Skipped test runner harness; not scientific operation. |
| `tests/testthat/test_DESeq.R` | Inspected; `deseq2-wald-workflow`, `deseq2-rank-deficiency`; repeated variants/error checks consolidated. |
| `tests/testthat/test_LRT.R` | Inspected; `deseq2-lrt`, `deseq2-single-cell`; repeated variants/error checks consolidated. |
| `tests/testthat/test_QR.R` | Inspected; `deseq2-wald-workflow`; repeated variants/error checks consolidated. |
| `tests/testthat/test_addMLE.R` | Inspected; `deseq2-beta-prior`, `deseq2-inspect-fit`; repeated variants/error checks consolidated. |
| `tests/testthat/test_betaFitting.R` | Inspected; `deseq2-ridge-reference`; repeated variants/error checks consolidated. |
| `tests/testthat/test_collapse.R` | Inspected; `deseq2-collapse`; repeated variants/error checks consolidated. |
| `tests/testthat/test_construction_errors.R` | Inspected; `deseq2-matrix-import`, `deseq2-rank-deficiency`; repeated variants/error checks consolidated. |
| `tests/testthat/test_design_matrix.R` | Inspected; `deseq2-rank-deficiency`, `deseq2-contrasts`, `deseq2-shrink`; repeated variants/error checks consolidated. |
| `tests/testthat/test_dispersions.R` | Inspected; `deseq2-dispersions`, `deseq2-custom-dispersion`, `deseq2-dispersion-reference`; repeated variants/error checks consolidated. |
| `tests/testthat/test_edge_case.R` | Inspected; `deseq2-wald-workflow`, `deseq2-inspect-fit`; repeated variants/error checks consolidated. |
| `tests/testthat/test_factors.R` | Inspected; `deseq2-rank-deficiency`; repeated variants/error checks consolidated. |
| `tests/testthat/test_fpkm.R` | Inspected; `deseq2-fpkm`, `deseq2-fpm`; repeated variants/error checks consolidated. |
| `tests/testthat/test_interactions.R` | Inspected; `deseq2-interactions`, `deseq2-shrink`; repeated variants/error checks consolidated. |
| `tests/testthat/test_lfcShrink.R` | Inspected; `deseq2-shrink`, `deseq2-shrink-fsos`; repeated variants/error checks consolidated. |
| `tests/testthat/test_linear_mu.R` | Inspected; `deseq2-dispersions`; repeated variants/error checks consolidated. |
| `tests/testthat/test_methods.R` | Inspected; `deseq2-size-factors`, `deseq2-outliers`; repeated variants/error checks consolidated. |
| `tests/testthat/test_model_matrix.R` | Inspected; `deseq2-rank-deficiency`, `deseq2-lrt`; repeated variants/error checks consolidated. |
| `tests/testthat/test_nbinomWald.R` | Inspected; `deseq2-t-wald`, `deseq2-beta-prior`; repeated variants/error checks consolidated. |
| `tests/testthat/test_optim.R` | Inspected; `deseq2-wald-workflow`; repeated variants/error checks consolidated. |
| `tests/testthat/test_outlier.R` | Inspected; `deseq2-outliers`; repeated variants/error checks consolidated. |
| `tests/testthat/test_parallel.R` | Inspected; `deseq2-dispersions`, `deseq2-wald-workflow`; repeated variants/error checks consolidated. |
| `tests/testthat/test_plots.R` | Inspected; `deseq2-pca`, `deseq2-ma`, `deseq2-gene-counts`, `deseq2-sparsity`; repeated variants/error checks consolidated. |
| `tests/testthat/test_results.R` | Inspected; `deseq2-contrasts`, `deseq2-threshold-tests`, `deseq2-custom-filter`, `deseq2-export`; repeated variants/error checks consolidated. |
| `tests/testthat/test_size_factor.R` | Inspected; `deseq2-size-factors`; repeated variants/error checks consolidated. |
| `tests/testthat/test_txi.R` | Inspected; `deseq2-tximport`, `deseq2-linked-txome`, `deseq2-abundance-counts`; repeated variants/error checks consolidated. |
| `tests/testthat/test_unmix.R` | Inspected; `deseq2-unmix`; repeated variants/error checks consolidated. |
| `tests/testthat/test_weights.R` | Inspected; `deseq2-weighted-fit`; repeated variants/error checks consolidated. |
| `tests/testthat/test_zero_zero.R` | Inspected; `deseq2-contrasts`; repeated variants/error checks consolidated. |
| `vignettes/DESeq2.Rmd` | Inspected all scientific sections/setup/FAQ; section queue below. Initial long output truncation recovered by bounded source segments. |
| `vignettes/library.bib` | Skipped bibliography; paper-specific methods/data not required for documented API inventory. |

## Main vignette section map

Every scientific heading below was inspected in source with upstream chunks. Container headings are grouped with children. Non-scientific help/acknowledgment/funding/install/session/reference sections were read in the main source and skipped as task units. Code comments beginning with # are not Markdown headings.

| Heading / sections | Status / units |
|---|---|
| Quick start; Why un-normalized counts?; The DESeqDataSet | Inspected; `deseq2-matrix-import`, `deseq2-wald-workflow` |
| Transcript abundance files and tximport / tximeta | Inspected; `deseq2-tximport`, `deseq2-abundance-counts` |
| Tximeta for import with automatic metadata | Inspected; `deseq2-tximeta` |
| Count matrix input | Inspected; `deseq2-matrix-import` |
| htseq-count input | Inspected; `deseq2-htseq-import` |
| SummarizedExperiment input | Inspected; `deseq2-airway-import` |
| Pre-filtering | Inspected; `deseq2-prefilter` |
| Note on factor levels | Inspected; `deseq2-matrix-import`, `deseq2-contrasts` |
| Collapsing technical replicates | Inspected; `deseq2-collapse` |
| About the pasilla dataset | Inspected; `deseq2-wald-workflow` |
| Differential expression analysis | Inspected; `deseq2-wald-workflow` |
| Log fold change shrinkage; Alternative shrinkage estimators; Extended section on shrinkage estimators | Inspected; `deseq2-shrink`, `deseq2-shrink-fsos` |
| Speed-up and parallelization thoughts | Inspected; `deseq2-wald-workflow`, `deseq2-single-cell` |
| p-values and adjusted p-values; More information on results columns | Inspected; `deseq2-contrasts`, `deseq2-independent-filter`, `deseq2-export` |
| Independent hypothesis weighting | Inspected; `deseq2-ihw` |
| MA-plot | Inspected; `deseq2-ma` |
| Plot counts | Inspected; `deseq2-gene-counts` |
| Exporting results to CSV files | Inspected; `deseq2-export` |
| Multi-factor designs | Inspected; `deseq2-multifactor` |
| Count data transformations; Blind dispersion estimation; Extracting transformed values; Variance stabilizing transformation | Inspected; `deseq2-vst`, `deseq2-rlog`, `deseq2-frozen-transform` |
| Regularized log transformation | Inspected; `deseq2-rlog` |
| Effects of transformations on the variance | Inspected; `deseq2-variance-diagnostics`, `deseq2-norm-transform` |
| Heatmap of the count matrix | Inspected; `deseq2-gene-heatmap` |
| Heatmap of the sample-to-sample distances | Inspected; `deseq2-sample-distances` |
| Principal component plot of the samples | Inspected; `deseq2-pca` |
| Wald test individual steps | Inspected; `deseq2-size-factors`, `deseq2-dispersions`, `deseq2-wald-workflow` |
| Control features for estimating size factors | Inspected; `deseq2-size-factors` |
| Contrasts (workflow and theory) | Inspected; `deseq2-contrasts` |
| Interactions | Inspected; `deseq2-interactions` |
| Time-series experiments | Inspected; `deseq2-time-series` |
| Likelihood ratio test | Inspected; `deseq2-lrt` |
| Recommendations for single-cell analysis | Inspected; `deseq2-single-cell`, `deseq2-zinbwave` |
| Approach to count outliers; Count outlier detection | Inspected; `deseq2-outliers` |
| Dispersion plot and fitting alternatives; Local or mean dispersion fit; Supply a custom dispersion fit | Inspected; `deseq2-dispersion-plot`, `deseq2-dispersions`, `deseq2-custom-dispersion` |
| Independent filtering of results | Inspected; `deseq2-independent-filter` |
| Tests of log2 fold change above or below a threshold | Inspected; `deseq2-threshold-tests` |
| Access to all calculated values | Inspected; `deseq2-inspect-fit` |
| Sample-/gene-dependent normalization factors | Inspected; `deseq2-normalization-matrix` |
| Model matrix not full rank; Linear combinations; Levels without samples | Inspected; `deseq2-rank-deficiency` |
| Group-specific condition effects, individuals nested within groups | Inspected; `deseq2-nested-design` |
| The DESeq2 model; Changes compared to DESeq; Methods changes since2014; Expanded model matrices | Inspected; `deseq2-wald-workflow`, `deseq2-dispersions`, `deseq2-beta-prior`, `deseq2-shrink` |
| Independent filtering and multiple testing; Filtering criteria; Why does it work? | Inspected; `deseq2-independent-filter`, `deseq2-filter-simulation` |
| FAQ: NA p-values; unfiltered results; VST/rlog for differential testing | Inspected; `deseq2-contrasts`, `deseq2-outliers`, `deseq2-vst` |
| FAQ: batches in PCA | Inspected; `deseq2-batch-visualization` |
| FAQ: normalized counts and design variables | Inspected; `deseq2-size-factors`, `deseq2-normalization-matrix` |
| FAQ: paired samples | Inspected; `deseq2-nested-design`, `deseq2-airway-analysis` |
| FAQ: multiple groups together or pairs | Inspected; `deseq2-heterogeneous-groups` |
| FAQ: contrast many groups; no replicates | Inspected; `deseq2-contrasts`, `deseq2-wald-workflow` |
| FAQ: continuous covariate | Inspected; `deseq2-continuous-design` |
| FAQ: LRT gives one comparison; exact DESeq steps | Inspected; `deseq2-lrt`, `deseq2-wald-workflow` |
| FAQ: benchmarking other tools | Inspected; `deseq2-benchmark` |
| Rich visualization/reporting: regionReport, Glimma, pcaExplorer, iSEE/iSEEde, DEvis | Inspected package/API leads in vignette; skipped recursive frontend inventories as ancillary presentation/reporting. Scientific result/plot operations already mapped. |
| FAQ official Galaxy tool | Inspected link to galaxyproject/tools-iuc tools/deseq2 and Tool Shed revision d983d19fbbab; skipped platform wrapper expansion, no distinct scientific analysis established. |

## External bounded collections and revisions

### thelovelab/rnaseqGene at 0d7e27dde3ca9875cf94770ed9de35e346dd3676

Tree not truncated. Source collection:

- `.gitignore`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `DESCRIPTION`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `inst/CITATION`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `vignettes/bibliography.bib`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `vignettes/rnaseqGene.Rmd`: inspected completely; scientific sections listed below.
### tavareshugo/tutorial_DESeq2_contrasts at 1b3db8d307ac7a336dd49233d207a0948aa1563e

Tree not truncated. Source collection:

- `.gitignore`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `README.md`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `DESeq2_contrasts.Rproj`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `DESeq2_contrasts.Rmd`: inspected completely; scientific sections listed below.
- `DESeq2_contrasts.md`: skipped duplicate render; source read.
### mikelove/zinbwave-deseq2 at 6c843b89cd917050eb8d7a07c94e37663af2e55f

Tree not truncated. Source collection:

- `.gitignore`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `README.md`: skipped project/manifest/citation/orientation metadata; Rmd setup was sufficient for unit state.
- `zinbwave-deseq2.Rmd`: inspected completely; scientific sections listed below.
- `zinbwave-deseq2.knit.md`: skipped duplicate render; source read.
- `zinbwave-deseq2_files/figure-html/*.png`: skipped stored plot artifacts; not execution evidence.

External scientific section queue (inspected, deduplicated where repeated):

- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Quantifying with Salmon: inspected → `deseq2-airway-salmon`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Reading in data with tximeta; SummarizedExperiment; loadfullgse: inspected → `deseq2-airway-tximeta`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, DESeqDataSet ...; makedds; Pre-filtering; Differential expression analysis; Plotting results: inspected → `deseq2-airway-full-analysis`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Sample distances; poisdistheatmap: inspected → `deseq2-poisson-distance`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, PCA plot using Generalized PCA; glmpca: inspected → `deseq2-glm-pca`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, MDS plot; mdsvst,mdspois: inspected → `deseq2-mds`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Gene clustering; genescluster: inspected → `deseq2-variable-gene-cluster`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Annotating and exporting results: inspected → `deseq2-annotate`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Plotting fold changes in genomic space; gvizplot: inspected → `deseq2-genomic-results`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Removing hidden batch effects; Using SVA with DESeq2; svaplot: inspected → `deseq2-sva`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Using RUV with DESeq2; ruvplot: inspected → `deseq2-ruv`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Time course experiments; fissionDE,fissioncounts,fissionheatmap: inspected → `deseq2-fission-profiles`.
- https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd, Appendix; Updated details on quantification; Building the salmon index; Producing ... tximeta: inspected → `deseq2-airway-requantify`.
- https://github.com/tavareshugo/tutorial_DESeq2_contrasts/blob/1b3db8d307ac7a336dd49233d207a0948aa1563e/DESeq2_contrasts.Rmd, One factor, two levels; Extra: recoding the design: inspected → `deseq2-numeric-two`.
- https://github.com/tavareshugo/tutorial_DESeq2_contrasts/blob/1b3db8d307ac7a336dd49233d207a0948aa1563e/DESeq2_contrasts.Rmd, One factor, three levels; Extra: why not define a new group ...: inspected → `deseq2-numeric-pooled`.
- https://github.com/tavareshugo/tutorial_DESeq2_contrasts/blob/1b3db8d307ac7a336dd49233d207a0948aa1563e/DESeq2_contrasts.Rmd, Two factors with interaction: inspected → `deseq2-numeric-interaction`.
- https://github.com/tavareshugo/tutorial_DESeq2_contrasts/blob/1b3db8d307ac7a336dd49233d207a0948aa1563e/DESeq2_contrasts.Rmd, Three factors, with nesting; Extra: imbalanced design: inspected → `deseq2-numeric-nesting`.
- https://github.com/mikelove/zinbwave-deseq2/blob/6c843b89cd917050eb8d7a07c94e37663af2e55f/zinbwave-deseq2.Rmd, Simulate ...; Model zero component ...; Estimate size factors; Estimate dispersion and DE; Evaluate: inspected → `deseq2-zinbwave`.
- https://github.com/mikelove/zinbwave-deseq2/blob/6c843b89cd917050eb8d7a07c94e37663af2e55f/zinbwave-deseq2.Rmd, Plot dispersion estimates; plotDispEsts2 and subsequent eval=FALSE: inspected → `deseq2-zinbwave-trend`.

Remaining rnaseqGene sections (Introduction/Experimental data, DESeq2 import functions, SE/container diagrams, branching point, design construction, transformations, VST distances/PCA, results/BH, counts/MA/filtering, IHW, exporting, session info/references) inspected and consolidated into airway provenance, deseq2-airway-full-analysis, and component units. No separate task unit for diagrams/install advice. The Appendix is inspected through final tximeta warnings/session section; current v49 log is kept separate from supplied v29 gse.

## Data provenance queue and access outcomes

Mutable public metadata retrieved 2026-09-30, with no Git commit pin. Package metadata versions are evidence about those pages only and do not pin datasets consumed by old examples.

- pasilla release page: inspected package identity/PMID/GEO range/version/license; complete pinned seven-sample CSV read. Payload skipped. Six reported GEO accessions do not by themselves explain seven packaged libraries.
- airway release page: inspected study identity/version/license; detailed paired design and Salmon lineage recovered from pinned rnaseqGene source.
- tximportData release page and rendered data vignette: inspected GEUVADIS sample table, reference and Salmon sections; other quantifier/alevin/oarfish entries read as upstream/context leads, skipped separate operations as not consumed by mapped DESeq2 examples. Distinct unrelated fixtures not assigned to human GEUVADIS. Fly SRR1197474 reference mismatch recorded.
- fission release page and rendered provenance vignette: inspected study/source constructor, time/strain metadata and PomBase release; count/annotation payload skipped.
- legacy ReCount landing page: inspected Hammer/Bottomly study rows, ExpressionSet loader convention, shared preprocessing and Ensembl61 reference identification. Other study rows skipped: no consuming DESeq2 workflow in this collection.
- parathyroidSE: current experiment release page returned404; generic package URL cache miss; BioC3.18 metadata URL inaccessible through web tool. Named product retained with unknown accession/species/sample counts. No binary retrieval attempted.
- GitHub source discovery attempts for thelovelab/tximportData and Bioconductor/{airway,pasilla,parathyroidSE,fission} each returned404 for commits/tree. No replacement revision guessed; official linked Bioconductor metadata used.
- code.bioconductor.org/browse/tximportData/ inaccessible via web tool; rendered official vignette used. Its unversioned text differs from pinned DESeq2 fly test, so no unsupported resolution asserted.
- Bottomly meanDispPairs_bottomly.rda and bottomly_sumexp.RData absent from complete pinned tree; script consumer and disabled derivation inspected. This is an artifact availability limitation, not evidence the study lacks data.
- GENCODE/PomBase/Ensembl/org.Hs.eg.db records rely on exact references in inspected loading/processing sources. Their large reference assets were not retrieved.

## Source checks and caveats for authoring

- Pinned man/results.Rd says lfcThreshold overwrites LRT p-values; the same revision's R/results.R threshold branch stops for test=LRT, with test_results expecting error. This is a same-entity/same-version interface inconsistency. Choose explicit Wald tests; source inspection did not run either path.
- Main vignette threshold table lists four alternatives, while pinned API also has greaterAbsUPSHOT and greaterAbs2014. UPSHOT has a unimodality assumption and t-distribution restriction in tests.
- testsuite uses addMLE=TRUE with default unshrunk fitting and expects lfcMLE for shrinkage summaries. Modern tests reject that combination; do not present suite as currently executable unchanged.
- plotCounts returnData returns count+pseudocount, not logarithmic values. Fission suite's “log2 count” label is unsupported by current implementation.
- fpkm GR4 source coordinates yield500bp despite a comment naming2.5Kb. This discrepancy concerns only the constructed fixture, not real genes.
- rnaseqGene genomic-track code thresholds padj<.05 while caption says.1. RUV narrative proposes unadjusted~dex results but source carries adjusted~cell+dex state; SVA/RUV new designs are prepared without final DESeq calls.
- Numeric-contrast tutorial explicitly withdraws universality for partially crossed imbalanced designs. Simulated labels and its altered weights are not real biological discoveries.
- Simulated fixtures and benchmark comparisons require fixed seeds/software and scientifically justified scoring choices. Benchmark EBSeq LFC is substituted from edgeR; placeholders and native/BH FDR conventions differ.
- General plot/fit invariance tests were read as source evidence, not rerun. The source test suite is not a grading oracle automatically.

## Stopping and continuation

Stopping reason: mapped accessible package-centered scientific scope is covered: exported APIs, all scientific main-vignette sections, all analysis scripts/notebooks, all test source files, and the three directly linked scientific workflow/tutorial sources have been inspected or explicitly deduplicated. Remaining internal implementation, duplicate renders, governance/history and ancillary frontend/platform links do not add a distinct inspected scientific operation to this scope. No artificial unit quota or elapsed-time cutoff was used.

Specific unresolved continuation leads for later data preparation: recover an archived parathyroidSE source/release and study metadata; obtain the benchmark's empirical parameter/SE assets and prove their lineage; reconcile tximportData salmon_dm release98 versus rendered release92 description; verify exact package data versions/count dimensions, reference hashes and reuse terms. These are missing access/provenance/artifact facts, not pending uninspected core operations. Scientific execution, real-data suitability and resource/Harbor/grader validation are intentionally later work.

Final verification: both JSONL files parsed; required fields, unique IDs, allowed role values, dataset references and related-unit references passed. All final units were reviewed against inspected setup/processing/semantics after boundary decisions. Independent reference products and observation families remain separate. Source map statuses above reflect final IDs; no obsolete checkpoint IDs or pending already-inspected section remains.


Observed completion timestamp: 2026-09-30 17:37:37 UTC; approximately 29 minutes 29 seconds elapsed since the first measured timestamp. This duration was observed, not used as a stopping rule. Lightweight closing snapshot: load1=0.16, MemAvailable about4.19GiB at17:31:50. No substantial workers were started or left running.
