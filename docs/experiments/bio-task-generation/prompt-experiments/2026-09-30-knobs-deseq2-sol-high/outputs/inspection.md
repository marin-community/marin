# DESeq2 source inventory inspection

Pinned DESeq2 revision: `c62c60c6ff83fd84ce115cacd1c49827533f85a7`, DESCRIPTION version 1.53.5, LGPL >=3. Full tree response was not truncated. Independently pinned rnaseqGene: `0d7e27dde3ca9875cf94770ed9de35e346dd3676`, version 1.35.3, Artistic-2.0. Mutable official Bioconductor metadata was retrieved 2026-09-30 and is explicitly separate from pinned dependency versions.

The launch initially imposed deadline 15:38:51.591962 UTC. Before source inspection the runner corrected it to 15:39:50.627096 UTC. The runner subsequently withdrew the artificial cutoff mid-run, as the user rejected per-run time cutoffs. No replacement time or unit quota was used. This is a transitional pilot, not a clean model-comparison cell.

Only the supplied launch message, resolved prompt, source-access instructions and discovered external sources were read. No parent documentation, sibling experiments, review criteria, prior outputs or memory were inspected. No scientific computation, package installation/build, biological count/reference download, Harbor validation, model call or cloud job was performed. Small text source and sample-annotation previews were inspected. Initial node load was 0.13 with approximately 4.3 GiB available; final check was 0.10 with approximately 4.2 GiB available. No heavy workers were started or left running.

## Pinned repository entry map

Status is source-inspection status, not runtime validation. Uninspected implementation files are not evidence that behavior is absent.

| Entry | Status and final unit links |
|---|---|
| `.gitignore` | skipped: repository support/configuration or reporting artifact, no separate scientific task operation assigned |
| `CODE_OF_CONDUCT.md` | skipped: repository support/configuration or reporting artifact, no separate scientific task operation assigned |
| `CONTRIBUTING.md` | skipped: repository support/configuration or reporting artifact, no separate scientific task operation assigned |
| `DESCRIPTION` | inspected: package identity, dependencies and exported API orientation; all DESeq2 units |
| `NAMESPACE` | inspected: package identity, dependencies and exported API orientation; all DESeq2 units |
| `NEWS` | partly inspected: first 110 source lines; recent 1.45–1.53 changes, single-cell/glmGamPoi count mode. Older history not expanded because current operation interfaces are covered. |
| `R/AllClasses.R` | inspected: `deseq2-matrix-import`, `deseq2-tximport-import`, `deseq2-tximeta-import`, `deseq2-htseq-import` |
| `R/AllGenerics.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/RcppExports.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/core.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/expanded.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/fitNbinomGLMs.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/helper.R` | inspected: `deseq2-direct-outlier-replacement`, `deseq2-unmix`, `deseq2-fpm`, `deseq2-fpkm` |
| `R/lfcShrink.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/methods.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/parallel.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/plots.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/results.R` | inspected: `deseq2-airway-analysis`, `deseq2-hammer-analysis`, `deseq2-bottomly-analysis`, `deseq2-parathyroid-analysis` |
| `R/rlog.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `R/vst.R` | inspected: `deseq2-transform` |
| `R/wrappers.R` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `inst/CITATION` | inspected: canonical scientific citations; package context, no separate operation |
| `inst/extdata/pasilla_gene_counts.tsv.gz` | skipped: biological asset download prohibited; identity from vignette and tree metadata only |
| `inst/extdata/pasilla_sample_annotation.csv` | inspected: `deseq2-matrix-import`, `deseq2-multi-factor` |
| `inst/script/icobra.png` | skipped: rendered benchmark image is not execution evidence |
| `inst/script/icobra_benchmarks.R` | inspected: `deseq2-icobra-benchmark` |
| `inst/script/icobra_pkg_versions.txt` | skipped: repository support/configuration or reporting artifact, no separate scientific task operation assigned |
| `inst/script/makeSim.R` | inspected: `deseq2-icobra-benchmark` |
| `inst/script/runScripts.R` | inspected: `deseq2-icobra-benchmark` |
| `inst/script/testsuite.Rmd` | inspected: `deseq2-airway-analysis`, `deseq2-hammer-analysis`, `deseq2-bottomly-analysis`, `deseq2-parathyroid-analysis`, `deseq2-fission-analysis` |
| `inst/script/vst.nb` | skipped: derivation lead; documented VST operation covered without symbolic derivation |
| `inst/script/vst.pdf` | skipped: derivation lead; documented VST operation covered without symbolic derivation |
| `man/DESeq.Rd` | inspected: `deseq2-wald-analysis`, `deseq2-lrt`, `deseq2-single-cell`, `deseq2-outliers` |
| `man/DESeq2-package.Rd` | inspected: package identity, dependencies and exported API orientation; all DESeq2 units |
| `man/DESeqDataSet.Rd` | inspected: `deseq2-matrix-import`, `deseq2-tximport-import`, `deseq2-airway-import` |
| `man/DESeqResults.Rd` | inspected: `deseq2-wald-analysis` |
| `man/DESeqTransform.Rd` | inspected: `deseq2-transform` |
| `man/coef.Rd` | inspected: `deseq2-fission-profiles` |
| `man/collapseReplicates.Rd` | inspected: `deseq2-collapse-replicates` |
| `man/counts.Rd` | inspected: `deseq2-matrix-import`, `deseq2-fpm` |
| `man/design.Rd` | inspected: `deseq2-multi-factor`, `deseq2-rank-repair` |
| `man/dispersionFunction.Rd` | inspected: `deseq2-dispersion-fit`, `deseq2-frozen-transform` |
| `man/dispersions.Rd` | inspected: `deseq2-dispersion-fit` |
| `man/estimateBetaPriorVar.Rd` | inspected: `deseq2-lfc-shrink` |
| `man/estimateDispersions.Rd` | inspected: `deseq2-dispersion-fit` |
| `man/estimateDispersionsGeneEst.Rd` | inspected: `deseq2-dispersion-fit` |
| `man/estimateSizeFactors.Rd` | inspected: `deseq2-size-factors` |
| `man/estimateSizeFactorsForMatrix.Rd` | inspected: `deseq2-size-factors` |
| `man/fpkm.Rd` | inspected: `deseq2-fpkm` |
| `man/fpm.Rd` | inspected: `deseq2-fpm` |
| `man/lfcShrink.Rd` | inspected: `deseq2-lfc-shrink` |
| `man/makeExampleDESeqDataSet.Rd` | inspected: `deseq2-simulate` |
| `man/nbinomLRT.Rd` | inspected: `deseq2-lrt` |
| `man/nbinomWaldTest.Rd` | inspected: `deseq2-wald-analysis`, `deseq2-effect-threshold` |
| `man/normTransform.Rd` | inspected: `deseq2-transform` |
| `man/normalizationFactors.Rd` | inspected: `deseq2-normalization-matrix` |
| `man/normalizeGeneLength.Rd` | inspected: `deseq2-tximport-import`, `deseq2-normalization-matrix` |
| `man/plotCounts.Rd` | inspected: `deseq2-gene-count-plot` |
| `man/plotDispEsts.Rd` | inspected: `deseq2-dispersion-fit` |
| `man/plotMA.Rd` | inspected: `deseq2-lfc-shrink` |
| `man/plotPCA.Rd` | inspected: `deseq2-sample-qc` |
| `man/plotSparsity.Rd` | inspected: `deseq2-sparsity-diagnostic` |
| `man/priorInfo.Rd` | inspected: `deseq2-lfc-shrink` |
| `man/replaceOutliers.Rd` | inspected: `deseq2-direct-outlier-replacement` |
| `man/results.Rd` | inspected: `deseq2-independent-filter`, `deseq2-effect-threshold` |
| `man/rlog.Rd` | inspected: `deseq2-transform`, `deseq2-frozen-transform` |
| `man/show.Rd` | inspected: object display support, no separate scientific unit |
| `man/sizeFactors.Rd` | inspected: `deseq2-size-factors` |
| `man/summary.Rd` | inspected: `deseq2-wald-analysis` |
| `man/unmix.Rd` | inspected: `deseq2-unmix` |
| `man/varianceStabilizingTransformation.Rd` | inspected: `deseq2-transform`, `deseq2-frozen-transform` |
| `man/vst.Rd` | inspected: `deseq2-transform` |
| `src/DESeq2.cpp` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `src/Makevars` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `src/Makevars.win` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `src/RcppExports.cpp` | skipped: uninspected implementation detail; public scientific API already mapped. No inference of absent behavior. |
| `tests/testthat.R` | skipped: repository support/configuration or reporting artifact, no separate scientific task operation assigned |
| `tests/testthat/test_DESeq.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_LRT.R` | inspected: `deseq2-lrt` |
| `tests/testthat/test_QR.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_addMLE.R` | inspected: `deseq2-lfc-shrink` |
| `tests/testthat/test_betaFitting.R` | inspected: `deseq2-coefficient-oracle` |
| `tests/testthat/test_collapse.R` | inspected: `deseq2-collapse-replicates` |
| `tests/testthat/test_construction_errors.R` | inspected: `deseq2-matrix-import` |
| `tests/testthat/test_design_matrix.R` | inspected: `deseq2-rank-repair`, `deseq2-nested-design` |
| `tests/testthat/test_dispersions.R` | inspected: `deseq2-dispersion-oracle` |
| `tests/testthat/test_edge_case.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_factors.R` | inspected: `deseq2-multi-factor` |
| `tests/testthat/test_fpkm.R` | inspected: `deseq2-fpkm`, `deseq2-fpm` |
| `tests/testthat/test_interactions.R` | inspected: `deseq2-contrast-interaction` |
| `tests/testthat/test_lfcShrink.R` | inspected: `deseq2-lfc-shrink` |
| `tests/testthat/test_linear_mu.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_methods.R` | inspected: `deseq2-wald-analysis`, `deseq2-transform` |
| `tests/testthat/test_model_matrix.R` | inspected: `deseq2-rank-repair` |
| `tests/testthat/test_nbinomWald.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_optim.R` | inspected: `deseq2-coefficient-oracle` |
| `tests/testthat/test_outlier.R` | inspected: `deseq2-outliers` |
| `tests/testthat/test_parallel.R` | inspected: `deseq2-wald-analysis` |
| `tests/testthat/test_plots.R` | inspected: `deseq2-sample-qc` |
| `tests/testthat/test_results.R` | inspected: `deseq2-custom-filter`, `deseq2-independent-filter`, `deseq2-contrast-interaction`, `deseq2-effect-threshold` |
| `tests/testthat/test_size_factor.R` | inspected: `deseq2-size-factors` |
| `tests/testthat/test_txi.R` | inspected: `deseq2-linked-transcriptome` |
| `tests/testthat/test_unmix.R` | inspected: `deseq2-unmix` |
| `tests/testthat/test_weights.R` | inspected: `deseq2-observation-weights` |
| `tests/testthat/test_zero_zero.R` | inspected: `deseq2-wald-analysis` |
| `vignettes/DESeq2.Rmd` | inspected: `deseq2-matrix-import`, `deseq2-tximport-import`, `deseq2-tximeta-import`, `deseq2-htseq-import`, `deseq2-airway-import`, `deseq2-prefilter`, `deseq2-wald-analysis`, `deseq2-lfc-shrink`, `deseq2-multi-factor`, `deseq2-transform`, `deseq2-sample-qc`, `deseq2-gene-count-plot`, `deseq2-independent-filter`, `deseq2-ihw`, `deseq2-contrast-interaction`, `deseq2-lrt`, `deseq2-single-cell`, `deseq2-outliers`, `deseq2-dispersion-fit`, `deseq2-effect-threshold`, `deseq2-normalization-matrix`, `deseq2-nested-design`, `deseq2-rank-repair`, `deseq2-shrink-threshold`, `deseq2-batch-visualization`, `deseq2-export-report` |
| `vignettes/library.bib` | skipped: bibliography metadata repeats method references; no separate tool-use operation |

## Main vignette operation map

Operational sections and worked state were inspected in bounded chunks; each locator below maps to final records. Quick-start, introductory prerequisites, help, acknowledgments, funding, installation, session information and citations add no separate scientific use. Theory/model/changes sections were partly inspected as method corroboration rather than independently assigned operations. Generic FAQ repeats are folded into the relevant records: replicate requirements into Wald analysis, paired designs into multi-factor analysis, continuous covariates/group contrasts into contrasts, transformed-input and normalized-count caveats into transform/normalization, joint-versus-displayed LRT hypotheses into LRT, and Galaxy/benchmark leads into the external/script map.

| Final unit | Vignette section/chunk |
|---|---|
| `deseq2-matrix-import` | Count matrix input; loadPasilla, reorderPasila, matrixInput, addFeatureData |
| `deseq2-tximport-import` | Transcript abundance files and tximport / tximeta; txiSetup, txiFiles, tximport, txi2dds |
| `deseq2-tximeta-import` | Tximeta for import with automatic metadata; coldata/files/names setup; hidden and eval=FALSE chunks |
| `deseq2-htseq-import` | htseq-count input; htseqDirII (eval=FALSE) |
| `deseq2-airway-import` | SummarizedExperiment input; loadSumExp, sumExpInput |
| `deseq2-prefilter` | Pre-filtering; prefilter; Note on factor levels; factorlvl, relevel, droplevels |
| `deseq2-wald-analysis` | Differential expression analysis; deseq; Wald test individual steps |
| `deseq2-lfc-shrink` | Log fold change shrinkage for visualization and ranking; Alternative shrinkage estimators; man/lfcShrink.Rd usage, value and details |
| `deseq2-multi-factor` | Multi-factor designs; copyMultifactor, fixLevels, replaceDesign, multiResults, multiTypeResults |
| `deseq2-transform` | Count data transformations; Blind dispersion estimation; rlogAndVST; Effects of transformations on the variance |
| `deseq2-sample-qc` | Data quality assessment by sample clustering and visualization; heatmap, sampleClust, figHeatmapSamples, figPCA, figPCA2 |
| `deseq2-gene-count-plot` | Plot counts; plotCounts, plotCountsAdv |
| `deseq2-independent-filter` | p-values and adjusted p-values; Independent filtering of results; filtByMean/noFilt |
| `deseq2-ihw` | Independent hypothesis weighting; IHW eval=FALSE chunk |
| `deseq2-contrast-interaction` | Contrasts; Interactions; combineFactors, interFig, interFig2 |
| `deseq2-lrt` | Likelihood ratio test; simpleLRT, simpleLRT2; Time-series experiments; man/nbinomLRT.Rd usage/value/details |
| `deseq2-single-cell` | Recommendations for single-cell analysis |
| `deseq2-outliers` | Approach to count outliers; boxplotCooks |
| `deseq2-dispersion-fit` | Dispersion plot and fitting alternatives; dispFit, dispFitCustom |
| `deseq2-effect-threshold` | Tests of log2 fold change above or below a threshold; lfcThresh |
| `deseq2-normalization-matrix` | Sample-/gene-dependent normalization factors; normFactors, offsetTransform eval=FALSE |
| `deseq2-nested-design` | Model matrix not full rank; Group-specific condition effects, individuals nested within groups; groupeffect1-4 |
| `deseq2-rank-repair` | Model matrix not full rank; Linear combinations; Levels without samples |
| `deseq2-shrink-threshold` | Extended section on shrinkage estimators; apeThresh, ashThresh |
| `deseq2-batch-visualization` | FAQ Why after VST are there still batches in the PCA plot? eval=FALSE recipe |
| `deseq2-export-report` | Exploring and exporting results; Rich visualization and reporting; Exporting results to CSV files |

## External source and lead map

| Source | Status |
|---|---|
| rnaseqGene `0d7e27dde3ca9875cf94770ed9de35e346dd3676`: DESCRIPTION, CITATION, vignette | Inspected import/quantification, object-state branching, transformations/ordination, DE/results/plots, annotation/export, SVA/RUV, fission time-course and updated quantification appendix. Duplicate prefilter/DE/shrink/filter/CSV APIs remain mapped to existing DESeq2 operations. Biological stages and source-specific state differences are retained in distinct records below. Bibliography not expanded. |
| tximportData official package/vignette | Inspected GEUVADIS sample table, quantifier/reference version and example assets; Drosophila single-run Salmon fixture separately identified. Other quantifier/example datasets are context rather than inputs to selected units; not expanded. |
| pasilla official package/create_objects vignette | Inspected study metadata, alignments and exon-bin creation. Ensembl62 reference recorded. Gene TSV production remains unresolved; no inferred equivalence with exon counts. |
| airway official package/vignette | Inspected packaged object summary, eight-sample counts, Ensembl75 row ranges and counting recipe. Mutable reconstruction preview discrepancy is scoped to that preview. |
| fission official package page | Inspected stress time-course provenance and named strain metadata; source analysis in testsuite and rnaseqGene. |
| org.Hs.eg.db official package page | Inspected current version 3.23.1, Artistic-2.0 and AnnotationDbi dependency; does not pin the older workflow's database contents. |
| parathyroidSE official package paths | Release URL HTTP404; generic package URL cache miss and historical3.18 path unavailable. Package source accession/version/terms remain unresolved; this does not establish package absence. |
| Numeric-contrast tutorial `1b3db8d307ac7a336dd49233d207a0948aa1563e` | README and full scientific examples inspected; partially crossed/imbalanced warning and attribution requirement retained. Synthetic labels are not study observations. |
| zinbwave-deseq2 `6c843b89cd917050eb8d7a07c94e37663af2e55f` | Scientific workflow inspected: splatter/dropout simulation, weights, scran factors, LRT, evaluation and optional trend-transfer recipe. All observations generated. |
| Galaxy tools-iuc DESeq2 path `105182a2fdcdf7e912bb20eb59217cd94bcefa1d` | Tree oriented; get_deseq_dataset.R read, deseq2.R import/design/contrast/DE/export paths inspected; XML/macros/test-data not read. Count/HTSeq/tximport import, prefilter, batch-aware design, DE, shrinkage, all-pair contrasts and export repeat recorded scientific purposes, so no separate wrapper unit. Important wrapper state: shrinkage requires a named coefficient and is disabled for many_contrasts; output columns omit stat when shrinkage selected. |
| Glimma/iSEE/pcaExplorer/DEvis/regionReport links | Source interface mentions inspected; reporting/CSV purpose recorded. External apps/package internals not inspected or launched. |

| Distinct external unit | Repository | Source locator |
|---|---|---|
| `deseq2-airway-salmon` | thelovelab/rnaseqGene | Quantifying with Salmon; Appendix Updated details on quantification |
| `deseq2-airway-gse` | thelovelab/rnaseqGene | Reading in data with tximeta; SummarizedExperiment loadfullgse; design construction; airwayDE |
| `deseq2-count-ordination` | thelovelab/rnaseqGene | Sample distances; PCA plot using Generalized PCA; MDS plot |
| `deseq2-variable-gene-clustering` | thelovelab/rnaseqGene | Gene clustering; genescluster |
| `deseq2-annotate-results` | thelovelab/rnaseqGene | Annotating and exporting results; Exporting results |
| `deseq2-genomic-results` | thelovelab/rnaseqGene | Plotting fold changes in genomic space; gvizplot |
| `deseq2-sva-adjustment` | thelovelab/rnaseqGene | Removing hidden batch effects; Using SVA with DESeq2 |
| `deseq2-ruv-adjustment` | thelovelab/rnaseqGene | Using RUV with DESeq2 |
| `deseq2-fission-profiles` | thelovelab/rnaseqGene | Time course experiments; fissionDE, fissioncounts, fissionheatmap |
| `deseq2-zinbwave-analysis` | mikelove/zinbwave-deseq2 | Simulate single-cell count data; Model zero component; Estimate size factors; Estimate dispersion and DE; Evaluate simulated data |
| `deseq2-numeric-contrast` | tavareshugo/tutorial_DESeq2_contrasts | One factor three levels; Two factors with interaction; Three factors with nesting; README warning |

## Independent unit and dataset audit

Every final unit was reviewed against its own input state, claimed output, role, dataset identity and source locator; related units alone were not treated as evidence. Public API operations with generated or generic inputs remain valid discoveries without inventing an observed study. Separate import representations, treatment hypotheses, effect shrinkage, testing, diagnostics, time-specific profiles and report generation retain distinct scientific purposes. The audit corrected test/script source kinds, stale upstream-inspection caveats, RUV actual-versus-intended result state and the historical addMLE incompatibility.

Tests implement three distinct method components: custom hypothesis-filter callback, independent penalized coefficient IRLS/posterior calculation, and independent Cox-Reid dispersion posterior/derivative calculation. These are mixed implementation/application records. Standard use of a supplied dispersion function remains tool_use. No invented standalone tool_creation role was assigned.

| Unit | Role | Dataset links / input origin | Independently checked purpose |
|---|---|---|---|
| `deseq2-matrix-import` | tool_use | `pasilla` | Construct an aligned count-matrix DESeqDataSet |
| `deseq2-tximport-import` | tool_use | `tximportdata-geuvadis`, `gencode-human` | Import Salmon estimates to gene-level counts |
| `deseq2-tximeta-import` | tool_use | `tximportdata-geuvadis`, `gencode-human` | Import quantification with transcriptome metadata |
| `deseq2-htseq-import` | tool_use | `pasilla` | Construct from HTSeq gene-count files |
| `deseq2-airway-import` | tool_use | `airway`, `ensembl-human-75` | Construct from airway RangedSummarizedExperiment |
| `deseq2-prefilter` | tool_use | `pasilla` | Filter low-count features and set reference levels |
| `deseq2-wald-analysis` | tool_use | `pasilla` | Pasilla treatment differential-expression analysis |
| `deseq2-lfc-shrink` | tool_use | `pasilla` | Shrink and compare log fold changes |
| `deseq2-multi-factor` | tool_use | `pasilla` | Adjust Pasilla treatment effects for sequencing type |
| `deseq2-transform` | tool_use | `pasilla` | Variance-stabilize counts for exploratory analysis |
| `deseq2-sample-qc` | tool_use | `pasilla` | Visualize sample relationships and high-expression genes |
| `deseq2-gene-count-plot` | tool_use | `pasilla` | Inspect normalized counts for a selected gene |
| `deseq2-independent-filter` | tool_use | `pasilla` | Control low-information hypothesis filtering |
| `deseq2-ihw` | tool_use | `pasilla` | Weight hypotheses using mean count covariate |
| `deseq2-contrast-interaction` | tool_use | Generic or generated inputs; no observed dataset asserted | Extract group and genotype-specific condition contrasts |
| `deseq2-lrt` | tool_use | Generic or generated inputs; no observed dataset asserted | Test multiple effects using full/reduced models |
| `deseq2-size-factors` | tool_use | Generic or generated inputs; no observed dataset asserted | Estimate library normalization, including control genes |
| `deseq2-collapse-replicates` | tool_use | Generic or generated inputs; no observed dataset asserted | Sum technical sequencing runs |
| `deseq2-fpkm` | tool_use | Generic or generated inputs; no observed dataset asserted | Length-normalize fragment abundance |
| `deseq2-unmix` | tool_use | Generic or generated inputs; no observed dataset asserted | Estimate mixture contributions from pure expression profiles |
| `deseq2-single-cell` | tool_use | Generic or generated inputs; no observed dataset asserted | Adapt DESeq2 to individual-cell count inference |
| `deseq2-outliers` | tool_use | `pasilla` | Inspect influential counts and preserve replacement lineage |
| `deseq2-dispersion-fit` | tool_use | `pasilla` | Inspect and configure dispersion-mean fitting |
| `deseq2-effect-threshold` | tool_use | `pasilla` | Test scientifically sized effect thresholds |
| `deseq2-normalization-matrix` | tool_use | Generic or generated inputs; no observed dataset asserted | Incorporate gene- and sample-specific offsets |
| `deseq2-nested-design` | tool_use | Generic or generated inputs; no observed dataset asserted | Estimate group-specific paired condition effects |
| `deseq2-rank-repair` | tool_use | Generic or generated inputs; no observed dataset asserted | Diagnose confounding and remove empty design columns |
| `deseq2-simulate` | tool_use | Generic or generated inputs; no observed dataset asserted | Generate negative-binomial counts with known truth |
| `deseq2-sparsity-diagnostic` | tool_use | Generic or generated inputs; no observed dataset asserted | Diagnose concentration of counts in individual samples |
| `deseq2-fpm` | tool_use | Generic or generated inputs; no observed dataset asserted | Compare robust and total-count library scaling |
| `deseq2-frozen-transform` | tool_use | Generic or generated inputs; no observed dataset asserted | Apply transformations anchored to a prior reference experiment |
| `deseq2-direct-outlier-replacement` | tool_use | Generic or generated inputs; no observed dataset asserted | Replace influential counts using trimmed means |
| `deseq2-shrink-threshold` | tool_use | `pasilla` | Assess posterior false-sign-or-small effects |
| `deseq2-batch-visualization` | tool_use | Generic or generated inputs; no observed dataset asserted | Remove batch shifts from transformed visualization |
| `deseq2-airway-analysis` | tool_use | `airway` | Estimate dexamethasone effects controlling cell line |
| `deseq2-hammer-analysis` | tool_use | `hammer` | Analyze protocol effect controlling corrected time metadata |
| `deseq2-bottomly-analysis` | tool_use | `bottomly` | Analyze strain differences in Bottomly counts |
| `deseq2-parathyroid-analysis` | tool_use | `parathyroid` | Analyze selected parathyroid treatment after collapsing runs |
| `deseq2-fission-analysis` | tool_use | `fission` | Test strain-specific stress time-course trajectories |
| `deseq2-icobra-benchmark` | tool_use | `bottomly` | Compare differential-expression tools with empirical simulations |
| `deseq2-airway-salmon` | tool_use | `airway`, `gencode-human` | Quantify paired airway reads against versioned transcriptome |
| `deseq2-airway-gse` | tool_use | `airway`, `gencode-human` | Import full airway Salmon gene estimates and preserve length metadata |
| `deseq2-count-ordination` | tool_use | `airway`, `gencode-human` | Compare count-based and transformed sample dissimilarity |
| `deseq2-variable-gene-clustering` | tool_use | `airway` | Cluster relative expression of high-variance airway genes |
| `deseq2-annotate-results` | tool_use | `airway`, `org-hs-eg-db` | Map Ensembl genes and export annotated treatment results |
| `deseq2-genomic-results` | tool_use | `airway`, `gencode-human`, `org-hs-eg-db` | Plot treatment effects around a selected genomic locus |
| `deseq2-sva-adjustment` | tool_use | `airway` | Estimate hidden variation and prepare surrogate-variable design |
| `deseq2-ruv-adjustment` | tool_use | `airway` | Estimate unwanted variation using empirical controls |
| `deseq2-fission-profiles` | tool_use | `fission` | Characterize time-specific effects and gene response profiles |
| `deseq2-export-report` | tool_use | `pasilla` | Select and export interpretable results, or generate a summary report |
| `deseq2-observation-weights` | tool_use | Generic or generated inputs; no observed dataset asserted | Fit count models with observation-specific influence |
| `deseq2-custom-filter` | mixed | Generic or generated inputs; no observed dataset asserted | Implement an alternative hypothesis-filtering callback |
| `deseq2-zinbwave-analysis` | tool_use | Generic or generated inputs; no observed dataset asserted | Analyze simulated zero-inflated cells using observation weights |
| `deseq2-numeric-contrast` | tool_use | Generic or generated inputs; no observed dataset asserted | Compare composite groups through model-matrix contrasts |
| `deseq2-linked-transcriptome` | tool_use | `srr1197474-tximportdata`, `ensembl-drosophila-reference` | Import a custom linked transcriptome and summarize genes |
| `deseq2-coefficient-oracle` | mixed | Generic or generated inputs; no observed dataset asserted | Implement independent penalized coefficient checks |
| `deseq2-dispersion-oracle` | mixed | Generic or generated inputs; no observed dataset asserted | Implement independent dispersion-posterior checks |

Dataset identity audit: Pasilla/airway/fission are observed study packages; Bottomly/hammer/parathyroid have named source assets with incomplete independently verified provenance. GEUVADIS artificial A/B labels are not treatments. SRR1197474 is a separate biological fixture despite sharing tximportData. Technical collapsed runs, subsets, Salmon/STAR representations and benchmark parameter pairs remain stages of their named study rather than separate identities. GENCODE, Ensembl human, Ensembl Drosophila and org.Hs.eg.db are independently sourced reference products; release-specific assets are explicit and not interchangeable. Unrelated simulations have no umbrella dataset record. Package licenses do not establish underlying study/reference reuse terms.

## Access and scientific limitations

Large combined source responses occasionally truncated the displayed text; raw pinned vignette text was retained and operational sections re-read in bounded chunks. File locators use headings/chunk/symbol names or verified source-file lines, never browser display line numbers. A guessed rnaseqGene Bioconductor URL was unavailable; the correct official workflows URL then worked and GitHub source was independently pinned. One local patch-format error was corrected. No credential data was read or copied.

Source findings are provisional task-generation evidence, not executable performance results. Four historical testsuite compositions request addMLE after a default fit; the pinned results guard predicts failure because betaPrior is FALSE. Missing icobra parameter files, historical HTTP assets, unverified reference releases, package version mismatch and unresolved reuse terms require authoring-time validation. The Pasilla HTSeq/exon-bin stage mismatch, airway reconstruction-preview discrepancy, hidden tximeta skipMeta call, unevaluated report/IHW examples and RUV result-state ambiguity are retained rather than silently repaired.

## Validation and stopping reason

JSONL structural/reference validation and final-file-derived counts are recorded below after parsing the saved artifacts. No scientific test was run.

The useful scope of this source inventory is covered: the package's public scientific API/manual and operational vignette surface, its bundled study/benchmark scripts, its scientific regression examples, and directly linked distinct rnaseqGene, numeric-contrast and zero-inflation workflows were inspected and mapped. The Galaxy wrapper and remaining visualization leads repeat purposes already recorded; remaining low-level implementations, bibliography/history and remote package internals do not add a distinct source-supported use without broadening into package-by-package research. Remaining dataset bytes, historical asset compatibility, access/terms and execution checks are later task-authoring work and were not authorized here. Stopping follows this source audit, not elapsed time or a unit quota.

Final saved-artifact validation completed 2026-09-30 16:07:59 UTC. Parsed counts: **57 units**, **12 dataset/reference records**; roles **54 tool_use**, **3 mixed**, **0 tool_creation**. Both JSONL files parsed; all required keys present; identifiers unique; all dataset and related-unit references resolve; every source has a matching pinned revision and nonempty locator/URL; every unit appears in the audit map; no unused dataset records or stale pending-source markers remain. These are structural/source checks only. No task-owned worker remains running.
