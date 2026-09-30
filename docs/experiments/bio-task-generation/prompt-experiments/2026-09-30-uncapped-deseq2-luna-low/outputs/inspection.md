# DESeq2 source inventory

Repository: [thelovelab/DESeq2](https://github.com/thelovelab/DESeq2), pinned revision `c62c60c6ff83fd84ce115cacd1c49827533f85a7` (resolved revision supplied by runner). Retrieval date: 2026-09-30. Source reads used `gh api` at that revision. The recursive GitHub tree was complete (`truncated: false`). No repository clone, data download, R execution, or scientific analysis was performed.

## Scope and organization

The inspected package is DESeq2 1.53.5 at this revision (DESCRIPTION). It implements negative-binomial count modeling for sequencing assays, with its main vignette at `vignettes/DESeq2.Rmd`, R API implementations under `R/`, generated API manuals under `man/`, examples/scripts under `inst/script/`, and tests under `tests/testthat/`. The vignette covers import and setup, count modeling/results, transformations and QA, design variants, outlier/dispersion/normalization behavior, and theory/FAQ. This inventory records six units: Salmon import, Pasilla Wald/shrinkage, Pasilla multifactor analysis, transformation/PCA, documented LRT patterns, and an iCOBRA simulation benchmark.

## Source map and inspection queue

Statuses: `inspected` means source text was read for the listed unit(s), not executed. `pending` means useful source remains. Paths and line locators refer to pinned source-file lines where available.

| Location | Status | Coverage / remaining |
|---|---|---|
| `DESCRIPTION` | inspected | Package version, description, license, dependencies. |
| Recursive repository tree | inspected | Complete path/size map; listed small source/manual/test/script assets. Binary assets not opened. |
| `vignettes/DESeq2.Rmd` lines 254-359 | inspected: `deseq2-vignette-tximport-salmon` | tximportData setup, artificial labels, Salmon quant files, tx2gene, import, DESeqDataSet constructor. |
| `vignettes/DESeq2.Rmd` lines 400-505 | inspected: `deseq2-vignette-pasilla-wald-shrink` | Bundled files, sample alignment, matrix constructor and feature metadata. HTSeq section only partly mapped; specific chunks around lines 500-546 remain pending. |
| `vignettes/DESeq2.Rmd` lines 632-720 | inspected: `deseq2-vignette-pasilla-wald-shrink` | Technical-replicate caveat, Pasilla context, standard DESeq/results, shrinkage. |
| `vignettes/DESeq2.Rmd` lines 1116-1186 | inspected: `deseq2-vignette-pasilla-multifactor` | Adjusts sequencing type and extracts treatment/type effects. |
| `vignettes/DESeq2.Rmd` lines 1191-1440 | inspected: `deseq2-vignette-transform-pca` | Transformations, sample distances/clustering and PCA source chunks; airway SummarizedExperiment input at lines 551-566 is identified but not followed as a standalone analysis. |
| `vignettes/DESeq2.Rmd` lines 1442-1682 | partly inspected: `deseq2-vignette-pasilla-lrt` | Wald step outline, control features, contrasts, interactions, time-series, LRT patterns mapped via headings/chunk search; exact design/data contexts for lines 1442-1623 and time-series lines 1624-1660 remain pending. |
| `vignettes/DESeq2.Rmd` lines 1685-1836 | pending | Extended shrinkage estimators and threshold tests. |
| `vignettes/DESeq2.Rmd` lines 1837-2427 | partly inspected | Read recommendations for single-cell analysis (1837-1893), outlier handling (1894-1971), dispersion fit alternatives (1972-2020), thresholded LFC and result access (2090-2191), normalization factors (2192-2235). Independent filtering full examples, model-matrix rank/linear-combination/nested-group and missing-level sections remain pending. |
| `vignettes/DESeq2.Rmd` lines 2428-2692 | partly inspected | Theory headings, independent-filtering rationale and simulated example (2608-2692) mapped; theory, contrasts, expanded matrices, complete filters evidence remain pending. |
| `vignettes/DESeq2.Rmd` lines 2692-2974 | pending | FAQ/session info/references. |
| `man/DESeq.Rd` | inspected | Wrapper stages, Wald/LRT controls, input/output and example; source-backed unit references. |
| `man/results.Rd` | inspected | Contrasts, threshold hypotheses, Cook's cutoff, independent filtering and result columns; relevant to Pasilla unit. |
| `man/nbinomLRT.Rd` | inspected | Full/reduced models, prerequisite estimates and deviance test semantics. |
| `man/vst.Rd`, `man/rlog.Rd` | inspected | Transformation APIs and output/dependency descriptions. |
| Other `man/*.Rd` files in tree | pending | Especially `lfcShrink.Rd`, size-factor and dispersion APIs, constructors, collapseReplicates, plots and outlier replacement. |
| `inst/script/testsuite.Rmd` | partly inspected | All chunks enumerated/read in retrieved source; identifies airway, Pasilla, Hammer, Bottomly, Parathyroid, Fission analyses. Detailed downstream comparisons/plots and external dataset lineage remain pending. Fission LRT with strain:minute interaction is a distinct candidate. |
| `inst/script/runScripts.R` | inspected | DESeq2 Wald/LRT wrappers and competing tool wrappers; noteworthy NA/FDR handling. |
| `inst/script/makeSim.R` | inspected | Simulated count generation function. |
| `inst/script/icobra_benchmarks.R` | inspected: `deseq2-icobra-simulation-benchmark` | Benchmark design, reference loading, simulations, methods, metrics and output. Input RDA absent in repository tree. |
| `tests/testthat/*.R` | pending | Tree enumerated 29 test files; not explored as primary scientific operations in this pass. |

## Data findings and boundaries

- `deseq2-bundled-pasilla` links the repository's bundled gene-count matrix and its annotation CSV. The vignette identifies the organism and RNAi question, warns that sample row ordering initially differs from count-column order, and explicitly shows the alignment correction. No file content preview, sample count, accession, or underlying data-package production vignette was inspected.
- `tximportdata-salmon-demo` is a separate source example. The vignette says it uses `tximportData` package extdata, Salmon quantifications and a Gencode v27-named mapping file. It explicitly creates artificial A/B conditions for demonstration. Underlying experiment identity and terms remain unknown; it is not represented as an observed contrast.
- `bottomly-mean-dispersion-reference` is an incomplete reference product, not a study dataset fully available here. The benchmark script expects `meanDispPairs_bottomly.rda`; its inactive generation block points to absent `bottomly_sumexp.RData`, assigns/constructs metadata, and derives base means and gene-wise dispersions. No identity, study provenance, or values were checked inside either RDA.
- The `airway` package object is used as input in `inst/script/testsuite.Rmd` and in an input-constructor vignette section; it is a useful remaining data lead. It was not conflated with the three records above.
- `testsuite.Rmd` also names external package datasets `hammer`, `bottomly`, `parathyroidGenesSE`, and `fission`. They remain leads rather than dataset records in this partial inventory because their actual sources, study context, and loading transformations were not followed fully. Fission analysis has an LRT-specific question; parathyroid section filters time/treatment and collapses technical runs before fitting.

## Boundary and deduplication decisions

Pasilla baseline Wald/shrinkage and multi-factor analysis are separate units because they answer different model questions, but share one data product and the former's constructed object is a prerequisite for the latter. The transformation/PCA unit uses the same observations but a different downstream purpose; it is not a differential test. The generic LRT patterns are retained as a unit because the manual specifies an independently documented operation and useful test result, although vignette chunks are marked unevaluated. The simulation benchmark is a separate tool-use composition: generated observations and benchmark truth are explicitly distinguished from biological source data. Lower-level functions and individual plotting chunks were not split into units without a distinct scientific objective.

## Possible task directions (not validated)

Source-backed possibilities include count/metadata alignment and model fitting, condition-adjusted Pasilla contrasts, transformed-data PCA/QA, full-versus-reduced model testing, and empirical evaluation on generated counts with known truth. Data suitability, runtime, grader design, execution success, and Harbor compatibility are untested. The benchmark's absent reference RDA and external method package stack are substantial authoring dependencies.

## Failed access and stopping reason

No source retrieval command failed. The recursive tree and all requested small-file reads succeeded. No external package data source was followed beyond references in DESeq2 source. This pass stops with useful accessible work remaining because time was used to establish a source-backed initial inventory and the invocation requires returning these three files; the remaining leads above are concrete and do not represent an access or resource blocker. Elapsed wall time from first source read (`2026-09-30 15:43:56 UTC`) to final inventory write is approximately 2 minutes. Final file counts are recorded after lightweight validation in the handoff response.
