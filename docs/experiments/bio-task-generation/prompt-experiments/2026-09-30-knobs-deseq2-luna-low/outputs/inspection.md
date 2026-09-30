# DESeq2 source inventory inspection

Repository: https://github.com/thelovelab/DESeq2. Inspected content is pinned to commit `c62c60c6ff83fd84ce115cacd1c49827533f85a7` (GitHub tree API returned `truncated: false`). Retrieval date: 2026-09-30. Package DESCRIPTION reports version 1.53.5, package license LGPL (>= 3), and dependencies/suggests including SummarizedExperiment, BiocParallel, tximport, tximportData, apeglm, ashr, and visualization packages. Package license does not establish data reuse terms.

The inventory records five provisional tool-use units: Pasilla matrix-based differential expression (`deseq2-pasilla-matrix-wald`), its separate sequencing-type multifactor refit (`deseq2-pasilla-multifactor`), Salmon import from tximportData (`deseq2-tximport-salmon-demo`), Pasilla VST/QC (`deseq2-pasilla-vst-qc`), and a generic nested-model LRT (`deseq2-lrt-model-comparison`). The Pasilla differential, multifactor, and QC units share the same study observations and are linked. The Salmon example remains a separate, unresolved demonstration dataset: its condition labels are explicitly artificial. No source implementation change is present in these units.

## Source map and inspection queue

| Location | Status | Notes / continuation |
|---|---|---|
| GitHub recursive tree at pinned revision | inspected | Complete, not truncated. Revealed package metadata, R sources, man pages, tests, extdata, vignette, and scripts. |
| `DESCRIPTION`, `NAMESPACE` | inspected | Package version/license/dependency and exported interface inventory. |
| `vignettes/DESeq2.Rmd` quick start, lines 37-95 | inspected | Standard workflow and supported input constructors; workflow notes unprocessed count source required. |
| `vignettes/DESeq2.Rmd` tximport, lines 254-359 | inspected | Salmon imports, gene mapping, estimated counts, artificial condition labels. |
| `vignettes/DESeq2.Rmd` count matrix, lines 400-500 | inspected | Pasilla extdata loading, sample alignment, constructor; unit `deseq2-pasilla-matrix-wald`. |
| `vignettes/DESeq2.Rmd` htseq-count, lines 500-546 | pending | Has both evaluated/unevaluated portions; inspect source/table setup and distinguish demonstration state. |
| `vignettes/DESeq2.Rmd` SummarizedExperiment and prefilter/factor sections, lines 547-642 | pending | Input alternative and preprocessing guidance. |
| `vignettes/DESeq2.Rmd` Pasilla DE/results/shrinkage, lines 644-710; downstream result sections partly searched | inspected in part | Unit `deseq2-pasilla-matrix-wald`; p-value summary and independent filtering also inspected at lines 748-830, but remainder of results exposition not read fully. |
| `vignettes/DESeq2.Rmd` multifactor, lines 1083-1188 | inspected | Unit `deseq2-pasilla-multifactor`; sequencing type is covariate; does not establish confounding/identifiability. |
| `vignettes/DESeq2.Rmd` transformations/QC, lines 1270-1440 | inspected | Unit `deseq2-pasilla-vst-qc`; VST, sample distances and PCA. Heatmap code was read as part of this section. |
| `vignettes/DESeq2.Rmd` LRT, lines 1624-1682 | inspected | Unit `deseq2-lrt-model-comparison`; full/reduced model meaning and examples. |
| `vignettes/DESeq2.Rmd` outliers, lines 1894-1934 | inspected in part | Cook's distance behavior; remaining cutoff details and surrounding variants pending. |
| `vignettes/DESeq2.Rmd` independent filtering and threshold tests, lines 2021-2108 | inspected | Result filtering and threshold alternatives; no separate unit. |
| `vignettes/DESeq2.Rmd` remaining sections, including interactions, single-cell recommendations, normalization factors, theory/FAQ | pending | Search result locators include interactions around 1442-1623, single-cell around 1837, normalization factors around 2192-2237, theory/FAQ 2428-end. Read targeted contexts before deciding unit boundaries. |
| `man/DESeq.Rd` | inspected | Arguments and stages of DESeq, LRT full/reduced interface. |
| `inst/extdata/pasilla_sample_annotation.csv` | inspected | Full 527-byte sample sheet; seven rows; count matrix intentionally not downloaded or previewed. |
| `inst/extdata/pasilla_gene_counts.tsv.gz` | identified, not read | Tree size 167602 bytes; no biological data download. |
| `inst/script/*`, tests, other `man/*.Rd` | mapped, pending | Script/test collections enumerated from tree but not individually inspected. Useful targeted leads: `man/results.Rd`, `man/lfcShrink.Rd`, `man/vst.Rd`, `man/DESeqDataSet.Rd`, `tests/testthat/test_results.R`, `test_LRT.R`, `test_txi.R`, `test_outlier.R`, `inst/script/testsuite.Rmd`. |
| Bioconductor `pasilla` and `tximportData` packages | linked, not inspected | Vignette says detailed Pasilla data production is in the pasilla package vignette. Need package versioned source and terms for further provenance; do not infer from the link. |

## Source-backed findings and boundaries

The vignette says DESeq2 models counts with a negative-binomial generalized linear model and describes size-factor, dispersion, and coefficient estimation. Its standard Pasilla matrix example explicitly aligns count columns with metadata rows before constructing a DESeqDataSet. The study description reports Drosophila melanogaster cell cultures and RNAi knock-down of splicing factor pasilla. The sample sheet preview reports seven libraries: treated1-3 and untreated1-4, with single-read/paired-end labels and lane/read/exon-count fields. The actual count matrix was not examined.

The multifactor example copies the Pasilla object, relabels sequencing-type factor levels, and fits `~ type + condition`, placing condition last so default results address condition. This model choice is a documented example, not evidence that sequencing protocol is unconfounded or that a causal interpretation is valid. The VST/QC section describes visualization of sample patterns. It does not make PCA or clustering a significance test.

The Salmon import example links tximportData files and a GENCODE v27-named tx2gene mapping through tximport, then constructs a DESeqDataSet. Its two A/B condition values are explicitly artificial demonstration labels. The generic LRT unit documents a reusable operation without a dataset link; authoring must supply observations and a scientifically appropriate nested design.

## Boundary and evidence notes

DE analysis, optional LFC shrinkage, and result extraction remain one Pasilla unit because they form the documented inference workflow for one scientific comparison. Sequencing-type adjustment is separate because it changes the scientific model and covariate interpretation. VST/QC is separate because its output and purpose are exploratory sample assessment rather than differential inference. tximport is an independent upstream representation/import stage. The LRT is an API-level operation with no specific dataset, and its unit record retains `dataset_ids: []`.

Public package URLs were not treated as reuse rights. No tool was installed or executed, no model or Harbor task was run, and no scientific validation is claimed. Source inspection only establishes the documented examples and inspected metadata.

## Stop and remaining leads

Stopped at 2026-09-30T15:36Z after the available targeted reads, output drafting, and JSONL/reference validation; the assignment deadline was not reached. Several useful source sections remain pending in the map, including the upstream pasilla package provenance, HTSeq and SummarizedExperiment input paths, interaction/time-series examples, and remaining API pages/tests. Resume with the highest-value pending source while the stated time budget remains; do not treat this partial inventory as proof that other operations or datasets are absent.
