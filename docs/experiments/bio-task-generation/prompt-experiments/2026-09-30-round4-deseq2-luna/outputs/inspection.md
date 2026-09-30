# DESeq2 source inventory inspection

## Scope and revisions

Repository: `thelovelab/DESeq2`, GitHub URL https://github.com/thelovelab/DESeq2. Repository source inspected at pinned revision `c62c60c6ff83fd84ce115cacd1c49827533f85a7`, resolved by the runner through the commits API. `DESCRIPTION` reports package version 1.53.5 and LGPL (>=3). The recursive GitHub tree was not truncated. No local checkout or biological assets were downloaded.

The tree map included R implementation files (`R/`), Rd manuals (`man/`), testthat files (`tests/testthat/`), scripts (`inst/script/`), two Pasilla extdata files, and `vignettes/DESeq2.Rmd` plus `library.bib`. The vignette is the main operation map. Package DESCRIPTION, selected constructor/DESeq/results/lfcShrink/VST/rlog manuals, and `inst/script/testsuite.Rmd` excerpts were checked. Implementation and most tests/manual pages were not inspected.

## Inventory organization and source map

IDs in `units.jsonl` cover distinct documented uses; ID references resolve to `datasets.jsonl` as of the final local check.

| Source location | Status |
|---|---|
| `DESCRIPTION` | Inspected: package version, role/metadata, dependencies and license. |
| `vignettes/DESeq2.Rmd` lines 193-243, 411-496, 642-696, 2430-2480 | Inspected: raw integer count requirements, DESeqDataSet matrix setup/sample alignment, Pasilla study context, standard differential analysis/model semantics. Unit `deseq2-pasilla-standard-de`. |
| `vignettes/DESeq2.Rmd` lines 500-543; `man/DESeqDataSet.Rd` constructor arguments | Inspected: `DESeqDataSetFromHTSeqCount` demo code is marked unevaluated. Unit `deseq2-htseq-input-constructor`; input asset identity/stage remains uncertain. |
| `vignettes/DESeq2.Rmd` lines 547-565 | Inspected: airway object constructor only. Unit `deseq2-airway-se-import`. |
| `vignettes/DESeq2.Rmd` lines 256-359 | Inspected: tximport/Salmon example, tx2gene file, artificial A/B labels and gene-level import. Unit `deseq2-tximport-genelevel`. `tximeta` mini-example at lines 361-396 was seen but not separately inventoried. |
| `vignettes/DESeq2.Rmd` lines 703-719; `man/lfcShrink.Rd` | Inspected: apeglm shrinkage. Unit `deseq2-lfc-shrinkage`. |
| `vignettes/DESeq2.Rmd` lines 1116-1182 | Inspected: Pasilla type-adjusted multi-factor refit and contrasts. Unit `deseq2-pasilla-protocol-adjustment`. |
| `vignettes/DESeq2.Rmd` lines 1191-1289, 1356-1440; `man/vst.Rd`, `man/rlog.Rd` | Inspected: transformations and sample distance/PCA QC. Unit `deseq2-pasilla-transform-qc`. |
| `vignettes/DESeq2.Rmd` lines 1624-1681 and 2890-2923 | Inspected: time-series design advice and LRT semantics/examples; code is generic and unevaluated. Unit `deseq2-likelihood-ratio-test`. |
| `vignettes/DESeq2.Rmd` lines 1888-2017 | Inspected: Cook's distance behavior, plotDispEsts, dispersion alternatives/custom fit. No separate unit was saved before the deadline. |
| `vignettes/DESeq2.Rmd` lines 2021-2104 | Inspected: independent filtering and threshold-based Wald tests. No separate unit was saved before the deadline. |
| `vignettes/DESeq2.Rmd` lines 1442-1555 excerpt | Inspected: individual Wald steps, controlGenes, contrasts and interactions; these sections contain unevaluated or generic examples. No separate unit saved. |
| `inst/script/testsuite.Rmd` lines 57-130 excerpt | Inspected: benchmark-style full analyses on airway, Pasilla, recount2 hammer/bottomly, parathyroid, and fission datasets, including filtering/collapse and LRT. These are continuation leads, not inventoried units. |
| Remaining vignette sections and remaining manuals/tests/scripts | Pending/not inspected: source inventory tree is available in the pinned Git tree; prioritize data lineage and distinct analyses rather than enumerating every helper/test. |

## Dataset records and relationships

`pasilla-rnaseq` links the DESeq2-copied gene-count matrix and annotation CSV to the Pasilla study because the vignette says those assets were copied from the Pasilla package. Annotation rows show 7 samples and condition/sequencing type. The official mutable Bioconductor Pasilla preprocessing vignette retrieved 2026-09-30 (rendered package 1.40.0) identifies GSE18508, 3 knockdown/4 control biological replicates, and details exon-count preprocessing. That exon-count description does not establish the exact provenance of the copied DESeq2 per-gene matrix or the HTSeq-demo file semantics. A possible mismatch between the DESeq2 demo's HTSeq label and the external Pasilla vignette's DEXSeq exon-count recipe remains explicitly unresolved, rather than treated as a same-file contradiction.

`tximportdata-geuvadis` identifies the six-sample GEUVADIS quantification package using the official package manual (1.38.0 / Bioconductor 3.22 / commit 5c04cae in inspected metadata). Its Salmon output and mapping paths are in the DESeq2 vignette. Tutorial A/B condition labels are fabricated and not biological exposure metadata.

`airway-rnaseq` identifies packaged airway gene counts, GEO GSE52778, four human cell lines, dexamethasone/control conditions and RangedSummarizedExperiment format from the official Bioconductor manual (package 1.32.0 / Bioconductor 3.23 / commit deb0a4f). The distinct package `gse` Salmon object is kept out of this record's assets because it is a different processed representation.

No downloaded data preview, execution, model fit, result values, runtime, grading, or reuse-rights verification is claimed. Mutable official Bioconductor pages were retrieved on 2026-09-30; their reported commits/versions are preserved where available.

## Boundary decisions and continuation leads

The standard Pasilla fit, protocol-adjusted refit, post-fit LFC shrinkage, and transformed-data QC answer distinct scientific/output needs, so they are separate units linked to the same study. Constructor examples for airway, tximport and HTSeq are separately recorded because their input representations and prerequisites differ. The LRT is recorded as a documented tool-use method despite its generic unevaluated code. tximeta, threshold-based tests, independent filtering diagnostics, outlier handling, custom dispersion fitting, interaction/contrast examples, and remaining benchmark-script analyses are continuation leads; no claim is made that they were exhaustively covered.

## Final check and stop reason

`units.jsonl` contains 8 unit records and `datasets.jsonl` contains 3 dataset records, computed from the final files. Final local validation at 2026-09-30 15:28:07 UTC parsed both JSONL files: 8 unique units, 3 unique datasets, all required fields present, tool_role values valid, and all unit/data references resolve. The output directory contains only the three assigned files. The assigned window ended at 2026-09-30 15:24:50.634831 UTC; following the parent deadline reminder, the actual final stop/check time was 2026-09-30 15:28:07 UTC. Work stopped because the deadline had elapsed, not because the mapped source scope was exhausted. Remaining leads above require follow-up inventory work.
