# DESeq2 source inventory

Inspected repository `thelovelab/DESeq2` at pinned commit `c62c60c6ff83fd84ce115cacd1c49827533f85a7`, resolved by the runner through the GitHub commits API. Retrieval date: 2026-09-30. The recursive GitHub tree was untruncated. Package metadata describes an R/Bioconductor package for negative-binomial count modeling and differential expression (DESCRIPTION; LGPL >= 3). Source reads were lightweight GitHub API requests; no repository clone, dataset download, package installation, R execution, or scientific analysis was performed.

The inventory has **8 units** and **4 dataset/reference records**. The records cover tool use and one mixed simulation benchmark. Dataset IDs point only to assets supported by the inspected source. Pasilla study metadata comes from the vignette and official Bioconductor package description; other provenance and terms are explicitly limited where the sources do not establish them.

## Source map and inspection queue

### Repository tree and package metadata

- `DESCRIPTION`, `NAMESPACE`, `R/`, `man/`, `inst/extdata/`, `inst/script/`, `tests/testthat/`, `vignettes/`: tree inspected at pinned revision. DESCRIPTION inspected. This confirms an R package with public API documentation, bundled Pasilla fixtures, analysis scripts and a long vignette.
- `R/` implementations: **pending**. Useful follow-up: inspect call paths only for a specific authoring question; implementation reading is not required to treat documented APIs as tool use.
- `tests/testthat/`: **pending**. There are tests for DESeq, LRT, dispersion, shrinkage, outliers, results, size factors, interactions, weights and other behavior. No individual tests were inspected; do not infer their coverage beyond filenames.

### Main vignette `vignettes/DESeq2.Rmd`

Sections and code chunks enumerated from source headings/chunk labels. Status:

- Quick start; input/count semantics; DESeqDataSet construction: **inspected**, represented in `deseq2-pasilla-count-import-and-differential-expression` where supported by a concrete example.
- Transcript abundance files / tximport / tximeta: **partly inspected**. tximport chunks `txiSetup`, `txiFiles`, `tximport`, `txi2dds` inspected and recorded as `deseq2-tximport-salmon-to-gene-counts`. Tximeta section and its chunk details are **pending**.
- Count matrix: chunks `loadPasilla`, `showPasilla`, `reorderPasila`, `matrixInput`, `addFeatureData` inspected; included in Pasilla count-import unit.
- htseq-count and SummarizedExperiment inputs: **pending**. Chunks `htseqDirI`, `htseqDirII`, `loadSumExp`, `sumExpInput` remain useful input-path leads.
- Prefiltering, factor levels, technical-replicate collapse and Pasilla study description: **inspected selectively** for context. Standalone prefiltering and replicate-collapse units deferred because the pass focused on distinct scientific analyses rather than helper operations.
- Standard differential expression, result extraction and p-value adjustment: **inspected**, represented in the Pasilla count-import unit. IHW optional workflow is **pending** (chunk `IHW`, `eval=FALSE`).
- LFC shrinkage and MA-plot: **inspected**, unit `deseq2-lfc-shrinkage`; alternative estimators documented, but estimator-specific output was not run.
- Multifactor design and contrasts: **inspected** for Pasilla sequencing-type adjustment (`deseq2-pasilla-adjust-sequencing-type`). Other interactions and contrast forms are **pending**.
- VST/rlog, sample distances, heatmaps and PCA: **inspected**, unit `deseq2-transform-and-sample-qc`. Count transformations beyond the showcased functions and visualization variants remain open.
- Time series and LRT: **inspected** alongside test-suite fission example, unit `deseq2-lrt-model-comparison`.
- Outlier handling and independent filtering: **inspected**, unit `deseq2-outlier-and-independent-filtering`.
- Dispersion fitting alternatives, threshold tests, normalization-factor offsets, model-matrix rank issues, nested designs, full theory, remaining FAQ and session info: **pending**. Specific section names and chunk labels are in the source outline; these include `dispFit`, `dispFitCustom`, `lfcThresh`, `normFactors`, `offsetTransform`, `lineardep*`, `groupeffect*`, and `missingcombo*`.
- Single-cell recommendations: **pending**; section `Recommendations for single-cell analysis` was identified but not inspected. This is a potentially useful distinct use and should be an early continuation lead.

### API manuals (`man/`)

- `results.Rd`, `lfcShrink.Rd`, `vst.Rd`, and `estimateDispersions.Rd`: **inspected** as bounded source files for result semantics, shrinkage choices, fast VST behavior, and dispersion fitting. The three first manuals support the unit records; dispersion semantics are supporting context.
- Remaining manual pages: **pending**, including `DESeq.Rd`, `nbinomLRT.Rd`, `nbinomWaldTest.Rd`, `estimateSizeFactors.Rd`, `replaceOutliers.Rd`, `rlog.Rd`, `varianceStabilizingTransformation.Rd`, `unmix.Rd`, and plotting/import constructors. Enumerated from the pinned tree but not read individually.

### Analysis scripts under `inst/script/`

- `runScripts.R`, `makeSim.R`, `icobra_benchmarks.R`: **inspected**, one related simulated benchmark unit `deseq2-simulated-method-benchmark`.
- `testsuite.Rmd`: **partly inspected**. Its helper definitions and chunks for airway, Pasilla, Bottomly, parathyroid and fission were read. The benchmark's disabled Bottomly parameter-generation block and the fission LRT example support recorded units/data. Other dataset analyses in this script are **pending**: airway (`~ cell + dex`), Hammer (`~ Time + protocol` with recount2 retrieval), Bottomly (`~ strain`), and parathyroid (subset and collapse technical replicates). `vst.nb`, PDF/PNG outputs, package version text, and other script assets were not opened.

### Data and external supporting sources

- Pinned `inst/extdata` tree metadata: **inspected** for Pasilla count and annotation paths and sizes; files themselves were not downloaded or opened. The DESeq2 vignette sections describing import and sample alignment were read.
- Official Bioconductor Pasilla package page and data-generation vignette: search results inspected 2026-09-30; they report the Brooks et al. RNAi study and GEO accessions. Full external pages and exact terms were **not verified**.
- Official tximportData vignette: search result inspected 2026-09-30; it identifies six GEUVADIS samples. Sample IDs, exact release and data package terms remain **pending**.
- Fission and Bottomly package data sources: only references and use sites in `testsuite.Rmd` were inspected. Package releases, studies, accessions and terms are **pending**.

## Supported uses and relationships

The strongest source-backed task candidates are: aligning and fitting the bundled Pasilla count matrix; adjusting its treatment effect for sequencing type; importing Salmon transcript quantifications using tximport; producing transformed sample-QC summaries; applying a joint LRT to interaction terms; interpreting NA results from Cook's-distance and independent filtering behavior; and benchmarking multiple methods on simulated negative-binomial counts with known truth.

The Pasilla import, multifactor fit, shrinkage, transformed sample-QC, and result-filtering units share the same observed study counts but expose different analysis stages and questions. The LRT example uses the separate fission package dataset. The tximport demonstration uses six GEUVADIS quantifier outputs and a separate GENCODE transcript-to-gene reference. The benchmark's generated observations are synthetic; Bottomly contributes only mean/dispersion parameter pairs. No records claim that a task is authored, executable, graded, or validated.

## Access failures, scope decisions, and continuation

GitHub tree and raw-file requests succeeded. No inaccessible repository source was encountered. External Bioconductor material was only partially accessible through search-result text; exact pages, package releases and terms need direct follow-up. The repository's software license is not treated as proof of redistribution rights for its data.

The discovery pass stops with this partial queue rather than treating representative examples as full coverage. The highest-value next sources are the single-cell recommendation, tximeta and htseq/SummarizedExperiment input sections, interaction and custom normalization examples, then remaining test-suite datasets and their package provenance. Additional useful work includes IHW, size-factor control features, and the many specialized API manuals. The launch allowed lightweight public source reads but prohibited cloning, installing, executing analyses or downloading biological data; these constraints were respected. At the time this inventory was saved, the launch deadline had not yet been reached, so the recorded stopping reason is the bounded pass/current assignment handoff, not deadline exhaustion. No process-heavy work was launched.
