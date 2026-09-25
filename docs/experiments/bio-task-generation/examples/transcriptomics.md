# Transcriptomics examples

[Planning overview](../index.md) · [Authoring](../task-authoring.md) · [Requirements](../requirements.md)

Use transcriptomics to review the curation process before broadening it. These are candidate sketches, not newly built or runtime-validated tasks. The eventual corpus remains broad.

## Concrete sources

| Source | Useful material |
| --- | --- |
| [Snakemake STAR–DESeq2 workflow](https://github.com/snakemake-workflows/rna-seq-star-deseq2) | Connected alignment and differential-expression workflow; [recipe analysis](star-deseq2.md) |
| [nf-core/rnaseq](https://nf-co.re/rnaseq/3.27.0/) and [differentialabundance](https://nf-co.re/differentialabundance/2.0.0/) | Read processing produces counts and QC; downstream differential analysis is a separate workflow |
| [Galaxy reference-based RNA-seq tutorial](https://training.galaxyproject.org/training-material/topics/transcriptomics/tutorials/ref-based/tutorial.html) | Scientific framing, observed reads, counting and differential analysis |
| [Bioconductor rnaseqGene](https://bioconductor.org/packages/release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html) | The airway treatment study, paired design and native reference code |
| [Bioconductor RNAseq123](https://bioconductor.org/packages/release/workflows/vignettes/RNAseq123/inst/doc/limmaWorkflow.html) | Contrasts and gene-set analyses using limma, Glimma and edgeR |

## Candidate tasks

| Candidate | Inputs → outputs | Scientific work and verification |
| --- | --- | --- |
| Focused: paired treatment effect | Observed counts and metadata → effects, p-values and adjusted p-values | The airway example has four treated/untreated cell-line pairs. Specify a DESeq2 analysis accounting for pairing and contrast direction. Verify complete tables and reject omitted pairing or reversed effects. |
| Focused: fragment counting | Observed paired-end BAMs, matching GTF and library metadata → count matrix and assignment totals | Resolve strandedness, fragment counting and overlapping features under explicit conventions. Check native results and accounting; incorrect strand or read/fragment settings should fail. Select suitable observed inputs before building. |
| Connected: quantification to response | Transcript quantifications, transcript-to-gene mapping and metadata → gene summaries, fitted contrast and results | Exercise import, identity reconciliation and statistical analysis. Check intermediates and final results; reject dropped or misassigned samples. Quantification is supplied upstream. |
| Connected: response and pathways | Observed counts, metadata and pinned gene sets → contrast and pathway statistics | Couple differential analysis to a specified enrichment method. Check identifiers, tested-gene universe or ranked-list construction, and numerical outputs. Do not score narrative plausibility. |

For the paired-treatment example, the [rnaseqGene source](https://bioconductor.org/packages/release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html) supplies a concrete design for discussion. Repeated use of one familiar tutorial study offers limited diversity. Expand to other observed studies and supported experimental designs after the authoring process is understood.

## Scanpy: focused and integrated recipes

Scanpy's [preprocessing and clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html) analyzes two human bone-marrow samples. It covers import, QC, doublet detection, normalization, variable-gene selection, PCA, neighbors, UMAP, clustering, QC reassessment, annotation and marker testing. It discusses batch integration but does not perform it.

The following are proposed recipe boundaries inspired by that tutorial. Their scientific context is transcriptomics, with a single-cell tag. Operation labels describe the required work and do not create additional hierarchy levels.

| Recipe | Input → deliverable | Required work |
| --- | --- | --- |
| Quality control | Count matrices and metadata → filtered counts and QC report | Import, QC metrics, doublet scoring and declared filtering rules |
| Population discovery | Filtered counts → clusters and population summaries | Normalization, feature selection, PCA, neighbor graph and clustering |
| Population characterization | Expression and supplied clusters → marker evidence and supported labels | Statistical testing, expression summaries and annotation against curated evidence |
| Complete analysis | Counts and metadata → characterized populations | Compose the preceding stages, checking their interfaces and final scientific outputs |

For example, quality control exercises data preparation, descriptive analysis and quality control; population discovery exercises normalization, feature selection, dimensionality reduction and clustering. These operation labels remain provisional. Graph construction can be part of a clustering recipe without becoming a separate recipe. A characterization task supplied with clusters does not exercise clustering.

An integrated task must preserve cell identities, retain the count representation needed downstream and use the correct representation at each stage. Verify these connections as well as final results. UMAP and other plots can aid inspection without making visual appearance a reward. Compare cluster membership independently of cluster numbering, and validate numerical tolerances and reproducibility before release.

Annotation requires curated acceptable labels or an explicit assignment rule, including ambiguous cases. Reproducing a marker-scoring rule establishes correct execution of that rule; it does not by itself establish biological identity. Keep cluster marker characterization distinct from treatment-effect inference across biological replicates. The tutorial warns about cell-level independence and suggests sample-level aggregation for downstream differential analysis.

Generate instances from other compatible observed datasets and supported scientific questions. Focused and integrated tasks derived from the same study retain shared study and asset lineage. These sketches have not been executed or checked against the sandbox resource limits.

## Review questions

For a focused task and a connected task, compare the actual solver instructions, input previews and sizes, scientific decisions, native reference and executable acceptance criteria. Confirm that the connected task adds substantive integration and that the focused task still produces a useful scientific result.

Preserve shared-study, recipe and asset identifiers; these tasks do not establish independent biological coverage merely because their starting points differ. Starting from counts can preserve all biological replicates while avoiding expensive alignment.

The [Galaxy tutorial](https://training.galaxyproject.org/training-material/topics/transcriptomics/tutorials/ref-based/tutorial.html) offers approximately 5 MB FASTQ subsets for quick demonstrations alongside much larger files. Do not automatically use those subsets for differential inference. Validate scientific adequacy and runtime for every reduction.
