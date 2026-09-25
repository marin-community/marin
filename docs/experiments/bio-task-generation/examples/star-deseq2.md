# STAR–DESeq2: from a workflow to task recipes

[Planning overview](../index.md) · [Authoring](../task-authoring.md) · [Transcriptomics examples](transcriptomics.md)

Use the [Snakemake workflow catalog entry](https://snakemake.github.io/snakemake-workflow-catalog/docs/workflows/snakemake-workflows/rna-seq-star-deseq2.html) as a discovery source and the implementation as evidence for recipe design. Inspected on 2026-09-25: release `v3.1.1`, commit [`aa6b17edf3396230165c18709d04cd982bdaaa4c`](https://github.com/snakemake-workflows/rna-seq-star-deseq2/tree/aa6b17edf3396230165c18709d04cd982bdaaa4c). This is a source review and task proposal; no tasks or runtime measurements are produced here.

## What the source supplies

The workflow connects FASTQ processing to differential-expression results. Its sample sheet describes biological samples and analysis covariates; its unit sheet associates sequencing runs or lanes with samples, read files and strandedness. This separates technical units from biological replication. See the pinned [configuration](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/config/config.yaml).

| Stage | Implementation evidence | Scientific decisions to preserve |
| --- | --- | --- |
| Read preparation | [fastp rules](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/rules/trim.smk) | Read pairing, adapters and filtering policy |
| Alignment and quantification | [STAR rule](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/rules/align.smk) | Compatible genome/GTF, splice-aware alignment and gene counts |
| Alignment QC | [RSeQC and MultiQC rules](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/rules/qc.smk) | Library orientation, read distribution and alignment summaries |
| Matrix construction | [Count-matrix script](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/scripts/count-matrix.py) | Select the stranded count column, omit STAR summary rows and sum technical units by sample |
| Model fitting | [DESeq2 initialization](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/scripts/deseq2-init.R) | Match metadata to counts; set factor levels, design and covariates; normalize and fit |
| Contrasts and effect estimates | [DESeq2 results script](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/scripts/deseq2.R) | Contrast direction, adjusted p-values and `ashr` fold-change shrinkage |
| Exploration and annotation | [PCA](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/scripts/plot-pca.R), [gene symbols](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/workflow/scripts/gene2symbol.R) | `rlog` transformation before PCA; identifier mapping and missing mappings |

## Candidate recipes

All candidates have scientific context transcriptomics, with a bulk RNA-seq tag. Operations can include quality control, data preparation, sequence alignment, normalization, dimensionality reduction and statistical inference. Assign only operations required and checked by the chosen task.

| Recipe | Supplied inputs → required artifacts | Deterministic checks |
| --- | --- | --- |
| Read preparation | Observed FASTQs and library metadata → cleaned reads and QC table | Read identity, pairing, sequence/quality transformations and accounting under a declared fastp protocol |
| Alignment and gene counting | Prepared FASTQs, compatible genome/GTF and index → alignments, gene counts and mapping summary | Gene-count values, read identities and semantic alignment checks under a pinned STAR protocol |
| Library QC | Observed BAMs and matching annotation → orientation evidence and QC summary | RSeQC statistics under pinned sampling settings; explicit decision thresholds and an ambiguous outcome where needed |
| Sample count assembly | STAR gene-count files and unit metadata → sample count matrix and provenance table | Correct strand columns, exclusion of non-gene rows, exact lane sums and separate biological replicates |
| Differential expression | Observed counts and sample metadata → fitted-design record, normalized counts and complete contrast results | Sample alignment, estimable design, contrast direction, tested genes, numerical tolerances and missing-value semantics |
| Sequencing units to treatment response | Unit-level counts and study metadata → sample matrix and differential-expression results | Compose count assembly and inference; catch aggregation errors that change the scientific conclusion |
| Reads to treatment response | Observed FASTQs, genome/GTF and study metadata → QC, counts and differential-expression results | Compose upstream and downstream stages; validate interfaces and final outputs within declared resources |

Paired, unpaired and interaction analyses can require separate inference recipes because their design constraints and interpretations differ. The table describes candidate boundaries, not implemented generators. The integrated recipes need end-to-end validation even when their components pass independently.

For PCA, a future task could request transformed-data summaries and variance explained instead of judging an SVG. Gene-symbol mapping can be an additional stage using a pinned local mapping with explicit missing/duplicate policies; retain stable gene IDs. Neither stage must become its own recipe.

## A concrete task specification to develop

For the sequencing-units-to-response recipe, select an observed study with biological replicates and documented technical units. A candidate instruction is:

> Estimate the treatment effect from the supplied STAR gene-count files and study metadata. Combine sequencing units belonging to the same biological sample using the recorded library orientation. Fit the specified experimental design and treated-versus-control contrast. Submit the sample count matrix, unit-to-sample provenance, design matrix and complete differential-expression table, including adjusted p-values and shrunken log2 fold changes.

An actual instance must replace “specified experimental design” with an explicit model justified by that study, such as `~ donor + treatment` for an estimable paired design. Freeze filtering, DESeq2/ashr versions, result settings, reference level and numerical acceptance criteria before computing expected outputs. The upstream default model includes interactions among variables of interest; do not inherit that choice without checking its scientific meaning.

Run the pinned native packages in the reference. Add independent checks for unit sums, identity matching and contrast direction. Challenge the grader with wrong-strand counts, lanes treated as biological replicates, omitted pairing, reversed contrasts and unshrunken effects submitted as shrunken effects. Each challenge must actually change a checked result in the chosen instance. Expected outputs remain outside solver-visible inputs; no LLM judge is needed.

## Data and ten-instance variation

Search for compatible observed studies with accessible counts or reads, documented designs and redistribution eligibility. Start with one instance and target ten validated instances per scalable recipe. Track study count and variation in sample structures, technical-unit layouts and supported comparisons. Multiple contrasts from one study share lineage; relabeling samples or changing thresholds alone does not establish breadth.

Count-based tasks can use studies whose reads are too large for the sandbox. Read-based recipes need separately qualified input sets. An observed dataset without lane-level evidence cannot be presented as an observed multi-lane study. An artificial read partition must be labeled as an adaptation and does not create biological replication.

Do not treat upstream integration-test fixtures as scientific studies. In this release, the [test unit sheet](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/.test/config_basic/units.tsv) reuses the same FASTQ pair for A2 and B1, while the [sample sheet](https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/aa6b17edf3396230165c18709d04cd982bdaaa4c/.test/config_basic/samples.tsv) assigns them different conditions. That is useful for testing workflow execution, but cannot supply independent treatment/control observations for our tasks.

No input pool supporting ten instances has been selected or validated for this example. Record eligible candidates separately from accepted task instances. Existing [paired-treatment candidates](transcriptomics.md#candidate-tasks) can inform count-based authoring, but do not establish eligibility for every recipe above.

## Sandbox adaptations and candidate order

Use scientifically appropriate subsets to make both focused and integrated recipes practical. For FASTQs, select read pairs together and preserve sufficient coverage across biological replicates. For count-based analyses, retain an estimable experimental design and enough expression information for the declared inference. Record all reductions as adaptations and compute references on the exact delivered data.

Read subsampling reduces input size and alignment work, but does not shrink an unchanged genome index. A compact reference or selected region can support a scoped alignment/counting task if its annotation and inputs are compatible. Record that reduced mapping context and avoid claiming whole-genome conclusions. For treatment inference, check how feature or reference reduction affects normalization and the tested-gene universe. A compatible small-genome study is another option. Validate each choice against the 8 GiB memory and 10 GiB disk envelope.

Sample count assembly, differential expression and their composition offer useful first implementations, while a reduced read-to-result candidate tests full-pipeline feasibility. Do not defer a raw-read recipe solely because the source dataset is large. If subsetting cannot preserve the intended scientific question, choose a later input boundary or another dataset. Precomputed inputs do not count as solver-executed alignment. Resource suitability remains unmeasured for all these candidates.

Prepare an offline environment with pinned wrappers, packages, reads, genome/GTF and any index. Cap execution at the declared allocation: the upstream STAR and paired fastp rules request 24 and 8 threads respectively, so their resource configuration must be adapted. Inspect the actual dependency graph and peak disk use; a rule's presence alone does not imply it runs for every target.

The upstream gene-symbol script queries live BioMart, and reference/read retrieval can use Ensembl/SRA. Stage versioned assets and replace the live annotation lookup with a local mapping, recording that adaptation. Validate the selected Snakemake targets in Harbor. A task that only calls extracted native scripts can exercise the scientific operations, but should not be recorded as executing the Snakemake workflow itself.

The public recipe record should pin the source revision, selected stages, adaptations, input provenance and hashes, verifier contract and validation evidence. Implementation and data selection remain the next steps; this review does not launch them.
