# BixBench-Verified: flat competency review

Draft for review: 41 of 50 source questions have proposed executable checks and 18 provisional competencies. Task generation remains paused. No LLM judge is in scope.

An **analysis competency** is a reusable scientific analysis with a checkable outcome. A **workflow recipe** connects competencies to answer a scientific question on observed data. Filters, covariates, model options and denominators belong in the question-specific contract unless they change the analysis being assessed. These boundaries are open for review.

Use the same competency when two tasks require the same scientific analysis and a comparable output contract, even if species, tool or threshold changes. Split it when the scientific decision or required artifacts change substantially. A task can carry several competencies when its verifier checks the connected intermediate results.

Each question may require several competencies; it is counted once per assigned label. The counts overlap and do not measure unique studies, independent workflows or validated generated tasks.

[Versioned annotations, original questions and verification notes](../../experiments/post_training/bio_tasks/benchmark_competencies/bixbench-verified-50.json) · [Source inventory](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench-verified-50.json) · [Licensed source dataset](https://huggingface.co/datasets/phylobio/BixBench-Verified-50)

The naming follows [EDAM's separation of operations, topics, data and formats](https://edamontology.org/). These are local draft labels, not official EDAM terms.

## Ranked competencies

| Competency | Questions | Checkable outcome |
| --- | ---: | --- |
| Differential expression analysis | 8 | A complete gene-level contrast from a declared design, including normalization, effect estimates and statistical uncertainty. Shrinkage, filters and covariates are part of each question contract. |
| Measure branch-length properties of phylogenetic trees | 8 | Treeness, evolutionary rate, long-branch score, patristic distance or tree length using a pinned metric definition. |
| Measure signal and composition in sequence alignments | 5 | Per-alignment parsimony-informative counts/fractions or relative composition variability under declared site rules. |
| Test pathway overrepresentation | 5 | Gene universe, selected genes, pathway membership, contingency counts and enrichment statistics. |
| Compare independent groups with a rank-based test | 4 | Mann–Whitney U statistic and p-value with group order, sidedness and tie handling declared. |
| Estimate rank associations between biological measurements | 4 | Spearman coefficients or tests on matched observations, preserving ties and missingness policy. |
| Filter and summarize annotated cohort variants | 4 | Per-patient/gene burdens or annotation proportions with VAF, consequence and reference-call filters. |
| Compare measured colony morphology across conditions | 2 | Condition-level area/circularity summaries and contrasts; segmentation is not assumed. |
| Compute reference-length-normalized CpG densities | 2 | Unique-site counts divided by matched chromosome lengths, then chromosome-level comparisons. |
| Filter methylation observations while preserving row/site identity | 2 | Strict threshold masks, deduplicated site sets where requested, and retained/removed row counts. |
| Fit and interpret a binary logistic response model | 2 | Unpenalized or explicitly specified fits, outcome encoding, coefficient units and likelihood. |
| Multiple-testing correction | 2 | Adjusted values and decisions for a declared family of statistical tests, including comparisons between correction methods. |
| Assess how cohort or replicate exclusions change an analysis | 1 | Comparable reruns with changed inclusion sets and endpoint differences. |
| Calculate genome-wide read coverage with zero-depth bases | 1 | Depth sum and full reference-length denominator, with alignment/base inclusion rules. |
| Classify nucleotide substitutions and calculate Ts/Tv | 1 | Transition and transversion counts, allele policy and their ratio. |
| Identify complete shared single-copy orthologues | 1 | Per-proteome BUSCO completeness and copy status, then an intersection across proteomes. |
| Map reads to a reference with declared alignment settings | 1 | Reference-aligned reads with the requested mapper, reference and read-group metadata. |
| Transform expression data and interpret PCA variance | 1 | An oriented and transformed matrix, fitted components and explained-variance fractions. |

## Questions with proposed executable checks

These are prospective verifier designs. No new Harbor validation is claimed.

| Question | Short description | Competencies |
| --- | --- | --- |
| bix-6-q4 | Agreement between replicate CRISPR-screen p-values. | Estimate rank associations between biological measurements |
| bix-11-q1 | Difference in median treeness between fungi and animals. | Measure branch-length properties of phylogenetic trees |
| bix-11-q2 | Fraction of fungal gene trees above a treeness threshold. | Measure branch-length properties of phylogenetic trees |
| bix-12-q2 | Median fraction of informative alignment sites across fungal genes. | Measure signal and composition in sequence alignments |
| bix-12-q4 | Rank-test statistic comparing informative-site percentages across groups. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test |
| bix-12-q5 | Largest informative-site count among animal gene alignments. | Measure signal and composition in sequence alignments |
| bix-12-q6 | Rank-test statistic comparing raw informative-site counts across groups. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test |
| bix-14-q1 | Synonymous fraction among low-VAF coding variants in carriers. | Filter and summarize annotated cohort variants |
| bix-16-q1 | Gene with the strongest negative expression–essentiality association. | Estimate rank associations between biological measurements |
| bix-16-q3 | Count of genes with strong positive expression–essentiality association. | Estimate rank associations between biological measurements |
| bix-16-q4 | Fraction of genes with significant expression–essentiality rank association. | Multiple-testing correction; Estimate rank associations between biological measurements |
| bix-17-q2 | Median patient-level somatic-variant burden among carriers. | Filter and summarize annotated cohort variants |
| bix-18-q1 | Mean circularity of the genotype with the greatest mean colony area. | Compare measured colony morphology across conditions |
| bix-18-q3 | Percent reduction in mean colony area for a mutant relative to wild type. | Compare measured colony morphology across conditions |
| bix-20-q3 | Benign classification fraction among eligible annotated carrier variants. | Filter and summarize annotated cohort variants |
| bix-26-q3 | Selected-gene overlap with a KEGG pathway under iron depletion. | Differential expression analysis; Test pathway overrepresentation |
| bix-26-q5 | Pathways enriched under iron depletion but not in the comparison condition. | Differential expression analysis; Test pathway overrepresentation |
| bix-27-q5 | Variance explained by PC1 after a specified expression transformation. | Transform expression data and interpret PCA variance |
| bix-28-q3 | Median long-branch score for a specified fungal gene. | Measure branch-length properties of phylogenetic trees |
| bix-30-q3 | Ratio of significant miRNA counts under two multiple-testing corrections. | Multiple-testing correction |
| bix-34-q2 | Median patristic distance for a specified fungal gene. | Measure branch-length properties of phylogenetic trees |
| bix-34-q5 | Ratio of group medians of gene-level mean patristic distances. | Measure branch-length properties of phylogenetic trees |
| bix-35-q1 | Evolutionary rate of a specified animal orthologue. | Measure branch-length properties of phylogenetic trees |
| bix-35-q2 | Rank-test statistic comparing animal and fungal evolutionary rates. | Compare independent groups with a rank-based test; Measure branch-length properties of phylogenetic trees |
| bix-38-q1 | Fold change in median tree length between organism groups. | Measure branch-length properties of phylogenetic trees |
| bix-43-q2 | Reactome pathway enrichment odds ratio after a multi-group treatment contrast. | Differential expression analysis; Test pathway overrepresentation |
| bix-43-q4 | DEG/pathway membership overlap fraction after a treatment contrast. | Differential expression analysis; Test pathway overrepresentation |
| bix-45-q1 | Rank-test p-value comparing alignment composition variability. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test |
| bix-46-q4 | Differential-expression effect for a specified gene and mutant contrast. | Differential expression analysis |
| bix-47-q3 | Gene with the most non-reference variants in the oldest male carrier. | Filter and summarize annotated cohort variants |
| bix-49-q4 | Number of significant genes in a sex-adjusted disease contrast with effect shrinkage. | Differential expression analysis |
| bix-51-q2 | AIC of a treatment-response logistic model using only BMI. | Fit and interpret a binary logistic response model |
| bix-51-q8 | Age coefficient in a single-predictor treatment-response logistic model. | Fit and interpret a binary logistic response model |
| bix-52-q2 | Mean chromosome density of extreme-methylation, age-related CpG sites. | Compute reference-length-normalized CpG densities; Filter methylation observations while preserving row/site identity |
| bix-52-q6 | Chromosome with the greatest density of age-related CpG sites. | Compute reference-length-normalized CpG densities |
| bix-52-q7 | Number of methylation measurement rows removed by an extreme-value filter. | Filter methylation observations while preserving row/site identity |
| bix-53-q2 | Change in differential-gene count after excluding third replicates. | Differential expression analysis; Assess how cohort or replicate exclusions change an analysis |
| bix-53-q5 | Fraction of top enriched pathways with an oxidative-stress name match. | Differential expression analysis; Test pathway overrepresentation |
| bix-55-q1 | Complete single-copy orthologues shared by four proteomes. | Identify complete shared single-copy orthologues |
| bix-61-q2 | Whole-reference average depth after a specified bacterial read alignment. | Calculate genome-wide read coverage with zero-depth bases; Map reads to a reference with declared alignment settings |
| bix-61-q5 | Transition/transversion ratio in a bacterial sample. | Classify nucleotide substitutions and calculate Ts/Tv |

## Set aside for now

These questions have no competency assignment or count while their complete endpoint lacks an executable contract.

| Question | Short description | Reason |
| --- | --- | --- |
| bix-22-q1 | Immune cell type with the weakest gene-length/expression correlation. | Freeze span versus exon length, expression scale and eligibility; do not import Verified q4’s ≥10 filter into q1. |
| bix-22-q4 | Gene-length/expression correlation in expressed CD14 protein-coding genes. | Freeze gene-length definition and expression scale; the existing adaptation explicitly records its donor-CPM context. |
| bix-24-q2 | Whether up- or downregulated genes primarily explain metabolic pathway effects. | Primarily drives metabolic effects lacks a quantitative rule. Do not replace it with a label or claim full coverage without a new explicit contract. |
| bix-31-q2 | Sex-associated gene effect with batch adjustment and shrinkage. | The source combines batch-corrected counts and a batch covariate. Resolve count validity and double adjustment before authoring. |
| bix-32-q2 | Pathways enriched in the same direction across three mutant contrasts. | Use |LFC|>1.5 for genes. Freeze the otherwise underspecified significance rules and background; do not assign a gene LFC to a pathway. |
| bix-37-q1 | Tumor-to-normal abundance fold change for a protein. | Normalization, missing values and ratio-of-means versus paired ratios require an explicit contract. |
| bix-37-q4 | Log2 tumor-to-normal abundance fold change for a protein. | Freeze abundance scale and aggregation; mean log difference and log ratio of means can differ. |
| bix-41-q5 | Mixture whose measured colony phenotype is closest to the reference strain. | Closest is underspecified across two differently scaled measurements. Freeze scaling, distance and tie rule. |
| bix-54-q7 | Maximum colony area predicted by the selected nonlinear response model. | The question does not specify what best-fitting means. Freeze selection criterion, spline basis and observed frequency domain. |

## Boundaries to review

- Differential expression analysis is one competency here; shrinkage, design formula and filtering specify the task instance.
- Phylogenetic tree metrics currently share one competency; we could split it if the metrics require distinct assessment contracts.
- Spearman correlation and Mann-Whitney tests are counted as competencies. They might instead be cross-cutting statistical tags.
- Eligibility and denominator choices are required verifier checks, not separate competencies in this draft.
- A question requiring both differential expression and pathway enrichment counts toward both competencies. A generated task must validate the connected workflow.

Rebuild this Markdown and the HTML with `uv run python -m experiments.post_training.bio_tasks.coverage_site`.
