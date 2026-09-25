# BixBench-Verified: flat competency review

Draft for review. All 50 source questions are annotated; the 31 competency definitions are provisional. Task generation remains paused. Only executable rewards are eligible; no LLM judge.

An **analysis competency** is a reusable operation with an observable outcome. A **workflow recipe** connects several competencies to answer a scientific question. Generate connected workflows on independent real inputs, and use competencies to track breadth. Source questions may ask for just one endpoint from a shared workflow.

Focal labels describe the requested endpoint; supporting labels describe necessary components. Count a question once per label. Counts overlap and do not measure unique studies, independent workflows, scientific importance or validated generated tasks.

[Versioned annotations and verification notes](../../experiments/post_training/bio_tasks/benchmark_competencies/bixbench-verified-50.json) · [Source inventory](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench-verified-50.json)

The naming is informed by [EDAM's separation of operations, topics, data and formats](https://edamontology.org/). These are local draft labels, not official EDAM terms. The [ISCB framework](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae166/7903279) addresses broader professional competencies; our units are narrower, observable analysis operations.

## Ranked competency list

Total includes focal and supporting appearances. Generic supporting operations can dominate; inspect focal counts before deciding generation priorities. No weighting policy is selected.

| Competency | Focal questions | Total questions | Observable outcome |
| --- | ---: | ---: | --- |
| Define eligible observations and denominators | 0 | 26 | An inclusion mask and denominator that preserve the intended patient, gene, variant, site or measurement unit. |
| Summarize and compare grouped measurements | 11 | 14 | Means, medians, extrema, ratios or differences with explicit grouping and order of aggregation. |
| Estimate design-aware differential gene expression | 10 | 11 | Count-model contrasts with sample design, normalization, dispersion, effect estimates and uncertainty. |
| Adjust and compare multiple-testing procedures | 2 | 9 | A declared test family, adjusted p-values and rejection counts under the named procedure. |
| Measure branch-length properties of phylogenetic trees | 8 | 8 | Treeness, evolutionary rate, long-branch score, patristic distance or tree length using a pinned metric definition. |
| Compare direction-aware gene or pathway sets | 6 | 7 | Intersections, differences, membership counts or ranked-set subsets with explicit direction and significance rules. |
| Test pathway overrepresentation | 6 | 7 | Gene universe, selected genes, pathway membership, contingency counts and enrichment statistics. |
| Reconcile biological feature and sample identities | 0 | 6 | An auditable join across assays, annotations or replicates with duplicate and missing-identity policies. |
| Measure signal and composition in sequence alignments | 5 | 5 | Per-alignment parsimony-informative counts/fractions or relative composition variability under declared site rules. |
| Compare independent groups with a rank-based test | 4 | 4 | Mann–Whitney U statistic and p-value with group order, sidedness and tie handling declared. |
| Compare measured colony morphology across conditions | 3 | 4 | Condition-level area/circularity summaries and contrasts; segmentation is not assumed. |
| Estimate rank associations between biological measurements | 4 | 4 | Spearman coefficients or tests on matched observations, preserving ties and missingness policy. |
| Filter and summarize annotated cohort variants | 4 | 4 | Per-patient/gene burdens or annotation proportions with VAF, consequence and reference-call filters. |
| Shrink differential-expression effect estimates | 4 | 4 | Shrunken log-fold changes with estimator and coefficient specified; significance tests kept distinct. |
| Aggregate expression across the declared samples | 2 | 2 | Per-gene sample means on a stated scale and count-based eligibility filters. |
| Calculate protein abundance changes on the correct scale | 2 | 2 | Tumor/control abundance summaries, fold changes and log2 changes with normalization and aggregation stated. |
| Compare statistical models using a declared criterion | 2 | 2 | Likelihood, parameter count, AIC or another prespecified selection criterion. |
| Compute reference-length-normalized CpG densities | 2 | 2 | Unique-site counts divided by matched chromosome lengths, then chromosome-level comparisons. |
| Derive gene lengths from a declared annotation | 2 | 2 | Protein-coding gene identities and lengths under a frozen span/exon and transcript-selection rule. |
| Estimate linear associations between biological measurements | 2 | 2 | Pearson correlations on the declared feature population and measurement scale. |
| Filter methylation observations while preserving row/site identity | 2 | 2 | Strict threshold masks, deduplicated site sets where requested, and retained/removed row counts. |
| Fit and interpret a binary logistic response model | 2 | 2 | Unpenalized or explicitly specified fits, outcome encoding, coefficient units and likelihood. |
| Assess how cohort or replicate exclusions change an analysis | 1 | 1 | Comparable reruns with changed inclusion sets and endpoint differences. |
| Calculate genome-wide read coverage with zero-depth bases | 1 | 1 | Depth sum and full reference-length denominator, with alignment/base inclusion rules. |
| Classify nucleotide substitutions and calculate Ts/Tv | 1 | 1 | Transition and transversion counts, allele policy and their ratio. |
| Find a bounded optimum of a fitted response | 1 | 1 | An optimum over the allowed biological domain, including boundary checks. |
| Fit nonlinear biological response curves | 1 | 1 | Fitted polynomial or spline response functions under explicit design and basis choices. |
| Identify complete shared single-copy orthologues | 1 | 1 | Per-proteome BUSCO completeness and copy status, then an intersection across proteomes. |
| Interpret directional pathway evidence in biological context | 1 | 1 | An evidence-backed biological claim; an unrestricted narrative is outside the current reward scope. |
| Map reads to a reference with declared alignment settings | 1 | 1 | Reference-aligned reads with the requested mapper, reference and read-group metadata. |
| Transform expression data and interpret PCA variance | 1 | 1 | An oriented and transformed matrix, fitted components and explained-variance fractions. |

## Question-by-question draft

Summaries are paraphrases checked against the pinned Verified question hashes. An executable check is only proposed, not validated. `needs-definition` requires a frozen scientific contract. `defer-interpretation` excludes the open-ended endpoint.

| Question | Summary | Focal competencies | Supporting competencies | Reward disposition |
| --- | --- | --- | --- | --- |
| bix-6-q4 | Agreement between replicate CRISPR-screen p-values. | Estimate rank associations between biological measurements | Reconcile biological feature and sample identities | numeric-contract |
| bix-11-q1 | Difference in median treeness between fungi and animals. | Measure branch-length properties of phylogenetic trees; Summarize and compare grouped measurements | — | numeric-contract |
| bix-11-q2 | Fraction of fungal gene trees above a treeness threshold. | Measure branch-length properties of phylogenetic trees | Define eligible observations and denominators | numeric-contract |
| bix-12-q2 | Median fraction of informative alignment sites across fungal genes. | Measure signal and composition in sequence alignments; Summarize and compare grouped measurements | — | numeric-contract |
| bix-12-q4 | Rank-test statistic comparing informative-site percentages across groups. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test | — | numeric-contract |
| bix-12-q5 | Largest informative-site count among animal gene alignments. | Measure signal and composition in sequence alignments; Summarize and compare grouped measurements | — | numeric-contract |
| bix-12-q6 | Rank-test statistic comparing raw informative-site counts across groups. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test | — | numeric-contract |
| bix-14-q1 | Synonymous fraction among low-VAF coding variants in carriers. | Filter and summarize annotated cohort variants | Define eligible observations and denominators | numeric-contract |
| bix-16-q1 | Gene with the strongest negative expression–essentiality association. | Estimate rank associations between biological measurements | Reconcile biological feature and sample identities | numeric-contract |
| bix-16-q3 | Count of genes with strong positive expression–essentiality association. | Estimate rank associations between biological measurements | Reconcile biological feature and sample identities; Define eligible observations and denominators | numeric-contract |
| bix-16-q4 | Fraction of genes with significant expression–essentiality rank association. | Estimate rank associations between biological measurements; Adjust and compare multiple-testing procedures | Reconcile biological feature and sample identities; Define eligible observations and denominators | numeric-contract |
| bix-17-q2 | Median patient-level somatic-variant burden among carriers. | Filter and summarize annotated cohort variants; Summarize and compare grouped measurements | Define eligible observations and denominators | numeric-contract |
| bix-18-q1 | Mean circularity of the genotype with the greatest mean colony area. | Compare measured colony morphology across conditions; Summarize and compare grouped measurements | — | numeric-contract |
| bix-18-q3 | Percent reduction in mean colony area for a mutant relative to wild type. | Compare measured colony morphology across conditions; Summarize and compare grouped measurements | — | numeric-contract |
| bix-20-q3 | Benign classification fraction among eligible annotated carrier variants. | Filter and summarize annotated cohort variants | Define eligible observations and denominators | numeric-contract |
| bix-22-q1 | Immune cell type with the weakest gene-length/expression correlation. | Estimate linear associations between biological measurements; Derive gene lengths from a declared annotation; Aggregate expression across the declared samples | Reconcile biological feature and sample identities; Define eligible observations and denominators | needs-definition |
| bix-22-q4 | Gene-length/expression correlation in expressed CD14 protein-coding genes. | Estimate linear associations between biological measurements; Derive gene lengths from a declared annotation; Aggregate expression across the declared samples | Reconcile biological feature and sample identities; Define eligible observations and denominators | needs-definition |
| bix-24-q2 | Whether up- or downregulated genes primarily explain metabolic pathway effects. | Interpret directional pathway evidence in biological context | Estimate design-aware differential gene expression; Test pathway overrepresentation; Compare direction-aware gene or pathway sets; Define eligible observations and denominators; Adjust and compare multiple-testing procedures | defer-interpretation |
| bix-26-q3 | Selected-gene overlap with a KEGG pathway under iron depletion. | Estimate design-aware differential gene expression; Test pathway overrepresentation; Compare direction-aware gene or pathway sets | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | numeric-contract |
| bix-26-q5 | Pathways enriched under iron depletion but not in the comparison condition. | Estimate design-aware differential gene expression; Test pathway overrepresentation; Compare direction-aware gene or pathway sets | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | numeric-contract |
| bix-27-q5 | Variance explained by PC1 after a specified expression transformation. | Transform expression data and interpret PCA variance | — | numeric-contract |
| bix-28-q3 | Median long-branch score for a specified fungal gene. | Measure branch-length properties of phylogenetic trees; Summarize and compare grouped measurements | — | numeric-contract |
| bix-30-q3 | Ratio of significant miRNA counts under two multiple-testing corrections. | Adjust and compare multiple-testing procedures | Define eligible observations and denominators | numeric-contract |
| bix-31-q2 | Sex-associated gene effect with batch adjustment and shrinkage. | Estimate design-aware differential gene expression; Shrink differential-expression effect estimates | Define eligible observations and denominators | needs-definition |
| bix-32-q2 | Pathways enriched in the same direction across three mutant contrasts. | Estimate design-aware differential gene expression; Test pathway overrepresentation; Compare direction-aware gene or pathway sets | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | needs-definition |
| bix-34-q2 | Median patristic distance for a specified fungal gene. | Measure branch-length properties of phylogenetic trees; Summarize and compare grouped measurements | — | numeric-contract |
| bix-34-q5 | Ratio of group medians of gene-level mean patristic distances. | Measure branch-length properties of phylogenetic trees; Summarize and compare grouped measurements | — | numeric-contract |
| bix-35-q1 | Evolutionary rate of a specified animal orthologue. | Measure branch-length properties of phylogenetic trees | — | numeric-contract |
| bix-35-q2 | Rank-test statistic comparing animal and fungal evolutionary rates. | Measure branch-length properties of phylogenetic trees; Compare independent groups with a rank-based test | — | numeric-contract |
| bix-37-q1 | Tumor-to-normal abundance fold change for a protein. | Calculate protein abundance changes on the correct scale | Summarize and compare grouped measurements | needs-definition |
| bix-37-q4 | Log2 tumor-to-normal abundance fold change for a protein. | Calculate protein abundance changes on the correct scale | Summarize and compare grouped measurements | needs-definition |
| bix-38-q1 | Fold change in median tree length between organism groups. | Measure branch-length properties of phylogenetic trees; Summarize and compare grouped measurements | — | numeric-contract |
| bix-41-q5 | Mixture whose measured colony phenotype is closest to the reference strain. | Compare measured colony morphology across conditions | Summarize and compare grouped measurements | needs-definition |
| bix-43-q2 | Reactome pathway enrichment odds ratio after a multi-group treatment contrast. | Estimate design-aware differential gene expression; Test pathway overrepresentation | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | numeric-contract |
| bix-43-q4 | DEG/pathway membership overlap fraction after a treatment contrast. | Estimate design-aware differential gene expression; Test pathway overrepresentation; Compare direction-aware gene or pathway sets | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | numeric-contract |
| bix-45-q1 | Rank-test p-value comparing alignment composition variability. | Measure signal and composition in sequence alignments; Compare independent groups with a rank-based test | — | numeric-contract |
| bix-46-q4 | Differential-expression effect for a specified gene and mutant contrast. | Estimate design-aware differential gene expression | — | numeric-contract |
| bix-47-q3 | Gene with the most non-reference variants in the oldest male carrier. | Filter and summarize annotated cohort variants | Define eligible observations and denominators | numeric-contract |
| bix-49-q4 | Number of significant genes in a sex-adjusted disease contrast with effect shrinkage. | Estimate design-aware differential gene expression; Shrink differential-expression effect estimates | Define eligible observations and denominators; Adjust and compare multiple-testing procedures | numeric-contract |
| bix-51-q2 | AIC of a treatment-response logistic model using only BMI. | Fit and interpret a binary logistic response model; Compare statistical models using a declared criterion | Define eligible observations and denominators | numeric-contract |
| bix-51-q8 | Age coefficient in a single-predictor treatment-response logistic model. | Fit and interpret a binary logistic response model | Define eligible observations and denominators | numeric-contract |
| bix-52-q2 | Mean chromosome density of extreme-methylation, age-related CpG sites. | Filter methylation observations while preserving row/site identity; Compute reference-length-normalized CpG densities; Summarize and compare grouped measurements | Define eligible observations and denominators | numeric-contract |
| bix-52-q6 | Chromosome with the greatest density of age-related CpG sites. | Compute reference-length-normalized CpG densities | Define eligible observations and denominators | numeric-contract |
| bix-52-q7 | Number of methylation measurement rows removed by an extreme-value filter. | Filter methylation observations while preserving row/site identity | Define eligible observations and denominators | numeric-contract |
| bix-53-q2 | Change in differential-gene count after excluding third replicates. | Estimate design-aware differential gene expression; Shrink differential-expression effect estimates; Assess how cohort or replicate exclusions change an analysis | Define eligible observations and denominators | numeric-contract |
| bix-53-q5 | Fraction of top enriched pathways with an oxidative-stress name match. | Estimate design-aware differential gene expression; Shrink differential-expression effect estimates; Test pathway overrepresentation; Compare direction-aware gene or pathway sets | Define eligible observations and denominators | numeric-contract |
| bix-54-q7 | Maximum colony area predicted by the selected nonlinear response model. | Fit nonlinear biological response curves; Compare statistical models using a declared criterion; Find a bounded optimum of a fitted response | Compare measured colony morphology across conditions | needs-definition |
| bix-55-q1 | Complete single-copy orthologues shared by four proteomes. | Identify complete shared single-copy orthologues; Compare direction-aware gene or pathway sets | — | numeric-contract |
| bix-61-q2 | Whole-reference average depth after a specified bacterial read alignment. | Map reads to a reference with declared alignment settings; Calculate genome-wide read coverage with zero-depth bases | — | numeric-contract |
| bix-61-q5 | Transition/transversion ratio in a bacterial sample. | Classify nucleotide substitutions and calculate Ts/Tv | Define eligible observations and denominators | numeric-contract |

Review granularity first: merge labels that would lead to the same assessment, and split labels when they require different scientific decisions or output checks. Choose a connected workflow, freeze its artifact contract, then vary study, organism, assay and design using independent observed data. New instances must add substantive biological variation; changing labels alone adds no coverage.

Rebuild this Markdown and the HTML with `uv run python -m experiments.post_training.bio_tasks.coverage_site`.
