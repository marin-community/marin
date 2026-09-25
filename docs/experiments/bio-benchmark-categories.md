# Categories across ID benchmarks

3,572 candidate verifiable source records across 32 inventoried ID releases, out of 5,133 inventoried records. Excluded questions and unavailable instructions contribute no category counts.

Assignments are provisional requirements, not validated task coverage. Each question counts once per assigned category. Categories overlap; percentages need not sum to 100%. Releases are not deduplicated: overlapping benchmarks and shared protocols can inflate raw frequency. Release counts show breadth of representation, not the number of independent studies.

Use these counts to find candidate workflows for review, alongside scientific decisions, available observed inputs and an executable reward. They are not generation quotas.

## Analytical skills

| Category | Question records | % of eligible records | ID releases |
| --- | ---: | ---: | ---: |
| Biological database querying | 1200 | 33.6% | 16 |
| Model evaluation | 821 | 23.0% | 14 |
| Structured evidence extraction | 576 | 16.1% | 11 |
| Predictive regression | 389 | 10.9% | 12 |
| Statistical estimation | 388 | 10.9% | 21 |
| Mechanistic modeling and simulation | 352 | 9.9% | 2 |
| Evidence-based prioritization | 273 | 7.6% | 10 |
| Variant consequence annotation | 192 | 5.4% | 9 |
| Network analysis | 190 | 5.3% | 12 |
| Differential expression analysis | 187 | 5.2% | 17 |
| Cell identity annotation | 154 | 4.3% | 14 |
| Data preparation and reconciliation | 142 | 4.0% | 18 |
| Variant analysis | 138 | 3.9% | 15 |
| Scientific visualization | 101 | 2.8% | 8 |
| Enrichment analysis | 100 | 2.8% | 18 |
| Association analysis | 88 | 2.5% | 17 |
| Predictive classification | 85 | 2.4% | 7 |
| Chromatin accessibility and binding analysis | 80 | 2.2% | 11 |
| Clustering | 77 | 2.2% | 11 |
| Sequence and annotation analysis | 76 | 2.1% | 8 |
| Population composition analysis | 75 | 2.1% | 13 |
| Dimensionality reduction | 61 | 1.7% | 10 |
| Regression modeling | 58 | 1.6% | 10 |
| Spatial analysis | 57 | 1.6% | 13 |
| Molecular structure analysis | 53 | 1.5% | 7 |
| Normalization | 53 | 1.5% | 10 |
| Group comparison | 50 | 1.4% | 9 |
| Phylogenetic inference and tree analysis | 47 | 1.3% | 7 |
| Molecular representation and descriptors | 45 | 1.3% | 4 |
| Sequence alignment | 44 | 1.2% | 7 |
| Genomic interval analysis | 41 | 1.1% | 10 |
| Data integration | 40 | 1.1% | 9 |
| Sequencing quality and coverage analysis | 40 | 1.1% | 10 |
| Methylation analysis | 38 | 1.1% | 8 |
| Sequence construct analysis and design | 35 | 1.0% | 5 |
| Feature selection | 31 | 0.9% | 6 |
| Cell communication analysis | 30 | 0.8% | 6 |
| Multiple-testing correction | 29 | 0.8% | 5 |
| Image analysis | 28 | 0.8% | 13 |
| Trajectory and dynamics analysis | 27 | 0.8% | 9 |
| Data transformation | 25 | 0.7% | 8 |
| Orthology assessment | 23 | 0.6% | 8 |
| Dose–response and assay analysis | 21 | 0.6% | 5 |
| Differential abundance analysis | 18 | 0.5% | 6 |
| Alignment informativeness and composition | 17 | 0.5% | 3 |
| Expression quantification | 16 | 0.4% | 5 |
| Population genetic analysis | 16 | 0.4% | 6 |
| Mixture deconvolution | 11 | 0.3% | 3 |
| Survival analysis | 11 | 0.3% | 4 |
| Cell and feature quality control | 9 | 0.3% | 6 |
| Immune repertoire analysis | 9 | 0.3% | 4 |
| Signal and time-series analysis | 9 | 0.3% | 2 |
| Motif analysis | 8 | 0.2% | 6 |
| Taxonomic profiling | 7 | 0.2% | 2 |
| Sequence assembly and assessment | 5 | 0.1% | 4 |
| Sensitivity analysis | 3 | 0.1% | 3 |
| Model inspection | 2 | 0.1% | 1 |
| RNA structure analysis | 2 | 0.1% | 2 |
| Causal inference | 1 | 0.0% | 1 |

## Biological applications

| Category | Question records | % of eligible records | ID releases |
| --- | ---: | ---: | ---: |
| Biological knowledge resources | 1216 | 34.0% | 22 |
| Genomic variation | 1027 | 28.8% | 22 |
| Transcriptomics | 1002 | 28.1% | 27 |
| Genome and sequence analysis | 925 | 25.9% | 21 |
| Clinical and epidemiological analysis | 632 | 17.7% | 20 |
| Systems biology | 449 | 12.6% | 10 |
| Functional genomics | 415 | 11.6% | 21 |
| Chemical biology | 374 | 10.5% | 17 |
| Immunology | 237 | 6.6% | 22 |
| Epigenomics | 157 | 4.4% | 16 |
| Molecular structure | 125 | 3.5% | 14 |
| Molecular evolution | 103 | 2.9% | 11 |
| Biological imaging | 97 | 2.7% | 16 |
| Microbiology | 79 | 2.2% | 10 |
| Proteomics | 56 | 1.6% | 10 |
| Physiology | 47 | 1.3% | 8 |
| Ecology | 29 | 0.8% | 4 |
| Metabolomics | 21 | 0.6% | 6 |

## Inventory gaps

These ID sources have benchmark pages but no task manifest in this review. Their unknown task counts are not treated as zero skill demand.

- LABBench2
- BioSecBench-Surveillance
- BioSecBench-Function
- BioSecBench-Refusal — direct framing
- LifeSciBench
- TargetVal (Popper)
- ABC-Bench (SecureBio)
- ABLE — agentic biological AI tool use
- BioASQ — retrieval / end-to-end QA

Rebuild with `uv run python -m experiments.post_training.bio_tasks.coverage_site`.
