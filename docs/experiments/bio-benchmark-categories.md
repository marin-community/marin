# Categories across ID benchmarks

3,572 candidate verifiable source records across 32 inventoried ID releases, out of 5,133 inventoried records. Excluded questions and unavailable instructions contribute no category counts.

Assignments are provisional requirements, not validated task coverage. Each question counts once per assigned category. Categories overlap; percentages need not sum to 100%. Releases are not deduplicated: overlapping benchmarks and shared protocols can inflate raw frequency. Release counts show breadth of representation, not the number of independent studies.

Use these counts to find candidate workflows for review, alongside scientific decisions, available observed inputs and an executable reward. They are not generation quotas.

50 records have a question-level operation decomposition. 641 records have no operation annotations yet; other migrated operation lists can also be incomplete. 619 records have no resolved scientific context. 873 retain a broad context requiring a more specific review. All eligible records remain in the denominator; missing annotations are not zero demand.

## Operation families

| Category | Description | Question records | % of eligible records | ID releases |
| --- | --- | ---: | ---: | ---: |
| Information retrieval | Locate reference records or source material and extract specified facts with traceable provenance. | 1534 | 42.9% | 17 |
| Predictive modeling | Fit predictors, generate predictions or assess predictive performance under a declared evaluation policy. | 495 | 13.9% | 16 |
| Descriptive analysis | Compute counts, proportions, distribution summaries or other descriptive measurements. | 406 | 11.4% | 21 |
| Simulation | Compute model-generated outcomes under specified conditions and numerical conventions. | 352 | 9.9% | 2 |
| Data preparation | Clean, reconcile, normalize or transform inputs and prepare features for analysis. | 188 | 5.3% | 20 |
| Statistical inference | Estimate associations or effects, test hypotheses, correct multiple tests or quantify uncertainty. | 111 | 3.1% | 11 |
| Clustering and dimensionality reduction | Discover groups or lower-dimensional representations through clustering or dimensionality reduction. | 108 | 3.0% | 12 |
| Graph analysis | Calculate connectivity, paths, network structure or tree properties. | 65 | 1.8% | 11 |
| Spatial analysis | Analyze positions, distances, geometry, neighborhoods or spatial dependence. | 57 | 1.6% | 14 |
| Alignment and matching | Establish correspondence between ordered sequences, reference profiles or other representations. | 45 | 1.3% | 7 |
| Optimization | Find a solution maximizing or minimizing a declared objective under explicit constraints. | 1 | 0.0% | 1 |

## Scientific context

| Category | Description | Question records | % of eligible records | ID releases |
| --- | --- | ---: | ---: | ---: |
| Molecular function | Gene essentiality, genetic dependencies, functional annotations and protein activity. | 1014 | 28.4% | 22 |
| Genetic variation | Alleles, genotypes, inheritance and population variation. | 1003 | 28.1% | 21 |
| Gene expression | RNA abundance and expression differences, including targeted and transcriptome-wide measurements. | 794 | 22.2% | 26 |
| Disease and treatment response | Clinical phenotypes, disease progression, experimental or clinical treatment effects, drug sensitivity and resistance. | 520 | 14.6% | 20 |
| Nucleotide sequences | Nucleotide composition, coding regions, sequence annotations and constructs. | 390 | 10.9% | 19 |
| Small-molecule properties | Chemical structure, solubility, lipophilicity and related compound properties. | 145 | 4.1% | 5 |
| Evolutionary relationships | Homology, ancestry, divergence and conservation. | 87 | 2.4% | 9 |
| Molecular structure | RNA, protein and molecular-complex geometry. | 62 | 1.7% | 12 |
| Chromatin organization | Chromatin accessibility, occupancy and genomic contacts. | 51 | 1.4% | 10 |
| RNA processing | Splicing, maturation, stability and translation. | 50 | 1.4% | 4 |
| DNA methylation | Methylation measurements at sites and regions. | 39 | 1.1% | 9 |
| Molecular interactions | Protein interactions, binding affinity and ligand–receptor relationships. | 34 | 1.0% | 8 |
| Protein abundance | Protein quantities and abundance differences. | 34 | 1.0% | 10 |
| Morphology | Cell, colony and tissue shape, size and structural features. An image alone does not establish this context. | 32 | 0.9% | 8 |
| Ecology | Community composition, diversity, population abundance and population dynamics. | 26 | 0.7% | 2 |
| Metabolism | Metabolite abundance, biochemical reactions and metabolic flux. | 21 | 0.6% | 6 |
| Physiology | Functional measurements such as neural signals, breathing and growth. | 17 | 0.5% | 7 |
| Immune repertoires | Receptor sequences, clonotypes and repertoire diversity. Other immune-cell analyses require their measured context. | 9 | 0.3% | 4 |
| Protein modifications | Phosphorylation and other post-translational modifications. | 0 | 0.0% | 0 |

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
