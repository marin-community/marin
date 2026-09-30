# GitHub topics: September 29, 2026

[Discovery inventory](index.md) · [Structured data](inventory.json)

Download the [repository topic lists](data/repository-topics-2026-09-29.csv)
or the [complete topic frequencies](data/topic-frequencies-2026-09-29.csv).

Topics were collected after selection, without expanding the 95 candidates.
All 83 verified GitHub repositories were queried on September 29, 2026:
53 have topics, 30 have none, and their 341 assignments contain 214 distinct
topic strings. The remaining 12 candidates have no verified GitHub mapping in
this pass. The JSON contains every repository's exact tags, the full frequency
table, source query and response hash in `github_topic_analysis`.

The most frequent tags are `bioinformatics` (32 repositories), `genomics` (12),
`python` (10), `ngs` (6), and `bioconductor`, `bioconductor-package` and
`sequence-alignment` (5 each). `computational-biology` and `biology` appear
only once each. Counts describe tagging within this cohort, not the total
number of GitHub repositories carrying each topic.

These observed topic families provide search ideas. Each row's count is the
union of repositories carrying any listed tag, counted once within that row.
Rows overlap and are not a biological diversity classification.

| Search idea | Observed topic strings | Repositories in this cohort |
| --- | --- | ---: |
| Broad biology searches | `bioinformatics`, `genomics`, `computational-biology`, `biology` | 33 |
| Transcriptomics and single-cell data | `rna-seq`, `rnaseq`, `scrna-seq`, `single-cell-rna-seq`, `single-cell`, `single-cell-genomics`, `transcriptomics`, `gene-expression`, `transcriptome` | 7 |
| Microbial and community sequence analysis | `metagenomics`, `microbiome`, `amplicon`, `metabarcoding`, `taxonomy`, `taxonomic-classification`, `taxonomic-profiling`, `metagenome-assembly`, `bacterial-genomes` | 6 |
| Evolution and phylogenetic trees | `phylogenetics`, `phylogenetic-trees`, `evolution`, `comparative-genomics` | 4 |
| Chromatin assays | `chip-seq`, `atac-seq`, `dnase-seq`, `peak-caller` | 2 |
| Proteomics and protein structure | `proteomics`, `protein-identification`, `protein-structure`, `protein-structure-alignment` | 4 |
| Assembly and genome annotation | `genome-assembly`, `genome-assembly-evaluation`, `genome-annotation`, `gene-finding`, `transcriptome-assembly` | 3 |
| Functional enrichment | `enrichment-analysis`, `gene-set-enrichment`, `pathway-enrichment-analysis`, `gsea` | 2 |

The broad biology row reaches only 33 of the 83 GitHub repositories. SAMtools,
DESeq2, STAR, BEDTools and PLINK have no topics. Topics therefore provide
additional discovery paths while package metadata and documentation remain
necessary. Absence of a topic does not establish absence of a scientific use.

Preserve aliases when planning searches: `rna-seq` and `rnaseq` are distinct
strings, and pyBigWig uses the misspelling `bioinfomatics`. Language tags such
as `python`, file formats such as `fastq`, and generic tags such as `pipeline`
need a biological qualifier or subsequent relevance review. Starting from
these observed tags also inherits the current inventory's coverage bias.
