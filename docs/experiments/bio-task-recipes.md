# Implemented biology task recipes

The recipes below generate three examples each for a single train split. They cover scientific
operations inspired by the inspected source repositories plus additional computational biology domains.
Repository mappings describe the source of an operation; actual CLI/API execution is tracked separately
in the [catalog](bio-task-catalog.md#repository-execution-coverage). See [build instructions](bio-tasks.md).

All instances have independently checked references, input-reading solvers, and scientific negative
controls. Host checks do not establish container execution or scientific approval. The supplied format
profiles are bounded: text intermediates do not count as native BAM, H5AD, SRA, or OME-TIFF coverage.

Real-data recipes retain full matrices, structures and genomes or explicitly bounded observed read subsets, with biological lineage; the remaining recipes are simulated
correctness controls. See [provenance and limitations](bio-task-catalog.md#id-workflow-coverage-and-input-realism).

## Sequence

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-genome-cds-extraction` | medium | FASTA, GFF3 | strand, compound-cds, circular-coordinates | Biopython, Biostrings, pybedtools |
| `real-genome-translation` | medium | FASTA, GFF3 | compound-cds, genetic-code, alternative-start-codons | Biopython, Biostrings |
| `real-genome-gc3` | medium | FASTA, GFF3 | reading-frame, coding-composition, denominators | Biopython, Biostrings |
| `real-genome-codon-counts` | medium | FASTA, GFF3 | reading-frame, overlapping-genes, stop-codon-exclusion | Biopython, Biostrings |
| `real-genome-overlap` | medium | FASTA, GFF3 | interval-union, overlapping-genes, coordinate-conventions | BEDTools, GenomicRanges, pybedtools |
| `real-genome-promoters` | medium | FASTA, GFF3 | strand, circular-coordinates, boundary-clipping | BEDTools, HTSlib, Biopython, GenomicRanges, Biostrings, pybedtools |
| `real-genome-restriction-digest` | medium | FASTA, GFF3 | restriction-sites, circular-coordinates, fragment-lengths | Biopython |
| `strand-extraction` | easy | fasta-dna, gff3-single-exon | coordinates, strand, sequence-extraction, feature-parent-joins, sequence-identifiers | Biopython, pybedtools |
| `fasta-iupac-gc` | easy | fasta | ambiguity, wrapped-fasta, denominators | Additional domain coverage |
| `fasta-six-frame-translation` | easy | fasta | reading-frames, reverse-complement, genetic-code | Biostrings |
| `fasta-orf-selection` | easy | fasta | orf-selection, stop-codons, coordinates | Additional domain coverage |
| `fasta-restriction-fragments` | easy | fasta | restriction-sites, linear-boundaries | Additional domain coverage |
| `fasta-motif-hits` | easy | fasta | overlapping-motifs, strand-coordinates | Biostrings |
| `fasta-kmer-jaccard` | easy | fasta | canonical-kmers, set-similarity, ambiguous-bases | Additional domain coverage |
| `protein-molecular-mass` | easy | fasta, csv-header | residue-masses, terminal-water, units | Additional domain coverage |
| `protein-tryptic-digest` | easy | fasta, csv-header | enzyme-specificity, proline-exception, peptide-coordinates | Additional domain coverage |
| `protein-charge` | easy | fasta, csv-header | ionization, termini, ph | Additional domain coverage |
| `protein-hydropathy-windows` | easy | fasta, csv-header | sliding-windows, hydropathy, tie-breaking | Additional domain coverage |
| `dna-unique-mapping` | medium | fasta | reference-mapping, orientation, unmapped-queries | BLAST+, BWA, Bowtie 2, minimap2 |
| `protein-local-search` | medium | fasta | query-coverage, local-search, identity | BLAST+, DIAMOND |
| `alignment-sum-of-pairs` | medium | aligned-fasta | alignment-scoring, gap-policy, residue-correspondence | MAFFT, MUSCLE |
| `hmmer-domain-extraction` | medium | fasta, hmmer-domtblout | domtblout, alignment-versus-envelope, domain-identities | HMMER |
| `sequence-identity-clusters` | medium | fasta | sequence-clustering, transitive-membership, coverage-policy | MMseqs2 |
| `fasta-indexed-regions` | medium | fasta, fai, csv-header | fasta-index, line-wrapping, coordinate-conversion | HTSlib |

## Genomic intervals

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `interval-overlap` | easy | bed4, text-integer | coordinates, overlap, feature-identity | GenomicRanges, pybedtools |
| `bed12-exons` | medium | bed12 | block-offsets, strand-aware-exon-order | BEDTools, UCSC Kent utilities |
| `gtf-splicing` | medium | gtf-exons, fasta-dna | splicing, transcript-joins, strand | Biopython, Ensembl VEP |
| `gff-cds-translation` | medium | gff3-cds, fasta-dna | cds-phase, genetic-code, strand | Biopython, Biostrings |
| `bed-union-coverage` | medium | bed4, chrom-sizes | interval-union, coverage-denominators | BEDTools |
| `bed-nearest-features` | medium | bed4 | nearest-feature, coordinate-distance, ties | GenomicRanges |
| `bed-stranded-promoters` | medium | bed6, chrom-sizes | tss, strand, boundary-clipping | GenomicRanges |
| `bed-complement` | medium | bed4, chrom-sizes | interval-complement, chromosome-boundaries | BEDTools |
| `bedgraph-weighted-signal` | medium | bedgraph, bed4 | weighted-signal, uncovered-bases | deepTools, UCSC Kent utilities |
| `bedgraph-threshold-peaks` | medium | bedgraph | signal-threshold, peak-merging, boundary-semantics | MACS2 / MACS3 |

## Expression

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-rnaseq-library-qc` | medium | gene-count-tsv, sample-metadata-tsv | sample-identifiers, biological-replicates, raw-counts, library-normalization | edgeR |
| `real-rnaseq-cpm-filter` | medium | gene-count-tsv, sample-metadata-tsv | sample-identifiers, biological-replicates, raw-counts, library-normalization | edgeR |
| `real-rnaseq-size-factors` | medium | gene-count-tsv, sample-metadata-tsv | sample-identifiers, biological-replicates, raw-counts, library-normalization | DESeq2 |
| `real-rnaseq-normalized-contrast` | medium | gene-count-tsv, sample-metadata-tsv | sample-identifiers, biological-replicates, raw-counts, library-normalization | DESeq2 |
| `donor-counts` | medium | csv-header | sample-joins, raw-counts, biological-replication | Scanpy |
| `cell-fractions` | medium | csv-header | sample-joins, cohort-selection, denominators | Scanpy |
| `transcript-tpm` | medium | csv-header | transcript-joins, abundance-units, decoys | Salmon, kallisto |
| `matrixmarket-cell-qc` | medium | matrix-market-coordinate-integer, 10x-features-tsv, 10x-barcodes-tsv | sparse-matrix, feature-types, mitochondrial-fraction | STAR, Scanpy |
| `matrixmarket-log-normalization` | medium | matrix-market-coordinate-integer, 10x-features-tsv, 10x-barcodes-tsv | library-normalization, implicit-zeros, feature-identifiers | Seurat |
| `matrixmarket-feature-filtering` | medium | matrix-market-coordinate-integer, 10x-features-tsv, 10x-barcodes-tsv | detection-threshold, matrix-orientation, duplicate-symbols | Seurat |
| `splice-psi` | medium | csv-header | splice-junctions, isoform-proportion, zero-support | Additional domain coverage |
| `bulk-size-factors` | medium | csv-header | geometric-means, median-ratios, zero-counts | DESeq2 |
| `bulk-cpm-filter` | medium | csv-header | library-size, cpm, inclusive-threshold | edgeR |
| `differential-expression-bh` | medium | csv-header | multiple-testing, effect-direction, missing-pvalues | Additional domain coverage |
| `gtf-coverage-counts` | medium | gtf | coverage-to-counts, inclusive-exon-length, estimated-counts | StringTie |

## Sequencing reads

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-fastq-pair-filter` | medium | FASTQ | paired-read-qc, threshold-boundaries, native-fastq-output | fastp |
| `real-fastq-fixed-trim` | easy | FASTQ | paired-reads, sequence-quality-synchronization, native-fastq-output | cutadapt, fastp |
| `real-fastq-cycle-quality` | easy | FASTQ | phred-encoding, sequencing-cycles, quality-denominators | FastQC, MultiQC, fastp |
| `real-fastq-expected-errors` | medium | FASTQ | phred-probabilities, paired-read-qc, nonlinear-aggregation | Additional domain coverage |
| `real-fastq-quality-yield` | easy | FASTQ | phred-encoding, read-pair-denominators, quality-yield | Picard, FastQC, MultiQC, fastp |
| `paired-read-qc` | medium | fastq-phred33, json | read-quality, mate-identity, thresholds | FastQC, fastp |
| `sam-inclusion` | medium | sam1.6 | sam-flags, mapping-quality, unknown-quality | SAMtools, pysam |
| `sam-fragment-counts` | medium | sam1.6 | mate-identity, fragment-filtering, primary-alignments | Picard, deepTools |
| `sam-allele-pileup` | medium | sam1.6, bed4 | cigar-replay, base-quality, allele-counts | pysam |
| `sam-cigar-coverage` | medium | sam1.6, bed4 | cigar-consumption, depth, covered-bases | SAMtools, deepTools |
| `sam-junction-support` | medium | sam1.6 | splicing, junction-coordinates, read-deduplication | STAR |
| `fastq-adapter-trimming` | medium | fastq-phred33 | adapter-matching, partial-adapters, quality-synchronization | cutadapt, fastp |
| `fastq-quality-trimming` | medium | fastq-phred33 | three-prime-trimming, quality-thresholds, empty-reads | cutadapt |
| `umi-deduplication` | medium | csv-header | umi-identity, whitelists, molecule-counting | STAR |
| `paf-query-coverage` | medium | paf, csv-header | split-alignments, query-span-union, paf-strand | minimap2 |
| `sam-pair-concordance` | medium | sam1.6 | mate-identity, fragment-orientation, discordance | Bowtie 2, Picard |
| `fastqc-report-reconciliation` | medium | fastqc-data-text, fastq-phred33, csv-header | qc-aggregation, sample-identity, raw-report-reconciliation | FastQC, MultiQC |
| `sra-spot-export` | medium | csv-spot-ledger | spot-read-types, mate-routing, orphan-export | SRA Toolkit |

## Variants

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `genotype-alleles` | medium | vcf4.3, csv-header | ploidy, missing-calls, allele-denominators | VCFtools |
| `vcf-allelic-depth` | medium | vcf4.3 | format-fields, allele-indexing, missing-denominators | BCFtools |
| `vcf-site-filtering` | medium | vcf4.3 | filter-semantics, variant-types, missing-quality | GATK |
| `vcf-genotype-masking` | medium | vcf4.3 | genotype-quality, phasing, missingness | GATK |
| `vcf-multiallelic-splitting` | medium | vcf4.3 | number-a, number-r, genotype-recoding | BCFtools |
| `vcf-minimal-representation` | medium | vcf4.3 | allele-normalization, anchor-bases, vcf-coordinates | BCFtools |
| `variant-coding-consequences` | medium | vcf4.3, gff3, fasta | strand, coding-consequences, genetic-code | SnpEff, Ensembl VEP |
| `genotype-hwe` | medium | csv-header | allele-frequencies, equilibrium, expected-counts | PLINK / PLINK 2 |
| `vcf-sample-qc` | medium | vcf4.3 | sample-missingness, partial-genotypes, inclusive-cutoffs | PLINK / PLINK 2, VCFtools |

## Phylogeny

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-protein-alignment` | medium | FASTA, TSV, JSON | protein multiple-sequence alignment, affine-gap scoring, paralog awareness | MAFFT, MUSCLE |
| `alignment-sites` | easy | fasta-alignment, csv-header | site-states, missing-bases, parsimony-informative-sites | Additional domain coverage |
| `tree-branches` | easy | csv-header | root-conventions, branch-lengths, treeness | Additional domain coverage |
| `newick-distances` | medium | newick | newick, patristic-distance, shared-ancestry | FastTree |
| `newick-monophyly` | medium | newick | rooted-clades, mrca, missing-taxa | FastTree |
| `newick-pruning` | medium | newick | tree-pruning, branch-length-preservation | Additional domain coverage |
| `alignment-consensus` | medium | aligned-fasta | iupac, majority-consensus, ties | MUSCLE |
| `alignment-p-distance` | medium | aligned-fasta | pairwise-deletion, sequence-distance | Additional domain coverage |
| `alignment-column-filter` | medium | aligned-fasta | alignment-mask, gaps, ambiguity | MAFFT |
| `alignment-partitions` | medium | nexus-sets, aligned-fasta | nexus-charsets, codon-stride, partition-validation | IQ-TREE |
| `newick-split-support` | medium | newick, csv-header | unrooted-splits, replicate-support, root-invariance | FastTree, RAxML |

## Assembly and ecology

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `busco-summary` | medium | csv-header | ortholog-identity, deduplication, completeness-categories | BUSCO |
| `taxonomy-counts` | medium | csv-header | taxonomy, hierarchical-counts, unclassified-denominator | Kraken 2 |
| `assembly-nx` | medium | fasta | length-weighted-contiguity, genome-size, unreached-thresholds | SPAdes |
| `assembly-gap-runs` | medium | fasta | gap-runs, case-insensitivity, half-open-coordinates | SPAdes |
| `contig-depth-breadth` | medium | fasta, bedgraph | depth, coverage-breadth, zero-regions | Additional domain coverage |
| `taxonomic-lca` | medium | ncbi-taxdump-nodes-profile, csv-header | taxonomy-tree, common-ancestor, unclassified-hits | Kraken 2 |
| `bray-curtis` | medium | csv-header | community-dissimilarity, library-normalization, empty-samples | Additional domain coverage |
| `alpha-diversity` | medium | csv-header | richness, entropy, effective-species | Additional domain coverage |

## Imaging and spatial

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `image-measurements` | medium | json | object-identity, pixel-spacing, image-measurement | Additional domain coverage |
| `image-threshold-components` | medium | pgm-p2, json-metadata | connected-components, pixel-centers, physical-area | Additional domain coverage |
| `image-background-correction` | medium | pgm-p2 | background-estimation, label-masks, signed-intensities | Additional domain coverage |
| `image-colocalization` | medium | pgm-p2 | regionwise-correlation, channel-pairing, centering | Additional domain coverage |
| `spatial-neighbor-enrichment` | medium | csv-header | physical-neighbors, mixing-fractions, self-exclusion | Additional domain coverage |
| `spatial-region-counts` | medium | csv-header | spatial-boundaries, type-counts, zero-combinations | Additional domain coverage |
| `image-dice-iou` | medium | pgm-p2 | segmentation-overlap, empty-masks, foreground-denominator | Additional domain coverage |

## Statistics

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-clinical-kaplan-meier` | medium | clinical TSV, survival TSV | patient joins, randomized cohort, censoring, tied events | Additional domain coverage |
| `real-clinical-adjusted-cox` | medium | clinical TSV, survival TSV | complete cases, Cox regression, Breslow ties, adjusted uncertainty | Additional domain coverage |
| `real-clinical-paired-visits` | medium | clinical TSV, longitudinal TSV | patient joins, nearest visits, missingness, paired changes | Additional domain coverage |
| `enrichment-universe` | medium | gmt, gene-lists | background-universe, hypergeometric-tail, multiple-testing | Additional domain coverage |
| `design-estimability` | medium | csv-header | model-rank, confounding, valid-stopping | DESeq2 |
| `paired-treatment-effect` | medium | csv-header | paired-design, technical-replicates, missing-visits | limma |
| `adjusted-linear-effect` | medium | csv-header | covariate-adjustment, signed-effects, ols | limma |
| `odds-ratio-contingency` | medium | csv-header | effect-direction, zero-cells, odds-ratio | Additional domain coverage |
| `kaplan-meier` | medium | csv-header | risk-sets, right-censoring, tied-events | Additional domain coverage |
| `diagnostic-thresholds` | medium | csv-header | positive-class, threshold-boundaries, precision-recall | Additional domain coverage |
| `permutation-mean-test` | medium | csv-header | exchangeability, exact-permutation, two-sided-tests | Additional domain coverage |

## Structures and proteomics

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `real-mmcif-chain-geometry` | medium | mmCIF | experimental-structures, author-residue-identifiers, alternate-conformers, coordinate-geometry | Biopython |
| `real-mmcif-contact-degree` | medium | mmCIF | experimental-structures, author-residue-identifiers, alternate-conformers, coordinate-geometry | Biopython |
| `pdb-ca-distances` | medium | pdb3.3-atom-profile | fixed-width-atoms, alternate-locations, insertion-codes | Additional domain coverage |
| `mmcif-chain-centroids` | medium | mmcif-atom-site | atom-site-loop, author-versus-label-ids, models | Additional domain coverage |
| `pdb-contact-map` | medium | pdb3.3-atom-profile | residue-contacts, local-neighbor-exclusion, distance-units | Additional domain coverage |
| `pdb-backbone-dihedrals` | medium | pdb3.3-atom-profile | backbone-geometry, handedness, angle-conventions | Additional domain coverage |
| `pdb-radius-gyration` | medium | pdb3.3-atom-profile | centering, structural-size, atom-selection | Additional domain coverage |
| `peptide-target-decoy-fdr` | medium | csv-header | target-decoy, monotone-qvalues, competition | Additional domain coverage |
| `protein-coverage` | medium | fasta, csv-header | peptide-mapping, repeated-sequences, union-coverage | Additional domain coverage |

## Networks

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `pathway-reachability` | medium | sbml-level3-version2-core, csv-header | cosubstrate-logic, reversible-reactions, closure | Additional domain coverage |
| `stoichiometric-balance` | medium | sbml-level3-version2-core, csv-header | stoichiometric-matrix, flux-residual, boundary-species | Additional domain coverage |
| `reaction-mass-balance` | medium | sbml-level3-version2-core, csv-header | chemical-formulas, element-balance, charge | Additional domain coverage |
| `interaction-components` | medium | sif | undirected-network, isolates, duplicate-edges | Additional domain coverage |
| `regulatory-path-signs` | medium | csv-header | signed-paths, contradictory-regulation, dag | Additional domain coverage |
| `network-shortest-paths` | medium | csv-header | weighted-paths, directionality, unreachable-nodes | Additional domain coverage |

## Assays and metabolomics

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `mgf-precursor-neutral-mass` | medium | mascot-generic-format, csv-header | precursor-charge, proton-mass, mgf-metadata | Additional domain coverage |
| `mgf-fragment-matching` | medium | mascot-generic-format, csv-header | ppm-tolerance, signed-mass-error, peak-matching | Additional domain coverage |
| `mgf-total-ion-current` | medium | mascot-generic-format, csv-header | fragment-intensities, base-peak, precursor-exclusion | Additional domain coverage |
| `metabolite-isotope-correction` | medium | csv-header | isotopologues, matrix-orientation, calibration | Additional domain coverage |
| `dose-response-ic50` | medium | csv-header | log-concentration, interpolation, no-extrapolation | Additional domain coverage |
| `plate-control-normalization` | medium | csv-header | plate-specific-controls, replicates, unclipped-response | Additional domain coverage |
| `qpcr-delta-delta-ct` | medium | csv-header | technical-replicates, reference-gene, fold-direction | Additional domain coverage |

## Workflow and identifiers

| Recipe | Difficulty | Supplied formats | Skills | Repository operations |
|---|---|---|---|---|
| `sample-sheet-lanes` | medium | csv-header | sample-key-joins, technical-lanes, include-policy | DESeq2, Snakemake, Nextflow, kallisto, nf-core/tools |
| `enrichment-identifier-mapping` | medium | csv-header, gmt, gene-lists | ambiguous-mapping, aliases, universe-order | Additional domain coverage |

