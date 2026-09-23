# Widely used computational biology and bioinformatics software

Draft global top 50 for identifying repositories relevant to computational
biology or bioinformatics. This is one ranking across all interfaces and
implementation languages; there are no per-language or per-analysis quotas.

## Ranking and metric notes

- The order is a rough adoption ranking, not a mathematically precise
  leaderboard. It prioritizes observed package downloads, then broad community
  use, paper citations, and repository activity.
- **Bioconda downloads** are lifetime package downloads reported by the
  [Anaconda API](https://api.anaconda.org/) on **2026-07-28**. They are the most
  comparable usage signal across this list, but they include CI, repeated
  environment creation, and transitive dependencies. They also miss installs
  from apt, Homebrew, Docker, vendor binaries, source builds, modules on HPC
  systems, and institutional mirrors.
- **PyPI downloads** are trailing-month file downloads. They include downloads
  made through pip and usually uv when uv uses PyPI, but they are not unique
  users or installations. uv has no separate public per-package counter.
- **Stars** are repository stars on **2026-07-28**. `—` means the canonical
  repository is not on GitHub or does not expose a comparable star count.
- **Citations** are approximate citations to one canonical software paper,
  mostly from OpenAlex on **2026-07-28**. They are version- and paper-dependent,
  and `—` means there is no clear package-specific canonical paper. Citation
  counts are a scientific-impact signal, not a direct usage count.
- The list is deliberately CLI-heavy because that is what the cross-ecosystem
  usage evidence supports. “Interface” is metadata only and did not create
  separate ranking buckets.

## Global top 50

| Rank | Software (repository) | Interface | Analysis or role | Download evidence | Stars | Canonical-paper citations |
|---:|---|---|---|---:|---:|---:|
| 1 | [BLAST+](https://github.com/ncbi/ncbi-cxx-toolkit-public) | C++ CLI | Local sequence similarity search | Bioconda 3.13M | 100 | [~23.5k](https://doi.org/10.1186/1471-2105-10-421) |
| 2 | [SAMtools](https://github.com/samtools/samtools) | C CLI | SAM/BAM/CRAM manipulation and alignment statistics | Bioconda 8.99M | 1.9k | [~68.2k](https://doi.org/10.1093/bioinformatics/btp352) |
| 3 | [BWA](https://github.com/lh3/bwa) | C CLI | Short-read alignment | Bioconda 2.16M | 1.8k | [~63.4k](https://doi.org/10.1093/bioinformatics/btp324) |
| 4 | [Bowtie 2](https://github.com/BenLangmead/bowtie2) | C++ CLI | Short-read alignment | Bioconda 3.47M | 805 | [~61.9k](https://doi.org/10.1038/nmeth.1923) |
| 5 | [DESeq2](https://github.com/thelovelab/DESeq2) | R/Bioconductor | Differential expression from count data | Bioconda 658k | 472 | [~102k](https://doi.org/10.1186/s13059-014-0550-8) |
| 6 | [STAR](https://github.com/alexdobin/STAR) | C++ CLI | Splice-aware RNA-seq alignment | Bioconda 1.70M | 2.2k | [~57.0k](https://doi.org/10.1093/bioinformatics/bts635) |
| 7 | [BEDTools](https://github.com/arq5x/bedtools2) | C++ CLI | Genomic interval arithmetic | Bioconda 3.75M | 1.0k | [~30.9k](https://doi.org/10.1093/bioinformatics/btq033) |
| 8 | [GATK](https://github.com/broadinstitute/gatk) | Java CLI | Variant discovery, genotyping, and processing | Bioconda 1.21M | 2.0k | [~30.3k](https://doi.org/10.1101/gr.107524.110) |
| 9 | [pysam](https://github.com/pysam-developers/pysam) | Python API | SAM/BAM/CRAM, VCF/BCF, FASTA, and tabix I/O | Bioconda 10.52M | 902 | — |
| 10 | [MAFFT](https://github.com/GSLBiotech/mafft) | CLI | Multiple sequence alignment | Bioconda 1.42M | 93 | [~48.9k](https://doi.org/10.1093/molbev/mst010) |
| 11 | [HMMER](https://github.com/EddyRivasLab/hmmer) | C CLI | Profile-HMM protein sequence search | Bioconda 2.29M | 416 | [~7.6k](https://doi.org/10.1371/journal.pcbi.1002195) |
| 12 | [Seurat](https://github.com/satijalab/seurat) | R API | Single-cell and spatial omics analysis | CRAN; no stable lifetime total | 2.8k | [~17.1k](https://doi.org/10.1016/j.cell.2019.05.031) |
| 13 | [minimap2](https://github.com/lh3/minimap2) | C CLI/API | Long-read, assembly, and spliced alignment | Bioconda 1.42M | 2.2k | [~17.2k](https://doi.org/10.1093/bioinformatics/bty191) |
| 14 | [BCFtools](https://github.com/samtools/bcftools) | C CLI | VCF/BCF querying, calling, filtering, and transformation | Bioconda 4.36M | 880 | [~16.3k](https://doi.org/10.1093/gigascience/giab008) |
| 15 | [Picard](https://github.com/broadinstitute/picard) | Java CLI/API | SAM/BAM processing and sequencing metrics | Bioconda 2.92M | 1.1k | — |
| 16 | [Snakemake](https://github.com/snakemake/snakemake) | Python DSL/CLI | Reproducible workflow orchestration | Bioconda 1.84M | 2.8k | [~3.2k](https://doi.org/10.1093/bioinformatics/bts480) |
| 17 | [HTSlib](https://github.com/samtools/htslib) | C API/CLI | Core high-throughput sequencing formats and indexing | Bioconda 8.28M | 939 | [~16.3k, shared ecosystem paper](https://doi.org/10.1093/gigascience/giab008) |
| 18 | [FastQC](https://github.com/s-andrews/FastQC) | Java CLI/GUI | Raw sequencing-read quality control | Bioconda 1.42M | 611 | — |
| 19 | [Biopython](https://github.com/biopython/biopython) | Python API | General sequence, alignment, phylogenetic, and structure analysis | Bioconda 441k; PyPI ~11M/month | 5.1k | [~5.9k](https://doi.org/10.1093/bioinformatics/btp163) |
| 20 | [Nextflow](https://github.com/nextflow-io/nextflow) | Groovy DSL/CLI | Portable, scalable workflow orchestration | Bioconda 598k | 3.5k | [~4.3k](https://doi.org/10.1038/nbt.3820) |
| 21 | [DIAMOND](https://github.com/bbuchfink/diamond) | C++ CLI | Fast protein similarity search | Bioconda 2.44M | 1.3k | [~15.4k](https://doi.org/10.1038/nmeth.3176) |
| 22 | [PLINK / PLINK 2](https://github.com/chrchang/plink-ng) | C/C++ CLI | GWAS, genotype QC, and population genetics | Bioconda 278k combined | 508 | [~36.5k](https://doi.org/10.1086/519795) |
| 23 | [Scanpy](https://github.com/scverse/scanpy) | Python API | Single-cell gene-expression analysis | Bioconda 147k; PyPI 984k/month | 2.5k | [~9.5k](https://doi.org/10.1186/s13059-017-1382-0) |
| 24 | [MultiQC](https://github.com/MultiQC/MultiQC) | Python CLI/API | Aggregate QC and pipeline reports | Bioconda 863k; PyPI 67k/month | 1.5k | [~10.7k](https://doi.org/10.1093/bioinformatics/btw354) |
| 25 | [edgeR](https://git.bioconductor.org/packages/edgeR) | R/Bioconductor | Differential expression from count data | Bioconda 630k | — | [~44.9k](https://doi.org/10.1093/bioinformatics/btp616) |
| 26 | [limma](https://git.bioconductor.org/packages/limma) | R/Bioconductor | Differential expression and linear modeling | Bioconda 806k | — | [~43.7k](https://doi.org/10.1093/nar/gkv007) |
| 27 | [SPAdes](https://github.com/ablab/spades) | C++/Python CLI | Short-read and metagenome assembly | Bioconda 851k | 955 | [~27.5k](https://doi.org/10.1089/cmb.2012.0021) |
| 28 | [IQ-TREE](https://github.com/iqtree/iqtree2) | C++ CLI | Maximum-likelihood phylogenetics | Bioconda 1.09M | 334 | [~27.9k](https://doi.org/10.1093/molbev/msu300) |
| 29 | [FastTree](https://github.com/morgannprice/fasttree) | C++ CLI | Approximate maximum-likelihood phylogenetics | Bioconda 1.47M | 41 | [~16.1k](https://doi.org/10.1371/journal.pone.0009490) |
| 30 | [RAxML](https://github.com/stamatak/standard-RAxML) | C CLI | Maximum-likelihood phylogenetics | Bioconda 1.40M | 350 | [~30.0k](https://doi.org/10.1093/bioinformatics/btu033) |
| 31 | [cutadapt](https://github.com/marcelm/cutadapt) | Python CLI/API | Adapter and primer trimming | Bioconda 1.52M; PyPI 44k/month | 586 | [~36.3k](https://doi.org/10.14806/ej.17.1.200) |
| 32 | [Salmon](https://github.com/COMBINE-lab/salmon) | C++/Rust CLI | Transcript-level RNA-seq quantification | Bioconda 623k | 920 | [~13.9k](https://doi.org/10.1038/nmeth.4197) |
| 33 | [kallisto](https://github.com/pachterlab/kallisto) | C++ CLI | Pseudoalignment and transcript quantification | Bioconda 273k | 768 | [~11.2k](https://doi.org/10.1038/nbt.3519) |
| 34 | [StringTie](https://github.com/gpertea/stringtie) | C++ CLI | Transcript assembly and quantification | Bioconda 776k | 526 | [~15.7k](https://doi.org/10.1038/nbt.3122) |
| 35 | [fastp](https://github.com/OpenGene/fastp) | C++ CLI | FASTQ QC, filtering, and adapter trimming | Bioconda 725k | 2.4k | [~30.4k](https://doi.org/10.1093/bioinformatics/bty560) |
| 36 | [SRA Toolkit](https://github.com/ncbi/sra-tools) | C++ CLI/API | Download and transform NCBI SRA sequencing data | Bioconda 727k | 1.4k | — |
| 37 | [deepTools](https://github.com/deeptools/deepTools) | Python CLI/API | Sequencing coverage, QC, and visualization | Bioconda 1.21M | 762 | [~9.2k](https://doi.org/10.1093/nar/gkw257) |
| 38 | [VCFtools](https://github.com/vcftools/vcftools) | C++/Perl CLI | VCF filtering, summaries, and population-genetic statistics | Bioconda 633k | 562 | [~18.0k](https://doi.org/10.1093/bioinformatics/btr330) |
| 39 | [SnpEff](https://github.com/pcingola/SnpEff) | Java CLI/API | Variant-effect annotation and prediction | Bioconda 617k | 310 | [~12.8k](https://doi.org/10.4161/fly.19695) |
| 40 | [Ensembl VEP](https://github.com/Ensembl/ensembl-vep) | Perl CLI/API | Variant consequence annotation | Bioconda 427k | 566 | [~8.8k](https://doi.org/10.1186/s13059-016-0974-4) |
| 41 | [MACS2 / MACS3](https://github.com/macs3-project/MACS) | Python CLI/API | ChIP-seq and ATAC-seq peak calling | Bioconda 248k combined | 781 | [~20.2k](https://doi.org/10.1186/gb-2008-9-9-r137) |
| 42 | [MMseqs2](https://github.com/soedinglab/MMseqs2) | C++ CLI | Large-scale protein search and clustering | Bioconda 601k | 2.1k | [~5.3k](https://doi.org/10.1038/nbt.3988) |
| 43 | [MUSCLE](https://github.com/rcedgar/muscle) | C++ CLI | Multiple sequence alignment | Bioconda 642k | 285 | [~46.8k](https://doi.org/10.1093/nar/gkh340) |
| 44 | [Kraken 2](https://github.com/DerrickWood/kraken2) | C++ CLI | K-mer-based metagenomic taxonomic classification | Bioconda 205k | 924 | [~7.3k](https://doi.org/10.1186/s13059-019-1891-0) |
| 45 | [BUSCO](https://gitlab.com/ezlab/busco) | Python CLI | Genome, transcriptome, and gene-set completeness | Bioconda 416k | 58 | [~14.7k](https://doi.org/10.1093/bioinformatics/btv351) |
| 46 | [UCSC Kent utilities](https://github.com/ucscGenomeBrowser/kent) | C CLI collection | liftOver; 2bit, BigWig, BigBed, BED, and genome-browser formats | Bioconda 51k–594k per utility | 275 | [~1.5k for BigWig/BigBed](https://doi.org/10.1093/bioinformatics/btq351) |
| 47 | [GenomicRanges](https://github.com/Bioconductor/GenomicRanges) | R/Bioconductor | Genomic interval representation and operations | Bioconda 1.77M | 47 | [~5.1k](https://doi.org/10.1371/journal.pcbi.1003118) |
| 48 | [Biostrings](https://github.com/Bioconductor/Biostrings) | R/Bioconductor | DNA, RNA, and amino-acid sequence operations | Bioconda 1.74M | 69 | — |
| 49 | [pybedtools](https://github.com/daler/pybedtools) | Python API | Python interface to BEDTools and genomic intervals | Bioconda 1.35M | 330 | [~609](https://doi.org/10.1093/bioinformatics/btr539) |
| 50 | [nf-core/tools](https://github.com/nf-core/tools) | Python CLI; Nextflow ecosystem | Develop, lint, launch, and manage community pipelines | Bioconda 150k | 318 | [~4.1k for nf-core](https://doi.org/10.1038/s41587-020-0439-x) |

## Interpretation cautions

- Do not add downloads from different registries or time windows. They are
  separate evidence, not parts of a shared denominator.
- Library downloads can be inflated by transitive dependencies. HTSlib,
  `pysam`, GenomicRanges, and Biostrings are especially likely to be installed
  as dependencies of other software.
- CLI totals can be understated when projects distribute binaries directly or
  are preinstalled on clusters. BLAST+, GATK, PLINK, the SRA Toolkit, and the
  UCSC Kent utilities are likely affected.
- Repository matching should search dependency manifests and workflow/container
  definitions as well as source imports. For the UCSC collection, also search
  individual executable names such as `liftOver`, `bedGraphToBigWig`,
  `bigWigToBedGraph`, `faToTwoBit`, `twoBitToFa`, `bedToBigBed`, and
  `bigBedToBed`.
- Snakemake, Nextflow, and nf-core are important bioinformatics signals but are
  workflow infrastructure, not biological analyses by themselves.
