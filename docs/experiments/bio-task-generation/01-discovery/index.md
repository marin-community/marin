# Source discovery

[Planning overview](../index.md) · [Discovery process](../discovery.md)

This directory contains results from source discovery, the first stage of
computational biology task generation for [issue #9257](https://github.com/marin-community/marin/issues/9257).
It accepts repositories and versioned source archives. The findings are discovery
evidence; no task authoring or execution validation is claimed.

| Page or artifact | Contents |
| --- | --- |
| [Expansion to 200 per ranking](ranking-expansion.md) | Marginal diversity and overlap: 672 distinct sources, including 337 new to the combined baseline |
| [Additional ranked sources](ranking-additions.md) | Ranks 101–200 for each approach, with scores, GitHub stars, topics and source types |
| [Four top-100 rankings](ranking-comparison.md) | Overlap, topic diversity and source types across Bioconda, Bioconductor, PyPI and GitHub |
| [Ranked source lists](ranking-lists.md) | All 400 ranking positions, covering 335 distinct sources |
| [Inventory and source screening](#candidate-inventory) | The 95 candidates, their scientific uses and adoption measurements |
| [Adoption analysis](adoption.md) | Correlations, metric coverage, sensitivity checks and limitations |
| [GitHub topics](topics.md) | Observed topic strings, frequencies and search ideas |
| [Structured inventory](inventory.json) | Canonical records, provenance, dated observations and exact analysis results |
| [Adoption table](data/adoption-2026-09-29.csv) | Compact export of the 95 candidates and four adoption measures |

The stage number places discovery before source inspection, task authoring and
validation. Within this stage, filenames describe the artifacts. Observation
dates and cohort membership are recorded in the data and reports. Keep a dated
copy of the cohort and its measurements before replacing them with a larger
sample. The original 95-source inventory, four top-100 lists and four top-200
lists are retained separately; the expansion compares the latter two.

## September 29, 2026 inventory

The September 29, 2026 package-led pass identifies **95 software candidates**
for [issue #9257](https://github.com/marin-community/marin/issues/9257):
93 have repository links and two have official versioned source distributions.
It retains the 50 earlier software entries and adds 45. Entries are candidates
screened through package metadata and selected documentation. No tasks were
created, packages installed, or candidate runtimes measured in this pass.

The [structured inventory](inventory.json) contains package-level
counts, recipe URLs and hashes, Bioconductor release metadata, documentation
links, role labels, and unresolved questions. GitHub metadata and stars are
verified for 83 entries, including 18 GitHub repositories linked by Bioconductor
release metadata. The inventory retains 25 Bioconductor source links, three
GitLab links, and the two source archives. PyPI downloads are verified for
19 entries (20 packages).

The [earlier inventory](https://github.com/marin-community/marin/blob/d3f09bbb3ba2e74c4c3073ed20087cd5f97139af/docs/experiments/computational_biology_bioinformatics_packages.md)
preserves its July 28 adoption evidence. Its approximate global ranking is not
carried forward. The initial screen displayed the top 110 package rows from
each download table, then manually selected additions and adjacent tools.
The 95 entries form a reviewed batch, not a strict top-95 ranking or a scientific
coverage threshold. There is no composite adoption score. This adoption comparison
holds the 95-source cohort fixed; the separate ranking analyses select sources
from broader pools and compare diversity at depths of 100 and 200.

Discovery accepts public source repositories and versioned source archives.
A Git repository is not required. Record the source URL, revision or release,
and available checksums for either route. MEME Suite and Entrez Direct are
high-priority candidates for further inspection.

The [adoption comparison](adoption.md) finds little rank agreement between GitHub stars
and Bioconda downloads in this cohort (Spearman ρ = 0.10, n = 82). Bioconda and
Bioconductor agree more (ρ = 0.45, n = 25). PyPI overlaps only 19 candidates;
correlations cannot support a single interchangeable popularity measure.

## Metrics and provenance

| Observation | Definition and scope |
| --- | --- |
| Bioconda, September 29, 2026 | Cumulative recorded downloads from the [official package summary](https://github.com/bioconda/bioconda-stats/blob/cd491b0a4c9a7e80069c8894fbd369d8b07ceddc/package-downloads/anaconda.org/bioconda/packages.tsv), containing 12,740 packages. The [collector](https://github.com/bioconda/bioconda-stats/blob/main/src/package_downloads/stats_from_anaconda_org.py) sums nonnegative file counters across versions, builds and platforms for main-label conda artifacts. |
| Bioconductor, data as of September 28, 2026 | [Download score](https://www.bioconductor.org/packages/stats/): average monthly distinct IPs over September 2025 through August 2026. The retrieved score table contains 3,118 package entries. This is not a count of distinct users across the whole year. |
| Package-to-source mapping | [Bioconda recipe revision](https://github.com/bioconda/bioconda-recipes/tree/254ba2d4bcda7fe6ed2baa586bac6c35885a4b10/recipes), individually pinned conda-forge recipes, and [Bioconductor 3.23 metadata](https://bioconductor.org/packages/3.23/bioc/VIEWS). The JSON preserves selected fields and source hashes. |
| GitHub stars, September 29, 2026 | Current `stargazers_count` from each identified [repository API](https://docs.github.com/en/rest/repos/repos#get-a-repository). Bioconductor mappings use package `URL` or `BugReports` links. No arbitrary GitHub mirrors were substituted for other source hosts. |
| PyPI, August 30–September 28, 2026 | Thirty complete calendar days of downloads from [PyPIStats](https://pypistats.org/api/), excluding known mirrors. Package identity, daily counts, API URLs and response hashes are retained in the JSON. A July 1–September 28 window supports the 90-day sensitivity check. |

A recent-window Bioconda download total has not been computed. The cumulative
counter is useful for initial discovery, but it favors older packages and
includes dependency installations and automation. Channel migrations, mirrors,
containers and other installation methods also affect interpretation. Do not
combine it with the Bioconductor score or infer researcher counts from either.

Packages sharing a repository retain separate counters. PLINK/PLINK 2,
MACS2/MACS3, Snakemake variants and the sampled UCSC utilities each occupy one
entry. For correlation only, use the largest recorded package counter per
entry within each registry; it is a proxy, not a repository download total.
The [adoption analysis](adoption.md) also excludes entries with multiple package
counters as a sensitivity check.
Libraries such as HTSlib and GenomicRanges remain useful candidates, with
dependency-driven adoption called out. General Perl and compression dependencies
were not selected solely because they have high counts.

## Source identity and inspection findings

- MAFFT's [official source page](https://mafft.cbrc.jp/alignment/software/source.html)
  points to `gitlab.com/sysimm/mafft`; the earlier inventory linked a GitHub mirror.
- The current `iqtree` recipe points to IQ-TREE 3. The earlier IQ-TREE 2 URL is
  retained as historical provenance in the JSON.
- MAFFT, Biopython and Scanpy have current conda-forge recipes. Their Bioconda
  counters cover only the Bioconda distribution history.
- The current Bioconductor release describes scuttle as legacy utilities.
  Review its present role and alternatives before selecting scientific examples.
- HTSeq is marked as a fork by GitHub. Its package recipe identifies
  `htseq/htseq`; resolve that relationship during deeper inspection.
- MEME Suite 5.5.9 and Entrez Direct 26.0.20260719 have official source archives;
  both URLs returned HTTP 200 to header requests on September 29. Their pinned
  Bioconda recipes record archive checksums. Archive contents have not been
  downloaded or rehashed in this pass. Both are included in the main inventory.
- DADA2's [tutorial](https://benjjneb.github.io/dada2/tutorial.html) links observed
  paired-end 16S mouse-gut reads and connects sequence processing to community
  analysis with phyloseq. ViennaRNA has worked structure examples, and sourmash
  has a tutorial index. Inputs were not downloaded or checked for redistribution.

## Candidate inventory

Rows are alphabetical within role. `B` is the Bioconda cumulative counter,
`C` is the Bioconductor score, and `P30` is the PyPI download count over
August 30–September 28, 2026. An em dash means no verified measurement.
Multiple packages retain individual counters in the JSON; `max*` marks the
largest recorded package counter used for correlation. Stars link to the
measured GitHub repository, which can differ from the primary source link.
Documentation links are inspection starting points; example data remain
uninspected except where noted in the JSON.

### Scientific tools

| Repository or source archive | Scientific use | B | C | Stars | P30 | Documentation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [BBTools](https://github.com/bbushnell/BBTools) | Read alignment, filtering and sequence processing | 1,949,531 | — | [95](https://github.com/bbushnell/BBTools) | — | [docs](https://bbmap.org) |
| [BCFtools](https://github.com/samtools/bcftools) | VCF/BCF querying, calling, filtering, and transformation | 4,661,321 | — | [896](https://github.com/samtools/bcftools) | — | [docs](https://github.com/samtools/bcftools) |
| [BEDOPS](https://github.com/bedops/bedops) | Genomic interval operations | 256,797 | — | [375](https://github.com/bedops/bedops) | — | [docs](https://bedops.readthedocs.io) |
| [BEDTools](https://github.com/arq5x/bedtools2) | Genomic interval arithmetic | 4,074,077 | — | [1,052](https://github.com/arq5x/bedtools2) | — | [docs](http://bedtools.readthedocs.org/) |
| [BLAST+](https://github.com/ncbi/ncbi-cxx-toolkit-public) | Local sequence similarity search | 3,382,435 | — | [102](https://github.com/ncbi/ncbi-cxx-toolkit-public) | — | [docs](https://blast.ncbi.nlm.nih.gov/doc/blast-help) |
| [Bowtie 2](https://github.com/BenLangmead/bowtie2) | Short-read alignment | 3,869,511 | — | [812](https://github.com/BenLangmead/bowtie2) | — | [docs](https://github.com/BenLangmead/bowtie2/blob/v2.5.5/README.md) |
| [BUSCO](https://gitlab.com/ezlab/busco) | Genome, transcriptome, and gene-set completeness | 455,570 | — | — | — | [docs](https://busco.ezlab.org/busco_userguide.html) |
| [BWA](https://github.com/lh3/bwa) | Short-read alignment | 2,285,259 | — | [1,769](https://github.com/lh3/bwa) | — | [docs](https://github.com/lh3/bwa/blob/v0.7.19/README.md) |
| [CheckM](https://github.com/Ecogenomics/CheckM) | Microbial genome quality assessment | 127,670 | — | [411](https://github.com/Ecogenomics/CheckM) | 660 | [docs](https://ecogenomics.github.io/CheckM) |
| [clusterProfiler](https://git.bioconductor.org/packages/clusterProfiler) | Functional enrichment of omics results | 202,009 | 27,991 | [1,239](https://github.com/YuLab-SMU/clusterProfiler) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/clusterProfiler/inst/doc/clusterProfiler.html) |
| [cutadapt](https://github.com/marcelm/cutadapt) | Adapter and primer trimming | 1,583,771 | — | [587](https://github.com/marcelm/cutadapt) | 38,448 | [docs](https://cutadapt.readthedocs.io/) |
| [dada2](https://git.bioconductor.org/packages/dada2) | Amplicon sequence inference | 722,834 | 3,655 | [558](https://github.com/benjjneb/dada2) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/dada2/inst/doc/dada2-intro.html) |
| [deepTools](https://github.com/deeptools/deepTools) | Sequencing coverage, QC, and visualization | 1,234,197 | — | [767](https://github.com/deeptools/deepTools) | 7,403 | [docs](https://deeptools.readthedocs.io/en/latest) |
| [DESeq2](https://git.bioconductor.org/packages/DESeq2) | Differential expression from count data | 751,452 | 31,328 | [479](https://github.com/thelovelab/DESeq2) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/DESeq2/inst/doc/DESeq2.html) |
| [DIAMOND](https://github.com/bbuchfink/diamond) | Fast protein similarity search | 3,028,204 | — | [1,324](https://github.com/bbuchfink/diamond) | — | [docs](https://github.com/bbuchfink/diamond/wiki) |
| [edgeR](https://git.bioconductor.org/packages/edgeR) | Differential expression from count data | 732,091 | 29,733 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf) |
| [Ensembl VEP](https://github.com/Ensembl/ensembl-vep) | Variant consequence annotation | 460,056 | — | [575](https://github.com/Ensembl/ensembl-vep) | — | [docs](https://useast.ensembl.org/info/docs/tools/vep/script/vep_tutorial.html) |
| [fastp](https://github.com/OpenGene/fastp) | FASTQ QC, filtering, and adapter trimming | 819,201 | — | [2,435](https://github.com/OpenGene/fastp) | — | [docs](https://github.com/OpenGene/fastp/blob/v1.3.7/README.md) |
| [FastQC](https://github.com/s-andrews/FastQC) | Raw sequencing-read quality control | 1,503,503 | — | [621](https://github.com/s-andrews/FastQC) | — | [docs](http://www.bioinformatics.babraham.ac.uk/projects/fastqc/) |
| [FastTree](https://github.com/morgannprice/fasttree) | Approximate maximum-likelihood phylogenetics | 1,633,787 | — | [45](https://github.com/morgannprice/fasttree) | — | [docs](https://morgannprice.github.io/fasttree) |
| [fgsea](https://git.bioconductor.org/packages/fgsea) | Gene-set enrichment | 222,156 | 28,193 | [459](https://github.com/alserglab/fgsea) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/fgsea/inst/doc/fgsea-tutorial.html) |
| [FreeBayes](https://github.com/freebayes/freebayes) | Haplotype-based variant calling | 929,823 | — | [880](https://github.com/freebayes/freebayes) | — | [docs](https://github.com/freebayes/freebayes/blob/v1.3.10/README.md) |
| [GATK](https://github.com/broadinstitute/gatk) | Variant discovery, genotyping, and processing | 1,238,705 | — | [2,003](https://github.com/broadinstitute/gatk) | — | [docs](https://www.broadinstitute.org/gatk/) |
| [GSVA](https://git.bioconductor.org/packages/GSVA) | Gene-set variation analysis | 96,383 | 11,755 | [249](https://github.com/rcastelo/GSVA) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/GSVA/inst/doc/GSVA_proteomics.html) |
| [HMMER](https://github.com/EddyRivasLab/hmmer) | Profile-HMM protein sequence search | 2,591,956 | — | [425](https://github.com/EddyRivasLab/hmmer) | — | [docs](http://hmmer.org/documentation.html) |
| [HyPhy](https://github.com/veg/hyphy) | Comparative sequence analysis and molecular evolution | 1,314,315 | — | [271](https://github.com/veg/hyphy) | — | [docs](https://hyphy.org) |
| [IQ-TREE](https://github.com/iqtree/iqtree3) | Maximum-likelihood phylogenetics | 1,140,975 | — | [166](https://github.com/iqtree/iqtree3) | — | [docs](http://www.iqtree.org/doc) |
| [kallisto](https://github.com/pachterlab/kallisto) | Pseudoalignment and transcript quantification | 308,370 | — | [772](https://github.com/pachterlab/kallisto) | — | [docs](https://pachterlab.github.io/kallisto/manual.html) |
| [Kraken 2](https://github.com/DerrickWood/kraken2) | K-mer-based metagenomic taxonomic classification | 223,847 | — | [941](https://github.com/DerrickWood/kraken2) | — | [docs](https://github.com/DerrickWood/kraken2/blob/2.17.2/docs/MANUAL.markdown) |
| [LAST](https://gitlab.com/mcfrith/last) | Sequence alignment | 3,736,648 | — | — | — | [docs](https://gitlab.com/mcfrith/last/-/blob/1654/doc/last-cookbook.rst) |
| [limma](https://git.bioconductor.org/packages/limma) | Differential expression and linear modeling | 924,968 | 42,915 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/limma/inst/doc/usersguide.pdf) |
| [MACS2 / MACS3](https://github.com/macs3-project/MACS) | ChIP-seq and ATAC-seq peak calling | 226,316 max* | — | [785](https://github.com/macs3-project/MACS) | 12,286 max* | [docs](https://macs3-project.github.io/MACS) |
| [MAFFT](https://gitlab.com/sysimm/mafft) | Multiple sequence alignment | 1,502,496 | — | — | — | [docs](http://mafft.cbrc.jp/alignment/software/) |
| [Mash](https://github.com/marbl/Mash) | Sequence distance estimation with sketches | 327,700 | — | [453](https://github.com/marbl/Mash) | — | [docs](https://mash.readthedocs.io/en/latest) |
| [MEME Suite](https://meme-suite.org/meme/meme-software/5.5.9/meme-5.5.9.tar.gz) | Sequence motif discovery and scanning | 3,472,737 | — | — | — | [docs](https://meme-suite.org/meme/doc/overview.html) |
| [minimap2](https://github.com/lh3/minimap2) | Long-read, assembly, and spliced alignment | 1,540,079 | — | [2,251](https://github.com/lh3/minimap2) | — | [docs](https://lh3.github.io/minimap2/minimap2.html) |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | Large-scale protein search and clustering | 672,817 | — | [2,148](https://github.com/soedinglab/MMseqs2) | — | [docs](https://github.com/soedinglab/mmseqs2) |
| [MultiQC](https://github.com/MultiQC/MultiQC) | Aggregate QC and pipeline reports | 927,852 | — | [1,493](https://github.com/MultiQC/MultiQC) | 24,301 | [docs](https://docs.seqera.io/multiqc/) |
| [MUSCLE](https://github.com/rcedgar/muscle) | Multiple sequence alignment | 729,183 | — | [290](https://github.com/rcedgar/muscle) | — | [docs](https://drive5.com/muscle5) |
| [PeptideShaker](https://github.com/CompOmics/peptide-shaker) | Proteomics identification interpretation | 1,427,146 | — | [56](https://github.com/CompOmics/peptide-shaker) | — | [docs](https://github.com/compomics/peptide-shaker/blob/master/README.md) |
| [phyloseq](https://git.bioconductor.org/packages/phyloseq) | Microbiome community analysis | 305,728 | 7,797 | [652](https://github.com/joey711/phyloseq) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/phyloseq/inst/doc/phyloseq-analysis.html) |
| [Picard](https://github.com/broadinstitute/picard) | SAM/BAM processing and sequencing metrics | 2,982,579 | — | [1,079](https://github.com/broadinstitute/picard) | — | [docs](http://broadinstitute.github.io/picard/) |
| [PLINK / PLINK 2](https://github.com/chrchang/plink-ng) | GWAS, genotype QC, and population genetics | 191,219 max* | — | [521](https://github.com/chrchang/plink-ng) | — | [docs](https://www.cog-genomics.org/plink/) |
| [Prokka](https://github.com/tseemann/prokka) | Prokaryotic genome annotation | 263,543 | — | [995](https://github.com/tseemann/prokka) | — | [docs](https://github.com/tseemann/prokka) |
| [QUAST](https://github.com/ablab/quast) | Genome assembly quality assessment | 401,704 | — | [521](https://github.com/ablab/quast) | 3,423 | [docs](https://quast.sourceforge.net/docs/manual.html) |
| [RAxML](https://github.com/stamatak/standard-RAxML) | Maximum-likelihood phylogenetics | 1,549,290 | — | [351](https://github.com/stamatak/standard-RAxML) | — | [docs](http://sco.h-its.org/exelixis/web/software/raxml/index.html) |
| [Salmon](https://github.com/COMBINE-lab/salmon) | Transcript-level RNA-seq quantification | 832,906 | — | [932](https://github.com/COMBINE-lab/salmon) | — | [docs](https://combine-lab.github.io/salmon) |
| [SAMtools](https://github.com/samtools/samtools) | SAM/BAM/CRAM manipulation and alignment statistics | 9,707,033 | — | [1,964](https://github.com/samtools/samtools) | — | [docs](https://github.com/samtools/samtools) |
| [Scanpy](https://github.com/scverse/scanpy) | Single-cell gene-expression analysis | 149,005 | — | [2,577](https://github.com/scverse/scanpy) | 667,672 | [docs](https://scanpy.readthedocs.io/) |
| [scater](https://git.bioconductor.org/packages/scater) | Single-cell quality control and exploratory analysis | 304,978 | 11,044 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/scater/inst/doc/overview.html) |
| [scran](https://git.bioconductor.org/packages/scran) | Single-cell normalization and statistical analysis | 328,350 | 8,655 | [48](https://github.com/MarioniLab/scran) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/scran/inst/doc/scran.html) |
| [SearchGUI](https://github.com/CompOmics/searchgui) | Proteomics identification search engines | 758,919 | — | [48](https://github.com/CompOmics/searchgui) | — | [docs](https://github.com/compomics/searchgui/blob/master/README.md) |
| [SEPP](https://github.com/smirarab/sepp) | Phylogenetic placement | 1,294,936 | — | [96](https://github.com/smirarab/sepp) | — | [docs](https://github.com/smirarab/sepp/blob/v4.5.6/README.md) |
| [SeqKit](https://github.com/shenwei356/seqkit) | FASTA and FASTQ processing | 911,309 | — | [1,596](https://github.com/shenwei356/seqkit) | — | [docs](https://bioinf.shenwei.me/seqkit) |
| [seqtk](https://github.com/lh3/seqtk) | FASTA and FASTQ processing | 862,851 | — | [1,565](https://github.com/lh3/seqtk) | — | [docs](https://github.com/lh3/seqtk/blob/v1.5/README.md) |
| [Seurat](https://github.com/satijalab/seurat) | Single-cell and spatial omics analysis | — | — | [2,805](https://github.com/satijalab/seurat) | — | [docs](https://satijalab.org/seurat/) |
| [SingleR](https://git.bioconductor.org/packages/SingleR) | Reference-based cell-type annotation | 52,024 | 6,291 | [205](https://github.com/SingleR-inc/SingleR) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/SingleR/inst/doc/SingleR.html) |
| [SnpEff](https://github.com/pcingola/SnpEff) | Variant-effect annotation and prediction | 656,246 | — | [313](https://github.com/pcingola/SnpEff) | — | [docs](http://snpeff.sourceforge.net/) |
| [sourmash](https://github.com/sourmash-bio/sourmash) | Genome and metagenome comparison with sketches | 310,321 | — | [557](https://github.com/sourmash-bio/sourmash) | 9,608 | [docs](https://sourmash.readthedocs.io/) |
| [SPAdes](https://github.com/ablab/spades) | Short-read and metagenome assembly | 920,567 | — | [961](https://github.com/ablab/spades) | — | [docs](https://ablab.github.io/spades) |
| [STAR](https://github.com/alexdobin/STAR) | Splice-aware RNA-seq alignment | 1,777,223 | — | [2,257](https://github.com/alexdobin/STAR) | — | [docs](https://github.com/alexdobin/STAR/blob/2.7.11b/doc/STARmanual.pdf) |
| [StringTie](https://github.com/gpertea/stringtie) | Transcript assembly and quantification | 873,646 | — | [534](https://github.com/gpertea/stringtie) | — | [docs](https://ccb.jhu.edu/software/stringtie/index.shtml?t=manual) |
| [sva](https://git.bioconductor.org/packages/sva) | Surrogate variables and batch effects | 133,792 | 10,633 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/sva/inst/doc/sva.pdf) |
| [UCSC Kent utilities](https://github.com/ucscGenomeBrowser/kent) | liftOver; 2bit, BigWig, BigBed, BED, and genome-browser formats | 618,051 max* | — | [278](https://github.com/ucscGenomeBrowser/kent) | — | [docs](https://github.com/ucscGenomeBrowser/kent/blob/v482_base/README) |
| [VCFtools](https://github.com/vcftools/vcftools) | VCF filtering, summaries, and population-genetic statistics | 769,297 | — | [565](https://github.com/vcftools/vcftools) | — | [docs](https://vcftools.github.io) |
| [ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) | RNA secondary structure prediction | 1,487,092 | — | [436](https://github.com/ViennaRNA/ViennaRNA) | 60,397 | [docs](http://www.tbi.univie.ac.at/RNA/) |
| [VSEARCH](https://github.com/torognes/vsearch) | Sequence clustering and metagenomic sequence analysis | 995,945 | — | [762](https://github.com/torognes/vsearch) | — | [docs](https://torognes.github.io/vsearch) |

### Scientific libraries

| Repository or source archive | Scientific use | B | C | Stars | P30 | Documentation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [BamTools](https://github.com/pezmaster31/bamtools) | BAM manipulation | 725,131 | — | [432](https://github.com/pezmaster31/bamtools) | — | [docs](https://github.com/pezmaster31/bamtools/wiki) |
| [Biopython](https://github.com/biopython/biopython) | General sequence, alignment, phylogenetic, and structure analysis | 444,273 | — | [5,215](https://github.com/biopython/biopython) | 3,815,234 | [docs](http://biopython.org) |
| [Biostrings](https://git.bioconductor.org/packages/Biostrings) | DNA, RNA, and amino-acid sequence operations | 1,902,417 | 55,981 | [70](https://github.com/Bioconductor/Biostrings) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/Biostrings/inst/doc/Biostrings2Classes.pdf) |
| [cyvcf2](https://github.com/brentp/cyvcf2) | VCF parsing | 1,199,768 | — | [450](https://github.com/brentp/cyvcf2) | 133,754 | [docs](https://brentp.github.io/cyvcf2) |
| [DendroPy](https://github.com/jeetsukumaran/DendroPy) | Phylogenetic trees and comparative data | 972,166 | — | [237](https://github.com/jeetsukumaran/DendroPy) | 87,310 | [docs](https://dendropy.org) |
| [GenomicFeatures](https://git.bioconductor.org/packages/GenomicFeatures) | Gene models and transcript annotations | 412,368 | 18,930 | [26](https://github.com/Bioconductor/GenomicFeatures) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/GenomicFeatures/inst/doc/GenomicFeatures.html) |
| [GenomicRanges](https://git.bioconductor.org/packages/GenomicRanges) | Genomic interval representation and operations | 1,957,521 | 54,735 | [47](https://github.com/Bioconductor/GenomicRanges) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/GenomicRanges/inst/doc/ExtendingGenomicRanges.pdf) |
| [HTSeq](https://github.com/htseq/htseq) | Read counting and sequencing-data processing | 1,113,088 | — | [109](https://github.com/htseq/htseq) | 13,602 | [docs](https://htseq.readthedocs.io/en/latest) |
| [HTSlib](https://github.com/samtools/htslib) | Core high-throughput sequencing formats and indexing | 9,118,220 | — | [953](https://github.com/samtools/htslib) | — | [docs](http://www.htslib.org/) |
| [pybedtools](https://github.com/daler/pybedtools) | Python interface to BEDTools and genomic intervals | 1,569,302 | — | [330](https://github.com/daler/pybedtools) | 123,860 | [docs](https://daler.github.io/pybedtools) |
| [pyBigWig](https://github.com/deeptools/pyBigWig) | bigWig signal access | 1,074,729 | — | [251](https://github.com/deeptools/pyBigWig) | 158,577 | [docs](https://github.com/deeptools/pyBigWig/blob/0.3.26/README.md) |
| [pyfaidx](https://github.com/mdshw5/pyfaidx) | Indexed FASTA access | 976,993 | — | [489](https://github.com/mdshw5/pyfaidx) | 328,711 | [docs](https://pypi.org/project/pyfaidx) |
| [pysam](https://github.com/pysam-developers/pysam) | SAM/BAM/CRAM, VCF/BCF, FASTA, and tabix I/O | 12,282,748 | — | [911](https://github.com/pysam-developers/pysam) | 1,014,605 | [docs](https://github.com/pysam-developers/pysam) |
| [rtracklayer](https://git.bioconductor.org/packages/rtracklayer) | Genome annotation file import and export | 634,870 | 24,920 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/rtracklayer/inst/doc/rtracklayer.pdf) |
| [scuttle](https://git.bioconductor.org/packages/scuttle) | Legacy single-cell analysis utilities | 504,671 | 12,844 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/scuttle/inst/doc/userguide.html) |
| [ShortRead](https://git.bioconductor.org/packages/ShortRead) | FASTQ processing and quality assessment | 749,843 | 7,516 | [8](https://github.com/Bioconductor/ShortRead) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/ShortRead/inst/doc/Overview.html) |
| [treeio](https://git.bioconductor.org/packages/treeio) | Phylogenetic tree formats and metadata | 392,644 | 30,493 | [105](https://github.com/YuLab-SMU/treeio) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/treeio/inst/doc/treeio.html) |
| [tximport](https://git.bioconductor.org/packages/tximport) | Transcript-to-gene quantification import | 209,963 | 5,262 | [145](https://github.com/thelovelab/tximport) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/tximport/inst/doc/tximport.html) |
| [VariantAnnotation](https://git.bioconductor.org/packages/VariantAnnotation) | Variant annotation and VCF processing | 225,530 | 9,757 | — | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/VariantAnnotation/inst/doc/ensemblVEP.html) |

### Visualization tools

| Repository or source archive | Scientific use | B | C | Stars | P30 | Documentation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [ComplexHeatmap](https://git.bioconductor.org/packages/ComplexHeatmap) | Annotated heatmaps for molecular measurements | 177,953 | 22,841 | [1,557](https://github.com/jokergoo/ComplexHeatmap) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/ComplexHeatmap/inst/doc/complex_heatmap.html) |
| [ggtree](https://git.bioconductor.org/packages/ggtree) | Phylogenetic tree annotation and visualization | 230,231 | 33,070 | [935](https://github.com/YuLab-SMU/ggtree) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/ggtree/inst/doc/ggtree.html) |

### Retrieval tools

| Repository or source archive | Scientific use | B | C | Stars | P30 | Documentation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [biomaRt](https://git.bioconductor.org/packages/biomaRt) | BioMart annotation retrieval | 414,010 | 24,443 | [51](https://github.com/Huber-group-EMBL/biomaRt) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/biomaRt/inst/doc/accessing_ensembl.html) |
| [Entrez Direct](https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/versions/26.0.20260719/edirect.tar.gz) | NCBI database retrieval and record transformation | 1,950,018 | — | — | — | [docs](https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/versions/26.0.20260719/README) |
| [GEOquery](https://git.bioconductor.org/packages/GEOquery) | GEO study retrieval | 105,905 | 14,795 | [118](https://github.com/seandavi/GEOquery) | — | [docs](https://bioconductor.org/packages/3.23/bioc/vignettes/GEOquery/inst/doc/GEOquery.html) |
| [SRA Toolkit](https://github.com/ncbi/sra-tools) | Download and transform NCBI SRA sequencing data | 792,214 | — | [1,368](https://github.com/ncbi/sra-tools) | — | [docs](https://github.com/ncbi/sra-tools/wiki) |

### Workflow infrastructure

| Repository or source archive | Scientific use | B | C | Stars | P30 | Documentation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [Nextflow](https://github.com/nextflow-io/nextflow) | Portable, scalable workflow orchestration | 643,483 | — | [3,495](https://github.com/nextflow-io/nextflow) | — | [docs](https://github.com/nextflow-io/nextflow) |
| [nf-core/tools](https://github.com/nf-core/tools) | Develop, lint, launch, and manage community pipelines | 155,524 | — | [323](https://github.com/nf-core/tools) | 22,894 | [docs](https://nf-co.re) |
| [Snakemake](https://github.com/snakemake/snakemake) | Reproducible workflow orchestration | 2,027,238 max* | — | [2,881](https://github.com/snakemake/snakemake) | 153,724 | [docs](https://snakemake.readthedocs.io/en/stable) |

## Adoption leads to inspect

`harpy`, `genenotebook` and `genoboo` also appeared among highly downloaded
packages. Their counters are recorded in the JSON, but their scientific purpose
and independent use have not been inspected. Their download counts alone do not
establish priority.

## Remaining discovery coverage

[Bioconductor](https://bioconductor.org/about/) is centered on R and distributes
most components as R packages. Discovery through its catalog therefore
emphasizes R-based analysis. [Bioconda](https://bioconda.github.io/) distributes
bioinformatics software across languages, including standalone command-line
tools. Continue discovery across Python packages, command-line tools, workflows
and paper-analysis code to cover scientific uses beyond Bioconductor. Language
and distribution channel are descriptive metadata, not eligibility requirements.

This pass is concentrated on packaged tools and scientific libraries. Follow
their tutorials, dependency manifests and citations into reusable workflows and
paper-analysis repositories. The
[Snakemake workflow catalog](https://snakemake.github.io/snakemake-workflow-catalog/),
[nf-core pipelines](https://nf-co.re/pipelines),
[Galaxy training material](https://training.galaxyproject.org/) and
[Bioconductor workflows](https://bioconductor.org/packages/release/workflows/)
provide complementary discovery routes. A low-download paper repository may
still provide an observed dataset and a well-defined scientific question.

Continue recording source identity, scientific use, adoption evidence,
documentation, data leads and uncertainties. Source discovery can proceed while
a separate prototype investigates task extraction from one known repository;
candidate status does not depend on that prototype succeeding.
