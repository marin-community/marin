# Computational biology task catalog

This is the maintained repository, skill, and format inventory for
[Marin #9257](https://github.com/marin-community/marin/issues/9257). The issue holds
program decisions, collection policy, and results; this file holds the detailed
coverage and source evidence. [Generator documentation](bio-tasks.md) lists the
implemented recipes and how to inspect their examples.

The implementation has **144 recipes across 13 domains**, with one task per
recipe by default and one train split. The [implemented recipe matrix](bio-task-recipes.md)
records the supplied formats, skills, and source-repository mappings. Sections below
also retain candidate capabilities beyond the current implementation. Source inspection
and host oracle validation do not establish runtime validity or training benefit.
Scientific review, native tool execution, and Harbor container validation remain separate checks.

Update this catalog when a source assessment, required skill, or format profile
changes. Preserve revision pins and limitations. Keep the generator's
`source_inventory.json` synchronized with repository assessments, and its recipe
metadata synchronized with implemented coverage. Corpus manifests pin code and
inventory hashes; branch links show the latest catalog, while commit permalinks
identify the catalog used by a release.

## Adoption metadata

The original [50-package inventory](computational_biology_bioinformatics_packages.md)
is vendored unchanged, including rank, interface, role, download evidence, stars,
canonical-paper citation counts/links, and metric caveats. Its counts are a
**2026-07-28 snapshot**, not newly measured figures. Bioconda lifetime downloads,
PyPI monthly downloads, stars, and paper citations are distinct signals; retain
missing values and avoid summing them into one score. Use adoption to inform
coverage, alongside scientific value and verifiability.

The numbered assessments below correspond to the original 50 rows. The machine
inventory retains the reported metric strings, snapshot date, path, and hash.
If metrics are refreshed, record dates/windows and sources for the new observations;
preserve this original snapshot for comparisons.

## Repository execution coverage

All 50 repositories have explicit scientific-operation mappings. The user-facing
requirement includes **actual CLI/API use for all 50**, with independent verification
of biological outputs. A recipe mapping, a successful import, or a version command
does not satisfy that requirement. Runtime tests must execute a bounded data operation,
record the package version and environment digest, and retain the output and grading result.

The authoring recipes have input-reading Python, R or native-tool oracles.
Native CLI/API execution is recorded separately below. **39 of 50 packages have passing native checks**: 34 on three reference cases each,
IQ-TREE, FastTree and RAxML on one shared observed COX1 alignment, and HMMER and MMseqs2 on a complete observed bacterial proteome. The first CoreWeave run passed 22 packages; correction batches
on an existing reserved TRC host in `us-central2` passed 12 more, with outputs
retrieved from regional GCS. MAFFT and MUSCLE each passed three real-protein alignments. MAFFT required the
same version from conda-forge after a Bioconda channel-priority conflict; the failed
installation remains in the evidence. Eleven repositories still need scripts. Picard and fastp passed on
observed ENA ERR266411 read pairs; fastp verification checks complete output FASTQ
records as well as the JSON selection summary.
The [machine-readable evidence](../../experiments/post_training/bio_tasks/native_validation.json)
indexes checksum-pinned files under `native_validation_runs/`, retaining resolved
package artifacts, input/output hashes, commands, grades and failures. Earlier host
checks remain as separate runs. The COX1 oracle reran all three packages in 1,028.8 seconds, checking three complete
trees, 3,916 distance rows, three RF comparisons and ten rejected artifact controls.
That case represents 94 source accessions through 89 distinct proteins and a fixed
558-column MAFFT alignment. Different search optimizations and a single-gene sample
do not establish species-tree truth or justify ranking raw likelihoods. These checks
do not establish complete benchmark workflows. The COX1 task also passed in Harbor:
one fresh oracle finished in 622.3 seconds, and changed-tree and missing-distance-row
controls received zero with correct summaries. The enclosing worker failed on a
redundant sandbox deletion (HTTP 409); a subsequent paginated check found no owned
sandboxes remaining. The [container record](../../experiments/post_training/bio_tasks/container_validation.json)
retains both the passing validation and the cleanup failure.
Snakemake executes included lanes grouped by sample; MultiQC parses supplied
FastQC reports before comparison with raw FASTQ counts. FastQC passed all three
cases; Nextflow passed after adding `ps` for its metrics collector. API/CLI usage
follows the [Snakemake](https://snakemake.readthedocs.io/en/stable/executing/cli.html),
[MultiQC](https://docs.seqera.io/multiqc/usage/scripts),
[FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/Help/3%20Analysis%20Modules/1%20Basic%20Statistics.html),
and [Nextflow](https://training.nextflow.io/latest/side_quests/splitting_and_grouping/) documentation.
The maintained machine record is `experiments/post_training/bio_tasks/repository_coverage.json`.
Downloads, stars, and citations remain available in the unchanged
[original 50-package inventory](computational_biology_bioinformatics_packages.md).

| # | Repository | Implemented scientific operations | Actual CLI/API execution |
|---:|---|---|---|
| 1 | [BLAST+](https://github.com/ncbi/ncbi-cxx-toolkit-public/blob/cf49184dc38476b1c9f605c38f47758a96e72d6b/src/algo/blast/unit_tests/api/bl2seq_unit_test.cpp) | `dna-unique-mapping`, `protein-local-search` | 3 reference checks passed (`dna-unique-mapping`) |
| 2 | [SAMtools](https://github.com/samtools/samtools/blob/664e3b5098a12bd5faca637fdc111ba90c21e135/doc/samtools-depth.1) | `sam-cigar-coverage`, `sam-inclusion` | 3 reference checks passed (`sam-cigar-coverage`) |
| 3 | [BWA](https://github.com/lh3/bwa/blob/d82444c17edc2384420409f85557c6ae84019732/example.c) | `dna-unique-mapping` | 3 reference checks passed (`dna-unique-mapping`) |
| 4 | [Bowtie 2](https://github.com/BenLangmead/bowtie2/blob/58e34bffd389d7ead6542b439784d9def92c6172/scripts/test/regressions.py) | `sam-pair-concordance`, `dna-unique-mapping` | 3 reference checks passed (`dna-unique-mapping`) |
| 5 | [DESeq2](https://github.com/thelovelab/DESeq2/blob/9e885b581380291797f2777145c395f50aaaa72b/tests/testthat/test_model_matrix.R) | `design-estimability`, `bulk-size-factors`, `sample-sheet-lanes`, `real-rnaseq-size-factors`, `real-rnaseq-normalized-contrast`, `real-rnaseq-differential-expression`, `real-rnaseq-go-enrichment`, `real-rnaseq-population-interaction` | 3 size-factor, 6 fitted-model/enrichment and 1 population-interaction checks passed |
| 6 | [STAR](https://github.com/alexdobin/STAR/blob/b1edc1208d91a53bf40ebae8669f71d50b994851/extras/tests/scripts/checkCellReadsStats_vsMatrix.awk) | `matrixmarket-cell-qc`, `sam-junction-support`, `umi-deduplication` | 3 reference checks passed (`dna-unique-mapping`) |
| 7 | [BEDTools](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/test/bed12tobed6/test-bed12tobed6.sh) | `bed12-exons`, `bed-union-coverage`, `bed-complement`, `real-genome-overlap`, `real-genome-promoters` | 3 reference checks passed (`bed12-exons`) |
| 8 | [GATK](https://github.com/broadinstitute/gatk/blob/0cde69eed30339f5978cbb1ac6e5cf3662f9e1f8/src/test/java/org/broadinstitute/hellbender/tools/walkers/filters/VariantFiltrationIntegrationTest.java) | `vcf-site-filtering`, `vcf-genotype-masking` | 3 reference checks passed (`vcf-site-filtering`) |
| 9 | [pysam](https://github.com/pysam-developers/pysam/blob/ba2e6c124398bdcd963db741d6f01164fed4f9b7/tests/AlignmentFilePileup_test.py) | `sam-allele-pileup`, `sam-inclusion` | 3 reference checks passed (`sam-cigar-coverage`) |
| 10 | [MAFFT](https://github.com/GSLBiotech/mafft/blob/26ecaba0130b533cf06a29200f0fb40829c00101/test/script) | `alignment-sum-of-pairs`, `alignment-column-filter`, `real-protein-alignment`, `real-cox1-tree-comparison` | 3 reference checks passed (`real-protein-alignment`) |
| 11 | [HMMER](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/testsuite/i13-msa-integrity.pl) | `hmmer-domain-extraction`, `real-proteome-domain-search` | 1 observed proteome reference and artifact check passed |
| 12 | [Seurat](https://github.com/satijalab/seurat/blob/586015abde10618ecb32d3fe632267a83317a08d/tests/testthat/test_data_manipulation.R) | `matrixmarket-log-normalization`, `matrixmarket-feature-filtering` | 3 reference checks passed (`matrixmarket-log-normalization`) |
| 13 | [minimap2](https://github.com/lh3/minimap2/blob/3c28777e7e2dcc90f825de1b9f17a89cca7d4452/example.c) | `paf-query-coverage`, `dna-unique-mapping` | 3 reference checks passed (`dna-unique-mapping`) |
| 14 | [BCFtools](https://github.com/samtools/bcftools/blob/edf7fd96c5da562ecfd99fb7f9e4b9eb597aeae8/test/fill-tags-VAF.out) | `vcf-allelic-depth`, `vcf-multiallelic-splitting`, `vcf-minimal-representation` | 3 reference checks passed (`vcf-allelic-depth`) |
| 15 | [Picard](https://github.com/broadinstitute/picard/blob/c2a483d497d1b0fe6d0ab518b1b32fe98fad0741/src/test/java/picard/sam/FilterSamReadsTest.java) | `sam-fragment-counts`, `sam-pair-concordance`, `real-fastq-quality-yield` | 3 reference checks passed (`real-fastq-quality-yield`) |
| 16 | [Snakemake](https://github.com/snakemake/snakemake/blob/91763d644db0a6051c40014fa8ffad340f7d39a0/tests/test_expand.py) | `sample-sheet-lanes` | 3 reference checks passed (`sample-sheet-lanes`) |
| 17 | [HTSlib](https://github.com/samtools/htslib/blob/d3cc9553d89dc34239afb7145b06c0dd818c0219/test/faidx/faidx.tst) | `fasta-indexed-regions`, `real-genome-promoters` | 3 reference checks passed (`fasta-indexed-regions`) |
| 18 | [FastQC](https://github.com/s-andrews/FastQC/blob/87fb3364a2f37115833d678648926d41e184f0b1/uk/ac/babraham/FastQC/Modules/SequenceLengthDistribution.java) | `paired-read-qc`, `fastqc-report-reconciliation`, `real-fastq-cycle-quality`, `real-fastq-quality-yield` | 3 reference checks passed (`fastqc-report-reconciliation`) |
| 19 | [Biopython](https://github.com/biopython/biopython/blob/08fc09086afe0b57215d2515660e0c032b55c0dd/Tests/test_SeqFeature.py) | `strand-extraction`, `gtf-splicing`, `gff-cds-translation`, `real-mmcif-chain-geometry`, `real-mmcif-contact-degree`, `real-genome-cds-extraction`, `real-genome-translation`, `real-genome-gc3`, `real-genome-codon-counts`, `real-genome-promoters`, `real-genome-restriction-digest`, `real-cox1-tree-comparison` | 3 reference checks passed (`gtf-splicing`) |
| 20 | [Nextflow](https://github.com/nextflow-io/nextflow/blob/17f18779266767b16bca51af71522e28adf5cff6/modules/nextflow/src/test/groovy/nextflow/extension/GroupTupleOpTest.groovy) | `sample-sheet-lanes` | 3 reference checks passed (`sample-sheet-lanes`) |
| 21 | [DIAMOND](https://github.com/bbuchfink/diamond/blob/5e25acaf40e6b9883636c5c564306fe77210de53/CMakeLists.txt) | `protein-local-search` | 3 reference checks passed (`protein-local-search`) |
| 22 | [PLINK / PLINK 2](https://github.com/chrchang/plink-ng/blob/a25a0d6438b61b1951cd6d7eb209db8b79687581/2.0/Tests/TEST_GRM_MAF/run_tests.sh) | `vcf-sample-qc`, `genotype-hwe` | 3 reference checks passed (`vcf-sample-qc`) |
| 23 | [Scanpy](https://github.com/scverse/scanpy/blob/0d5fd16234865619d2f5097d33fc4281900a2bc2/tests/test_qc_metrics.py) | `matrixmarket-cell-qc`, `donor-counts`, `cell-fractions`, `real-singlecell-read-qc` | 3 small checks and 1 full observed study passed |
| 24 | [MultiQC](https://github.com/MultiQC/MultiQC/blob/fdc68d394849f69b67b6e6e13ebe907504ed534b/multiqc/modules/samtools/tests/test_flagstat.py) | `fastqc-report-reconciliation`, `real-fastq-cycle-quality`, `real-fastq-quality-yield` | 3 reference checks passed (`fastqc-report-reconciliation`) |
| 25 | [edgeR](https://github.com/bioconductor-source/edgeR/blob/8986864d8f92dac37925ef641fcd6c4161130551/R/cpm.R) | `bulk-cpm-filter`, `real-rnaseq-library-qc`, `real-rnaseq-cpm-filter` | 3 reference checks passed (`bulk-cpm-filter`) |
| 26 | [limma](https://github.com/bioconductor-source/limma/blob/14eabaeb695945b45ceb885ac8d4c61232639ea5/R/contrasts.R) | `adjusted-linear-effect`, `paired-treatment-effect` | 3 reference checks passed (`adjusted-linear-effect`) |
| 27 | [SPAdes](https://github.com/ablab/spades/blob/808b87dade1300ecaa712429955ccba7bfb286f4/src/projects/spades/pipeline/spades_pipeline/supplemetary/check_test_script.py) | `assembly-nx`, `assembly-gap-runs` | Pending |
| 28 | [IQ-TREE](https://github.com/iqtree/iqtree2/blob/a00094e03d1ae984e1497e16738f91514df8c366/example/example.nex) | `alignment-partitions`, `real-cox1-tree-comparison` | 1 observed COX1 reference check passed |
| 29 | [FastTree](https://github.com/morgannprice/fasttree/blob/a5a2723ea1e64faf3da7ea514521cfa348891add/CompareTree.pl) | `newick-distances`, `newick-monophyly`, `newick-split-support`, `real-cox1-tree-comparison` | 1 observed COX1 reference check passed |
| 30 | [RAxML](https://github.com/stamatak/standard-RAxML/blob/36ec36110631c34692abcd4f24ca7b3e2fea742a/usefulScripts/bsBranchLengths.pl) | `newick-split-support`, `real-cox1-tree-comparison` | 1 observed COX1 reference check passed |
| 31 | [cutadapt](https://github.com/marcelm/cutadapt/blob/4927632f7c546dd290c53501c8417f909252befe/tests/test_trim.py) | `fastq-adapter-trimming`, `fastq-quality-trimming`, `real-fastq-fixed-trim` | 3 reference checks passed (`fastq-adapter-trimming`) |
| 32 | [Salmon](https://github.com/COMBINE-lab/salmon/blob/5515b7f05a90341b6652adfdb807e7cf14295518/crates/salmon-cli/tests/output_contract.rs) | `transcript-tpm` | Pending |
| 33 | [kallisto](https://github.com/pachterlab/kallisto/blob/4e9f29cf3b021260415430c057a22469ca081391/test/Snakefile) | `transcript-tpm`, `sample-sheet-lanes` | Pending |
| 34 | [StringTie](https://github.com/gpertea/stringtie/blob/d1dc38ddb681089b2e8faaabdcfee772af6fb033/prepDE.py3) | `gtf-coverage-counts` | Pending |
| 35 | [fastp](https://github.com/OpenGene/fastp/blob/8a2397b6628ae14127efdb7566f67fc05f9aea56/src/filter.cpp) | `paired-read-qc`, `fastq-adapter-trimming`, `real-fastq-pair-filter`, `real-fastq-fixed-trim`, `real-fastq-cycle-quality`, `real-fastq-quality-yield` | 3 reference checks passed (`real-fastq-pair-filter`) |
| 36 | [SRA Toolkit](https://github.com/ncbi/sra-tools/blob/434ae787c86e32e7faa5e80417a342c365fa03b0/test/external/fasterq-dump/fq_tests/split3.sh) | `sra-spot-export` | Pending |
| 37 | [deepTools](https://github.com/deeptools/deepTools/blob/cde2aa7938cb4af6fe28de1504f94f6928344342/pydeeptools/deeptools/test/test_countReadsPerBin.py) | `bedgraph-weighted-signal`, `sam-cigar-coverage`, `sam-fragment-counts` | Pending |
| 38 | [VCFtools](https://github.com/vcftools/vcftools/blob/1f87a83402ffd17ea2723a456b7edf5d80b4a22c/src/perl/fill-an-ac) | `genotype-alleles`, `vcf-sample-qc` | 3 reference checks passed (`vcf-sample-qc`) |
| 39 | [SnpEff](https://github.com/pcingola/SnpEff/blob/1db15998ea6aad93a35848aca0f6cba81cd36738/src/test/java/org/snpeff/snpEffect/testCases/unity/TestCasesSnps.java) | `variant-coding-consequences` | Pending |
| 40 | [Ensembl VEP](https://github.com/Ensembl/ensembl-vep/blob/cee181c2a1bb31900a0b7526168c67577fb23928/t/AnnotationSource_File_GTF.t) | `variant-coding-consequences`, `gtf-splicing` | Pending |
| 41 | [MACS2 / MACS3](https://github.com/macs3-project/MACS/blob/ece08963b6a30f4de0c5a5e684513f876b788d2c/test/test_Pileup.py) | `bedgraph-threshold-peaks` | 3 reference checks passed (`bedgraph-threshold-peaks`) |
| 42 | [MMseqs2](https://github.com/soedinglab/MMseqs2/blob/d401e78c2d18a822cdb1527d7464a043f6035a15/data/workflow/easycluster.sh) | `sequence-identity-clusters`, `real-proteome-clustering` | 1 observed proteome reference and artifact check passed |
| 43 | [MUSCLE](https://github.com/rcedgar/muscle/blob/29aa0671d0e46c862457749c7f2d87f29007b8eb/test_scripts/check_results.py) | `alignment-sum-of-pairs`, `alignment-consensus`, `real-protein-alignment` | 3 reference checks passed (`real-protein-alignment`) |
| 44 | [Kraken 2](https://github.com/DerrickWood/kraken2/blob/8c190b1b668825935dbf6dee5f969227dc8269bb/src/reports.cc) | `taxonomy-counts`, `taxonomic-lca` | Pending |
| 45 | [BUSCO](https://gitlab.com/ezlab/busco/-/blob/cd071053c38c5060f75d0b370cb66c4edc8e59a1/src/busco/busco_tools/hmmer.py) | `busco-summary` | Pending |
| 46 | [UCSC Kent utilities](https://github.com/ucscGenomeBrowser/kent/blob/0f58b0eef93be6d6d3b26b9e2b99261d558d67df/src/utils/bedGraphToBigWig/tests/makefile) | `bedgraph-weighted-signal`, `bed12-exons` | 3 reference checks passed (`bedgraph-weighted-signal`) |
| 47 | [GenomicRanges](https://github.com/Bioconductor/GenomicRanges/blob/44c311c711b9a5a5d6db070a8f3210819e4bc9de/inst/unitTests/test_findOverlaps-methods.R) | `interval-overlap`, `bed-nearest-features`, `bed-stranded-promoters`, `real-genome-overlap`, `real-genome-promoters` | 3 reference checks passed (`interval-overlap`) |
| 48 | [Biostrings](https://github.com/Bioconductor/Biostrings/blob/fb0cd89830abd054cf2681d6bc6c929981e07b21/tests/testthat/test-translate.R) | `fasta-six-frame-translation`, `gff-cds-translation`, `fasta-motif-hits`, `real-genome-cds-extraction`, `real-genome-translation`, `real-genome-gc3`, `real-genome-codon-counts`, `real-genome-promoters` | 3 reference checks passed (`fasta-six-frame-translation`) |
| 49 | [pybedtools](https://github.com/daler/pybedtools/blob/efb8534c11ca6b45a6cd173ff3b3d1bf754e34a1/pybedtools/test/test_1.py) | `strand-extraction`, `interval-overlap`, `real-genome-cds-extraction`, `real-genome-overlap`, `real-genome-promoters` | 3 reference checks passed (`strand-extraction`) |
| 50 | [nf-core/tools](https://github.com/nf-core/tools/blob/eb2f709090f4054f45437c34049ea2068567c339/tests/pipelines/test_schema.py) | `sample-sheet-lanes` | Pending |



## Package reference run

The first 2026-09-23 CoreWeave run installed all 30 environments and attempted 90 cases.
Of those, 68 passed, 10 completed with answer mismatches, and 12 failed execution.
All requested cases must pass before a package counts as checked; record the number of cases. The reference
scripts ran on one CPU worker with serial environments and no automatic retries.
Private references stayed local; grading used the frozen contracts without
changing tolerances. That run used small component fixtures.

| Package | Cases passing | Observed failure |
|---|---:|---|
| Seurat | 0/3 | An all-zero barcode produces null normalized values; the task requires zeros. |
| minimap2 | 2/3 | The 37-base reads in one instance produce no accepted full-length placements under the current options. |
| BCFtools | 0/3 | Printed VAF fractions have six significant digits and fail the numerical contract. |
| HTSlib | 0/3 | `pkg-config` cannot resolve the required `zlib` package. |
| Nextflow | 0/3 | The task-metrics collector requires `ps`, which is absent. |
| PLINK 2 | 0/3 | The requested 512 MiB workspace is below its 640 MiB minimum. |
| limma | 0/3 | R data-frame row names add `_row` to the JSON, violating the schema. |
| VCFtools | 0/3 | It rejects the VCFv4.3 header and accepts v4.0 through v4.2. |

These failures distinguish environment requirements, output serialization, and
package semantics from the independent scientific contract. Correct the reference
scripts or environment definitions and rerun affected cases before changing their
status. Review VCF version compatibility and zero-count behavior explicitly;
do not discard those inputs or relax verification to claim a pass.

TRC correction runs raised the total to 30/30 implemented packages passing three
cases each; the table above preserves the first run's failures. Corrections retain
empty Seurat columns as zero, remove limma row names, lower minimap2's alignment-score
threshold for short controls, add HTSlib development metadata and Nextflow's `ps`,
and use PLINK's minimum workspace and normalize its `#IID` header.
The first TRC launch failed before package installation because `/tmp` is mounted
`noexec`. Executable package environments now use the task work mount. Later
runs exposed missing `liblzma` development metadata and an absent `CC` variable;
HTSlib compilation now invokes the GCC executable supplied by its package environment.
Earlier failures remain in the per-run evidence, alongside subsequent results.
The VCFtools reference explicitly projects the supported diploid SNP/GT records to VCF 4.2 and rejects other profiles.
The BCFtools reference now queries integer AD values and computes their exact
ratios in Python. That checks native AD extraction; it does not validate the
`fill-tags` plugin's floating-point VAF output. No answer tolerance was changed.

## Skills

| Seed family | Skills to exercise | Independent verification |
|---|---|---|
| Strand-aware sequence extraction | Coordinate conventions, strand, transcript structure | Slice and complement generated FASTA; check exact sequence and coordinates. GC alone misses reverse-complement errors. |
| Interval overlap | Boundary rules, feature identity, deduplication | Direct interval arithmetic on fresh peaks/exons; distinguish qualifying features from overlapping pairs. |
| BAM inclusion | Flag/quality filters, read versus fragment counting | Compare selected records and counts against a generated read ledger, independently of library defaults. |
| Donor-level count aggregation | Sample/feature joins, raw counts, biological replication | Check every donor/gene integer and contributing cell count. Pooling donors, means, or transformed X must fail. |
| Patient-level cell fractions | Population, visit, and denominator selection | Check numerators and denominators from generated annotations; vary cell yields and specify zero-denominator behavior. |
| Design identifiability | Covariate adjustment, confounding, valid stopping decisions | Check estimability of the requested contrast against the reference design. Accept equivalent coding; dropping batch to force a result must fail. |
| Adjusted patient association | ID alignment, missingness, signed effects, adjustment | Independently fit a prescribed model on fresh tables; check included patients, slope, and sample size. |
| Enrichment background | Measured-gene universe, term eligibility, multiple testing | Independently calculate contingency counts, hypergeometric tails, and Benjamini–Hochberg adjustment for the full eligible term set. |
| Informative alignment sites | Site-state counting, gaps, ambiguous bases | Directly count columns under explicit conventions; distinguish variable from informative sites. |
| Tree branch summaries | Internal/terminal branches, root conventions | Sum branches from generated trees; check internal and total lengths as well as their ratio. |
| Read QC and trimming | Quality encodings, adapter overlap, mate identity, threshold boundaries | Independently compute retained IDs and sequence/quality slices from generated reads. Specify error and paired-read policies. |
| Local search and mapping | Strand, alignment coordinates, identity, coverage, ambiguity | Use planted unique matches and independently replay alignment operations. State acceptable ties; a match to one aligner's output is insufficient. |
| Genotype QC | Allele versus sample denominators, missing calls, ploidy, filter scope | Count called alleles and selected samples/variants from a generated genotype ledger. Distinguish global from analysis-specific filtering. |
| Expression normalization and transcript joins | Effective library size, transcript/gene identity, abundance units | Recompute CPM/TPM and gene-level sums from supplied counts, lengths, and mappings. Estimated abundance is not observed molecule count. |
| Coding variant consequences | Transcript choice, CDS phase, strand, genetic code | Reconstruct reference and alternate codons on fresh annotated sequences and compare translations; report per-transcript consequences. |
| Coverage and signal intervals | Read/fragment/base counting, shifts, normalization, zero bins | Accumulate generated intervals directly, then apply specified scaling and threshold/merge rules. Avoid image-based grading. |
| Assembly and alignment assessment | Sequence completeness, equivalent representations, residue correspondences | Compare full sequences or independently defined correspondence scores. Length, N50, or residue preservation alone cannot establish reconstruction correctness. |
| Sequence clustering | Identity/coverage criteria, partition membership, representative choice | Use small, clearly separated generated families; check all memberships and sequences modulo allowed representative choices. |
| Taxonomic and completeness summaries | Hierarchical counts, ortholog identity, overlapping categories | Sum assignments through a frozen taxonomy; count BUSCO ortholog IDs rather than hit rows, with explicit denominators. |
| Workflow and format repair | Sample/lane joins, coordinate conversion, input validity, artifact identity | Check manifest-derived groups and complete biological outputs on fresh small inputs, alongside schema and format checks. |
| Image-derived measurements | Object identity, physical scale, channels, intensity and spatial aggregation | Generate labeled masks and intensity arrays; independently sum pixels and coordinates to check object areas, centroids, and per-sample summaries. State connectivity, boundary handling, and pixel spacing; no visual judge. |

The [scikit-image example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html)
supplies imaging operations. Distinguish measuring supplied masks from segmenting images.

## ID workflow coverage and input realism

The 142 recipes include twenty-seven using real observations and 115 simulated component
controls. They do **not** establish coverage of complete ID benchmark workflows. Repository count, format count, and successful
package checks measure different things from workflow coverage. The following
assessment uses public benchmark descriptions and the program's prior source
inspection; it is a qualitative gap analysis, not a benchmark coverage score.

The observed protein-domain workflow searches all 4,403 reviewed proteins of
E. coli K-12 with three pinned Pfam models. HMMER 3.4 found six domains; Biopython
reference measurements and an independent parser agree on domain boundaries,
sequences, scores and all protein coverage rows. Corrupted domain sequences and
false coverage on an unmatched protein fail. The
[native record](../../experiments/post_training/bio_tasks/native_validation_runs/a619ee74f5b8.json)
retains exact model/proteome hashes and the serial package build. The reference
and validation each took under one second on a reserved TRC CPU. Only the supplied
profiles are searched; this is not complete functional annotation. Both phases
passed before the parent job failed during unrelated single-cell packaging due
to a missing SciPy import dependency. That failure is preserved. Full-corpus package checks now pass.
A fresh Harbor oracle completed in 0.8 seconds; separate verifiers accepted its
complete outputs and rejected changed domain sequence and false zero-hit coverage
with identical correct summaries.

The full observed GSE81682 QC workflow supplies 1,920 cells and 46,170 features.
Actual Scanpy and an independent streaming oracle agree on both complete QC tables
and all 17,332,418 retained matrix entries; changing one count fails with correct
summaries. Reference computation took 97.8 seconds, oracle computation 219.4 seconds
and artifact grading 149–151 seconds on a reserved TRC CPU host. The
[native record](../../experiments/post_training/bio_tasks/native_validation_runs/40ed93019b52.json)
pins inputs, packages, artifacts and resources. This checks cell/gene filtering
and identity-preserving sparse export. Normalization, annotation, clustering and
donor-level comparisons remain separate workflow gaps. The one-recipe package build
passed its oracle and all 15 negative controls in 591 seconds with 971 MiB peak RSS.
The fresh Harbor oracle completed in 108.2 seconds; the separate verifier took
78.3 seconds. Changed-count and missing-feature-row controls received zero with
identical correct summaries. All six single-cell/domain cases passed. The wrapper
failed after observing a sandbox still being deleted; an independent paginated
check found none remaining. The
[container evidence](../../experiments/post_training/bio_tasks/container_validation.json)
preserves both outcomes and the durable regional archive. The inspected biological
source does not establish exact benchmark independence.

The observed proteome clustering task runs MMseqs2 Linclust on all 4,403 proteins,
then extracts 4,184 unchanged native representatives. All membership rows and
cluster quantities are checked, including 4,029 singletons and 155 multimember
clusters. Biopython reference measurements and a fresh native run with independent
standard-library measurements agree; nine missing, duplicated or changed-artifact
controls fail with correct summaries. Reference execution took 1.2 seconds and
oracle plus artifact checks took 3.9 seconds. The
[native record](../../experiments/post_training/bio_tasks/native_validation_runs/9d28dd15e082.json)
pins all 26 package artifacts and retains the first launcher dependency failure.
These heuristic similarity clusters do not establish orthology or function.
The packaged task passed two positive checks and 17 negative controls in 6.1 seconds
with 221 MiB peak RSS; its archive is preserved in regional GCS. A fresh Harbor
oracle completed in 2.1 seconds; its separate verifier accepted all artifacts.
Changed residues, a missing representative and a wrong membership each received
zero with byte-identical correct summaries. Cleanup completed with no owned
sandboxes remaining. Benchmark-lineage screening and scientific review remain open.

The [agentic source inventory](../../experiments/post_training/bio_tasks/benchmark_sources.json)
records all 48 benchmark/protocol rows from the spreadsheet's Agentic (Harbor) tab:
42 provisional ID and six OOD. Protocol variants and subsets overlap; these are
not counts of independent datasets. The non-agentic tab is outside this program.
Terminal-Bench-Science (full/lite), GeneBench (original/Pro), BioMysteryBench and
SciCode retain their OOD assignments and cannot supply training task formulations.

The [task-level index](../../experiments/post_training/bio_tasks/benchmark_coverage.json)
links one versioned file per inspected benchmark. Each ID record includes a stable
task ID, workflow pattern, required stages and formats, recipe mappings, evidence
and remaining gaps. Source revisions and hashes identify the inspected metadata;
benchmark answers and biological fixtures are excluded from training authoring.

| Benchmark inventory | Assessed ID task/protocol records | Advertised full suite, when larger |
| --- | ---: | ---: |
| [BixBench](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench.json) | 205 | — |
| [CompBioBench](../../experiments/post_training/bio_tasks/benchmark_tasks/compbiobench.json) | 100 | — |
| [BiomniBench-DA](../../experiments/post_training/bio_tasks/benchmark_tasks/biomnibench-da.json) | 50 | — |
| [Biomni-Eval1, tool-enabled](../../experiments/post_training/bio_tasks/benchmark_tasks/biomni-eval1.json) | All 433 IDs and source categories; 100 sequence/database endpoints assessed | — |
| [BioAgent](../../experiments/post_training/bio_tasks/benchmark_tasks/bioagent.json) | 10 | — |
| [scBench](../../experiments/post_training/bio_tasks/benchmark_tasks/scbench.json) | 6 | 195 |
| [SpatialBench](../../experiments/post_training/bio_tasks/benchmark_tasks/spatialbench.json) | 16 | 159 |
| [EpiBench](../../experiments/post_training/bio_tasks/benchmark_tasks/epibench.json) | 7 | 106 |
| [Bio-Task Bench](../../experiments/post_training/bio_tasks/benchmark_tasks/bio-task-bench.json) | 34 | — |
| [ScienceAgentBench, bioinformatics](../../experiments/post_training/bio_tasks/benchmark_tasks/scienceagentbench.json) | 27 | — |
| [CORE-Bench, biomedical](../../experiments/post_training/bio_tasks/benchmark_tasks/core-bench.json) | 25 | — |
| [DiscoveryBench, biology](../../experiments/post_training/bio_tasks/benchmark_tasks/discoverybench.json) | 26 | — |
| [VariantBench](../../experiments/post_training/bio_tasks/benchmark_tasks/variantbench.json) | 8 (6 main, 2 supplemental) | 118 main |
| [Liu et al., single-cell](../../experiments/post_training/bio_tasks/benchmark_tasks/liu-single-cell.json) | 63 (50 main, 13 additional datasets) | — |
| [sc-HeurekaBench](../../experiments/post_training/bio_tasks/benchmark_tasks/sc-heurekabench.json) | 130 (64 open-answer, 66 multiple-choice) | — |
| [scBench-Long](../../experiments/post_training/bio_tasks/benchmark_tasks/scbench-long.json) | 4 | Unverified |
| [SpatialBench-Long](../../experiments/post_training/bio_tasks/benchmark_tasks/spatialbench-long.json) | 4 | Unverified |
| [BixBench-Verified-50](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench-verified-50.json) | 50 (all overlap original IDs) | — |
| [CellBench](../../experiments/post_training/bio_tasks/benchmark_tasks/cellbench.json) | 50 analysis-planning contexts | — |
| [BioML-bench](../../experiments/post_training/bio_tasks/benchmark_tasks/biomlbench.json) | 406 registry definitions; 24 selected by the released experiment | — |
| [BixBench3](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench3.json) | 20 IDs; 19 full prompts and 131 output artifacts assessed, one objective excluded | — |
| [BAISBench](../../experiments/post_training/bio_tasks/benchmark_tasks/baisbench.json) | 193 discovery questions across 41 studies; 15 annotation dataset identities | — |
| [PromptBio-Bench](../../experiments/post_training/bio_tasks/benchmark_tasks/promptbio-bench.json) | All 244 task definitions; 27 workflow families and 316 required-output declarations | — |
| [BioXArena](../../experiments/post_training/bio_tasks/benchmark_tasks/bioxarena.json) | All 76 task IDs, catalog objectives and scorers; full per-task prompts and data uninspected | — |
| [TxBench-PP](../../experiments/post_training/bio_tasks/benchmark_tasks/txbench-pp.json) | All 12 public prompts; input schemas and grader behavior uninspected | 100 |
| [TxBench-Ab](../../experiments/post_training/bio_tasks/benchmark_tasks/txbench-ab.json) | All six public prompts; inputs, ground truth and graders withheld | 100 |
| [TxBench-OD](../../experiments/post_training/bio_tasks/benchmark_tasks/txbench-od.json) | All four public prompts; inputs, ground truth and graders withheld | 113 |
| [DrugDiscoveryBench](../../experiments/post_training/bio_tasks/benchmark_tasks/drugdiscoverybench.json) | All 82 complete public prompts and endpoints assessed | 82; full-release ID equality unverified |

Of these 2,306 ID task/protocol/definition records, 47 have manually assessed component mappings,
2,258 are unmapped and one is excluded from authoring. None is marked workflow-validated. Other eligible sources
still need task-level inspection. The 90
[BioMysteryBench identifiers](../../experiments/post_training/bio_tasks/benchmark_tasks/biomysterybench.json)
are tracked separately as OOD, without workflow patterns or training mappings.

BiomniBench-DA `da-20-4` now maps to the population-interaction component on
GSE60450: all twelve libraries, 18,418 fitted genes, design and contrast tables,
within-population effects and a direct test of response differences. The
[native run](../../experiments/post_training/bio_tasks/native_validation_runs/5436d9bd3811.json)
passed two positive and 33 negative controls. Its two mouse populations and
developmental stages do not cover the four-human-cell-type drug-response endpoint,
dose/time/vehicle selection, HDF5 handling or biological interpretation.
Packaged-task checks, container execution and complete lineage screening remain pending.

Biomni-Eval1 has ten source categories and 20 assessed workflow patterns. Its
compound task name and task-specific ID identify an evaluation item; the global
row ID is a different field. The 100 LAB-Bench-derived questions cover sequence
operations and database queries. Other categories include association evidence,
perturbation screens and phenotype-based prioritization; their instance-specific
scientific evidence remains unreviewed. The inspected evaluator uses deterministic
answer checks, including a permissive gene-list intersection check. It has not
been executed, and no recipe coverage is assigned. Answers and benchmark sequence
fixtures are excluded from authoring.

DrugDiscoveryBench adds 27 patterns covering structural chemistry, assay selection,
multisource target evidence, sequence analysis and connected omics decisions.
Six questions require attachments absent from the public preview; four shared
structure accessions are tracked separately from task counts.
Full-release rubrics use an LLM judge; independent tasks need quantitative artifact
contracts. No mappings are assigned from category or tool-name similarity.

LABBench2's pinned dataset card declares 1,912 rows in its `all` configuration.
The public harness and card are inspected, but existing access lacks the dataset
gate grant. Its task rows are not included in the inventory count. Category
subsets and image/PDF variants must be reconciled before counting distinct tasks.

The COX1 recipe supplies per-tree treeness, length and pair-distance components
for 18 original BixBench endpoints and five overlapping Verified-50 protocols.
These 23 mappings share one biological input. Cross-gene and biological-group
aggregation, native PhyKIT agreement and benchmark-lineage screening remain
pending; method replicates do not substitute for gene or biological replicates.

The three TxBench public manifests and nontruncated repository trees agree on
12 preclinical, six antibody and four oligonucleotide examples, despite each README
claiming seven. Every public prompt has an assessed workflow pattern; the full
100/100/113-task suites are not enumerated. Grader type names do not establish
executable scoring behavior. The oligonucleotide PK answer schema omits several
outputs requested in its narrative, and the liver-safety ranking scores only IDs.
One preclinical multi-assay package is explicitly labeled synthetic. Fresh tasks
need complete artifact contracts and independent observed inputs. Source prompts,
answers, trajectories and biological fixtures remain outside training authoring.

BioXArena's 76 task identities agree across source graders, four launch lists
and the data-release manifest. Catalog objectives and executable scorer endpoints
are assessed; full prompts, input schemas and sample submissions remain
uninspected. The inventory distinguishes prediction metrics from broader workflow
names, records correlation masking and row-order requirements, and flags the
mitochondria-counting metric conflict: the catalog says MAE, while the pinned
grade function computes Spearman correlation. All 76 remain unmapped. The public
scorers use deterministic numerical metrics or exact labels, with no LLM judge.

PromptBio-Bench includes all 131 bioinformatics and 113 data-science task
definitions at the pinned release. Every question and required-output declaration
has been inspected and paraphrased into a task-specific workflow pattern. Input
formats inferred from filenames are labelled; biological input contents, evaluator
files and reference answers were not downloaded. All 244 tasks remain unmapped.
The source permits 60 minutes per task; that is a timeout, not measured runtime.
Adapted tasks still require a measured solve within 30 minutes. Modelling tasks
can contain internal train/test cohorts while the released task dataset keeps its
single train split. Plotting tasks need checks of the underlying results and
scientific encodings; an existing image file is insufficient.

BAISBench question stems have manually assessed workflow patterns. Task1 has
dataset identities and a broad annotation pattern; its scoring implementation
and input schemas remain uninspected. Discovery answer-choice grading does not
check biological artifacts, and some questions require external functional or
mechanistic evidence beyond expression matrices.

ScienceAgentBench includes all 27 Bioinformatics-domain tasks from its verified
release; 75 tasks from other disciplines are outside the selected scope. CORE-Bench
includes all 25 Medical Sciences capsules among 90 released capsules. Its three
reproduction protocols share capsule identity; each capsule is counted once.
Capsule patterns use public metadata; analysis scripts and raw schemas remain
uninspected. Source train/test partitions do not create additional splits here.
DiscoveryBench records 26 query variants grouped into ten claims across two
biological studies; these variants are not independent workflow coverage.
VariantBench exposes six main-suite examples and two supplemental neoantigen
examples. Keep the supplemental examples outside the main-suite denominator.
The Liu single-cell inventory has 50 main tasks and 13 additional dataset cases.
Its 13 `data1` prompts exactly duplicate main prompts and remain aliases. Prompt
hint variants do not add tasks; several saved embeddings do not establish the
biological endpoint described by their task.
Its gimVI task also identifies an excluded candidate input lineage: the 3,005-cell
Zeisel cortex data used by `scvi.data.cortex`. The source manifest records this
exclusion and primary evidence; downloading another copy would not make the
biological observations independent.

sc-HeurekaBench's six full, lite and TU protocol files contain 130 distinct
formulation/input records sharing 52 insight groups across 13 studies. Exact
aliases share one record; changed questions or declared input descriptions
retain separate records. These counts do not measure independent workflows.
Its open-answer judge cannot supply our acceptance rule: each adapted task needs
quantitative artifacts, deterministic grading and independent observed inputs.

scBench-Long and SpatialBench-Long each expose four public tasks. Their patterns
include paired RNA/TCR analysis, RNA–chromatin integration, cross-species
interaction hypotheses, cohort-design audits, lineage-guided spatial contrasts,
and section-aware spatial nulls. Published model-run counts are not task counts.
Coherent observed subsets must preserve biological replication and comparison
structure while fitting the solve-time limit.

BixBench-Verified-50 metadata was read through existing authorized gated access.
Its 50 IDs overlap original BixBench across 33 capsules; 17 question texts differ
and 33 are unchanged. Per-record links preserve the relationship without implying
50 additional workflows. Revisions affect model design, filtering, units and
denominators; capsule equivalence has not been verified.

CellBench's 50 research contexts request proposed single-cell analyses. Their
LLM judge compares proposals with hidden analysis ideas; the protocol does not
validate executed biological artifacts. Context-derived analysis topics are
recorded separately from executable requirements. Adapting a context requires an
independent observed study and a quantitative endpoint. The pinned agent runner
also references three CSV files absent from the release; source execution was not
attempted.

The first 815 ID records have assessed workflow prerequisites or source planning
requirements, with the distinction explicit. BioML-bench adds all 406 registry
folder definitions from its pinned alpha release; its experiment list selects 24.
Among 405 parseable configurations, 30 configured IDs occur twice; one further
configuration has invalid YAML. Folder IDs, configured IDs, shared assay lineage,
release selection and configuration defects remain separate fields. Five single-cell
protocols and two shared assay cross-validation protocols have been assessed;
individual assay interpretation and the remaining configuration-derived patterns
need review. The spatial-gene protocol uses simulated observations; the communication
protocol uses activity proxies. Neither supplies real training inputs or a biological
ground truth by itself. Public assay labels also mean that a prediction-file score
cannot establish whether each fold was excluded during fitting.

BixBench3 contributes 20 research-scale paper tasks. Nineteen full prompts have
assessed workflow stages and all 131 required output artifacts reviewed for identities,
row universes, units and relationships. The inventory records these decisions per task.
One pathogen-enhancement objective remains an excluded identifier. The source
grades artifacts deterministically and uses a separate process judge; our acceptance
rules require quantitative artifacts. Its reported runs average roughly eight hours,
so adaptations need an explicit analysis boundary and measured runtime. Multi-assay
joins, technical versus biological replication, missing observations and complete
output tables must survive that reduction. Simulated community-growth trajectories
remain distinct from observed microbial measurements.
Static review of the shared grader found that numeric agreement uses shared rows
and columns, duplicate keys retain the first row, and missing-value masks can omit
pairs. Adapted contracts require explicit completeness, uniqueness and missing-value
checks. Task-specific grading configurations packaged with ground truth remain
uninspected; the source benchmark has not been executed.

TargetVal's pinned task constructor and runner have been inspected, but its gene-level
IDs depend on external score tables and remain uninventoried. Requested sample counts
and repeated model runs are not a task manifest. Its sequential testing pattern needs
explicit nulls, denominators and stopping rules; permuted-data controls remain separate
from observed biological inputs. The source inventory records these limits without
adding task or execution coverage.

For BixBench,
shared workflow-family stages are distinguished from individual question endpoints;
this inventory does not add execution coverage. Four RNA-seq component mappings now
link the executed DESeq2/GO workflows and their Harbor artifact checks; ontology
simplification, endpoint-specific designs and biological source independence remain
unresolved. The inspection browser retains those validation references and records
generated-example checks separately in `example_validation_evidence`. These timings
are solver-check runtimes, not teacher latency measurements.

Twenty-seven recipes supply unchanged biological observations or declared observed subsets:

- **GSE60450:** 27,179 genes and 12 libraries from mouse mammary basal/luminal cells,
  with two biological replicates per population and stage. Tasks cover library QC,
  CPM filtering, DESeq2-convention size factors, and descriptive normalized contrasts.
  Counts and metadata require an identifier join because their orders differ.
  Normalization uses the full matrix; a 256-gene reporting panel bounds the answer.
  These component contrasts do not claim fitted differential-expression significance.
  Two connected workflows additionally fit all retained genes with DESeq2 and carry
  directional selections into a frozen GO-BP universe. Their six native cases pass
  complete QC, gene and term-table checks. Both pass in Harbor with the pinned
  Haswell BLAS kernel, and changed-intermediate controls fail. The earlier GO
  probability-field mismatch remains recorded; scientific and benchmark-lineage
  review remain open.
- **PDB 1UBQ, 1CRN and 4HHB:** complete deposited mmCIF inputs for per-chain geometry
  and residue contact degree. Construction references parse the paired PDB deposits;
  independent solvers parse mmCIF. Model, alternate-conformer and author-ID rules
  are explicit.
- **NC_001422.1, NC_001416.1 and NC_001604.1:** complete PhiX, lambda and T7 genomes.
  Seven recipes exercise CDS parts, overlapping genes, alternative starts, codon
  composition, upstream windows and circular restriction fragments. All 144 CDSs
  reproduce deposited protein translations using Biopython 1.86; independent solvers
  read the converted GFF3 and unchanged FASTA.
- **ERR266411:** 6,000 observed PhiX read pairs in three disjoint archive-order
  blocks. Five recipes cover paired filtering, fixed trimming, cycle quality,
  expected errors and yield. Filtering and trimming require native FASTQ outputs
  with exact ordered IDs, bases and qualities. These technical subsets do not
  establish biological replication or mixed-community metagenomic coverage.

- **UniProt globins:** six full-length proteins per task selected from ten reviewed
  alpha/beta globin entries, with species and sequence versions retained. Produce
  an aligned FASTA under an explicit affine-gap BLOSUM62 objective. Independent
  Biopython pairwise scores bound the objective; a separate center-star solution
  checks feasibility. Different qualifying alignments pass. This does not establish
  orthology or a species phylogeny. UniProt data are [CC-BY-4.0](https://www.uniprot.org/help/license).

- **UniProt COX1:** 94 reviewed metazoan accessions represented by 89 distinct,
  unchanged proteins in a fixed 558-column MAFFT alignment. The connected workflow
  infers three gene trees and compares their complete branches, distances and
  unrooted bipartitions. A separate Biopython calculation checks the input-reading
  oracle's measurements. The native oracle passed in 17.1 minutes and the Harbor
  oracle in 10.4 minutes with unchanged tolerances. Benchmark-lineage screening
  remains pending.

- **Mayo PBC study:** 418 baseline records and 1,945 follow-up observations. Three
  recipes cover randomized-cohort selection, composite-endpoint Kaplan-Meier
  estimation, adjusted Cox regression, and complete-pair visit comparisons. Cox
  references use Statsmodels and an independent standard-library solver. The
  longitudinal source contains corrected values and extended follow-up; its
  baselines are kept separate from the original baseline table. Missingness may
  be informative. Native R execution and scientific review remain pending.

[Source provenance](../../experiments/post_training/bio_tasks/data_sources.json)
records retrieval URLs, checksums, licenses and unchanged-content recompression.
Earlier genome/read accession checks found no matches in inspected permitted benchmark metadata;
full artifact-lineage screening is still pending, so these are authoring candidates.
No claim of independence is made merely because a query uses a different seed.

| Provisional ID benchmark | Current relevant components | Missing dependent workflow behavior |
|---|---|---|
| [BixBench](https://github.com/Future-House/BixBench) | Cohort selection, expression summaries, enrichment, phylogeny and image measurements | Discover and join files, choose the eligible biological population, perform the analysis, and derive a requested result from its outputs. Current tasks usually prescribe each operation separately. |
| [CompBioBench](https://github.com/Genentech/compbiobench-runner) | Common sequence, interval and expression operations | Broader tool selection, data acquisition and multi-step analyses. The offline corpus does not claim coverage of internet-dependent tasks. |
| [BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) | Patient/sample joins, composition, adjusted effects, expression and variant summaries | Connect raw/layer selection, QC, biological replication, contrast/model fitting, multiple testing, and the final association or composition result. |
| [BioAgent Bench](https://arxiv.org/abs/2601.21800) | Read QC, mapping conventions, transcript arithmetic, variant filtering, taxonomy and assembly summaries | Actual RNA-seq, variant-calling and metagenomics pipelines, including dependency repair and distractor/corrupt-input handling. Supplied alignments, domain hits and taxonomy assignments omit their upstream inference. |

Track workflow coverage with a fresh-data recipe, a dependency graph, independently
checked intermediate/final artifacts, actual tool execution, and measured attempt
time. Composing outputs must change the downstream answer: running independent
small commands in sequence does not establish workflow reasoning. No current row
above has been certified as an end-to-end workflow match.

Target real biological data for every task in the final training release; record
any necessary exception. Keep tiny corner-case inputs as correctness controls
outside that release. The inspected HMMER example
has one 65-residue protein and two supplied domain hits; its purpose is coordinate
extraction, not HMM search. Matrix Market examples have five features and four
cells. Such fixtures make errors inspectable but permit shortcuts and do not
exercise realistic data handling. In the 2026-09-23 reviewed 345-task build,
supplied inputs have a median size of 231 bytes, a 95th percentile of 1,829 bytes,
and a maximum of 7,835 bytes, excluding instructions and private references.
This is an authoring fixture corpus, not an approved training release.

For exported training instances, choose sizes appropriate to each operation and
include realistic sample structure, sparsity, noise, ambiguity, missingness and
file joins. One reviewed task per recipe is the default. Additional tasks need a
meaningful difference in data, study design or scientific decision; renaming
records, multiplying counts or shifting scores does not establish new coverage.
Choose input scale to preserve the biological problem and measure execution cost. Maintain the single
train split and the 30-minute attempt limit. Size targets remain uncalibrated
until measured on the selected task environment.

Large matrices, sequences and call sets require native output artifacts with
streaming or bounded deterministic checks. Keep JSON summaries within the 2 MiB
answer limit. Separate contracts verify complete FASTA records, FASTQ reads,
alignments, TSV tables, Newick trees and sparse MatrixMarket counts. Unaligned
FASTA verification checks every identity and residue while accepting record
reordering, line wrapping and letter-case changes. The full-study single-cell
workflow uses sparse matrix artifacts; the older normalization controls still
emit gene–cell pairs in JSON and must remain small. Do not silently truncate
results or relax scientific checks.

An independent read-only review on 2026-09-23 identified the component/workflow
gap, small inputs, insufficient method discrimination, and repeated targets across
seeds. It inspected representative code and examples; it did not scientifically
certify all recipes or run all packages. Review-driven fixture changes add
complete coding annotations with consistent reference lengths, asymmetric
normalization outliers, rank-deficient estimable designs, unequal
technical replication, and varying filtering/topology/FDR/mapping decisions.
Native package execution and biological workflow review remain separate gates.

## Biological data formats

Treat format literacy as an explicit skill axis and coverage dimension: identify,
read, validate, query, write, and convert native biological files while preserving
meaning. Use small bounded operations and conversions within analysis tasks.

| Format family | Semantics and candidate tasks |
|---|---|
| FASTA / FASTQ | Wrapped records, stable IDs, sequence alphabets, quality encoding, sequence/quality correspondence, mate pairing; extract sequences or filter paired reads. |
| BED / BED12, bedGraph / bigWig | Zero-based half-open intervals, strand, block-relative offsets, contig names, sorted/indexed queries; expand exons or convert annotations without coordinate drift. |
| SAM / BAM / CRAM | Headers/reference identity, flags, CIGAR reference/query consumption, clipping, mates, tags, quality filters, sort/index consistency; recover spans or count eligible fragments. |
| GFF3 / GTF | One-based closed coordinates, distinct attribute grammars, IDs/parents, multi-exon ordering, CDS phase/strand; join annotations to FASTA or convert to BED. |
| VCF / BCF | Header types/cardinalities, REF/ALT order, genotype allele indices, ploidy, phase, missing calls, site versus genotype filters; subset samples or compute allele counts without corrupting metadata. |
| PDB / PDBx-mmCIF | Atom/residue/chain identity, author versus standardized IDs, insertion codes, alternate locations, models, occupancy and coordinate units; select atoms or calculate specified distances with explicit selection rules. |
| Newick / NEXUS / Stockholm | Tree rooting/branch lengths, labels, alignment gaps and annotations; parse, summarize, or convert with declared equivalences. |
| Matrix Market / 10x, H5AD / AnnData, HDF5 / Zarr | Sparse orientation, implicit zeros, feature/barcode joins, axes, raw versus transformed layers, categorical/missing metadata; subset or aggregate without misaligning observations. |
| OME-TIFF / OME-NGFF | Axis/channel order, physical pixel spacing, labels and multiscale metadata; measure supplied masks and preserve units when reading or converting images. |

Pin format versions/profiles and stage necessary documentation offline. Use
[HTS specifications](https://samtools.github.io/hts-specs/),
[UCSC format conventions](https://genome.ucsc.edu/FAQ/FAQformat.html),
[Sequence Ontology's GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md),
and [wwPDB format documentation](https://www.wwpdb.org/documentation/file-format)
with its [mmCIF guide](https://mmcif.wwpdb.org/docs/user-guide/guide.html).

Verify syntax plus complete biological records against independent construction
ledgers. For conversions, check semantic equivalence and declared metadata retention;
byte identity and parser success alone are insufficient. Include boundary, missing,
and malformed cases paired with valid cases. Record input/output format profiles
and operations in task metadata and the browser. Report native-format coverage
separately: CSV/JSON summaries do not count as BAM, H5AD, Newick, or OME-TIFF tasks.

## Inspected repository and analysis sources

The 2026-09-22 source review covers all 50 repositories in the supplied inventory.
Representative tests, examples, workflow files, or implementations were read for
concrete operations, executable references, and limitations. The numbering preserves
the inventory, not implementation priority. These source inspections did not execute the upstream tasks or environments.
Locally validated recipes are listed in the generator documentation.

Each linked file is pinned to the inspected revision. For edgeR and limma, the
primary Bioconductor repositories were also inspected directly at the recorded
primary revisions (edgeR `f47ecb9ab434419a6474a70ab3cbc9c675706766`;
limma `57a8de7296ad733ac25d3e3c01de3fdddcd0a9ae`); the links expose older
browsable source snapshots. In edgeR,
offset handling differs between those revisions, reinforcing the need to pin the
chosen environment and specify normalization inputs.

| # | Repository and inspected evidence | Candidate seed and skill | Executable verification and limitations |
|---:|---|---|---|
| 1 | [BLAST+](https://github.com/ncbi/ncbi-cxx-toolkit-public), `cf49184`. [Pairwise alignment tests](https://github.com/ncbi/ncbi-cxx-toolkit-public/blob/cf49184dc38476b1c9f605c38f47758a96e72d6b/src/algo/blast/unit_tests/api/bl2seq_unit_test.cpp) | Local similarity search: find planted unique matches and report strand, coordinates, and identity. | Enumerate matches in fresh sequences; check aligned residues and spans. The inspected test uses accession-backed sequences: replace those inputs. E-values and heuristic recall need a separately specified contract. |
| 2 | [SAMtools](https://github.com/samtools/samtools), `664e3b5`. [Depth semantics and toy SAM](https://github.com/samtools/samtools/blob/664e3b5098a12bd5faca637fdc111ba90c21e135/doc/samtools-depth.1) | Read-depth calculation with explicit flag, base-quality, deletion, and overlapping-mate rules. | Compute depth from a generated SAM/CIGAR ledger. The manual distinguishes depth from mpileup defaults, including swapped quality-option meanings; matching their defaults is not an independent check. |
| 3 | [BWA](https://github.com/lh3/bwa), `d82444c`. [BWA-MEM API example](https://github.com/lh3/bwa/blob/d82444c17edc2384420409f85557c6ae84019732/example.c) | Map fresh, uniquely placed reads; recover coordinates, orientation, and edit operations. | Check against planted placements and independently replay CIGARs. Include reverse-strand and unmapped cases. Repeats, clipping, secondary alignments, and MAPQ prevent a universal exact-output oracle. |
| 4 | [Bowtie 2](https://github.com/BenLangmead/bowtie2), `58e34bf`. [Paired/unmapped-output regressions](https://github.com/BenLangmead/bowtie2/blob/58e34bffd389d7ead6542b439784d9def92c6172/scripts/test/regressions.py) | Partition paired reads into concordant and nonconcordant outputs without losing sample or mate identity. | Check every read ID and mate against a generated ledger, including deliberately discordant pairs. Upstream count-conservation tests motivate the seed but counts alone can accept swapped reads. |
| 5 | [DESeq2](https://github.com/thelovelab/DESeq2), `9e885b5`. [Model-matrix tests](https://github.com/thelovelab/DESeq2/blob/9e885b581380291797f2777145c395f50aaaa72b/tests/testthat/test_model_matrix.R) | Diagnose confounded designs and construct estimable contrasts; collapse technical sequencing runs by biological sample. | Use independent linear algebra for estimability and integer sums for collapseReplicates. Preserve biological replicates. Full dispersion estimation, shrinkage, and fitted DE tests require stronger numerical references. |
| 6 | [STAR](https://github.com/alexdobin/STAR), `b1edc12`. [STARsolo matrix/statistics checker](https://github.com/alexdobin/STAR/blob/b1edc1208d91a53bf40ebae8669f71d50b994851/extras/tests/scripts/checkCellReadsStats_vsMatrix.awk) | Reconcile barcode-level UMI and detected-gene summaries with a sparse count matrix. | Recompute per-barcode sums and nonzero features, including absent barcodes and unique/multimapping policy. The script prints discrepancies; turn them into complete failing checks. Full splice alignment additionally needs a genome index. |
| 7 | [BEDTools](https://github.com/arq5x/bedtools2), `614e9a5`. [BED12 block-expansion tests](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/test/bed12tobed6/test-bed12tobed6.sh) | Expand transcript blocks into exon intervals, preserving strand-aware exon numbering; compose with overlap or sequence extraction. | Use direct arithmetic on generated block starts and sizes. Check complete interval tuples and reverse-strand exon order. BED end coordinates are excluded; feature counts and overlap-pair counts differ. |
| 8 | [GATK](https://github.com/broadinstitute/gatk), `0cde69e`. [VariantFiltration integration tests](https://github.com/broadinstitute/gatk/blob/0cde69eed30339f5978cbb1ac6e5cf3662f9e1f8/src/test/java/org/broadinstitute/hellbender/tools/walkers/filters/VariantFiltrationIntegrationTest.java) | Apply a supplied variant-quality and interval-mask policy; distinguish site from allele/genotype filtering. | Evaluate predicates and interval membership independently on generated VCF records. Check FILTER labels and retained records, including missing annotations and threshold boundaries. Filtering does not establish variant-calling accuracy. |
| 9 | [pysam](https://github.com/pysam-developers/pysam), `ba2e6c1`. [Pileup tests](https://github.com/pysam-developers/pysam/blob/ba2e6c124398bdcd963db741d6f01164fed4f9b7/tests/AlignmentFilePileup_test.py) | Count eligible bases or alleles at specified loci with explicit read and base filters. | Use a generated per-read ledger and CIGAR interpreter. Inspect overlapping mates, orphans, deletions, and BAQ policy. pysam and SAMtools share underlying code; their agreement is not independent validation. |
| 10 | [MAFFT](https://github.com/GSLBiotech/mafft), `26ecaba`. [Alignment regression commands](https://github.com/GSLBiotech/mafft/blob/26ecaba0130b533cf06a29200f0fb40829c00101/test/script) | Construct or repair a small multiple alignment under a specified scoring or correspondence rule. | Check residue preservation plus independently established aligned residue pairs or a small exact scoring objective; allow equivalent alignments. The supplied repository is a 2020 snapshot; its bytewise regression diffs do not establish biological optimality. |
| 11 | [HMMER](https://github.com/EddyRivasLab/hmmer), `9acd8b6`. [HMM alignment/domain integrity test](https://github.com/EddyRivasLab/hmmer/blob/9acd8b6758a0ca5d21db6d167e0277484341929b/testsuite/i13-msa-integrity.pl) | Extract and reconcile protein domain sequences from a supplied domain table and alignment. | Slice fresh FASTA using domain coordinates, remove alignment gaps, and compare IDs and residues. This checks domain bookkeeping; full profile-search significance requires a pinned profile, search space, and statistical reference. |
| 12 | [Seurat](https://github.com/satijalab/seurat), `586015a`. [Matrix manipulation tests](https://github.com/satijalab/seurat/blob/586015abde10618ecb32d3fe632267a83317a08d/tests/testthat/test_data_manipulation.R) | Join single-cell features and apply prescribed library-size log normalization or row scaling. | Compute joins and log1p(count / cell total × scale) independently; check IDs, sparse zeros, and every output value. Counts, normalized data, and scaled layers are distinct; this does not grade cell-type interpretation. |
| 13 | [minimap2](https://github.com/lh3/minimap2), `3c28777`. [Mapping and PAF/CIGAR example](https://github.com/lh3/minimap2/blob/3c28777e7e2dcc90f825de1b9f17a89cca7d4452/example.c) | Recover long-read alignment spans and orientation on fresh references, including split or inverted segments. | Check placements and aligned strings against generated segments; validate PAF spans and CIGAR consumption. The example explicitly emits CIGARs without soft/hard clips. Pin presets; equal-scoring placements need equivalence rules. |
| 14 | [BCFtools](https://github.com/samtools/bcftools), `edf7fd9`. [Allelic-depth fixture and expected output](https://github.com/samtools/bcftools/blob/edf7fd96c5da562ecfd99fb7f9e4b9eb597aeae8/test/fill-tags-VAF.out) | Calculate per-alternate-allele fractions and total alternate fraction from multiallelic depth fields. | Compute AD_alt / sum(AD) and sum(AD_alt) / sum(AD) from fresh integers. Test missing fields and zero denominators. Site-level inclusion and per-sample genotype filtering need separate checks. |
| 15 | [Picard](https://github.com/broadinstitute/picard), `c2a483d`. [Read-list filtering tests](https://github.com/broadinstitute/picard/blob/c2a483d497d1b0fe6d0ab518b1b32fe98fad0741/src/test/java/picard/sam/FilterSamReadsTest.java) | Select or exclude read pairs using a manifest while retaining correct records and metadata. | Compare the full multiset of retained SAM records against a fresh ledger; check mates and counts. The upstream test constructs paired reads. Define inclusion by read name explicitly; duplicate marking would require its own tie-breaking policy. |
| 16 | [Snakemake](https://github.com/snakemake/snakemake), `91763d6`. [Wildcard expansion tests](https://github.com/snakemake/snakemake/blob/91763d644db0a6051c40014fa8ffad340f7d39a0/tests/test_expand.py) | Repair sample/lane pairing and workflow dependencies; avoid accidental Cartesian products. | Derive expected input/output mappings from a fresh sample sheet and run a bounded biological step. The tests distinguish product from zip expansion. A correct file list or successful dry run alone does not verify downstream biology. |
| 17 | [HTSlib](https://github.com/samtools/htslib), `d3cc955`. [FASTA/FASTQ indexing and retrieval tests](https://github.com/samtools/htslib/blob/d3cc9553d89dc34239afb7145b06c0dd818c0219/test/faidx/faidx.tst) | Retrieve indexed regions correctly across wrapped lines, sequence descriptions, and compressed files. | Compare extracted bases and qualities to direct slicing of generated records; test absent IDs and coordinate boundaries. HTSlib also underlies SAMtools/pysam, so use an independent parser for the oracle. |
| 18 | [FastQC](https://github.com/s-andrews/FastQC), `87fb336`. [Sequence-length module](https://github.com/s-andrews/FastQC/blob/87fb3364a2f37115833d678648926d41e184f0b1/uk/ac/babraham/FastQC/Modules/SequenceLengthDistribution.java) | Compute read-length distributions and apply a supplied QC policy to generated FASTQ. | Count lengths independently and reconcile report bins and totals; include empty or variable-length records. A FastQC warning is a configured heuristic, not ground truth that a biological sample is unusable. |
| 19 | [Biopython](https://github.com/biopython/biopython), `08fc090`. [Feature extraction and translation tests](https://github.com/biopython/biopython/blob/08fc09086afe0b57215d2515660e0c032b55c0dd/Tests/test_SeqFeature.py) | Extract compound, strand-aware features and translate explicitly defined coding sequences. | Use direct slicing, complementing, exon concatenation, and a frozen codon table. Check source-record IDs, frame, and alternative genetic codes. Reject unresolved remote references under an explicit input contract. |
| 20 | [Nextflow](https://github.com/nextflow-io/nextflow), `17f1877`. [Grouping and remainder tests](https://github.com/nextflow-io/nextflow/blob/17f18779266767b16bca51af71522e28adf5cff6/modules/nextflow/src/test/groovy/nextflow/extension/GroupTupleOpTest.groovy) | Repair sample-key grouping and lane aggregation in a small workflow. | Compare complete groups and final per-sample artifacts to a manifest-derived oracle. The tests show that incomplete groups can disappear unless remainders are handled. Make the intended missing-lane policy explicit. |
| 21 | [DIAMOND](https://github.com/bbuchfink/diamond), `5e25aca`. [CTest search cases](https://github.com/bbuchfink/diamond/blob/5e25acaf40e6b9883636c5c564306fe77210de53/CMakeLists.txt) | Search a tiny protein database and select hits by explicit identity and coverage criteria. | Validate planted matches and alignment-derived identity/coverage; use a supplied score table for pure filtering variants. CTest compares saved outputs, while the legacy test command is a no-op at this revision. E-values and heuristic ties need separate treatment. |
| 22 | [PLINK / PLINK 2](https://github.com/chrchang/plink-ng), `a25a0d6`. [Genotype-frequency/PCA filtering tests](https://github.com/chrchang/plink-ng/blob/a25a0d6438b61b1951cd6d7eb209db8b79687581/2.0/Tests/TEST_GRM_MAF/run_tests.sh) | Choose variants and samples by allele frequency and missingness; keep analysis-specific and global filters distinct. | Compute called-allele denominators and selected IDs from fresh genotypes. State ploidy, founder, and missing-call rules. Do not grade PCA by raw eigenvector signs or assume a PCA-only filter also changes frequency reports. |
| 23 | [Scanpy](https://github.com/scverse/scanpy), `0d5fd16`. [Single-cell QC tests](https://github.com/scverse/scanpy/blob/0d5fd16234865619d2f5097d33fc4281900a2bc2/tests/test_qc_metrics.py) | Calculate per-cell counts, detected genes, and supplied gene-set fractions; aggregate raw counts by donor. | Direct sums and nonzero counts on fresh matrices, plus independent donor/cell ledgers. Compare IDs and denominators. Vary raw versus transformed layers; do not reuse benchmark PBMC fixtures. |
| 24 | [MultiQC](https://github.com/MultiQC/MultiQC), `fdc68d3`. [Flagstat parsing tests](https://github.com/MultiQC/MultiQC/blob/fdc68d394849f69b67b6e6e13ebe907504ed534b/multiqc/modules/samtools/tests/test_flagstat.py) | Combine sample QC reports while retaining sample identity and QC-passed/failed denominators. | Generate reports from a hidden count ledger; check every metric and sample mapping, including N/A and filename collisions. Parsing a report correctly does not validate the upstream alignment. |
| 25 | [edgeR](https://git.bioconductor.org/packages/edgeR); primary `f47ecb9`, linked mirror `8986864`. [CPM implementation](https://github.com/bioconductor-source/edgeR/blob/8986864d8f92dac37925ef641fcd6c4161130551/R/cpm.R) | Normalize counts with explicitly supplied library sizes/factors and a declared expression filter. | Check non-log CPM = count × 1e6 / effective library size and selected genes. Specify offsets and normalization factors; zero libraries need a defined error. This short recipe does not substitute for replicate-aware differential expression. |
| 26 | [limma](https://git.bioconductor.org/packages/limma); primary `57a8de7`, linked mirror `14eabae`. [Contrast implementation](https://github.com/bioconductor-source/limma/blob/14eabaeb695945b45ceb885ac8d4c61232639ea5/R/contrasts.R) | Fit prescribed linear models and signed contrasts with correct sample IDs and covariates. | Use an independent least-squares calculation on fresh measurements; check contrast estimates, estimability, and unmoderated uncertainty. Gene-wise weights and empirical-Bayes moderation require additional method-specific references. |
| 27 | [SPAdes](https://github.com/ablab/spades), `808b87d`. [Assembly smoke-test checker](https://github.com/ablab/spades/blob/808b87dade1300ecaa712429955ccba7bfb286f4/src/projects/spades/pipeline/spades_pipeline/supplemetary/check_test_script.py) | Reconstruct a deliberately unambiguous tiny genome, or calculate assembly summaries from supplied contigs. | For reconstruction, compare full sequences modulo permitted orientation/circular rotation and check missing/extra contigs. N50 and contig-length checks alone are insufficient: the inspected smoke test checks number and length, not sequence identity. |
| 28 | [IQ-TREE](https://github.com/iqtree/iqtree2), `a00094e`. [Partition example and test generator](https://github.com/iqtree/iqtree2/blob/a00094e03d1ae984e1497e16738f91514df8c366/example/example.nex) | Validate codon-position or gene partitions and prepare correctly labeled alignment/model inputs. | Enumerate expected site membership, coverage, and overlap from generated annotations. Short tasks can stop at this contract. Full tree inference needs likelihood/topology criteria; the generating tree is not guaranteed to be the fitted maximum-likelihood tree. |
| 29 | [FastTree](https://github.com/morgannprice/fasttree), `a5a2723`. [Split and branch comparison script](https://github.com/morgannprice/fasttree/blob/a5a2723ea1e64faf3da7ea514521cfa348891add/CompareTree.pl) | Compare tree bipartitions and summarize branch lengths while respecting rooting and taxon identity. | Canonicalize leaf-set splits and sum lengths independently on generated Newick trees. The script documents duplicate/root-edge problems for rooted trees; do not copy it as the oracle or compare raw Newick strings. |
| 30 | [RAxML](https://github.com/stamatak/standard-RAxML), `36ec361`. [Bootstrap branch-length workflow](https://github.com/stamatak/standard-RAxML/blob/36ec36110631c34692abcd4f24ca7b3e2fea742a/usefulScripts/bsBranchLengths.pl) | Summarize branch support or lengths across a supplied set of replicate trees. | Match canonical bipartitions across fresh trees and compute support denominators and branch summaries directly. The script supplies a workflow boundary, not an independent answer. This source is standard RAxML 8, distinct from RAxML-NG. |
| 31 | [cutadapt](https://github.com/marcelm/cutadapt), `4927632`. [Adapter-trimming tests](https://github.com/marcelm/cutadapt/blob/4927632f7c546dd290c53501c8417f909252befe/tests/test_trim.py) | Trim declared adapters and synchronize sequence/quality strings, including partial matches and paired reads. | Generate insert-plus-adapter reads and independently compute allowed cut positions under an explicit error/overlap rule. Test unchanged reads and boundaries. Upstream tests expose deletion-versus-substitution cases; a simple substring oracle cannot grade every fuzzy match. |
| 32 | [Salmon](https://github.com/COMBINE-lab/salmon), `5515b7f`. [Synthetic decoy/output-contract tests](https://github.com/COMBINE-lab/salmon/blob/5515b7f05a90341b6652adfdb807e7cf14295518/crates/salmon-cli/tests/output_contract.rs) | Join transcript quantification outputs to a gene map; check decoy exclusion and abundance normalization. | Use fresh supplied quantification tables for exact ID joins, sums, and TPM arithmetic. Quantifier estimates are not latent simulated molecule counts. Inspected main is Salmon 2.0 Rust; legacy 1.x indexes/options require a separate environment. |
| 33 | [kallisto](https://github.com/pachterlab/kallisto), `4e9f29c`. [Quantification/BUS workflow](https://github.com/pachterlab/kallisto/blob/4e9f29cf3b021260415430c057a22469ca081391/test/Snakefile) | Construct a transcript/sample abundance matrix or repair paired-input wiring for quantification. | Independently check transcript IDs, sample columns, explicit missing-transcript policy, and TPM from supplied counts/effective lengths. The bundled workflow requests quant/BUS artifacts but does not itself establish quantification accuracy. |
| 34 | [StringTie](https://github.com/gpertea/stringtie), `d1dc38d`. [Coverage-to-count matrix helper](https://github.com/gpertea/stringtie/blob/d1dc38ddb681089b2e8faaabdcfee772af6fb033/prepDE.py3) | Convert annotated transcript coverage into transcript/gene count estimates with correct exon lengths and sample joins. | Independently compute inclusive GTF exon lengths and ceil(coverage × length / read length), then aggregate by gene. Check annotation consistency. These are estimated counts; they are not exact observed molecule counts. |
| 35 | [fastp](https://github.com/OpenGene/fastp), `8a2397b`. [Read filtering implementation](https://github.com/OpenGene/fastp/blob/8a2397b6628ae14127efdb7566f67fc05f9aea56/src/filter.cpp) | Apply explicit quality, N-content, length, and complexity filters to fresh FASTQ. | Compute per-read predicates and retained IDs independently, preserving paired-read policy. Check strict versus inclusive boundaries and filter ordering. Automatic adapter detection adds ambiguity, so use declared adapters for exact trimming tasks. |
| 36 | [SRA Toolkit](https://github.com/ncbi/sra-tools), `434ae78`. [Split-3 export regression](https://github.com/ncbi/sra-tools/blob/434ae787c86e32e7faa5e80417a342c365fa03b0/test/external/fasterq-dump/fq_tests/split3.sh) | Export biological reads from a small staged SRA fixture with correct paired and orphan outputs. | Compare IDs, sequences, qualities, and mate assignment to an independently authored spot/read ledger. Upstream fastq-dump/fasterq-dump agreement is only a cross-check. Stage the fixture locally; live accession downloads add network, provenance, and setup cost. |
| 37 | [deepTools](https://github.com/deeptools/deepTools), `cde2aa7`. [Explicit coverage-bin tests](https://github.com/deeptools/deepTools/blob/cde2aa7938cb4af6fe28de1504f94f6928344342/pydeeptools/deeptools/test/test_countReadsPerBin.py) | Count reads/fragments in genomic bins under a declared extension and duplicate policy. | Use direct interval arithmetic from a generated alignment ledger and check every bin, including zero bins. Separate read counts, covered bases, and normalization denominators; matching a heatmap image is unnecessary. |
| 38 | [VCFtools](https://github.com/vcftools/vcftools), `1f87a83`. [Allele-count recalculation helper](https://github.com/vcftools/vcftools/blob/1f87a83402ffd17ea2723a456b7edf5d80b4a22c/src/perl/fill-an-ac) | Recalculate AC/AN and genotype QC after selecting samples from a VCF. | Count called alleles directly, including multiallelic and partially missing genotypes under explicit rules. Check retained sample IDs and header cardinalities; a fixed two-alleles-per-sample denominator fails missing and haploid cases. |
| 39 | [SnpEff](https://github.com/pcingola/SnpEff), `1db1599`. [Generated transcript/SNP tests](https://github.com/pcingola/SnpEff/blob/1db15998ea6aad93a35848aca0f6cba81cd36738/src/test/java/org/snpeff/snpEffect/testCases/unity/TestCasesSnps.java) | Classify coding SNVs as synonymous, missense, stop gained/lost, or start changes on fresh transcripts. | Reconstruct strand-aware codons and compare reference/alternate translation with an independent codon table. Check transcript ID and allele orientation. SnpEff impact categories do not establish pathogenicity; freeze annotation and genetic code. |
| 40 | [Ensembl VEP](https://github.com/Ensembl/ensembl-vep), `cee181c`. [Custom-GTF annotation tests](https://github.com/Ensembl/ensembl-vep/blob/cee181c2a1bb31900a0b7526168c67577fb23928/t/AnnotationSource_File_GTF.t) | Annotate variants against a supplied miniature FASTA/GTF, retaining per-transcript consequences. | Independently derive exon/CDS positions and codon changes; accept all requested transcripts rather than one arbitrary consequence. The source tests transcript construction and sequence explicitly. Pin assembly, transcript version, phase, and consequence vocabulary. |
| 41 | [MACS2 / MACS3](https://github.com/macs3-project/MACS), `ece0896`. [Synthetic strand/shift pileup tests](https://github.com/macs3-project/MACS/blob/ece08963b6a30f4de0c5a5e684513f876b788d2c/test/test_Pileup.py) | Build fragment pileups or call intervals above a supplied signal threshold. | Use independent interval accumulation, strand-specific shifts, scaling, and explicit merging rules. Verify full bedGraph/peak intervals. Such checks do not establish calibrated significance for the full MACS peak-calling model. |
| 42 | [MMseqs2](https://github.com/soedinglab/MMseqs2), `d401e78`. [Clustering workflow and export](https://github.com/soedinglab/MMseqs2/blob/d401e78c2d18a822cdb1527d7464a043f6035a15/data/workflow/easycluster.sh) | Cluster a complete observed proteome and reconcile native membership, sequence versions, cluster statistics and representative FASTA files. | The observed task reproduces pinned Linclust assignments with one thread and explicit identity, coverage and clustering modes. Check every member and unchanged representative sequence. Similarity clusters do not establish orthology or function; greedy clustering is not an all-pairs identity guarantee. |
| 43 | [MUSCLE](https://github.com/rcedgar/muscle), `29aa067`. [Alignment Q/TC regression checks](https://github.com/rcedgar/muscle/blob/29aa0671d0e46c862457749c7f2d87f29007b8eb/test_scripts/check_results.py) | Score or repair a small multiple alignment using known residue correspondences. | Check residue preservation and independently specified pair/column correspondence scores. The inspected regression accepts relative Q/TC thresholds on benchmark alignments; do not reuse those fixtures or treat those thresholds as universal biological truth. |
| 44 | [Kraken 2](https://github.com/DerrickWood/kraken2), `8c190b1`. [Taxonomic report implementation](https://github.com/DerrickWood/kraken2/blob/8c190b1b668825935dbf6dee5f969227dc8269bb/src/reports.cc) | Compute direct and clade-level counts from read assignments and a frozen miniature taxonomy. | Sum assignments up the tree, count paired fragments once, and check unclassified denominators. Ancestor clade counts overlap and must not be summed as disjoint abundances. Full classification additionally needs a pinned reference database. |
| 45 | [BUSCO](https://gitlab.com/ezlab/busco), `cd07105`. [Completeness table/summary implementation](https://gitlab.com/ezlab/busco/-/blob/cd071053c38c5060f75d0b370cb66c4edc8e59a1/src/busco/busco_tools/hmmer.py) | Reconcile complete single-copy, duplicated, fragmented, and missing ortholog categories. | Count unique BUSCO IDs against a supplied lineage inventory: C = S + D and N = S + D + F + M. Multiple hits for a duplicated ortholog do not create multiple orthologs. This checks summary accounting; sequence-level completeness needs frozen models and cutoffs. |
| 46 | [UCSC Kent utilities](https://github.com/ucscGenomeBrowser/kent), `0f58b0e`. [Format round-trip and alias tests](https://github.com/ucscGenomeBrowser/kent/blob/0f58b0eef93be6d6d3b26b9e2b99261d558d67df/src/utils/bedGraphToBigWig/tests/makefile) | Convert small genome-browser tracks while preserving coordinates, values, and chromosome identity. | Compare canonical intervals/values to independently generated tracks; include unknown and mixed chromosome aliases. TwoBit tasks can check exact sequence and masking. Match individual executables; utility-specific formats, float precision, and licenses differ. |
| 47 | [GenomicRanges](https://github.com/Bioconductor/GenomicRanges), `44c311c`. [Overlap and count/subset tests](https://github.com/Bioconductor/GenomicRanges/blob/44c311c711b9a5a5d6db070a8f3210819e4bc9de/inst/unitTests/test_findOverlaps-methods.R) | Translate interval conventions and distinguish hits, counts, and qualifying features, including compound features. | Enumerate all expected interval pairs directly with explicit strand and adjacency rules; compare IDs and multiplicities. GRanges uses one-based closed ranges, unlike BED. Cross-tool agreement requires a correct conversion. |
| 48 | [Biostrings](https://github.com/Bioconductor/Biostrings), `fb0cd89`. [Translation and ambiguous-codon tests](https://github.com/Bioconductor/Biostrings/blob/fb0cd89830abd054cf2681d6bc6c929981e07b21/tests/testthat/test-translate.R) | Translate coding sequences under declared genetic-code, initiation, ambiguity, and frame rules. | Enumerate codons using a frozen table; check exact peptides or specified errors. Tests distinguish resolvable ambiguous codons, X, initiation handling, and incomplete terminal codons. State these policies in the task. |
| 49 | [pybedtools](https://github.com/daler/pybedtools), `efb8534`. [Synthetic stranded-extraction tests](https://github.com/daler/pybedtools/blob/efb8534c11ca6b45a6cd173ff3b3d1bf754e34a1/pybedtools/test/test_1.py) | Extract strand-aware sequences or compose feature filtering with interval arithmetic in Python. | Check bases and feature IDs by direct arithmetic on fresh BED/FASTA; preserve stream contents and repeated records. The wrapper invokes BEDTools, so comparing the two is not independent verification. |
| 50 | [nf-core/tools](https://github.com/nf-core/tools), `eb2f709`. [Pipeline parameter-schema tests](https://github.com/nf-core/tools/blob/eb2f709090f4054f45437c34049ea2068567c339/tests/pipelines/test_schema.py) | Repair a bounded pipeline configuration or module contract using a fresh sample sheet and specified biological output. | Validate schema and sample mappings, then check the small workflow output independently. Lint/schema success establishes configuration properties only; existing nf-core/rnaseq tests provide a downstream composition example. |

These are bounded operations, not full-inference oracles. Search collections by
executable as well as package: UCSC examples include `liftOver`, `bedGraphToBigWig`,
`bigWigToBedGraph`, `faToTwoBit`, `twoBitToFa`, `bedToBigBed`, and `bigBedToBed`.
The inspected Kent fixtures cover only track/sequence operations. Workflow systems
still require downstream biological-output checks.

## Additional inspected analysis and benchmark sources

| Source | Inspected material and use |
|---|---|
| clusterProfiler, `fa71a3e` | [`enricher`](https://github.com/YuLab-SMU/clusterProfiler/blob/fa71a3ef739a1ef44e695e69823126c94c703bdb/R/enricher.R): custom term/gene mappings and universe; verify hypergeometric/BH results independently. |
| PhyKIT, `3e59b12` | [Informative sites](https://github.com/JLSteenwyk/PhyKIT/blob/3e59b123e6ffd1ba1f298603a4dc4e7b04c69b0e/tests/integration/alignment/test_parsimony_informative_sites_integration.py) and [treeness](https://github.com/JLSteenwyk/PhyKIT/blob/3e59b123e6ffd1ba1f298603a4dc4e7b04c69b0e/tests/integration/tree/test_treeness_integration.py): explicit expected outputs; use fresh alignments/trees and direct counting. |
| [rnaseqGene](https://bioconductor.org/packages/release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html) | Count-matrix entry point and paired design; use fresh counts, excluding airway/GSE52778 from approved training inputs. |
| Galaxy Training Network, `f0bbf97` | [RNA-seq tutorial](https://github.com/galaxyproject/training-material/blob/f0bbf971bbc35cc75a725c794084bf47e58a5313/topics/transcriptomics/tutorials/ref-based/tutorial.md): bounded steps and scientific rationale adapted to terminal tasks. |
| nf-core/rnaseq, `1f03b53` | [Tests](https://github.com/nf-core/rnaseq/blob/1f03b53ef799e298f60c813440e961e867017043/tests/default.nf.test) and [usage](https://nf-co.re/rnaseq/usage): workflow repair/composition; stub success checks wiring only. |
| statsmodels/NumPy | [Generated-data OLS](https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html) and [least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html): fresh patient tables and signed adjusted effects checked independently. |

BioAgent's inspected [corrupt-input](https://github.com/bioagent-bench/bioagent-experiments/blob/35eed91676062d98d98562556ec6cce326fd7d93/ablation/generate_corrupt_data.py)
and [decoy](https://github.com/bioagent-bench/bioagent-experiments/blob/35eed91676062d98d98562556ec6cce326fd7d93/ablation/generate_decoys.py)
generators provide robustness design references. Independently author new variants;
exclude benchmark fixtures from training. The scoring sources are assessed above.

Preserve code attribution and check data/teaching assets separately. Inspected
licenses include MIT (BEDTools, pysam, PhyKIT, nf-core/rnaseq), BSD-3-Clause (Scanpy),
LGPL >=3 (DESeq2), and Artistic-2.0 (clusterProfiler). BCFtools has MIT/GPL choices
with build-dependent qualifications; Biopython has its own license with some
dual-licensed files. BioAgent Bench and BiomniBench-DA benchmark artifacts are
CC-BY-4.0; underlying data retain their own terms. No root license was found for
BioAgent experiments; use it as a read-only design reference and independently
author implementations rather than copying its code.
