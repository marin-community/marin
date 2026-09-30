# bedtools2 source inventory

## Scope and repository map

Repository: [arq5x/bedtools2](https://github.com/arq5x/bedtools2). The GitHub repository page identifies `master` and shows `docs/`, `data/`, `genomes/`, `scripts/`, `src/`, `test/`, and `tutorial/`; its README describes genomic set operations over BAM, BED, GFF/GTF, and VCF and a command-line suite. Public pages inspected on 2026-09-29 did not expose the commit SHA for `master`. Inventory revision fields therefore record the mutable branch and retrieval date; this revision gap remains.

| Collection | Location | Inspection status | Resulting units | Uninspected leads |
|---|---|---|---|---|
| Project overview | [README](https://github.com/arq5x/bedtools2) | Inspected summary, performance description, license and citation | Context for all units | Full README details and license text not independently examined |
| Tutorial analysis | [tutorial/bedtools.md](https://github.com/arq5x/bedtools2/blob/master/tutorial/bedtools.md) | Inspected setup, interval overlap, merge, genome coverage, chained workflow, Jaccard and matrix workflow sections | `bedtools2-tutorial-intersect-annotation`, `bedtools2-tutorial-merge-exons`, `bedtools2-tutorial-genome-coverage`, `bedtools2-tutorial-dnase-jaccard` | Tutorial's other sections on window/closest/subtract and end-of-document content; tutorial is longer than inspected sections |
| Tool reference | [intersect.rst](https://github.com/arq5x/bedtools2/blob/master/docs/content/tools/intersect.rst), [getfasta.rst](https://github.com/arq5x/bedtools2/blob/master/docs/content/tools/getfasta.rst), [genomecov docs](https://bedtools.readthedocs.io/en/latest/content/tools/genomecov.html), [example usage](https://bedtools.readthedocs.io/en/latest/content/example-usage.html) | Intersect semantics and sorted behavior; getfasta sequence extraction options; genomecov output modes; merge and closest examples | `bedtools2-tutorial-intersect-annotation`, `bedtools2-tutorial-genome-coverage`, `bedtools2-doc-getfasta`, supporting merge context | Other command references and API/source implementations not examined; Read the Docs `latest` build SHA not captured |
| Source implementation and tests | `src/`, `test/` | Repository directories identified only | None | Enumerate operation source files and tests for edge cases, parameters, and executable grading oracles |
| Bundled data and references | `data/`, `genomes/` | Repository directories identified only | No bundled files accepted as data records | Inspect filenames, metadata and bounded previews to determine whether test fixtures or references identify further tasks |
| Scripts and workflows | `scripts/`, `.github/workflows/` | Directory names identified only | None | Inspect scripts for real multi-stage biological analyses; workflows appear likely build/test focused but were not inspected |

## Scientific uses found

Source-backed findings include genomic overlap annotation (including minimum and reciprocal overlap, strand, non-overlap selection, counts, and multiple B files); merging intervals and aggregating their contributing annotations; genome-wide coverage histograms, per-base depth and BEDGRAPH; extraction of interval or block-spliced reference sequence; and comparison of DNase hypersensitivity intervals using base-pair Jaccard values across samples. The tutorial presents the latter as input to a 20-sample comparison matrix and suggests clustering/PCA downstream.

The first four are operations on supplied genomic files. The DNase example is a composed analysis using bedtools plus shell tools, GNU parallel, a downloaded matrix script, and R packages. These records describe capabilities and example workflows, not validated benchmark tasks, scientific conclusions, or successful executions.

## Data inventory and links

The tutorial setup names separate downloads for `cpg.bed`, `exons.bed`, `gwas.bed`, `genome.txt`, and `maurano.dnaseI.tgz`. They are represented separately where distinct biological/reference roles are apparent; no shared observation provenance is inferred from being in one tutorial. ChromHMM is listed separately as a tutorial-named input because its source is not identified by the setup excerpt. Documentation also uses `NA18152.bam` as a 1000 Genomes coverage example, without a URL or accession. Dataset records preserve these leads and explicitly state that assets were not downloaded or previewed, so size, exact build/version, source observation links, and redistribution terms remain unknown.

`getfasta` documents an inline toy FASTA/BED example only. It is not treated as an identifiable biological dataset; its unit has empty `dataset_ids` and describes the requirement for a compatible reference and interval input.

## Organization and author handoff

Stable unit IDs derive from repository path/operation identity. Units link shared inputs and adjacent operations: interval intersections, merges and coverage can form an interval analysis chain; Jaccard is a comparative use of interval sets; getfasta is a distinct sequence-extraction operation. Dataset references are only used when a cited example names the data role. An author may select a narrow operation such as strand-aware interval annotation, or combine merge and coverage, but the source pass does not determine task questions or grading contracts.

Useful next inspections are: (1) tutorial's remaining feature examples and any source-linked tasks; (2) `data/` and `genomes/` inventories with bounded fixture previews; (3) implementation and tests for intersect, merge, genomecov, jaccard and getfasta; (4) the DNase archive and cited Maurano paper for sample/accession/build provenance; and (5) possible datasets for `closest` or `window` examples.

## Limits and stopping point

This was a time-boxed lightweight source pass. At stop, it had inspected GitHub-rendered overview/documentation and tutorial excerpts only; it did not clone or build the repository, inspect code or tests, download/preview biological files, install software, execute bedtools, or verify task runtime. GitHub exposed a mutable branch but not an immutable commit SHA in the inspected metadata, and archive URLs/current availability and terms were not validated. Source coverage is therefore partial. The run stopped after identifying and recording five units and seven candidate supporting data records within the assigned short investigation window; code, tests, repository fixtures, and tutorial sections listed above remain the main continuation leads.
