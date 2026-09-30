# Source inventory: arq5x/bedtools2

## Scope and revision

Inspected pinned commit `614e9a5c5935ab86e873dab9072fbbaf003c1b7e`, resolved by the runner. The recursive Git tree response was not truncated. The repository is a C++ command-line suite for genome-feature arithmetic over BED, GFF/GTF, VCF and BAM, with reference/interval transformations, coverage summaries, set comparisons, randomization and conversion commands. Documentation describes UNIX-stream composition. The repository README cites Quinlan and Hall (2010), *BEDTools: a flexible suite of utilities for comparing genomic features*, Bioinformatics 26(6), 841–842.

Inventory records are in `units.jsonl` (8 units) and `datasets.jsonl` (8 records). IDs derive from source identity or named asset, not enumeration. No command was executed; documentation examples and tree metadata are source evidence only.

## Source map and inspection queue

| Collection / location | Status and records | Specific remaining leads |
|---|---|---|
| `README.md`; `docs/content/overview.rst` | Inspected summary and tool catalogue. Establishes toolkit purpose, supported formats, citation, and broad operation families. | The full tool catalogue entries remain to inspect individually. |
| `docs/content/tools/intersect.rst`; `docs/content/example-usage.rst`; `tutorial/bedtools.md` | Inspected overlap semantics and examples; unit `bedtools2.doc.tools.intersect`. Tutorial also inspected through descriptions of its files, setup, and intersection sections. | Later tutorial sections on coverage, merging, subtraction, closest, and other tools were not inspected. |
| `docs/content/tools/coverage.rst` | Inspected overview, output semantics and documented strand/histogram examples; unit `bedtools2.doc.tools.coverage`. | Remaining options and later doc sections; consider BAM-backed real examples and test assets. |
| `docs/content/tools/genomecov.rst` | Inspected overview, input requirements, options and beginning of default behavior; unit `bedtools2.doc.tools.genomecov`. | Remaining default/output examples and tests. |
| `docs/content/tools/fisher.rst` | Inspected opening examples, evaluation caveats, and option summary; unit `bedtools2.doc.tools.fisher`. | Inspect `test/fisher/` data/script and implementation if statistical task design proceeds. |
| `docs/content/tools/reldist.rst` | Inspected stated purpose, citation, options, and exon/AluY, exon/GERP and self-comparison examples; unit `bedtools2.doc.tools.reldist`. Linked records: three repository example assets. | Exact source/assembly/release for the three bundled annotations; inspect test suite if needed. |
| `docs/content/tools/jaccard.rst` | Inspected definition, example, and overlap-threshold behavior; unit `bedtools2.doc.tools.jaccard`. | Implementation and tests should resolve denominator wording before a grader is designed. |
| `docs/content/tools/multiinter.rst` | Inspected input requirements, output fields, worked example and `-empty`; unit `bedtools2.doc.tools.multiinter`. | Additional examples/tests and use with identified biological datasets. |
| `docs/content/tools/getfasta.rst` | Inspected sequence extraction, formats, strand and split-block semantics with synthetic example; unit `bedtools2.doc.tools.getfasta`. | FASTA fixture/data provenance and tests; related `maskfasta` operation. |
| `tutorial/bedtools.md` | Inspected setup and data descriptions; intersect; sorted/multi-file intersections; merge; complement; genomecov; zero-coverage target workflow; Jaccard comparisons/PCA; puzzle list. Units: `bedtools2.doc.tools.intersect`, `bedtools2.doc.tools.merge`, `bedtools2.doc.tools.complement`, `bedtools2.doc.tools.genomecov`, `bedtools2.tutorial.uncovered-capture-regions`, `bedtools2.tutorial.dnase-jaccard-pca`. | The later puzzle leads (closest GWAS-to-exon, exon counts per windows, enhancer overlap, SNP feature fractions, splice-site flanks, randomized Jaccard, ChromHMM state base-pair totals) are pending; inspect their linked command docs and source/data before promoting. External tutorial URLs remain unverified. |
| `data/` repository files | Tree metadata inspected, not file content. Inspected three entries by name/size because reldist docs directly use them: `refseq.chr1.exons.bed.gz` (456,335 bytes), `aluY.chr1.bed.gz` (129,766), `gerp.chr1.bed.gz` (1,128,077). | Other example data include `knownGene.hg18.chr21.bed` (122,154 bytes), its `.short.bed` (19,867), `simpleRepeats.chr1.bed.gz` (614,807); identity, purpose and provenance pending. No data downloaded. |
| `docs/content/tools/*.rst` command catalogue | Partial: inspected intersect, coverage, genomecov, fisher, reldist, jaccard, multiinter, getfasta, merge and subtract. Units include merge and complement where tutorial examples show direct exon-derived scientific outputs. | Pending distinct candidates from overview: annotate, closest, map, maskfasta, multicov, pairToBed/pairToPair, random/shuffle, unionbedg, summary, cluster and others. |
| `src/` C++ command implementations | Tree inventory only; no implementation body inspected. | Inspect implementation and tests for any unit selected for task authoring, especially input edge semantics/statistic definitions. |
| `test/` fixtures and shell scripts | Tree inventory only; no test bodies or fixture content inspected. | Bounded examples and regression semantics by operation; distinguish synthetic fixtures from observations. |

## Source-backed findings

- BEDTools applies interval set operations and transformations to genomic feature files and supports composing commands on standard streams (`README.md`, `overview.rst`).
- `intersect` provides shared intervals, original records, counts, and absence filtering; the tutorial uses human CpG islands against RefSeq exons.
- Feature coverage (`coverage`) and genome-wide coverage (`genomecov`) answer different questions and have distinct output/input contracts.
- `reldist` is linked to named bundled chromosome 1 example files and frames distribution shape as evidence about spatial association. Its docs compare RefSeq exons with AluY repeats and GERP conserved elements.
- `fisher` infers possible interval counts heuristically. The documentation explicitly notes p-value sensitivity/inflation and recommends simulation validation.
- `jaccard` and `multiinter` provide distinct summaries: set similarity and per-segment membership across multiple sets.
- `getfasta` turns interval coordinates into sequence outputs given a matching reference FASTA.
- The 2017–2018 tutorial describes a Maurano et al. fetal DNaseI example plus UCSC Table Browser-derived human annotations; the underlying tutorial archive and BED assets were not retrieved.

## Dataset reconciliation and limits

The tutorial's `exons.bed` and the repository's `data/refseq.chr1.exons.bed.gz` remain separate: assembly, source version, and identity are not established. CpG, exon, GWAS, and ChromHMM tutorial assets are separate records because they are different products with different biological meanings. The three `data/` assets used by `reldist` are also separate products; they are linked as inputs only to the documented reldist use. Repository public access and top-level MIT licensing do not establish redistribution rights for bundled biological annotations. No access URL was opened and no dataset bytes were fetched.

## Why this pass stopped

This remains a partial discovery pass rather than a repository-wide survey. The main tutorial is now inspected through its final puzzle list, and the pass has covered 12 source-supported units across distinct operation families. I stopped with the supplied 15:19:22.092150 UTC deadline approaching; at the final check it had not yet elapsed. Pending scope includes most command docs and implementation/tests, external tutorial asset retrieval/provenance, and follow-up of the listed puzzle questions. No execution, Harbor validation, or task validation is claimed.
