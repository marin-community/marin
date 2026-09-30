# Source inventory: arq5x/bedtools2

Inspected repository revision: `614e9a5c5935ab86e873dab9072fbbaf003c1b7e` (runner-resolved). GitHub source reads occurred 2026-09-29. This is a command-line toolkit for genomic interval arithmetic across BED, BAM, GFF/GTF, VCF and related formats; documentation describes comparisons, transformations, coverage, set statistics and composed shell workflows.

## Source map

| Collection | Location | Status and selected units | Open leads |
|---|---|---|---|
| Main purpose and supported formats | `README.md`; `docs/content/overview.rst` | Inspected overview and tool catalog. | Individual tools listed in overview need systematic coverage. |
| Quick start and worked examples | `docs/content/quick-start.rst`; `docs/content/example-usage.rst` | Inspected quick-start and examples for intersect, closest, subtract, merge, coverage and more; selected `bedtools2-intersect-gene-repeat`. | Continue remaining example sections, especially mapping, genome coverage and multi-file comparisons. |
| Tool reference | `docs/content/tools/summary.rst`, `jaccard.rst`, `fisher.rst`; also `intersect.rst`, `coverage.rst`, `getfasta.rst`, `genomecov.rst`, `closest.rst`, `map.rst`, `multiinter.rst` retrieved/read in bounded selections | Detailed review of summary, Jaccard and Fisher; recorded `bedtools2-summary-repeatmasker-qc`, `bedtools2-jaccard-sets`, `bedtools2-fisher-interval-enrichment`. Other tool docs inspected in part, not independently inventoried. | Many command references remain: annotate, complement, cluster, map, coverage, genomecov, getfasta, multiinter, reldist, etc. |
| Tutorial | `tutorial/bedtools.md` | Read tutorial setup/data context, intersect and Jaccard analysis through pairwise matrix, PCA/heatmap and puzzles; selected `bedtools2-tutorial-dnase-jaccard`. | Remaining puzzles and middle tutorial sections merit inspection; exact section outline was not completed. |
| Bundled data | `data/` tree listing | Inventory metadata showed six small-to-moderate BED tracks (`aluY`, `gerp`, `knownGene.hg18`, `refseq.chr1.exons`, `simpleRepeats`); contents not previewed. | Identify provenance/build and intended examples by tracing docs and bounded file previews. |
| Tests and fixtures | `test/`, including intersect, coverage, fisher, jaccard and numerous command-specific directories | File listing inspected; selected Fisher README only. Test fixtures are not assumed to be biological study datasets. | Read script semantics and inspect small fixture content to identify educational/scientific scenarios. |
| Implementation | `src/` | Tree listing showed command-specific implementations and shared interval/BAM utilities; implementation was not deeply read. | Inspect implementation only when needed for undocumented semantics or precise output behavior. |
| External studies/data | Tutorial-linked Maurano archive; UCSC hg38 Simple Repeats and chromInfo URLs | URLs and claims read in docs; external assets were not fetched. Dataset records capture known metadata and access gaps. | Check availability, snapshot/version, complete metadata, exact terms and assembly. |

## Findings and boundaries

The inspected sources support interval overlap filtering/counting, nearest-feature queries, interval set similarity, overlap enrichment, and chromosome-level distribution QC. The best connected workflow found is the tutorial comparison of fetal tissue DNase hypersensitivity interval sets using pairwise Jaccard and exploratory R plots. The summary documentation separately gives a concrete GRCh38 Simple Repeats QC example. The inventory keeps these as separate scientific questions and links only datasets explicitly used by documentation.

`bedtools2-maurano-fetal-dnase` is observed study data according to the tutorial, but its archive was not downloaded. UCSC Simple Repeats is an annotation track and chromInfo is a reference product, therefore they have separate records. Bundled `data/` tracks and tests were not conflated with tutorial datasets because their biological provenance/use was not traced. Example filenames alone were not treated as identifiable datasets.

Source-backed details are the described inputs, outputs, operations, assumptions and tutorial claims. Potential task ideas (reproducible tissue similarity, interval enrichment validation, annotation QC) remain hypotheses for authoring; no task execution, Harbor compatibility, runtime, grader determinism or dataset release eligibility was established.

## Inaccessible sources and stopping point

All selected GitHub reads at the supplied revision succeeded except the tutorial-heading extraction command, which failed because ripgrep does not allow a literal newline in its regex. The tutorial itself was read through GitHub raw output, but its complete outline was not enumerated. No external URLs were fetched. The recursive tree output was truncated by the interface after exposing top-level docs, source and test inventories; this did not prevent identifying their locations.

The pass stopped after representative documentation and use cases were selected; remaining repository scope is broad rather than exhausted. There was no enforced time or resource interruption. At stop, `/proc/loadavg` was `0.01 0.04 0.04`; `MemAvailable` was about 4.93 GB. The ten-minute assignment window was the practical inspection budget; further useful progress would be systematic inspection of remaining tool docs and tutorial/test sections.
