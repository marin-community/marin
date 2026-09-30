# Source inventory: arq5x/bedtools2

Inspected on 2026-09-29. Repository identity: https://github.com/arq5x/bedtools2. The project describes bedtools as a CLI suite for set operations on genomic intervals in formats including BAM, BED, GFF/GTF, and VCF. The inspected repository content was on `master`, but an immutable commit SHA could not be recovered: GitHub pages were readable in the browser, while `git ls-remote` failed because this VM could not resolve github.com. The official documentation pages inspected are the 2.31.0 documentation; they may not match the repository's current source revision.

## What the inventory covers

The inventory records five source-backed units: interval intersection and annotation; complement/genome-wide coverage; target/window coverage; pairwise fetal DNase interval similarity and matrix exploration; and the tutorial's linked exercise set. They are supported chiefly by the 2017–2018 tutorial at `tutorial/bedtools.md` and the bedtools 2.31.0 tool documentation. The examples show tool use and shell composition, rather than standalone biological analysis code implemented by the repository.

Data records distinguish the tutorial's fetal DNaseI assay intervals, UCSC-derived reference annotations/genome sizes, GWAS-associated SNP intervals, and generic BAM/capture-region inputs. The tutorial calls the 20 fetal tissue DNaseI files samples from brain, heart, intestine, kidney, lung, muscle, skin, and stomach and attributes the work to Maurano et al. 2012. Annotation track provenance is more limited: it says those BED files came from the UCSC Table Browser but does not preserve exact table queries, release versions, or builds. Generic BAM examples have no linked data source and are marked as unknown candidate inputs rather than as an actual study.

## Inspected sources and scientific work

- Repository README: project scope, interval formats, and set-operation framing. https://github.com/arq5x/bedtools2
- Tutorial synopsis/setup/data description: tutorial downloads `maurano.dnaseI.tgz`, cpg.bed, exons.bed, gwas.bed, genome.txt, and hesc.chromHmm.bed; it names the 20 fetal tissue samples and descriptions of the UCSC-derived tracks. Tutorial locator: “Synopsis”, “Setup”, and “What are these files?” in https://github.com/arq5x/bedtools2/blob/master/tutorial/bedtools.md.
- Tutorial intersection section: CpG/exon overlaps; original paired features, overlap bases, per-A counts, absence calls, fractional overlap, sorted performance, and multiple B databases.
- Tutorial complement/genomecov sections: non-exonic intervals using genome sizes; genome-wide depth histograms and BEDGRAPH; a zero-coverage capture-target pipeline example.
- Official `coverage` and advanced usage docs: breadth/depth over windows or exons, full-containment filter, BAM conversion, proper-pair filtering, and strand-separated examples. These are generic workflows without identified study data.
- Tutorial Jaccard/matrix/PCA section: same-tissue vs cross-tissue DNase interval similarity, 400 pairwise comparisons, matrix preparation, PCA and heatmap. The source explicitly calls PCA a toy example and notes heatmaps better fit the distance-like similarity.
- Tutorial closing puzzles: ten concrete questions cover complement, closest-feature distances, window counts, complete overlaps, SNP annotation fractions, splice flanks, Jaccard, shuffle nulls, and ChromHMM state base-pair totals.
- Official `intersect`, `genomecov`, `jaccard`, `coverage`, and general suite documentation for semantics and options. URLs and precise heading locators are recorded per unit.

The Jaccard analysis is kept separate from generic interval intersection because it asks a distinct cross-sample similarity question and feeds a matrix-level analysis. Its PCA/heatmap stages remain one composite unit because the tutorial presents them as downstream summaries of the same pairwise measurements. The ten tutorial puzzles are grouped as one source unit to avoid manufacturing ten records from question bullets; an author can split them into separate tasks with explicit graders.

## Leads not inspected or verified

- The repository's full source tree, tests, command implementations, and all tool-specific docs were not surveyed. The tool suite list in the 2.31.0 documentation is a lead for further units, including closest, merge, shuffle, multiinter, map, groupby, nuc, and getfasta.
- The tutorial-linked answer page, GNU parallel site, make-matrix.py script, blog post on exome coverage, and Maurano paper/data accession were not opened.
- No datasets were downloaded, no files were previewed, and no commands were run. Data sizes, hashes, exact sample manifest, exact assembly, access details, license/redistribution terms, and runnable environments remain unknown.
- GitHub commit identity remains unresolved. The source inventory uses `master` plus an explicit SHA gap; do not treat it as a pinned content revision.

## Useful next work and boundaries

The strongest self-contained task path is the tutorial's fetal DNase set similarity: obtain and inventory the sample files and manifests, pin build and processing provenance, then define deterministic Jaccard outputs and matrix checks. The annotation puzzles can be assembled from pinned UCSC tracks after choosing exact assemblies and explicit enhancer/promoter state definitions. Coverage tasks need a sourced BAM and capture/target set with read-filter and genome-build contracts. Public availability is not evidence of redistribution eligibility, and none of these source inspections demonstrates successful execution, Harbor compatibility, or a validated task.

## Stop reason

This discovery pass stopped at the assigned ten-minute boundary. The browser source reads and local source-revision check yielded sufficient documented operations and data leads for a useful partial inventory; repository-wide source/tests review, external data resolution, and runtime validation were outside this pass. No process-heavy local command, dataset download, Harbor run, model call, or cloud job was launched.
