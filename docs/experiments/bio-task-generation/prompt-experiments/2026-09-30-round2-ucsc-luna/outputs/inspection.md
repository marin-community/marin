# UCSC kent source inventory

Repository: `ucscGenomeBrowser/kent`; inspected revision `ad6dd2177ad20bea9e32563ee76a1c598bccb6d5`; retrieval date 2026-09-30. GitHub tree metadata was untruncated. Public GitHub API raw contents and bounded previews were used. No clone, build, scientific execution, Harbor run, or biological asset download was performed.

## Summary

Inspected operations cover Table Browser query/export/intersection, sequence retrieval from 2bit references, VCF allele/position filtering, assembly coordinate transfer, and variant projection to genomic/transcript/coding/protein HGVS forms. Sources include tutorials, command implementations, an operational workflow, and test assets. Source inspection supports candidate task concepts only; it does not establish task execution or reproducible grading.

`units.jsonl` contains 5 units with stable source-derived IDs. `datasets.jsonl` contains 4 records: a mutable hg38 reference download lead, a small SARS-CoV-2 VCF fixture, Personalis HGVS benchmark VCF fixtures, and a RefSeq test reference product. Data terms remain unresolved. Only small text previews were read; no operation was run.

## Source map and queue

| Collection / location | Coverage state | Resulting unit IDs | Remaining specific leads |
|---|---|---|---|
| `docs/tutorials/index.md`, `docs/tutorials/tableBrowserTutorial.md` | Inspected index and Table Browser tutorial: assembly, track/table, region/ID inputs, filter/intersection, output formats, summary. | `ucsc-kent-table-browser-query-export` | Linked Table Browser guide `https://genome.ucsc.edu/goldenPath/help/hgTablesHelp.html`; inspect operations/limits and identify an immutable data snapshot. |
| `docs/tutorials/gb101.md` | Inspected tutorial text on coordinate/gene/rsID/HGVS/sequence search, BLAT sequence lookup, track search, recommended sets, browser display. | None; introductory interface overlaps, no discrete executable task selected. | Inspect specific BLAT example and alignment semantics; track-search implementation; clinical variant track set and data snapshot. `gatewayTutorial.md` and `customTrackTutorial.md` are pending. |
| `src/utils/twoBitToFa/twoBitToFa.c` | Inspected usage/options, interval/list/BED paths, BED block and negative-strand logic. | `ucsc-kent-twobittofa-region-extraction` | Makefile and expected FASTA files pending; pin reference release and verify assembly/coordinate compatibility. |
| `src/utils/vcfFilter/vcfFilter.c`, `tests/makefile`, `tests/input/subset.vcf` | Inspected flag semantics, position and allele-count paths, test cases, VCF header/body preview. | `ucsc-kent-vcffilter-allele-count` | `exclude.vcf` and expected outputs pending. Original aligned FASTA and fixture terms/provenance need follow-up. |
| `src/hg/doc/liftOver.txt` | Inspected same- and cross-organism guidance, chain/net steps and warnings. | `ucsc-kent-liftover-chain-workflow` | Later workflow/output details and current command docs pending. Source is historical and explicitly says instructions are untested. |
| `src/hg/utils/vcfToHgvs/vcfToHgvs.c`, `tests/makefile`, selected test VCFs and `tests/PGTdb` tree | Inspected variant-to-transcript flow, database lookups and output fields, test load commands, Personalis agree/patch headers and leading records, component file paths/sizes. | `ucsc-kent-vcftohgvs-variant-consequences` | Inspect disagree header and expected outputs; verify benchmark provenance/terms, bundled annotation releases, and assembly compatibility. Symbolic alleles are a noted TODO. |
| `src/hg/genePredHisto`, `src/utils/bigWigSummary`, `src/utils/wigCorrelate`, `src/utils/bedGraphToBigWig`, conservation tools, broader repository | Tree/search metadata only; operation bodies not inspected. Repository tree is large and contains many utility collections. | None | Inspect command semantics/examples and associated data for transcript structure distributions, signal summaries/correlation, signal format conversion, phylogenetic conservation, BLAT and chain/net tools. README body not inspected. |

## Data links and boundaries

`twoBitToFa` names `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.2bit` as an example input. This is a mutable URL; asset size, release, checksum and terms remain unknown. The Table Browser tutorial names no specific track, so no dataset is linked to it.

The vcfFilter test fixture declares VCFv4.2, reference label `NC_045512v2`, and `faToVcf refPlus.unmapped.aligned.fasta stdout`; its header has 12 sample columns. Original FASTA, study, design and terms are unknown. The separate exclusion file's relationship to these observations was not checked.

The vcfToHgvs `PersonalisGroundTruth2017_agree.vcf` header names `JenniferYen`, the study key `A_variant_by_any_name:quantifying_annotation_discordance_across_tools_and_clinical_databases`, and UCSC test modification. It declares GRCh37/hs37d5; the `patches` file says hg38 and comments on patch sequence coordinates. The makefile separately loads chromosome, RefSeq transcript, alignment, CDS, peptide and RNA sequence tables into a test database. VCF inputs and co-loaded reference tables are separate dataset records. Exact upstream release, source provenance and terms remain unresolved.

## Stop evidence and validation

This is a targeted partial pass of a very large source tree. No access, resource or execution failure stopped work. I stopped after the selected distinct operations and supporting data leads had been inspected, leaving time for output validation; repository-wide discovery is not complete. Next sources should include the pending Table Browser guide, vcfToHgvs expected/disagree fixtures, twoBitToFa tests, conservation and signal analyses. The supplied deadline has not been reached (clock checked at 2026-09-30 14:23:56 UTC).

Both JSONL files parse. IDs are unique (5 units, 4 datasets); each dataset/unit reference resolves and every unit appears in the source map. No runtime estimate, tool execution, Harbor compatibility, deterministic grader, or license clearance was assessed. Candidate task ideas need authored input snapshots, defined outputs and an executable grading contract.
