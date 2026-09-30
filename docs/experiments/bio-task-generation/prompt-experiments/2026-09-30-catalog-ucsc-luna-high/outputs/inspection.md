# Source inventory inspection: ucscGenomeBrowser/kent

## Scope and records

Pinned repository revision: `ad6dd2177ad20bea9e32563ee76a1c598bccb6d5`; its recursive Git tree was reported untruncated. `README`, `src/README`, `docs/index.md`, and `docs/tutorials/index.md` orient the source: Genome Browser biological analysis/display, `src/utils` command-line tools, shared sequence/interval libraries, browser applications, tutorials, and linked services. Repository documentation says licensing varies by directory; public access is not reuse permission.

The final outputs contain **60 units and 21 dataset records**. All units are `tool_use`: these are uses of existing commands, hosted tools or analysis examples, not method implementation. Empty dataset links mean a named biological asset was not identified. Fixture records distinguish test input from study provenance; unknown terms remain explicitly unknown.

The live UCSC Utilities directory at <https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/> and linked `FOOTER.txt` were retrieved 2026-09-30. The footer reports source version 362, distinct from the pinned repository revision. Directory listing enumeration yielded 328 entries including the `blat/` and `multiz/` subdirectories. The file catalog is mapped entry by entry below; 45 entries have inspected unit records and 283 remain individually pending. The live catalog can change; presence in it does not prove any binary was executed or is installed.

## Pinned repository/tutorial source map

| Entry | Status |
|---|---|
| `README`, `src/README` | Inspected for scope, directory layout, licensing caveat, source organization. |
| `docs/index.md` | Inspected; tutorials, custom tracks, downloads, utilities and REST API links. |
| `docs/tutorials/index.md` | Inspected; enumerates tutorial sources below. |
| `docs/tutorials/tableBrowserTutorial.md` | Inspected; query, filter, intersection and output workflow in `kent-hg-tables-query`. |
| `docs/tutorials/customTrackTutorial.md` | Inspected for annotation formats, assembly and display workflow; no separate analysis promoted. |
| `docs/tutorials/gatewayTutorial.md` | Inspected for assembly discovery, sequence aliases and coordinates; no distinct analysis promoted. |
| `docs/tutorials/gb101.md` | Partial: navigation, search, BLAT/Table Browser and display modes; sections after “Viewing the Reverse Strand” pending. |
| `docs/browserSlideDecks.md` | Inspected; enumerates five deck collections. |
| `docs/slideDecks/tutorial1-basics/{presentation,tableOfContents}/index.html` | Partial: Table Browser ClinVar/GENCODE intersection and BLAT workflow inspected; remaining slides pending. |
| `docs/slideDecks/tutorial2-cancer/{presentation,tableOfContents}/index.html` | Partial: BRAF V600E and ClinVar/COSMIC/CIViC/TCGA leads found; remaining slides and data lineage pending. |
| `docs/slideDecks/tutorial3-clinical/{presentation,tableOfContents}/index.html` | Partial: BRCA2 c.8167G>C (p.Asp2723His) and TERT promoter c.-124C>T/C228T examples read; remaining expression, regulatory, prediction and frequency sections pending. TERT has unit `kent-tert-noncoding-driver-evidence`. |
| `docs/slideDecks/tutorial4-clinical-cases/{presentation,tableOfContents}/index.html` | Partial: POU1F1 upstream deletion recorded as `kent-pou1f1-regulatory-deletion-case`; MECP2, MAP2K2, TCF4, IGHMBP2, KCTD7, SHH/ZRS, RNU4-2, CNV, FGF14 and gallery remain pending. The linked POU1F1 session fetch returned cache miss; session data not inspected. |
| `docs/slideDecks/tutorial5-teaching/{presentation,tableOfContents}/index.html` | TOC inspected; molecular biology, variant, disease and evolution sessions pending. |
| `src/hg/liftOver/liftOver.c`, `src/hg/hgLiftOver/hgLiftOver.c`, `src/blat/blat.c`, `src/hg/hgPcr/hgPcr.c`, `src/utils/faCount/faCount.c`, `src/utils/twoBitToFa/twoBitToFa.c`, `src/utils/bedToBigBed/bedToBigBed.c`, `src/utils/bigBedToBed/bigBedToBed.c`, `src/utils/wigToBigWig/wigToBigWig.c` | Relevant entry points and usage/format portions inspected; units `kent-liftover`, `kent-blat-alignment`, `kent-insilico-pcr`, `kent-fa-count`, `kent-two-bit-to-fasta`, `kent-bed-to-bigbed`, `kent-bigbed-to-bed`, `kent-wig-to-bigwig`. This does not imply all implementation/tests inspected. |
| `src/utils/userApps/mkREADME.sh`, `src/utils/userApps/bigBedWigs.txt` | Inspected. README generator derives command list from installed binaries; guide explains indexed bigBed/bigWig and chromosome-size prerequisites. |
| `src/utils/userApps/fetchChromSizes` | Located but source not inspected. Earlier wrong path `src/utils/fetchChromSizes` returned 404; corrected path found in tree. |
| `src/utils/twoBitToFa/tests/makefile` | Guessed path returned 404; root-level `src/utils/twoBitToFa/makefile` was located but not inspected. |

## UCSC web catalog

`https://genome.ucsc.edu/util.html` was retrieved 2026-09-30. Each public entry has a disposition; inspected means the cited source portions are in the unit record and does not establish live execution.

| Public operation | Status |
|---|---|
| Genome Browser | Partial: `gb101.md`, custom-track and tutorial navigation/display sources; browser-track analyses and remaining guide sections pending. |
| BLAT | Inspected: `kent-blat-alignment`. |
| In-Silico PCR | Partial: `kent-insilico-pcr`; result/catalog and entry point, detailed request parameters pending. |
| Table Browser | Inspected: `kent-hg-tables-query`. |
| LiftOver | Inspected: `kent-liftover`; no specific chain/data product chosen. |
| Gene Sorter | Inspected: `kent-gene-sorter`; relation ranking, filtering, output; source-data release unresolved. |
| Genome Graphs | Inspected: `kent-genome-graphs`; graph upload/import, correlation, threshold regions and Gene Sorter handoff; no input dataset selected. |
| Data Integrator | Inspected: `kent-data-integrator-overlap-query`. |
| UShER | Inspected: `kent-usher-sarscov2-placement`; exact deployed reference tree/protobuf remains unidentified. |
| Gene Interactions | Inspected: `kent-gene-interaction-graph`; curated interaction sources and historical hgFixed tables. |
| VisiGene | Inspected: `kent-visigene-in-situ-image-search`; distinct collections, detailed context only for Allen. |
| DNA Duster | Catalog description inspected: `kent-dna-duster`; implementation, endpoint and options pending. |
| Protein Duster | Catalog description inspected: `kent-protein-duster`; implementation, endpoint and options pending. |
| Phylogenetic Tree PNG Maker | Inspected: `kent-phylo-png`; renderer/form and Newick grammar; repository `.nh` example pending. |
| REST `/getData/sequence` | Inspected: `kent-rest-sequence`; official guide describes interval and reverse-complement options. |
| Other REST operations | Individually pending: `/findGenome`, `/list/publicHubs`, `/list/ucscGenomes`, `/list/genarkGenomes`, `/list/hubGenomes`, `/list/files`, `/list/tracks`, `/list/chromosomes`, `/list/schema`, `/getData/track`, `/search`. Guide retrieved 2026-09-30; deployed revision unknown. |
| VAI | Inspected: `kent-variant-annotation-integrator`; official tools index is not its source; source/help describe variant formats, transcript effects and optional annotation sources. |

## Utilities command catalog: entry-level status

The mutable official footer/catalog was retrieved 2026-09-30. Each name below is a distinct entry; `blat/` and `multiz/` are directory entries. The 45 inspected commands resolve to a unit ID; all others remain individual pending leads, not a combined “other utilities” category.

| Command | Status |
|---|---|
| `ameme` | inspected: `kent-ameme-motif-discovery` |
| `bedClip` | inspected: `kent-bed-clip` |
| `bedCommonRegions` | inspected: `kent-bed-common-regions` |
| `bedCoverage` | inspected: `kent-bed-coverage` |
| `bedGeneParts` | inspected: `kent-bed-gene-parts` |
| `bedGraphToBigWig` | inspected: `kent-bedgraph-to-bigwig` |
| `bedIntersect` | inspected: `kent-bed-interval-intersection` |
| `bedItemOverlapCount` | inspected: `kent-bed-item-overlap-count` |
| `bedToBigBed` | inspected: `kent-bed-to-bigbed` |
| `bigBedToBed` | inspected: `kent-bigbed-to-bed` |
| `bigWigAverageOverBed` | inspected: `kent-bigwig-average-over-bed` |
| `bigWigCorrelate` | inspected: `kent-bigwig-correlate` |
| `bigWigSummary` | inspected: `kent-bigwig-summary` |
| `bigWigToBedGraph` | inspected: `kent-bigwig-to-bedgraph` |
| `faCount` | inspected: `kent-fa-count` |
| `faFilter` | inspected: `kent-fasta-filter` |
| `faRandomize` | inspected: `kent-fasta-randomize` |
| `faTrans` | inspected: `kent-fasta-translate` |
| `fastqStatsAndSubsample` | inspected: `kent-fastq-stats-subsample` |
| `featureBits` | inspected: `kent-feature-bits` |
| `findMotif` | inspected: `kent-find-motif` |
| `genePredCheck` | inspected: `kent-gene-pred-check` |
| `hgGcPercent` | inspected: `kent-gc-window-profile` |
| `gff3ToGenePred` | inspected: `kent-gff3-to-genepred` |
| `liftOver` | inspected: `kent-liftover` |
| `mafCoverage` | inspected: `kent-maf-coverage` |
| `mafFilter` | inspected: `kent-maf-filter-blocks` |
| `mafFetch` | inspected: `kent-maf-fetch-overlap` |
| `oligoMatch` | inspected: `kent-oligo-match` |
| `twoBitToFa` | inspected: `kent-two-bit-to-fasta` |
| `twoBitInfo` | inspected: `kent-two-bit-sequence-inventory` |
| `gtfToGenePred` | inspected: `kent-gtf-to-genepred` |
| `vai.pl` | inspected: `kent-variant-annotation-integrator` |
| `wigToBigWig` | inspected: `kent-wig-to-bigwig` |

| `pslStats` | inspected: `kent-psl-alignment-statistics` |

| `gff3ToPsl` | inspected: `kent-gff3-alignment-to-psl` |
| `bedToGenePred` | inspected: `kent-bed-to-genepred` |
| `bedToPsl` | inspected: `kent-bed-to-psl` |
| `bigWigMerge` | inspected: `kent-bigwig-merge` |
| `twoBitMask` | inspected: `kent-two-bit-mask` |

| `faSize` | inspected: `kent-fa-size` |
| `faSomeRecords` | inspected: `kent-fasta-record-subset` |
| `genePredHisto` | inspected: `kent-genepred-feature-histograms` |
| `vcfToBed` | inspected: `kent-vcf-to-bed-annotations` |
| `hgvsToVcf` | inspected: `kent-hgvs-to-vcf` |

Pending entries (each command individually pending scientific-semantic inspection):

`addCols`
`autoDtd`
`autoSql`
`autoXml`
`ave`
`aveCols`
`axtChain`
`axtSort`
`axtSwap`
`axtToMaf`
`axtToPsl`
`bamToPsl`
`barChartMaxLimit`
`bedExtendRanges`
`bedGraphPack`
`bedJoinTabOffset`
`bedJoinTabOffset.py`
`bedMergeAdjacent`
`bedPartition`
`bedPileUps`
`bedRemoveOverlap`
`bedRestrictToPositions`
`bedSort`
`bedToExons`
`bedWeedOverlapping`
`bigBedInfo`
`bigBedNamedItems`
`bigBedSummary`
`bigChainBreaks`
`bigChainToChain`
`bigGenePredToGenePred`
`bigGuessDb`
`bigHeat`
`bigMafToMaf`
`bigPslToPsl`
`bigWigCat`
`bigWigCluster`
`bigWigInfo`
`bigWigToWig`
`binFromRange`
`blastToPsl`
`blastXmlToPsl`
`blat/`
`calc`
`catDir`
`catUncomment`
`chainAntiRepeat`
`chainBridge`
`chainCleaner`
`chainFilter`
`chainMergeSort`
`chainNet`
`chainPreNet`
`chainScore`
`chainSort`
`chainSplit`
`chainStitchId`
`chainSwap`
`chainToAxt`
`chainToBigChain`
`chainToMaf`
`chainToPsl`
`chainToPslBasic`
`checkAgpAndFa`
`checkCoverageGaps`
`checkHgFindSpec`
`checkTableCoords`
`chopFaLines`
`chromGraphFromBin`
`chromGraphToBin`
`chromToUcsc`
`clusterGenes`
`clusterMatrixToBarChartBed`
`colTransform`
`countChars`
`cpg_lh`
`crTreeIndexBed`
`crTreeSearchBed`
`dbDbToHubTxt`
`dbSnoop`
`dbTrash`
`endsInLf`
`estOrient`
`expMatrixToBarchartBed`
`faAlign`
`faCmp`
`faFilterN`
`faFrag`
`faNoise`
`faOneRecord`
`faPolyASizes`
`faRc`
`faSplit`
`faToFastq`
`faToTab`
`faToTwoBit`
`faToVcf`
`fastqToFa`
`fetchChromSizes`
`fixStepToBedGraph.pl`
`fixTrackDb`
`gapToLift`
`gencodeVersionForGenes`
`genePredCompare`
`genePredFilter`
`genePredSingleCover`
`genePredToBed`
`genePredToBigGenePred`
`genePredToFakePsl`
`genePredToGtf`
`genePredToMafFrames`
`genePredToProt`
`gensub2`
`getRna`
`getRnaPred`
`gmtime`
`headRest`
`hgBbiDbLink`
`hgFakeAgp`
`hgFindSpec`
`hgGoldGapGl`
`hgLoadBed`
`hgLoadChain`
`hgLoadGap`
`hgLoadMaf`
`hgLoadMafSummary`
`hgLoadNet`
`hgLoadOut`
`hgLoadOutJoined`
`hgLoadSqlTab`
`hgLoadWiggle`
`hgSpeciesRna`
`hgTrackDb`
`hgWiggle`
`hgsql`
`hgsqldump`
`hicInfo`
`htmlCheck`
`hubCheck`
`hubClone`
`hubPublicCheck`
`hubtools`
`ixIxx`
`lastz-1.04.00`
`lastz_D-1.04.00`
`lavToAxt`
`lavToPsl`
`ldHgGene`
`liftOverMerge`
`liftUp`
`linesToRa`
`localtime`
`mafAddIRows`
`mafAddIRowsStream`
`mafAddQRows`
`mafFrag`
`mafFrags`
`mafGene`
`mafMeFirst`
`mafNoAlign`
`mafOrder`
`mafRanges`
`mafSpeciesList`
`mafSpeciesSubset`
`mafSplit`
`mafSplitPos`
`mafToAxt`
`mafToBigMaf`
`mafToBigMafSummary`
`mafToPsl`
`mafToSnpBed`
`mafsInRegion`
`makeTableList`
`maskOutFa`
`matrixClusterColumns`
`matrixMarketToTsv`
`matrixNormalize`
`matrixToBarChartBed`
`mktime`
`mrnaToGene`
`multiz/`
`netChainSubset`
`netClass`
`netFilter`
`netSplit`
`netSyntenic`
`netToAxt`
`netToBed`
`newProg`
`newPythonProg`
`nibFrag`
`nibSize`
`overlapSelect`
`para`
`paraFetch`
`paraHub`
`paraHubStop`
`paraNode`
`paraNodeStart`
`paraNodeStatus`
`paraNodeStop`
`paraSync`
`paraTestJob`
`parasol`
`positionalTblCheck`
`pslCDnaFilter`
`pslCat`
`pslCheck`
`pslDropOverlap`
`pslFilter`
`pslHisto`
`pslLiftSubrangeBlat`
`pslMap`
`pslMapPostChain`
`pslMrnaCover`
`pslPairs`
`pslPartition`
`pslPosTarget`
`pslPretty`
`pslProtToRnaCoords`
`pslRc`
`pslRecalcMatch`
`pslRemoveFrameShifts`
`pslReps`
`pslScore`
`pslSelect`
`pslSomeRecords`
`pslSort`
`pslSortAcc`
`pslSpliceJunctions`
`pslSplitOnTarget`
`pslSwap`
`pslToBed`
`pslToBigPsl`
`pslToChain`
`pslToPslx`
`pslxToFa`
`qaToQac`
`qacAgpLift`
`qacToQa`
`qacToWig`
`raSqlQuery`
`raToLines`
`raToTab`
`randomLines`
`rmFaDups`
`rmskAlignToPsl`
`rowsToCols`
`sizeof`
`spacedToTab`
`splitFile`
`splitFileByColumn`
`sqlToXml`
`strexCalc`
`stringify`
`subChar`
`subColumn`
`tabFmt`
`tabQuery`
`tabToTabDir`
`tailLines`
`tdbQuery`
`tdbRename`
`tdbSort`
`textHistogram`
`tickToDate`
`toLower`
`toUpper`
`trackDbIndexBb`
`transMapPslToGenePred`
`trfBig`
`twoBitDup`
`ucscApiClient`
`udr`
`validateFiles`
`validateManifest`
`varStepToBedGraph.pl`
`webSync`
`wigCorrelate`
`wigEncode`
`wordLine`
`xmlCat`
`xmlToSql`

## Data, boundaries and access

- VisiGene products remain separate: Allen Brain Atlas, Jackson GXD, Mahoney, GENSAT and NIBB. Only Allen source detail supports eight-week-old male mouse sagittal-brain mRNA in-situ probes; other collections need image-level provenance recovery.
- UShER's wuhCor1 / NC_045512.2 reference and external configured phylogeny are separate products; exact tree/protobuf release is unknown. Historical guide sequence counts are not presented as current.
- `faCount` tests name FASTA and 2bit inputs, but matching count outputs do not prove identical bases. `twoBitToFa` supports BED-block concatenation, intron omission and minus-strand reverse-complement; fixture origins unknown.
- Test BED, MAF and FASTA assets are fixture records only. Some inputs' bytes were not opened, so biological provenance, release and reuse terms remain unknown.
- The POU1F1 example identifies hg38 chr3:87,279,612–87,288,322 and an approximately 3 kb upstream homozygous deletion with regulatory/conservation/expression context; linked session fetch failed. TERT is a noncoding c.-124C>T/C228T promoter example; exact session assets remain pending.
- `kent-phylo-png` covers renderer behavior; a repository `.nh` file remains unread. The invalid `fetchChromSizes` and `twoBitToFa` test paths returned GitHub 404; a raw web fetch returned cache miss. No biological asset was downloaded.

## Validation, execution limits and stopping point

This is a partial source inventory. No build/install, scientific tool execution, live hosted submission, biological dataset download, Harbor run, model call or cloud job occurred. Source inspection does not establish run correctness, data access terms, suitability, runtime or grader feasibility.

The current JSONL counts are 60 units and 21 datasets. Approximate elapsed time is 69 minutes from the earlier recorded checkpoint estimate (~25 minutes at 16:33 UTC) through finalization at 17:15 UTC; exact start time was not captured. Stopping reason: a useful partial handoff, not exhaustion or a resource/access blocker. Remaining utility commands, API endpoints and slide sections listed above remain accessible inspection leads and were not completed in this pass.
