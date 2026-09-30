# bedtools2 source inventory

Inspected repository: [arq5x/bedtools2](https://github.com/arq5x/bedtools2), revision `614e9a5c5935ab86e873dab9072fbbaf003c1b7e` (runner-resolved immutable commit). Retrieval date: 2026-09-29. Repository files were read with `gh api` at that revision; no repository clone, dataset download, build, or execution was performed.

## Repository and scientific scope

The README and overview describe bedtools as a C++ command-line suite for arithmetic on genomic features in BED, GFF/GTF, VCF, and BAM formats. Operations include overlap queries, interval aggregation/transformation, sequence extraction/masking, coverage, and statistical summaries. The example-use guide demonstrates common commands and compositions. The tutorial supplies a substantial regulatory-genomics exercise based on fetal tissue DNase I hypersensitivity and annotation tracks. The recursive tree also shows source modules and many focused test fixture directories; tests were not inspected as scientific studies.

Inventory records are organized around distinct analysis questions rather than each flag: interval overlap/reporting, overlap enrichment, coverage profiles, and the multi-stage tutorial. Dataset records distinguish tutorial study/annotation inputs from repository-packaged interval examples. Scientific suitability, runtimes, deterministic grading and task validity remain untested.

## Source map and inspection queue

| Collection/location | Status | Resulting units | Uninspected leads / continuation |
|---|---|---|---|
| `README.md` and `docs/content/overview.rst` | Inspected at summary level | Context for all units | Overview tool catalog contains the broader tool set; inspect entries individually. |
| `docs/content/example-usage.rst` | Partially inspected; overlap and coverage sections read, remaining document not systematically enumerated | `bedtools2_intersect_interval_overlap`, `bedtools2_coverage_profile` | Continue remaining examples, especially closest, map, merge, shuffle and multi-command compositions; exact section coverage is incomplete. |
| `docs/content/tools/` | Tree enumerated in supplied inspection; selected manuals inspected | See queue below | 40 `.rst` entries enumerated; eighteen selected manuals are now recorded as inspected; the tutorial answer key was additionally read. |
| `tutorial/bedtools.md` | Inspected for synopsis, setup/data descriptions and analysis/puzzle material, including late tutorial PCA/heatmap and question list | `bedtools2_tutorial_dnase_pairwise_similarity`, `bedtools2_tutorial_nonexonic_nonpromoter_regions`, `bedtools2_tutorial_enhancer_covered_exons`, `bedtools2_tutorial_gwas_exonic_fraction`, `bedtools2_tutorial_gwas_enhancer_promoter_fraction`, `bedtools2_tutorial_largest_chromhmm_state` | Tutorial source is long; enumerate all headings and inspect any omitted sections, command outputs, and answer key. Resolve primary-study and file metadata. |
| `data/` | Tree paths and byte sizes inspected, contents not previewed | six independent packaged interval/reference asset records; provenance unresolved | Preview selected BED/gzip records, identify source citations and data terms before use. |
| `src/` and `test/` | Tree glimpsed only | None | Follow code/tests for any task chosen by author; code presence alone was not used to invent scientific units. |

### Tool-manual queue

`inspected` means the manual's stated purpose and relevant semantics/examples were reviewed sufficiently to record the unit; it does not mean exhaustive option coverage. All other listed entries remain pending.

| Manual | Status / unit |
|---|---|
| `annotate.rst` | inspected — `bedtools2_annotate_multi_tracks` |
| `bamtobed.rst` | pending |
| `bamtofastq.rst` | pending |
| `bed12tobed6.rst` | pending |
| `bedpetobam.rst` | pending |
| `bedtobam.rst` | pending |
| `closest.rst` | inspected — `bedtools2_closest_feature_distance` |
| `cluster.rst` | pending |
| `complement.rst` | inspected — `bedtools2_complement_uncovered_regions` |
| `coverage.rst` | inspected — `bedtools2_coverage_profile` |
| `expand.rst` | pending |
| `fisher.rst` | inspected — `bedtools2_fisher_overlap_enrichment` |
| `flank.rst` | inspected — `bedtools2_flank_feature_intervals` |
| `genomecov.rst` | inspected — `bedtools2_coverage_profile` |
| `getfasta.rst` | pending |
| `groupby.rst` | inspected — semantics support `bedtools2_tutorial_largest_chromhmm_state`; tutorial supplies no worked answer |
| `igv.rst` | pending |
| `intersect.rst` | inspected — `bedtools2_intersect_interval_overlap` |
| `jaccard.rst` | inspected — `bedtools2_interval_jaccard_similarity` |
| `links.rst` | pending |
| `makewindows.rst` | inspected — substantive doc is a stub; tutorial answer yields `bedtools2_windowed_exon_counts` |
| `map.rst` | inspected — `bedtools2_map_interval_values` |
| `maskfasta.rst` | pending |
| `merge.rst` | inspected — `bedtools2_merge_clustered_intervals` |
| `multicov.rst` | pending |
| `multiinter.rst` | inspected — `bedtools2_multi_set_common_intervals` |
| `nuc.rst` | pending |
| `overlap.rst` | pending |
| `pairtobed.rst` | pending |
| `pairtopair.rst` | pending |
| `random.rst` | pending |
| `reldist.rst` | inspected — `bedtools2_relative_interval_distance` |
| `shift.rst` | pending |
| `shuffle.rst` | inspected — `bedtools2_shuffle_null_overlap` |
| `slop.rst` | pending |
| `sort.rst` | pending |
| `subtract.rst` | inspected — `bedtools2_subtract_interval_regions` |
| `summary.rst` | pending |
| `tag.rst` | pending |
| `unionbedg.rst` | pending |
| `window.rst` | inspected — `bedtools2_window_nearby_features` |

## Inspected findings and links

- `closest` supports nearest-feature joins and signed/strand-aware distances; the tutorial specifically asks mean GWAS-SNP-to-exon distance. `jaccard` summarizes base-pair set similarity and underpins the tutorial CpG versus enhancer/promoter question. `map` summarizes values of overlapping B intervals onto A features; `merge` combines nearby/overlapping loci and can aggregate annotations. `reldist` measures proximity even without overlap, with documented examples using the repository-packaged RefSeq exon, AluY and GERP tracks. `subtract` yields residual features (including the gene-minus-intron example), `complement` yields genome-uncovered regions, and `flank` constructs chromosome-bounded, strand-aware neighborhoods; the tutorial specifically suggests two-base exon flanks for splice sites. `annotate` reports coverage/counts across multiple annotation tracks, and `multiinter` partitions shared membership across multiple collections; the tutorial’s 20 DNase samples are a candidate multi-set input. `window` searches fixed neighborhoods and complements nearest-distance analysis. `shuffle` creates seeded or constrained genomic null intervals and is proposed in the tutorial as a promoter-randomization comparison. These form a related nearest-feature/similarity/null family, although task questions and assumptions differ.

- `intersect` supports overlap comparisons across interval collections and controls reporting (e.g. counts, original entries, fraction/reciprocal overlap); the examples name gene/read, exon/repeat, SV/segmental duplication and LINE/SINE questions. It is related to Fisher enrichment and the tutorial's overlap questions.
- `fisher` compares observed overlap counts with an interval-count contingency model using a genome-size file. Its own documentation says the possible-interval count is heuristic and recommends validating low p-values by simulation; it cites an evaluation based on canonical chr1 genes and shuffled intervals. No source dataset was identified for the toy 500-base worked example.
- `coverage` and `genomecov` document per-feature/window and genome-wide coverage, including split alignments and fragment coverage. The example guide illustrates 10 kb windows. This may compose with tutorial interval analysis, but the specific example filenames are not identified datasets.
- The tutorial describes 20 fetal tissue DNase I hypersensitivity BED files, plus separate CpG, RefSeq exon, GWAS and hESC ChromHMM tracks. Its puzzles support several distinct questions (e.g. GWAS-to-exon distance, annotation overlap fractions, enhancer/promoter comparison, shuffled controls, ChromHMM state base pairs). The inventory separates the pairwise 20-tissue Jaccard workflow and the independently selectable tutorial puzzles into operation-specific units; puzzle 10 retains an answer-key detail check as an open limitation. Tutorial identifies a Maurano et al. 2012 paper but precise accessions/build and current data rights remain unknown.

Dataset asset metadata for tutorial URLs is document-derived only. Packaged `data/` file sizes are from Git tree metadata and formats/semantics are not verified by reading the assets. Software license statements do not establish third-party dataset redistribution terms.

## Omissions, access and stopping point

No source retrieval errors occurred. The GitHub tree was successfully read and filtered to enumerate all 40 tool-manual paths. Large tree output was truncated in the initial listing, so a focused tree query was used for the manual list. No inaccessible source was established. External tutorial data URLs and linked scientific paper were not opened; these are open leads, not evidence of absence. Test suites, all source implementations, individual test fixtures, and the remaining 23 tool manuals were not inspected. The makewindows manual is a stub, but the tutorial puzzle and answer key provide its 500 kb exon-count workflow.

This reconciliation pass repaired the supplied inventory only. The source map and pending queue remain carried forward; source-backed records remain distinct from uninspected leads and unexecuted examples. The repository's command-manual queue and tutorial still contain useful uninspected material; the next worker should prioritize remaining example-usage sections and primary-study/data provenance, then reconcile new dataset links.

## Hypotheses for task author (not source-backed validation)

The tutorial's answer key provides explicit expected result examples and commands, which may help design output-file graders; independent execution and completeness checks are still required. Combining intersect plus Fisher could produce an enrichment analysis, and coverage operations could yield bedGraph artifacts. These are hypotheses only; source inspection did not establish dataset availability, task execution, Harbor compatibility, or a validated grader.

## Reconciliation status

This review read fixed-revision intersect and fisher manuals plus tutorial portions describing assets, interval examples, the fetal-tissue Jaccard workflow, and tutorial questions. Independently sourced tutorial products and unrelated packaged assets are separate records; old-to-new mappings and remaining unknowns are in `reconciliation.md`. Current inventory counts are 22 units and 13 data records, derived from output files. No assets were downloaded or commands from the scientific package executed.
