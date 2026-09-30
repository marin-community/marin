# Reconciliation report

Input inventory: `inputs/units.jsonl`, `inputs/datasets.jsonl`, and `inputs/inspection.md`, governed by `inputs/discovery-prompt.md`. Repository `arq5x/bedtools2`, revision `614e9a5c5935ab86e873dab9072fbbaf003c1b7e`. External tutorial asset URLs retain the source URLs in the data records; their revisions are unknown.

Sources inspected during this pass: `tutorial/bedtools.md` and `docs/content/example-usage.rst`, fetched with `gh api` at the supplied commit. Tutorial setup and data description (lines 30-70) establish one 20-file observed fetal tissue DNase collection and separately sourced UCSC Table Browser annotation/reference assets. Example-guide command blocks confirmed schematic interval names and representative intersect, closest, coverage, and shuffle use (lines 11-106, 176-205, 259-292, 314-333). No source-access errors occurred. Tool manuals, tutorial answers, README/overview, and tree metadata were carried forward from input inspection, not re-read here.

Material corrections:

- Split `bedtools2_tutorial_maurano_dnase_and_annotations` into `bedtools2_tutorial_maurano_dnase_observations`, `bedtools2_tutorial_cpg_islands`, `bedtools2_tutorial_refseq_exons`, `bedtools2_tutorial_gwas_snps`, `bedtools2_tutorial_genome_sizes`, and `bedtools2_tutorial_hesc_chromhmm`. The tutorial explicitly distinguishes the DNase files from four independent Table Browser products; `genome.txt` is separately downloaded reference metadata. Unit links were updated to the supported candidate inputs.
- Split `bedtools2_packaged_interval_examples` into six `bedtools2_fixture_*` candidate records, one per named file, because packaging and filenames do not establish dataset identity. The three assets explicitly paired by the reldist examples are linked there; other units do not inherit broad fixture links.
- Retained all 17 existing unit identifiers and scientific uses. Changed `bedtools2_windowed_exon_counts` from `mixed` to `tool_use`; the tutorial composes existing `makewindows` and `intersect` commands, with no method implementation. All other role explanations now consistently identify applying/composing existing tools.
- Schematic filenames remain input requirements and are not linked as identified biological data. Unknown access/version/build/terms and non-execution remain explicit.

No records were removed or merged. The old-to-new dataset mapping is recorded above; each exact file-to-record mapping is preserved in `datasets.jsonl`. The original source inventory had an internally inconsistent queue claim (40 manuals, 17 inspected, 23 pending); counting its pending list gives 24 and 41 total names. This discrepancy is preserved and flagged in `inspection.md` for repair against a fresh tree enumeration.

Unresolved: exact tutorial asset accessions, assembly/releases, live availability and rights; packaged fixture contents and provenance; remaining tool manuals and examples. These are recorded limitations or open discovery work; no specific contradictory scientific claim or lost use remains in the revised records. No data or code was executed, and there is no evidence of availability, task validity, Harbor compatibility, or redistribution eligibility.

Recommendation: `ready_for_authoring_review`. The revised inventory is internally reconciled for the requested checks, with explicit unknowns and a pending discovery queue. This recommendation does not certify task design or data release eligibility.

Validation: both JSONL files were parsed; required fields, unique IDs, unit-to-dataset links, and related-unit links were checked after writing. Counts are 17 units and 12 data/reference/fixture records. Source inventory review was completed before the ten-minute assignment deadline; stopping reason was completion of this lightweight reconciliation task.
