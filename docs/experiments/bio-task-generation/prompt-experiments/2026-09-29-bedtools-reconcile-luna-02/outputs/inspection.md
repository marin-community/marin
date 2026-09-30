# bedtools2 source inventory after reconciliation

Repository: [arq5x/bedtools2](https://github.com/arq5x/bedtools2), fixed revision `614e9a5c5935ab86e873dab9072fbbaf003c1b7e`; retrieval date 2026-09-29. This is a source inventory, not task validation. No assets were downloaded, previewed, or executed.

The inventory covers documented interval operations and one multi-question fetal tissue DNase regulatory-genomics tutorial workflow. Units remain centered on documented operations: overlap, enrichment, coverage, nearest feature, Jaccard, shuffle, map, merge, relative distance, subtract, complement, flank, annotate, multi-set intersection, window, and windowed exon counts. The tutorial workflow links independent source products only when supported as candidate inputs. Component tool operations remain independently selectable. All 17 units are classified `tool_use`: composing existing CLI commands or writing surrounding analysis does not implement the underlying bedtools method.

## Source map and queue

| Collection | Status | Unit records | Remaining leads |
|---|---|---|---|
| `README.md`, `docs/content/overview.rst` | Carried forward as inspected at summary level; not re-read this pass | Context for inventory | Tool catalog needs item-by-item coverage. |
| `docs/content/example-usage.rst` | Carried forward as partial; this pass read intersect, closest, coverage, shuffle examples (source sections confirmed by headings/commands) | Intersect, closest, coverage, shuffle and linked units | Remaining sections and command compositions. |
| `docs/content/tools/` | Carried forward: 17 selected manuals recorded inspected; fixed-revision tree count verified as 41 tool manuals, leaving 24 pending | Existing manual-backed units listed below | Pending manuals below. |
| `tutorial/bedtools.md` | Re-read this pass: setup and file descriptions (lines 30-70); previously recorded tutorial analysis/puzzle/answer-key sections are carried-forward evidence | `bedtools2_tutorial_dnase_regulatory_analysis`, `bedtools2_windowed_exon_counts`; other units cite relevant puzzles | Fully enumerate remaining tutorial headings, answer-key details, and primary-study provenance. |
| `data/` | Carried forward: fixed-revision tree paths and sizes; this pass did not preview content | Six separate fixture/reference candidates; reldist links only the RefSeq exon, AluY, and GERP candidates named in its manual | Preview and identify provenance, semantics, terms before task use. |
| `src/`, `test/` | Tree glimpse only, carried forward | None | Inspect implementation/tests for authored tasks. |

The carried-forward tool-manual queue from input inspection is retained and its count checked against the fixed-revision recursive Git tree (41 `.rst` paths): inspected are `annotate`, `closest`, `complement`, `coverage`, `fisher`, `flank`, `genomecov`, `intersect`, `jaccard`, `makewindows`, `map`, `merge`, `multiinter`, `reldist`, `shuffle`, `subtract`, and `window` (17 entries). Pending are `bamtobed`, `bamtofastq`, `bed12tobed6`, `bedpetobam`, `bedtobam`, `cluster`, `expand`, `getfasta`, `groupby`, `igv`, `links`, `maskfasta`, `multicov`, `nuc`, `overlap`, `pairtobed`, `pairtopair`, `random`, `shift`, `slop`, `sort`, `summary`, `tag`, and `unionbedg` (24 entries). The input inspection had a stale count (40 manuals and 23 pending); the fixed-revision tree confirms 41 manuals, matching 17 inspected plus 24 pending.

## Reconciliation observations

This pass read `tutorial/bedtools.md` at the assigned revision via `gh api`. Its setup identifies one `maurano.dnaseI.tgz` containing 20 fetal tissue DNase BED files and names five separate downloaded support files: `cpg.bed`, `exons.bed`, `gwas.bed`, `genome.txt`, and `hesc.chromHmm.bed`. At lines 53-68, the prose says the DNase files represent measured sites in 20 fetal tissue samples and explicitly attributes the latter four annotation files to UCSC Table Browser. Thus the observation bundle, four annotations, and chromosome-size metadata are separate records; no common study identity is inferred for them. The tutorial cites Maurano et al. 2012, but accession mapping and precise source versions remain unknown.

The same tutorial supports distinct scientific questions including overlap, nearest GWAS-SNP-to-exon distance, interval coverage/similarity, randomized promoter comparison, and exon counts in 500 kb windows. A broad tutorial unit was retained to preserve the workflow context, with independent questions called out for task author choice. The `makewindows` manual is a stub per carried-forward evidence; the tutorial puzzle/answer remains the evidence for windowed exon counts.

The example guide this pass confirmed representative operations for intersect (lines 11-106), closest (176-205), coverage (259-292), and shuffle (314-333). Its names such as `reads.bed`, `genes.bed`, `variants.bed`, and `windows10kb.bed` are schematic examples, not identified data assets. The Fisher 500-base example is similarly a toy example, not biological observations.

The packaged `data/` inventory was split into six candidate records because filename/directory co-location does not establish common observational identity. Their content, biological provenance, and exact relationships remain unknown; file formats inferred from suffixes are not treated as verified. `reldist` has links only to the packaged RefSeq exon, AluY and GERP assets explicitly named in its examples.

## Coverage, unresolved items, and stopping

The input carried forward 17 unit records and two dataset records; outputs contain 17 units and 12 data/reference/fixture records. All existing unit IDs were retained. Old dataset ID `bedtools2_tutorial_maurano_dnase_and_annotations` maps to `bedtools2_tutorial_maurano_dnase_observations`, `bedtools2_tutorial_cpg_islands`, `bedtools2_tutorial_refseq_exons`, `bedtools2_tutorial_gwas_snps`, `bedtools2_tutorial_genome_sizes`, and `bedtools2_tutorial_hesc_chromhmm`. Old `bedtools2_packaged_interval_examples` maps to six individual `bedtools2_fixture_*` candidate records; see the output records for exact asset mapping. No scientific use was removed.

The tutorial asset versions/builds, live availability, accessions, terms, and redistribution rights are unknown. Packaged fixture content was not read. No data execution, task validation, Harbor compatibility, or release eligibility was established. The remaining manual queue, complete tutorial enumeration, primary study check, and fixture provenance inspection are open. The assignment-specific reconciliation checks were completed before the ten-minute deadline; work stopped because the requested lightweight reconciliation pass and structural validation were complete, not because source access or resources failed. Further inspection of the queued materials is still useful before authoring.

Source retrieval in this pass succeeded for the tutorial and example guide at the fixed commit; no external biological source was opened. Evidence carried forward from the input inventory is identified above as carried-forward rather than freshly checked.
