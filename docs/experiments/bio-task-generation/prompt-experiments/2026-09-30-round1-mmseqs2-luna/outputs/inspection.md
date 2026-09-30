# MMseqs2 source inventory

Inspected repository: `soedinglab/MMseqs2`, revision `564f40d8857f4eca4e1dfe100c67c155b1933e70`, pinned before launch through the GitHub commits API. Retrieval date: 2026-09-30 UTC. The recursive Git tree reported `truncated: false`. All source links below point to that immutable revision. Source inspection only; no software execution, biological-data download, dependency installation, or task validation was performed.

## Repository and findings

The README describes a C++ command-line suite for sequence search and clustering across protein and nucleotide collections. Public uses include homology search, cascaded and linear-time clustering, taxonomic assignment, translated nucleotide/protein searches, reciprocal best hits, and iterative/profile search. `data/workflow/` contains shell workflows that compose lower-level MMseqs2 commands; `src/workflow/` and `src/util/` contain their implementations and utilities. The repository also contains a large vendored dependency tree and lower-level algorithm code; these were not broadly inventoried because this pass focused on user-facing scientific workflows.

The twelve records in `units.jsonl` cover: ordinary sequence search; protein/nucleotide clustering; sequence and contig taxonomy; translated search; reciprocal best hits; iterative sequence-profile search; target-profile search; iterative profile enrichment; proteome-level search/clustering; set-level multi-hit search; and incremental cluster update. These are provisional operation boundaries. Potential task questions include hit retrieval, sequence-family assignment, taxon assignment/contig aggregation, cross-collection reciprocal matching, proteome comparison, incremental clustering and iterative profile-based discovery. Those are hypotheses for task authoring, not validated benchmark tasks.

Five records are included: two bundled FASTA fixtures, the Powassan case tutorial reads, the ERR1384114 human gut metagenomic run, and the tutorial's historical 2018-03 Swiss-Prot sequence/mapping reference. The tutorial page is mutable and last edited 2026-09-07. Data metadata and access limits are recorded without downloading files. The pinned tree reports `examples/DB.fasta` as 11,434,968 bytes and `examples/QUERY.fasta` as 304,764 bytes. The README uses them in clustering and search examples. Their records, organisms, provenance, study design, relationship to each other, and sequence-data terms remain unknown; no file contents were fetched. The README names UniProtKB/Swiss-Prot as a searchable reference example and lists UniRef, NR, NT and PFAM in the database workflow, but this pass did not inspect releases, accessions, terms, or download metadata for those products, so they are not represented as concrete dataset records.

## Source map and inspection queue

| Collection/location | Status | Unit IDs / entries inspected | Uninspected leads |
|---|---|---|---|
| Repository overview: `README.md` | Inspected, selected sections | `mmseqs2_easy_search`, `mmseqs2_easy_clustering`, `mmseqs2_easy_taxonomy`, `mmseqs2_translated_search`, `mmseqs2_reciprocal_best_hits`, `mmseqs2_iterative_profile_search`, `mmseqs2_target_profile_search`; publications and setup references noted | Full publication-linked studies and tutorials; detailed parameter, output, taxonomy, and translated-search wiki sections |
| Workflow scripts: `data/workflow/` | Partial pass; inspected entries below | `easysearch.sh` → `mmseqs2_easy_search`; `easycluster.sh`, `linclust.sh`, `nucleotide_clustering.sh` → `mmseqs2_easy_clustering`; `easytaxonomy.sh`, `taxpercontig.sh` → `mmseqs2_easy_taxonomy`; `translated_search.sh` → `mmseqs2_translated_search`; `rbh.sh`, `easyrbh.sh` → `mmseqs2_reciprocal_best_hits`; `iterativepp.sh` → `mmseqs2_iterative_profile_search`; `searchtargetprofile.sh` → `mmseqs2_target_profile_search`; `enrich.sh` → `mmseqs2_profile_enrichment` | See per-entry queue below |
| Workflow implementations: `src/workflow/` | Pending | None inspected | EasySearch, Cluster/EasyCluster, Taxonomy, Linclust, Rbh, Search, Enrich, and related implementations to verify CLI parameter semantics and output details |
| Command implementations/utilities: `src/taxonomy/`, `src/multihit/`, `src/util/` | Pending | None inspected | Taxonomic aggregation and reporting, `result2rbh`, profile conversion and output formatting |
| Tests: `src/test/` | Pending | None inspected | Focused tests for taxonomy, profiles, alignment and RBH-related utility semantics; tests would establish fixtures/oracles but not biological provenance |
| Bundled sequence fixtures: `examples/` | Metadata only | `examples/DB.fasta` → `mmseqs2_examples_db_fasta`; `examples/QUERY.fasta` → `mmseqs2_examples_query_fasta` | Bounded header preview and source/licensing review during authoring; identify taxa and determine whether any fixture is suitable as observed biological data |
| Download/setup workflow: `data/workflow/databases.sh` and `createtaxdb.sh` | Pending | None | Enumerate available reference products, accessions/releases, download source, versioning, mapping requirements, and terms |
| Documentation/tutorials: GitHub Wiki linked by README | Partially inspected (mutable page, last edited 2026-09-07) | `mmseqs2_metagenomic_pathogen_case`, `mmseqs2_easy_taxonomy`, `mmseqs2_easy_clustering` | Other wiki pages and tutorials; resolve stable wiki snapshots or save page revision before citation |
| Linked publications listed in README | Not inspected beyond citations | None | Read papers and their supplementary datasets/workflows, especially clustering (2018), metagenomic contig taxonomy (2021), and GPU search (2025) |

### Workflow entry queue

Inspected entries have their unit IDs above. Other scripts were visible in the complete pinned repository tree and remain pending; their names are leads, not evidence that their operations have been inspected.

| Path | Status |
|---|---|
| `data/workflow/blastn.sh` | pending |
| `data/workflow/blastp.sh` | inspected, support for staged sensitivity search observed; no separate unit extracted because this is a configurable search composition overlapping ordinary search |
| `data/workflow/blastpgp.sh` | inspected; iterative search/profile composition overlaps `iterativepp.sh`, no duplicate unit |
| `data/workflow/cascaded_clustering.sh` | inspected → `mmseqs2_easy_clustering` |
| `data/workflow/clustering.sh` | pending |
| `data/workflow/createindex.sh` | inspected; treated as supporting setup, not a separate scientific analysis |
| `data/workflow/createtaxdb.sh` | pending |
| `data/workflow/databases.sh` | pending; high-value data/access lead |
| `data/workflow/easycluster.sh` | inspected → `mmseqs2_easy_clustering` |
| `data/workflow/easyproteomecluster.sh` | inspected → `mmseqs2_proteome_clustering` |
| `data/workflow/easyproteomesearch.sh` | inspected → `mmseqs2_proteome_clustering` |
| `data/workflow/easyrbh.sh` | inspected → `mmseqs2_reciprocal_best_hits` |
| `data/workflow/easysearch.sh` | inspected → `mmseqs2_easy_search` |
| `data/workflow/easytaxonomy.sh` | inspected → `mmseqs2_easy_taxonomy` |
| `data/workflow/enrich.sh` | inspected → `mmseqs2_profile_enrichment` |
| `data/workflow/iterativepp.sh` | inspected → `mmseqs2_iterative_profile_search` |
| `data/workflow/linclust.sh` | inspected → `mmseqs2_easy_clustering` |
| `data/workflow/linsearch.sh` | pending |
| `data/workflow/map.sh` | inspected; thin wrapper around search, no separate unit because README describes mapping as a high-similarity search mode |
| `data/workflow/multihitdb.sh` | inspected → `mmseqs2_multihit_set_search` |
| `data/workflow/multihitsearch.sh` | inspected → `mmseqs2_multihit_set_search` |
| `data/workflow/nucleotide_clustering.sh` | inspected → `mmseqs2_easy_clustering` |
| `data/workflow/pickconsensusrep.sh` | pending |
| `data/workflow/pickconsensusrepfast.sh` | pending |
| `data/workflow/rbh.sh` | inspected → `mmseqs2_reciprocal_best_hits` |
| `data/workflow/searchslicedtargetprofile.sh` | inspected; chunked profile search/computation strategy, deferred as implementation of profile search pending fuller user-facing semantics |
| `data/workflow/searchtargetprofile.sh` | inspected → `mmseqs2_target_profile_search` |
| `data/workflow/taxonomy.sh` | pending |
| `data/workflow/taxpercontig.sh` | inspected → `mmseqs2_easy_taxonomy` |
| `data/workflow/translated_search.sh` | inspected → `mmseqs2_translated_search` |
| `data/workflow/tsv2exprofiledb.sh` | pending |
| `data/workflow/update_clustering.sh` | inspected → `mmseqs2_update_clustering` |

`README.md` and shell script bodies are source evidence for the included units. GitHub Wiki pages and linked papers were not accessed. No inaccessible-source error was encountered. Deduplication grouped lower-level translated clustering with the general clustering record while explicitly retaining its distinct translation/ORF stages; easy and lower-level presentations of RBH, taxonomy, and search were likewise recorded once per scientific operation.

## Coverage and next work

This is a partial discovery pass. The next useful steps are `databases.sh`, taxonomy database setup, then the remaining workflow entries; inspect taxonomy/profile implementations and the tutorial page sections beyond the three inspected examples. In particular, a multi-hit analysis should not be assigned without resolving the set-to-member mapping built by `multihitdb.sh`. Profile enrichment requires a coherent target profile, target sequence DB and precomputed profile-search result DB.

The pass stopped with about 40 seconds remaining before the assigned cutoff (clock checked at 2026-09-30 14:06:48 UTC) so the partial records and queue could be saved and reconciled. Useful mapped entries remain; this is incomplete coverage. No command failure, access denial, or cutoff expiration caused the stop. No source inspection here establishes successful execution, Harbor compatibility, task suitability, runtime, license eligibility, or deterministic grading.
