# MMseqs2 source inventory inspection

Repository: [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2), pinned source revision `564f40d8857f4eca4e1dfe100c67c155b1933e70`. Public GitHub repository tree retrieved with `truncated=false`; output was filtered to README, `data/workflow/`, `examples/`, and `docs/` paths. README points users to GitHub Wiki tutorials. No repository clone/build, biological asset download, MMseqs2 execution, Harbor run, or model call was performed. Wiki material was fetched as GitHub HTML on 2026-09-30; its current wiki revision was not pinned, so findings using it have a revision gap. GitHub raw-file reads used pinned commit URLs.

## Inventory organization and relationships

Seven units currently cover: patient CSF read taxonomy and taxon filtering; a separate yeast contig taxonomy example; the patient-read protein follow-up; direct protein catalogue assembly from a human-gut SRA run; redundancy reduction on that catalogue; Pfam profile annotation of its cluster representatives, and abundance estimation by mapping translated read ORFs back to the protein catalogue. The four gut units are linked in assembly → clustering → profile-search order and share `mmseqs2-ds-err1384114` with distinct raw/derived processing stages. Patient taxonomy and its protein follow-up share the human-depleted CSF read dataset; the 2018-03 Swiss-Prot snapshot is kept separate from the live/current Swiss-Prot workflow reference. The yeast contig and two Swiss-Prot reference identities are separate. Pfam current release remains its own profile reference. The optional yeast proteins file is explicitly excluded from inputs because the tutorial says it is not used.

Tutorial-reported numerical outputs (for example the yeast-contig voting summary) are not execution results from this inspection. No suitability, scientific validity, execution, runtime, grading, access-rights, or Harbor compatibility validation is established.

## Entry-level source map and continuation queue

Statuses refer to whether source material was inspected for this inventory, not whether examples were executed.

| Collection / location | Status and coverage |
|---|---|
| Repository `README.md` at pinned commit | Partly inspected: description, publications, documentation links, installation/setup overview, search and taxonomy overview, supported modes, memory guidance. Remaining prose includes additional uninspected operational detail. |
| `data/workflow/` pinned tree listing | Enumerated. Inspected selected workflows: `easysearch.sh` (query/target DB preparation, search, conversion and optional taxonomy report); `easycluster.sh` (input DB, cluster, TSV, representative/all-sequence FASTA); `easytaxonomy.sh` (workflow path); `taxonomy.sh` (search plus LCA/result branch); `taxpercontig.sh` (ORF extraction/search-result/output stages); `clustering.sh` (redundancy prefilter and cluster merge); `linclust.sh` (k-mer matching, alignment/clustering, refinement); `createtaxdb.sh` (taxonomy dump and mapping setup); `createindex.sh` (translated/nucleotide index preparation); `translated_search.sh` (optional nucleotide ORF extraction, search, alignment offset restoration); `databases.sh` (selected Swiss-Prot branch and portions of download/setup). Remaining scripts, including reciprocal-best-hit, profile-search, enrichment, cascaded/update clustering, nucleotide clustering, sliced search, and other database variants, are pending. |
| Repository `examples/` | Tree shows large `DB.fasta` (11,434,968 bytes) and `QUERY.fasta` (304,764 bytes); skipped without preview because they are sizable example sequence assets and tutorials supplied more context-rich cases. No relationship to tutorial assets established. |
| GitHub Wiki `Tutorials` page | Partly inspected (retrieved 2026-09-30, current wiki page, revision not pinned): intro/setup; all patient pathogen taxonomy/visualization/filtering/protein follow-up sections; all contig taxonomy sections; human-gut workflow through assembly, clustering, indexed DB, Pfam annotation/filtering, custom clustering workflow and abundance analysis. The page has 703 rendered lines; no claim that all prose or code beyond these sections was inspected. |
| Wiki `Writing large scale sequence analysis workflows` remaining material | Inspected through abundance analysis and re-create-linclust text (sections around lines 386–626 of GitHub-rendered page). Uninspected remaining wiki material: conclusion/references only; these are not current operation leads. |
| External `soedinglab/metaeuk-regression` | Only inspected through the immutable raw URL and commit named by tutorial; asset content was not fetched. The tutorial says the adjacent protein file is not used. |
| External EBI Pfam, UniProt, ENA/SRA assets | URLs/identifiers were read in tutorial/repository workflow. Payloads and current records/releases were not fetched; asset metadata/terms pending. |

## Inspected leads and decisions

- Taxonomic search of patient CSF reads is a composed analysis using MMseqs2 commands. The tutorial explicitly reports removal of human reads with Kraken and a smaller read set; the exact SRA accession and downloadable-file lineage are unresolved, so the dataset record does not invent an accession.
- The yeast contig's optional 19-protein sequence fixture is not an input to the assignment workflow per the tutorial. Its source FASTA was not downloaded. The MMseqs2 tutorial output is reported as 32 filtered fragments, 32 labeled, 30 in agreement with assigned species/descendant, and 0.890 support; this is a tutorial claim, not reproduced evidence.
- `ERR1384114` is a distinct SRA run named by the tutorial. The paired ENA FASTQ assets and protein-assembly, MEGAHIT/Prodigal comparison, cluster, and Pfam stages are represented as processing lineage for this run, not independent observations.
- Plass, MEGAHIT, Prodigal, and optional HMMER use is classified as applying existing tools (`tool_use`). Tutorial workflow construction from MMseqs2 modules is also use, not implementation of the underlying scientific method.
- MMseqs2 tutorial and repository workflow evidence was cross-checked for easycluster and taxonomy/contig flow; no command or scientific claim was validated by execution.
- First attempt to open raw GitHub wiki markdown (`raw.githubusercontent.com/wiki/soedinglab/MMseqs2/Tutorials.md`) failed with a web-tool cache-miss message. GitHub rendered Wiki page was available and used instead; no attempt to infer inaccessible raw markdown details.
- A general web search for exact Piantadosi SRA identity did not resolve the accession; the SRA search page itself was not checked. Keep accession unresolved rather than guessing.

## Remaining useful continuation locations

- Wiki `Tutorials`, human-gut `Abundance analysis` is mapped and read and recorded as a separate unit; the source tutorial asks the author to assemble its ordered steps, and no workflow was executed. `Re-create the linclust workflow` remains a mapped lead but was not made a separate unit because it asks for re-implementation of the already inventoried clustering workflow; inspect if exploring algorithm reconstruction tasks.
- Review other distinct-use wiki tutorials and pinned `data/workflow/` scripts, especially reciprocal best hits, translated search, profile search, and enrichment, before treating source coverage as broad.
- Inspect external ERR1384114 metadata, the NC_001133.9 segment source file, and reference release/term pages only through small metadata reads if needed; do not download biological data for this inventory.

## Final consistency and stopping record

Final JSONL inventory: 7 units and 6 dataset/reference records. Data references resolve; unit IDs are unique and all seven unit role values are `tool_use`. Datasets remain separate when identity differs (CSF vs Swiss-Prot, dated vs current Swiss-Prot, yeast contig vs gut run, Pfam), while the gut read-to-derived-protein lineage is grouped as linked stages. Final clock check: 2026-09-30 15:30:42 UTC. The supplied inspection deadline (15:29:48 UTC) had passed; stopped further source inspection on the parent agent’s deadline reminder. This is a scoped inventory checkpoint, not a claim that all MMseqs2 operations were mapped. Remaining useful leads are listed above. Standard-library JSON parsing, unique-ID checks, required-field checks and all unit/data/unit-reference checks passed. The final inspection map was reread against both JSONL files.
