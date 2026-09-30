# mmseqs2: first cross-repository round

[Experiment index](../index.md)

Twelve units include a scientific pathogen workflow and separate reference data; an incorrect DOI and stale inspection prose remain.

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [frozen review criteria](review-criteria.md)
- Original [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The unchanged prompt from `37f8a601d2` ran in a fresh Luna context with a ten-minute
maximum. JSON structure and inventory references pass. Counts are 12 units
and 5 data records; counts alone do not measure quality.

The worker discovered the linked tutorials and preserved a composed pathogen
investigation as a tool-use unit, with taxonomy, read selection, protein assembly,
clustering, and annotation stages. It also found the gut-metagenome accession
ERR1384114 and linked the downstream clustering use. Bundled sequence fixtures,
clinical reads, gut reads, and the historical Swiss-Prot reference have distinct
records. Query and target fixture relationships are explicitly unknown.

The source map records the tutorial as inspected, but a later paragraph says
GitHub Wiki pages were not accessed. These contradictory statements survived
the final consistency check. The contig-taxonomy example is absorbed into a
broad taxonomy operation, with its concrete contig source still absent; the
report should clearly retain this as an uninspected data lead.

The pathogen dataset adds DOI `10.1093/cid/ciy029`, although it says the cited
paper was not inspected. The [paper itself](https://pmc.ncbi.nlm.nih.gov/articles/PMC5850433/)
identifies DOI `10.1093/cid/cix792`. This is an unsupported identifier, not a
source-access limitation. Do not infer verified provenance from a plausible URL.
The DOI check was added after inspecting the output.

The worker stopped about 40 seconds before its cutoff to save and reconcile
records, and accurately recorded this. No biological data or workflows were
executed. The parent inspected the pinned README and command registrations plus
the mutable tutorial page; this was a targeted evidence audit, not exhaustive
verification of every shell-workflow description.
