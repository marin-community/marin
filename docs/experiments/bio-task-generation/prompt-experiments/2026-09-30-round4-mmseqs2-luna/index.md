# MMseqs2: compact inspect-record-check prompt

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The shorter general prompt produces 7 units and 6 data records. Structural
checks pass. All seven units are appropriately labeled `tool_use`, including
the Plass assembly and downstream annotation workflows previously labeled
`mixed`. The gut workflow is split into assembly, clustering, annotation and
abundance units with upstream stages preserved. Pfam and both dated and current
Swiss-Prot products are retained despite missing release or terms metadata.

The patient study, yeast contig and gut run are separate from the reference
products. No wrong paper DOI is asserted. The seven-unit inventory focuses on
the wiki tutorials; broad standalone API/command coverage remains pending. An
SRA documentation landing page is included as an asset even though it identifies
no particular archived data object. That URL is a discovery lead, not a usable
asset locator. Parent review does not establish completeness of independent
taxonomy dependencies omitted from this pass.

Final counts agree, but the continuation section says abundance is both a
candidate for an additional unit and already recorded. The worker initially
recorded a 15:24 checkpoint as its stopping evidence while still active. After
a parent reminder, it records 15:30:42 against a 15:29:48 deadline, about 54s
late. These are residual source-map and stopping-state defects. No biological
data or scientific workflow was executed.
