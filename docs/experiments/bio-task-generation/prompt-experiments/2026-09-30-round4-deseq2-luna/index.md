# DESeq2: compact inspect-record-check prompt

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The shorter general prompt produces 8 units and 3 data records. Structural
checks pass. The worker retains artificial A/B labels, the Pasilla sample-order
repair, sequencing-type adjustment, raw-count versus transformed-state
requirements, and the distinction between LRT p-values and the displayed LFC.
The final counts and source map agree. Several inspected but unrecorded sections
are explicitly queued rather than claimed complete.

The independent GENCODE transcript-to-gene annotation is still embedded in the
GEUVADIS quantification record. This is the fourth consecutive DESeq2 trial
with that identity error, despite clearer general instructions in this version.
The source map explicitly leaves tximeta unrecorded, so its hidden-chunk handling
is not assessed. The report excludes airway's alternate representation because
it has a different processing stage; the prompt instead asks for grouping by
established observation lineage, which was not checked here.

The source revision is preserved separately from mutable external data-package
manual versions. External Pasilla exon-count preprocessing is correctly not
asserted to prove the copied gene-count matrix's pipeline. Parent review has
not independently verified every external package-manual identifier.

The parent sent a deadline reminder after 15:26 UTC. The worker reports a final
check at 15:28:07 against a 15:24:50.634831 deadline, an overrun of about 3m16s.
This is an assisted partial run, not equal-runtime evidence of improved coverage.
No scientific execution or Harbor validation occurred.
