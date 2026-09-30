# Scanpy: identical-prompt repeat

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The unchanged round 2 prompt produces 7 units and 7 data records. Structural
checks pass. The clustering unit now correctly retains `adata.layers["counts"]`,
which the previous trial incorrectly said was absent. PBMC3k raw/processed data
remain together while PBMC10k and PBMC68k have separate identities. Moran's I
and approximate-neighbor comparison add useful operations.

The cell-cycle unit names the independently sourced Tirosh gene set but has no
separate reference record for it. That reference was recorded in earlier runs;
its omission is a coverage regression, not evidence that it belongs to the
Nestorowa expression data. The integration record also leaves the PBMC query
provenance unresolved. These are incomplete handoffs despite coherent final
counts and source-map statuses.

The worker exceeded the supplied 15:19:22 UTC deadline. The parent sent a stop
reminder at approximately 15:21:35; the worker reports checking the clock at
15:22:15 before its final integrity check. This assisted run therefore exceeded
the limit by at least 2m53s. Its coverage cannot be compared as equal runtime.
No scientific execution or Harbor validation occurred.
