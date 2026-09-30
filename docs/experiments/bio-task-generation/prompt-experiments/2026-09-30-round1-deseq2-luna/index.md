# deseq2: first cross-repository round

[Experiment index](../index.md)

Eight units preserve example-specific designs and artificial labels, but reference identity and early stopping still need repair.

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [frozen review criteria](review-criteria.md)
- Original [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

The unchanged prompt from `37f8a601d2` ran in a fresh Luna context with a ten-minute
maximum. JSON structure and inventory references pass. Counts are 8 units
and 4 data records; counts alone do not measure quality.

The worker found the R Markdown vignette, public manuals, bundled-data metadata,
and analysis/benchmark scripts without Bioconductor-specific hints. Its Pasilla
records preserve sample-order reconciliation and sequencing-type adjustment.
The tximport record correctly warns that the demonstrated A/B labels are
artificial, and the simulation record distinguishes generated counts from the
empirical parameter source. This run therefore does not show that a special
Bioconductor discovery prompt is necessary.

The `tximportdata-geuvadis-demo` record groups expression quantifications and the
GENCODE transcript-to-gene annotation in one dataset. Its own asset description
calls the annotation independent. This violates the existing data-identity rule;
JSON reference validity does not catch it. The record also mixes a search-result
claim about six packaged samples with a claim of a two-sample demonstration.
The precise package release and sample table were not inspected, so the number
of samples used by this source remains unverified.

The source map is useful but still has many accessible leads. Its stopping
reason explicitly acknowledges remaining time and calls the stop a bounded
handoff. Completion was observed more than two minutes before the deadline.
That does not satisfy the prompt's stopping condition. The latest self-check
instructions did not reliably enforce either data identity or continuation.

The parent verified the sample-order and artificial-label statements against
[the pinned vignette](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/vignettes/DESeq2.Rmd).
The parent did not execute R or verify the full external data packages. Tool-role
classification of the custom simulation generator remains a boundary judgment,
not an independently established error.
