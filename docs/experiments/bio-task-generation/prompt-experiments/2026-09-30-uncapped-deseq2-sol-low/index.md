# DESeq2: uncapped Sol at low effort

[Experiment index](../index.md)

- [Configuration](run.json), [template](template.md), [resolved prompt](resolved-prompt.md)
- [Launch](launch-message.txt), [source access](source-access.txt), [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [data](outputs/datasets.jsonl), [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json), [source-map check](source-map-review.json), [worker final](worker-final.txt)
- [Runner metrics](runner-metrics.json), [compressed execution events](runner-events.jsonl.gz)

The fresh CLI run requested `gpt-6.1-sol` / `low` and produced 90 units and 19
data records in 1,827.92 seconds (30.47 minutes). There are 86 tool-use, three
mixed implementation/use, and one tool-creation units. JSONL structure and
references pass. Every final unit appears in the inspection map, and its
backtick-delimited unit IDs resolve. Peak child RSS was 315,104 KiB. No
scientific execution, biological payload download or Harbor validation occurred.

Coverage extends across exported APIs, the main vignette, analysis/benchmark
scripts, tests, a symbolic derivation and three linked scientific tutorials.
The event stream records the corresponding source reads. The final map gives
specific dispositions instead of leaving the core operation leads pending.
Remaining work concerns unavailable data/provenance, downstream validation and
explicitly excluded duplicate or supporting material. The parent found no
equivalent of the obvious premature stop in either Luna run. This is a sampled
coverage assessment, not proof of exhaustive discovery across all linked sources.

The shared probes retain useful scientific context: fabricated tximport condition
labels, separate reference products, hidden `skipMeta=TRUE` setup, sample ordering,
sequencing-type design, and raw versus transformed input stages. The six
GEUVADIS samples are supported by the cited release data vignette; the record
states that its dependency version is not pinned to DESeq2. That does not
contradict the newer two-sample revision inspected by the blocked Sol/high run.

A post-output source check confirms the RUV prerequisite: the linked workflow
proposes treatment-only results, while earlier code fits `~ cell + dex` and
later shrinks those results. The proposed treatment-only refit and final RUV
refit are not shown. The unit preserves those gaps. The worker also identifies
a same-revision discrepancy between `results.Rd` describing thresholded LRT
replacement and `R/results.R` rejecting that path unless Wald is selected.

Remaining handoff problems include copied metadata. The three R-test method
units and the Mathematica derivation all say `source_kind: R Markdown section`.
The derivation also inherits a generic DESeq2 1.53.5 runtime dependency despite
its historical symbolic source. Correct source links do not repair those fields.
The `deseq2-simulations` data record groups independent generated realizations
as one generator family, including a separately drawn Poisson vector. It labels
that distinction, but a family is coarser than the requested observation/product
identity; the author still needs an exact fixture or generation recipe per use.

The custom-filter record mentions its quantile scan and BH adjustment without
distinguishing two adjustment stages. The cutoff search calls `p.adjust` with
its default Holm method; the final assignment receives the method argument
(default BH). This description is incomplete, rather than the explicit false
BH-search claim found in the earlier Sol/high pilot. These post-output probes
were not supplied to the worker. Other claims in the 90 records were not all
independently audited.

The CLI reports 5,500,840 input tokens (5,271,680 cached), 32,134 output tokens
and 2,951 reasoning-output tokens. These are reported usage fields, not billed
cost or independently verified served settings. An invalid file-edit patch was
recovered within this sole model attempt. The worker's approximately 29-minute
estimate covers its measured working interval; the runner records launch to exit.
One run per setting on one repository cannot establish a general model ranking.

Parent sources: [pinned main vignette](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/vignettes/DESeq2.Rmd),
[linked workflow](https://github.com/thelovelab/rnaseqGene/blob/0d7e27dde3ca9875cf94770ed9de35e346dd3676/vignettes/rnaseqGene.Rmd),
[threshold API](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/man/results.Rd),
[threshold implementation](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/R/results.R),
[custom filter](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/tests/testthat/test_results.R),
and [R adjustment defaults](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
