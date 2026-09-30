# DESeq2: uncapped Luna at high effort

[Experiment index](../index.md)

- [Configuration](run.json), [template](template.md), [resolved prompt](resolved-prompt.md)
- [Launch](launch-message.txt), [source access](source-access.txt), [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [data](outputs/datasets.jsonl), [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json), [worker final](worker-final.txt)
- [Runner metrics](runner-metrics.json), [compressed execution events](runner-events.jsonl.gz)

The fresh CLI run requested `gpt-6-luna` / `high` with the same prompt and DESeq2
pin as Luna/low. It produced 23 tool-use units and 5 data/reference records in
1,453.79 seconds (24.23 minutes). JSONL structure and references pass. Peak child
RSS was 314,580 KiB. No scientific computation or Harbor validation occurred.

Compared with the single low-effort run, this inventory covers more input,
modeling and diagnostic operations, separates the GENCODE transcript-to-gene
reference from observations, preserves artificial condition labels, and labels
the unmix example as synthetic. It records the seven-row embedded Pasilla sample
table without treating a differently versioned external package's six GEO
accessions as a same-version contradiction. These are observed differences in
one pair of runs, not a general effort ranking.

Completion remains incomplete. The worker explicitly stops without a time,
resource or access blocker while tximeta, supplementary study/benchmark scripts
and other useful sources remain pending. Its claim that these remaining sources
do not introduce a distinct unrecorded workflow is unsupported: they were not
read. Earlier pinned-source reviews identified study analyses and the iCOBRA
benchmark in those scripts. The final count and source map are coherent, but
honest pending statuses do not make the requested exploration complete.

The tximport data evidence needs tighter version scoping. The unit uses archived
Bioconductor 3.11 documentation and acknowledges a dependency-version gap, yet
its input/data prose still says six samples are used by the DESeq2 example.
The parent separately verified a two-sample tximportData 1.41.1 revision. The six
archived samples are a valid source product; their applicability to an unpinned
current dependency is not established. This is a source-version qualification,
not a claim that the archived six-sample table is wrong.

Several archived-HTML evidence locators use browser display lines, including
line zero, rather than verified source-file lines. The prompt explicitly
forbids substituting browser display lines for source locators. Headings and
URLs still locate the material, but the numeric locators need repair.

The CLI reports 3,075,341 input tokens (2,917,120 cached), 53,275 output tokens
and 23,479 reasoning-output tokens. These are aggregate reported fields, not
billed cost. Elapsed time was about 5.8 times the single low-effort run; counts
and token totals do not establish quality or cost efficiency. Served model
settings were not independently verified. Parent source checks sampled the
shared import/data-identity/completion probes; they did not audit every field.

Parent sources: [pinned main vignette](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/vignettes/DESeq2.Rmd),
[archived tximportData vignette](https://mirror.dotsrc.org/bioconductor-releases/3.11/data/experiment/vignettes/tximportData/inst/doc/tximportData.html),
[two-sample newer metadata](https://github.com/bioc/tximportData/blob/59766ff72a3c545424a085049458f15a0df4fd2c/inst/extdata/samples.txt),
[study analyses](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/inst/script/testsuite.Rmd),
and [simulation benchmark](https://github.com/thelovelab/DESeq2/blob/c62c60c6ff83fd84ce115cacd1c49827533f85a7/inst/script/icobra_benchmarks.R).
