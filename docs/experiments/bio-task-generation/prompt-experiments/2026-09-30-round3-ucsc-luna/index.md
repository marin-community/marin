# ucsc: identical-prompt repeat

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

Round 3 repeats the exact round 2 prompt, source pin, model and retrieval setup
in a fresh context with an instructed ten-minute maximum. It produces
7 units and 2 data records. Structural checks pass.

Seven units include current liftOver help, interval signal aggregation,
monoploid VCF filtering, sequence extraction, annotation conversion, and browser
tutorials. Parent source checks confirm the monoploid-only limitation in
`vcfFilter.c` and the genePredExt limitation in `genePredToBed.c`; these are useful
prerequisites that earlier records did not always preserve. A parent's initially
guessed genePred path returned 404; the worker's cited path was correct.

The SARS-CoV-2 fixture record now absorbs the independent positional exclusion
annotation, reversing round 1's correct separation. The record says source
identity is unresolved, which is a reason to retain separate identities rather
than group by test packaging. The source map leaves `genePredToBed` pending even
though a final unit and evidence bullet describe its inspected implementation.
The final-state consistency problem therefore recurs under the identical prompt.

No named study is required for a documented operation or a named reference
candidate. The worker appropriately retains hg38 with a mutable URL and unknown
release. GUI tutorials remain allowed because no binaries-only focus was passed.
The worker reports stopping after the supplied deadline; exact overrun is not
available from its final response. No commands were scientifically executed.

Markdown trailing spaces were normalized for required lint. The [exact original](original-inspection.json) retains the worker text and SHA-256; no semantic edits were made.
