# mmseqs2: identical-prompt repeat

[Experiment index](../index.md)

- [Run configuration](run.json), [template](template.md), and [resolved prompt](resolved-prompt.md)
- [Launch instructions](launch-message.txt), [source access](source-access.txt), and [review criteria](review-criteria.md)
- Worker [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Structural checks](structural-review.json) and [worker final response](worker-final.txt)

Round 3 repeats the exact round 2 prompt, source pin, model and retrieval setup
in a fresh context with an instructed ten-minute maximum. It produces
11 units and 7 data records. Structural checks pass.

The repeat recovers a dated Swiss-Prot reference and named incomplete reference
products that round 2 omitted. Source access remains partial: two wiki pages
failed in the browser, while the tutorials and pinned workflow scripts were
read. The same prompt therefore produces materially different data coverage.

Both composed metagenomic workflows are classified `mixed` solely because they
combine existing MMseqs2, Plass, or annotation tools. Their stated basis provides
no tool implementation or algorithm change. This directly contradicts the role
rule and regresses from rounds 1 and 2. The taxonomy data record also combines
an NCBI taxonomy dump with an independently sourced UniProt mapping product,
explicitly calling the latter separate in its asset relationship. These need
separate identities. General Swiss-Prot and its dated snapshot are also separate
records; unresolved release relationships should be reconciled without guessing
that every current and historical record is identical.

Useful unit boundaries, reference stage information, and truthful source gaps
remain. Source discovery reportedly stopped 43 seconds before deadline to
reserve final validation, a plausible finalization interval. The roles and data
identity rules are not reliable despite their repeated wording and valid JSON.
