# bedtools trial with a source map and clarified data identity

[Experiment index](../index.md)

Status: completed and reviewed. This trial follows the
[second bedtools run](../2026-09-29-bedtools-luna-02/index.md). The two changes
under review are an explicit source map and a clearer distinction between an
identifiable dataset, an independent reference product, and a schematic input.

A fresh `gpt-6-luna` worker received the frozen general prompt, a ten-minute
budget, and the same operational envelope. It saw no prior outputs, parent
history, or repository-specific review criteria.

- [Run configuration](run.json)
- [Generic prompt snapshot](template.md) and [resolved prompt](resolved-prompt.md)
- [Exact launch message](launch-message.txt)
- [Review criteria](review-criteria.md), unchanged from the second trial
- Original outputs: [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Independent structural checks and output hashes](structural-review.json)
- [Worker's final message](worker-final.txt)

The exact served model version, effective inherited reasoning effort, context,
output and compaction limits, tokens, and cost are unreported by the runner.

## Results

Five units and seven data records passed the independent field, identity, and
reference checks. Unit count did not increase; the differences concern the
structure and supporting evidence.

The source map distinguishes inspected tutorial and reference material from
uninspected code, tests, and bundled data. CpG, exon, GWAS, ChromHMM, and genome
size records are now separate. The `getfasta` unit describes toy inputs without
creating a dataset record. The DNase Jaccard unit links only its DNase dataset,
removing the unrelated reference link seen in trial 02. The stopping explanation
describes a completed partial pass without claiming the deadline was exhausted.

These improvements are observations from one fresh sample. The run still leaves
most command references and external scientific examples uninspected. It also
misses source details available in the tutorial: ChromHMM and DNase download
links, the UCSC provenance of the annotations, and R code examples for
downstream PCA and heatmaps. See the pinned
[setup and data description](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/tutorial/bedtools.md#L39)
and [matrix exploration](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/tutorial/bedtools.md#L612).
Unknown values are preferable to fabricated details, but these omissions reduce
handoff quality.

The DNase workflow is labeled `mixed` solely because it uses shell commands,
a matrix script, and R plotting alongside bedtools. That basis describes
analysis scripting; it does not identify implementation of a new bioinformatics
tool. The prompt names tool-use categories without defining this boundary
explicitly. Source revisions also remain unresolved.

## Next decision

Keep this prompt fixed for a [Scanpy transfer trial](../2026-09-29-scanpy-luna-01/index.md).
Check whether useful unit boundaries, input state, data identities, and tool-role
classification transfer to APIs and notebooks. Do not interpret the bedtools
improvements as proof that the source-map instruction caused them. Role
definitions and source-reading completeness remain questions for the next review.
