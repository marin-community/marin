# bedtools trial with an inspection queue

[Experiment index](../index.md)

The worker returned 17 units and two overgrouped data records. It continued
inspection close to the deadline and found additional operations, including
`map`. Data identity and tool-role classification regressed. This run adds an
entry-level inspection queue and a final
budget check to the general find-units prompt. For a bounded source collection,
the worker must enumerate members with locations and mark each as pending,
inspected with unit IDs, or skipped with a reason. This tests whether making
remaining work concrete helps avoid stopping after representative examples.

The direct comparison is [trial 05](../2026-09-29-bedtools-luna-05/index.md).
The source revision, source-access instructions, reviewer probes, model request,
fresh context, and ten-minute budget are unchanged. No bedtools-specific
guidance was added to the prompt or launch instructions.

- [Run configuration](run.json)
- [Generic prompt](template.md) and [resolved prompt](resolved-prompt.md)
- [Exact launch message](launch-message.txt)
- [Source-access instructions](source-access.txt)
- [Review criteria](review-criteria.md)
- Raw [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and
  [inspection report](outputs/inspection.md)
- [Structural checks, queue audit, and output hashes](structural-review.json)
- [Worker final response](worker-final.txt)

Both JSONL files parse, contain the required top-level fields, have unique
identifiers, and have resolving unit/data references. The queue enumerates 41
manuals, matching an independent read of the
[pinned source tree](https://github.com/arq5x/bedtools2/tree/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/docs/content/tools).
It marks 17 inspected and 24 pending. The worker's prose and final response
incorrectly report 40 total and 23 pending; the raw output is unchanged.
These are inspection claims, not independent verification of every tool read.

The inventory now includes map, merge, closest, relative distance, subtract,
complement, flank, annotate, multiinter, window, and shuffle, alongside the
previous overlap/statistics/coverage themes. The map record's aggregation
semantics, sorted inputs, and missing-value behavior agree with its
[manual](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/docs/content/tools/map.rst).
The worker also follows a documentation stub to a tutorial answer key for a
windowed exon-count workflow. This is a source-backed candidate; no commands
were executed. Unlike trial 05's early representative sample, this pass ends
near the supplied deadline with a concrete continuation queue.

The broader inventory has material semantic failures:

- One data record combines the DNase study, CpG islands, RefSeq exons, GWAS
  variants, ChromHMM states, and genome sizes. Another groups independent
  annotations by their shared `data/` directory. Both violate the prompt's
  identity rule, even though their prose acknowledges distinct sources. Units
  such as nearest-exon analysis inherit links to the whole tutorial bundle.
- The regulatory tutorial and windowed exon-count workflow are classified as
  `mixed` because they compose shell/R analysis and existing tools. Their own
  explanations describe tool use under the prompt's explicit definitions.
- The tutorial is again one broad record. It mentions PCA/heatmap in evidence
  and dependencies but does not preserve the explicit DNase matrix workflow
  boundary from trial 05. Greater operation coverage did not preserve every
  useful workflow record.

The queue is a useful mechanism to retain for further testing. This sample
does not establish overall prompt quality: data identity and role errors are
disqualifying for an inventory handed directly to authors. It also has not
been transferred to Scanpy. Effective model settings and full tool traces remain
unreported, and one run cannot attribute every difference to the queue change.

The next comparison should hold the discovery prompt fixed and test a separate
reconciliation pass over its output, with the same source evidence. Review
whether that pass separates data products, repairs role labels and links, and
preserves distinct workflow boundaries without losing the discovered operations.
This follow-up has not run; no new prompt revision was made after this review.
