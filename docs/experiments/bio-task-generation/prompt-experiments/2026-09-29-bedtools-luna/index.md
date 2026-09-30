# bedtools unit-discovery trial

[Experiment index](../index.md)

A fresh Luna worker produced three units and one data record from the general
find-units prompt. The output passed structural checks, but coverage, data
identity, and source references need improvement. This single exploratory run
does not establish whether the prompt generalizes across repositories.

## Question and protocol

What source inventory does the general prompt produce for bedtools without the
parent conversation or repository-specific hints?

The worker received the prompt at Marin commit
`72008dd68247318a367a840a4f41e27fb15ff7e1`, with `{{REPO}}` and `{{REPO_URL}}`
resolved to `arq5x/bedtools2` and its repository URL. It had a ten-minute budget
for lightweight source inspection. The operational instructions allowed web and
GitHub reads, small source reads, bounded previews, and JSON validation; they
excluded clones, builds, installation, biological dataset downloads, Harbor
runs, cloud jobs, further model calls, and subagents.

| Setting | Recorded value |
| --- | --- |
| Requested model | `gpt-6-luna`; exact served version unreported |
| History | Fresh context, no parent conversation or planning documents |
| Reasoning effort | Inherited; effective value unreported |
| Context, output, and compaction limits | Not exposed by the launch tool |
| Concurrency and attempts | One child, one attempt, zero retries |
| Preparation observed | 2026-09-29 17:11:54 America/New_York |
| Completion observed | 2026-09-29 17:15:45 America/New_York |
| Deadline | 2026-09-29 17:21:54 America/New_York |
| Tokens and cost | Not recorded |

The observed interval is a wall-clock upper bound of less than four minutes;
exact worker start and finish times were not captured. Structural checks were
specified by the prompt. Scientific findings below came from review after the
run; there were no predeclared scientific acceptance thresholds or comparator.

The worker used a mutable `master` view. A separate parent check resolved HEAD
to `614e9a5c5935ab86e873dab9072fbbaf003c1b7e`; that revision was used for review
and was not supplied to the worker. It does not pin all the worker's retrievals.

## Evidence

- [Run configuration, limitations, and artifact hashes](run.json)
- [Exact resolved prompt](resolved-prompt.md)
- Original outputs: [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection note](outputs/inspection.md)
- [Structural check results](structural-review.json)

The original outputs are preserved byte for byte, including the errors discussed
below. The full agent/tool trace and exact launch-message text were not exported;
the run record preserves the available configuration and operational summary.
This report and its normalized run record were assembled after the experiment.

## Results

All three units cite the same tutorial: CpG–exon intersection, multi-track
annotation, and exon merge/complement. Both JSONL files parsed, contained the
requested top-level fields, had unique identifiers, and had resolving
dataset/unit references. These checks establish structural validity.

The scientific review found:

1. Coverage omitted a connected DNase similarity analysis later in the tutorial:
   pairwise Jaccard comparisons, a 20-by-20 matrix, and clustering. The tutorial
   also has an exon-coverage example; the worker said it had not found or
   inspected a concrete `genomecov` workflow.
   See the pinned [coverage section](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/tutorial/bedtools.md#L440)
   and [DNase similarity section](https://github.com/arq5x/bedtools2/blob/614e9a5c5935ab86e873dab9072fbbaf003c1b7e/tutorial/bedtools.md#L525).
2. The single data record combines DNase study inputs, annotation tracks, and
   unrelated repository fixtures because they share a tutorial or packaging.
   This conflicts with the prompt's instruction to preserve distinct biological
   observations and supporting reference identities. It also gives units data
   links that include assets their cited commands do not use.
3. Source references retain mutable `master`, and approximate line numbers do
   not match raw-file coordinates. For example, CpG intersection appears around
   lines 103–206, not the reported 275–368. Headings would have provided better
   locators where file coordinates were unavailable.
4. The inspection note claims the deadline or budget was exhausted. Completion
   was observed more than six minutes before the deadline, so the saved timing
   does not support that stopping explanation.

The basic operation semantics are grounded in the tutorial. The worker marked
uncertainties about data assembly, access, terms, and execution, and qualified
the scientific interpretation of exon complements. No task was authored or
validated in Harbor.

## Interpretation and next decision

The result exposed a boundary ambiguity: the worker excluded some documented
operations because it lacked a complete biological workflow and ready data.
The revised prompt permits operations as units, follows relevant source links,
and records relationships between focused operations and larger analyses. It
also clarifies data identity, coverage reporting, and source references.

These revisions are hypotheses. The data-grouping instruction was already
present, so its violation may reflect instruction following rather than missing
guidance. Retrieval and stopping behavior could also depend on the worker or
runner. One trial cannot distinguish these causes.

For bedtools, manual review should look for documented operations such as `map`
and `intersect`, plus scientific uses linked from its usage-example collection.
Those expectations were added after this run and were not worker inputs. Keep
such expectations in the testbed while evaluating the same general prompt on
other repository types.

At the initial review, the candidate prompt had not been run. The subsequent
[second bedtools trial](../2026-09-29-bedtools-luna-02/index.md) records that
comparison. Transfer to other repository types remains a separate question.
