# Scanpy trial with source fallbacks and data reconciliation

[Experiment index](../index.md)

Status: completed and reviewed. This trial follows the
[first Scanpy run](../2026-09-29-scanpy-luna-01/index.md). The prompt now explicitly
reconciles dataset identities across sources, tries source files for important
blocked or incomplete rendered pages, and defines the tool-role categories.

A fresh `gpt-6-luna` worker received one resolved prompt, the same ten-minute
budget and operational envelope, and no parent history or reviewer hints.

- [Run configuration](run.json)
- [Generic prompt](template.md) and [resolved prompt](resolved-prompt.md)
- [Exact launch message](launch-message.txt)
- [Review criteria](review-criteria.md), carried over from the first Scanpy trial
- Original outputs: [units](outputs/units.jsonl), [datasets](outputs/datasets.jsonl), and [inspection](outputs/inspection.md)
- [Independent structural checks and output hashes](structural-review.json)
- [Worker's final message](worker-final.txt)

The carried-over criteria describe the first run as a transfer probe; the
comparison here is between Scanpy trials 01 and 02. The substantive reviewer
probes are unchanged. Effective model settings and usage remain unreported.

## Results and interpretation

Three tutorial units and three data/reference records pass the independent
structural checks. The worker combines shared PBMC3k observations used by the
legacy and Pearson-residual tutorials into one record and keeps transformed
representations as assets. It separates PBMC10k observations and the tutorial's
marker reference. All units are classified as tool use.

This is narrower coverage than the first Scanpy run: integration, trajectory,
and standalone API operations remain leads. The worker stops after one source
map sweep and three tutorial contexts, with time still remaining before its
deadline. The stopping note refers to the investigation window as insufficient,
but the captured completion time alone does not establish budget exhaustion.

The explicit source-fallback instruction did not recover the complete current
clustering notebook. A unit is accepted from indexed excerpts with its dataset
identity unresolved, and repository revision recovery still fails. Raw browser
line numbers reappear despite the source-locator instruction. These are
continuing evidence and access limitations.

The exact earlier `pbmc3k_processed` loader is now an uninspected lead, so this
sample does not demonstrate that its prior duplicate record was directly fixed.
It does demonstrate consolidation of PBMC3k observations across two other
tutorial contexts. No overall improvement or causal effect is established by
one sample per prompt.

## Next change

The next prompt makes the stopping condition explicit: spend remaining budget
on distinct useful leads, ending when the mapped scope is inspected, remaining
leads repeat recorded work, or a concrete limit prevents progress. The
[fourth bedtools trial](../2026-09-29-bedtools-luna-04/index.md) checks this together
with the intervening source, role, and data-identity changes. This comparison
does not isolate the stopping instruction alone.
