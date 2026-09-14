# NeMo Gym and TaskTrove coverage audit (historical `.8`)

**Superseded for current TaskTrove work by the Clean `.9` inventory and
reviews:** [tasktrove-release-09-inventory-2026-09-14.md](tasktrove-release-09-inventory-2026-09-14.md),
[clean09-review-a.md](clean09-review-a.md), and
[clean09-review-b.md](clean09-review-b.md).**

Effort: medium. Date: 2026-09-13. This bounded static audit uses two
fresh-context GPT-5.6 Terra investigations. The coordinating agent synthesized
their findings without inspecting individual source rows. No new environment,
verifier, live model, or full dataset import was executed.

TaskTrove already provides substantial NeMo-family coverage, but the current
TaskCompendium imports cover only part of it. The first implementation wave
should add four diagnostic cases: instruction constraints, code-as-answer,
non-executing predicted actions, and seeded domain tools. Reuse the existing
answer, executable-code, and R2E controls. Do not add a second repository-repair
integration just to demonstrate an already-tested execution pattern.

## Scope and evidence

The requested clean release is
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8`.
Its manifest and ledger were inaccessible during this audit. Clean-release row
counts and complete membership remain unknown. Producer keep/drop decisions,
raw source-directory presence, and recorded local sample provenance are separate
facts. The local POC labels its samples as `.8` and records fixture hashes, but
the archives themselves do not independently attest clean-release membership.
The 44-instance TaskCompendium sample and current HF Viewer counts must not be
substituted for release counts.

The comparison pins [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym/tree/1e668906d2e69a9e8ee9aaafc60050a4025d9688)
and the [.8 producer](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/pipeline.py).
The raw TaskTrove revision is `02923004846e4e73862c20962f823a6d05100e7a`.
Upstream HF revisions in the catalog describe current public datasets; they do
not prove row parity with historical TaskTrove conversions.

The [upstream catalog](nemo-catalog.md) and
[TaskTrove inventory/fidelity notes](tasktrove-inventory.md) contain the primary
source links and bounded sample evidence. [coverage.json](coverage.json) joins
14 families with explicit overlap levels, policy evidence, support status, and
uncertainty. [suite.json](suite.json) is a proposed case manifest, not a runnable
or validated suite.

## Coverage matrix

“Keep” and “drop” below refer to the pinned producer policy. They do not claim
observed row counts in the inaccessible clean release. Generic Codeforces and
R2E coverage demonstrate execution patterns; they are not NeMo dataset joins.

| NeMo family | TaskTrove evidence | TaskCompendium boundary | Next action |
| --- | --- | --- | --- |
| MCQA | Keep; exact local sample join | Answer importer exists | Reuse control |
| Math with judge | Keep for named math families | These NeMo converters are not imported | Bounded importer follow-up |
| Instruction following | Keep IFEval/structured families | Importer missing | First-wave constraint case |
| Competitive code | Keep direct NeMo competitive source | Generic Codeforces/pytest supported; direct NeMo source missing | First-wave code-answer case |
| Function-call pivot | Drop frozen next-call task | Final-action submission missing | First-wave predicted-action case |
| Workplace | Raw source exists; clean ok membership unknown | Stateful domain tools missing | First-wave state case |
| SWE pivot | Drop; predicted action has no repo | Same action-submission gap as function pivot | Avoid duplicate first-wave work |
| SWE-Gym/SWE-bench | Direct join unproven; other SWE sources differ | R2E already exercises repo/process/test pattern | Reuse R2E control |
| Google Search | Drop; promised search tool absent | Live service adapter missing | Later service case |
| Reasoning Gym | Keep with library scorer | Importer/lowering unsupported | Generator-provenance follow-up |
| Structured Outputs | Older text-output samples confirmed | JSON/XML supported with known verifier defects; current NeMo v4 differs | Supplement final-action case |
| Calendar v2 | Keep static schedule checker | Importer missing; no reactive runtime needed | Static-history follow-up |
| Indian Banking | No proven source join | Reactive user simulator unsupported | Keep deferred |
| Indirect injection | Drop weak negative-only checker | Seeded tools plus task-completion checking needed | Follow Workplace |

## Corrections to the initial proposal

**Predicted calls do not execute tools.** The
[function-call pivot](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/single_step_tool_use_with_argument_comparison/app.py)
scores a proposed next action. Its batch mode compares an unordered collection;
it does not demonstrate concurrent tool execution. The source's default
extra-call leniency also needs an explicit case. A frozen next-action problem
can be a valid task even though TaskTrove's agentic-task policy dropped it.
Label it as prediction, and retain the original scoring limitations.

**Calendar v2 is static during evaluation.** The
[calendar server](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/calendar/app.py)
consumes materialized conversation history and checks a final schedule. Preserve
that history as supplied input; do not invent interactive turns. The
[Indian Banking server](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/indian_banking/app.py)
has the live customer simulator that requires a future conversational interface.

**Structured-output versions have different contracts.** Current NeMo
[Structured Outputs v4](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/structured_outputs/README.md)
scores one final function call without executing it. Its declared objective is
schema adherence. Local TaskTrove rows join an older text-output dataset, not
this v4 artifact. A format-only score may match the source objective while still
being inadequate for a task that requests factual extraction. Preserve this
distinction when recording verifier parity and correctness defects.

**Code generation does not imply agent-side compute.** The direct NeMo coding
source is already represented in TaskTrove's converter policy, but is not
accepted by the current TaskCompendium importer. Add its code-answer submission
path with an isolated private checker. Existing generic code tasks are useful
controls, but do not establish this source's parity. The upstream
[code checker](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/code_gen/app.py)
is a remote service; its presence alone does not establish safe local execution.

## Minimal first implementation wave

Acquire one source seed for each case below, then exercise diagnostic outcome
variants. Exact source revisions, fixture paths, row IDs or canonical hashes,
and acceptance checks are in [suite.json](suite.json). A source `selector` names
the row ID; an optional `fixture_selector` supplies an exact JSONL index/hash
when there is no stable ID. Null `fixture_selector` uses the concrete source
selector; secondary acquisition candidates must be made unambiguous before use.
The records have not been
materialized into new TaskCompendium specifications or executed.

1. **Instruction constraints:** upstream fixture `id=17616`. Check all-pass,
   one-constraint failure, binary/fractional aggregation, and unsupported or
   failing evaluator behavior. This expands importer/verifier coverage.
2. **Code as answer:** the pinned NeMo competitive-code fixture. Require no
   agent process capability merely because verification runs a program. Test
   correct, wrong, and empty code in an isolated checker; preserve every
   retained hidden test. This expands submission/importer coverage.
3. **Predicted action:** the pinned function-pivot fixture. Preserve name and
   typed arguments, distinguish action submission from execution, and test
   cardinality. Batch semantics need an explicitly selected batch case; the
   single-call seed does not establish batch coverage. This adds a submission
   contract, not a domain-tool environment.
4. **Stateful domain tools:** Workplace fixture `id=0`. Test correct and wrong
   mutations, no-op failure, tool-error recovery, trace call/result correlation,
   and fresh/concurrent session isolation. This is the first new execution
   provider interface.

Keep the existing MCQA row 1972, generic code row 0, and SymPy R2E repair as
controls. The secondary queue covers Structured Outputs v4 final-call output,
Calendar, generator provenance, and untrusted tool observations. Live search is
later; reactive Banking stays explicitly unsupported. True concurrent tool
mutation semantics are untested and require an ordering/atomicity contract.

## Validity and acceptance

Compare to the original verifier as a parity baseline, but do not confuse parity
with correctness. Source errors and timeouts sometimes become score zero; record
such behavior explicitly before mapping it to TaskCompendium's failure statuses.
Do not silently tighten a weak source checker or silently inherit a false pass.

Known TaskTrove rows 16636 and 16634 remain in the
[problematic-task ledger](../../logbooks/taskcompendium-problematic-tasks.md).
Their structural checkers accept semantic false positives; row 16636 also lacks
facts required by its task. This audit does not repair them or reinterpret them
as current NeMo v4 tasks. The existing tasktrove-2 chat-code follow-up also remains
open. Task suitability near Snowball's approximately 2B active scale is unmeasured.

The suite requires that task-private verifier data remain private and that a
provider or rendering change not redefine correctness. A valid wrong answer,
malformed submission, invalid task, and infrastructure failure remain distinct
outcomes. Stateful cases additionally require isolated mutable state per run.

## Negative leads and reproducibility limits

- Direct S3 access lacked credentials; clean `.8` totals are unknown. The
  TaskTrove Marina viewer was checked as an alternative, but it serves the raw
  HF dataset rather than the requested clean release.
- HF Viewer totals differ from producer-era input counts and are not a historical
  clean-release index.
- Current public HF revisions do not resolve GitLab-only source artifacts or
  prove historical TaskTrove joins.
- A source-directory-name search cannot establish content-level absence of
  Indian Banking.
- Source inspection establishes expected behavior, not successful execution of
  the new cases. No live models or grading services were launched.
