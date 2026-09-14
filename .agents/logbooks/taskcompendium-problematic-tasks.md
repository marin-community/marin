---
topic: taskcompendium-problematic-tasks
description: Confirmed task defects and principles for the next cheap-model repair sweep.
author: dlwh
---

# TaskCompendium problematic tasks

Track concrete defects for the cheap-model repair sweep. Recording an issue does
not repair or remove a task. Keep the evidence entries below; update this index
as fixes are validated. The first focused sweep was static and made no source
changes.

## Review principles

- **Semantic coverage:** Verification must assess every nontrivial requirement
  in the task. Valid syntax, required fields, and types do not establish faithful
  extraction, factual correctness, or satisfaction of behavioral requirements.
- **Sufficient specification:** Supply the information and rules needed to
  determine acceptable answers. Open-ended tasks may admit many answers, but
  required facts must be supplied, obtainable through declared resources/tools,
  or explicitly left to the solver's choice. Define missing-data behavior where
  needed; do not silently require invented facts.
- **Consistent contracts:** Instructions, output requirements, reference answers,
  and checks must agree. Preserving a source checker is not evidence of validity.
- **Approximate verification:** Cheap inexact checks combined with an LLM judge
  are an option to evaluate. Record each component's coverage, known blind spots,
  and the rule for combining results. A schema check alone must not certify
  semantic correctness. A judge cannot recover facts absent from an underspecified
  task. Keep judge infrastructure failure distinct from an incorrect answer.

## Current index

The authoritative review target is TaskTrove Clean release `2026.09.10.9` at
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`. The sampled `.9`
findings are recorded below the historical entries. The older `.8` rows remain
as evidence for the earlier POC and must not be treated as a review of `.9`.

### Clean `.9` sampled rows

The first release-balanced sample selected exactly two task paths per each of
the 43 kept source subsets. The independent review reports are
[`clean09-review-a.md`](../projects/taskcompendium-coverage/clean09-review-a.md)
and
[`clean09-review-b.md`](../projects/taskcompendium-coverage/clean09-review-b.md).

| Task/source row | Status | Problem | Principle | Next sweep action |
| --- | --- | --- | --- | --- |
| `laion__nemotron-gym-instruction-following-structured-v3` (`if-structured-700bbfb7ae3d.tar.gz`, `if-structured-a252a3bbe924.tar.gz`) | Reject; Clean `.9` | Required fields are absent from the supplied documents, while the public contract permits arbitrary schema-valid values and the verifier checks structure only. | Semantic coverage; sufficient specification | Recover aligned source facts/schema and content checks, or reject the source family. |
| `laion__nemotron-gym-structured-outputs-v4/if-structured-v2-529e76d630c5.tar.gz` | Reject; Clean `.9` | XML verifier requires only `<output/>` and has no schema resource, so an empty output bypasses the documented extraction and datatype requirements. | Semantic coverage; consistent contracts | Require a source-backed XML schema/content checker, or reject. |
| `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` (`task_1139`, `task_6529`) | Open; Clean `.9` | `nl2bash_check.py` compares an order-insensitive multiset although instructions and expected files require sorted output. | Semantic coverage; consistent contracts | Repair the checker to enforce ordering or reject affected rows; audit related tasks. |

### Clean `.9` representation and lowering gaps

These are not confirmed source-checker defects, but they are concrete reasons
the sampled tasks cannot yet be exported faithfully by the current
TaskCompendium implementation.

| Sampled source/tasks | Gap | Principle | Follow-up |
| --- | --- | --- | --- |
| DCAgent pytest families (`curriculum-*`, `e2egit-*`, `multifile-*`, `pymethods2test-*`, `stack-pytest-*`, `unitsyn-*`) | Tasks use source-specific test paths such as `/tests/test_solution.py` and expose test scaffolding or expected-test behavior; the importer assumes `/tests/test_curriculum.py`. | Preserve task semantics and keep evaluation machinery private. | Make the pytest lowering resource/path-driven and scrub verifier/test-generation delivery text. |
| DCAgent code-contests and Taco samples (`code_contests-*`, `taco-*`) | Complete Python batch programs are coherent, but the current coding lowering assumes the narrower Codeforces/C++17 path and has no `taco` converter. | One semantic task must lower to a compatible execution contract without changing language or I/O. | Add source-backed Python stdio lowering or retain as unsupported. |
| Codeforces `codeforces-07764` and ARC-AGI samples (`arc-induct-*`, `arc-trans-*`) | Constructive/special-judge or candidate-code runtimes are present, while public wrappers expose verifier, gold, or held-out-test machinery. | Hide verifier internals while preserving the required execution capability. | Add dedicated private runtime lowerings; scrub public delivery text. |

All other sampled `.9` rows are classified in the two reports as coherent
but needing an importer/prompt/lowering repair, unsupported by the current
execution model, or keepable after correcting the release/verifier pins. No
source archive or checker was modified during this review.

### Historical `.8` index

The rows below come from the superseded TaskTrove release `2026.09.10.8` at
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8`.

| Task/source row | Status | Problem | Principle | Next sweep action |
| --- | --- | --- | --- | --- |
| `tasktrove-16636` | Open; still included | Schema-only verification accepts invented facts. The document gives general home-buying advice, but required fields describe a particular purchase absent from the document. | Semantic coverage; sufficient specification | Recover an aligned source document/schema and semantic checks, or reject. An explicitly derived schema-generation task is a separate option with changed instructions and provenance. |
| `tasktrove-16634` | Open; still included | XML verification checks required names, accepting fabricated values and nonnumeric text for integer fields. | Semantic coverage; consistent contracts | Check extracted values against the document, enforce field types, and assess extraction completeness. |
| `tasktrove-2` | Rejected; interactive protocol unsupported | The source is an interactive stdio problem, but the static importer only supports batch stdio and final workspace submission. | Preserve all interaction semantics; do not silently lower an interactive task to a batch task. | Add a dedicated interactive execution adapter before reconsidering this row. |
| TaskTrove row `16158` | Open; still included | The prompt requires descending frequency order, but the checker compares an order-insensitive multiset and permits extra non-error records. | Semantic coverage; consistent contracts | Repair the checker or reject after confirming the source ordering requirement. Audit related rows `16159` and `16160` for the same permissive checker. |
| TaskTrove row `1488` | Open; still included | The prompt requires `/app/segno.py`, while the source test imports `app.segno`; the required submission path and checker import path may contradict. | Sufficient specification; consistent contracts | Confirm the runner behavior and source intent, then repair or reject. |
| TaskTrove row `113` | Already rejected | Instructions request descending order; expected output is ascending. | Consistent contracts | Check source intent before repairing the expected answer; otherwise retain rejection. |
| TaskTrove row `16635` | Already rejected | Instructions require quoted values, but the schema requires boolean and numeric values. | Consistent contracts | Resolve the contradictory output requirements from source evidence; otherwise retain rejection. |

## Evidence entries

### 2026-09-12 — tasktrove-16636: fabricated extraction receives full reward

Inspected the local schema 0.3 specification. Its verifier is `json-schema` with
`schema.json`; it has no content-correctness check. The document supplies no
specific city, budget, inspection results, or escrow provider, yet the schema
requires these details.

Submitted a schema-valid object containing city `Atlantis`, interest rate `99`,
and invented inspection findings. `grade_attempt`, using assistant-final plain
text extraction, returned `status=graded`, `reward=1.0`, `reason=valid`.
The [saved candidate and result](taskcompendium-evidence/tasktrove-16636.json)
include the specification hash. The source archive is
[`json-row-16636.tar.gz`](../../lib/taskcompendium/tests/fixtures/structured/json-row-16636.tar.gz).
This confirms acceptance of one semantically unsupported answer; it does not
measure the false-acceptance rate of the broader family.

Repairing verification alone is insufficient here. First make the task
answerable, then check semantic fidelity as well as structure. Do not invent
missing source facts during normalization.

### 2026-09-12 — existing rejections: rows 113 and 16635

The [rejection ledger](../../lib/taskcompendium/examples/poc/rejections.jsonl)
records row `113` as `broken_grader` (descending instruction, ascending answer)
and row `16635` as `unrecoverable_source` (quoted-value instruction conflicts with
boolean/number schema types). These are excluded already; retain them as examples
of defects the next sweep should detect.

## Next cheap-model sweep

Start with the current example set in
[`examples/poc/manifest.json`](../../lib/taskcompendium/examples/poc/manifest.json)
(43 specifications and three rejected rows) and the open rows above.
Confirm any larger corpus scope when launching the sweep. Related structured-output
tasks initially means examples using `json-schema` or `xml-elements` verification.

For each task, enumerate its nontrivial requirements and map them to checks.
Flag uncovered requirements, unavailable facts, and contradictory constraints.
Record a concrete counterexample where possible, preserving valid formatting
while changing a meaningful value or behavior. Check related structured-output
tasks for the same defect; their status remains unreviewed until inspected.

Propose a repair or rejection with source evidence. If a task is intentionally
redefined, record it as derived rather than silently changing its meaning.
Validate repairs with both acceptable answers and semantically wrong answers
that pass cheap structural checks. Append the decision, evidence, and validation
result here, then regenerate affected exports and datasets in that sweep.


### 2026-09-12 — tasktrove-16634: XML field names pass without correct values

The Sikorsky S-9 task requests extraction of explicitly stated facts. Its
`xml-elements` verifier requires six names: `aircraft_name`, `role`,
`manufacturer`, `designer`, `first_flight_year`, and `number_built`.

A document with those elements, invented aircraft details, `not a year` for the
flight year, and `not a number` for the number built received `status=graded`,
`reward=1.0`, `reason=names_present`. The
[saved candidate and result](taskcompendium-evidence/tasktrove-16634.json) include
the specification hash. The verifier accepted incorrect facts and values that
violate the schema's integer requirements. Required field names also do not
establish that the answer includes the additional source-supported schema fields
requested by the instruction to extract all matching information.

The source document supplies the required aircraft facts; the missing-input
problem identified in `16636` is not established here. Repair should assess
semantic fidelity, types, and completeness. The task remains included pending
the next sweep.

### 2026-09-13 — focused cheap-model sweep: ordering and import-path defects

A fresh Luna review of the current ledger and related checked-in fixtures found
three additional concrete issues. Row `16158` requires descending frequency
order, but `nl2bash_check.py::_score` compares an order-insensitive multiset and
allows extra non-error records. Rows `16159` and `16160` use the same permissive
checker, although their prompts do not establish the same ordering defect.

Row `1488` appears to require `/app/segno.py` while its source test imports
`app.segno`; this needs a runner confirmation before changing the task. The
sweep found no safe importer-only repair and made no source changes. Keep these
rows open pending semantic repair or rejection.

The same sweep confirmed that stdio row `2` is explicitly interactive. The
static TaskCompendium importer now rejects it as `unsupported_environment`
instead of producing a non-equivalent Harbor package. The canonical POC was
regenerated with 43 specifications, 86 exports, and three rejections.


### 2026-09-12 — tasktrove-2: support chat-only program generation

Requested follow-up: make this task expressible with no required agent environment,
then demonstrate lowerings to chat, ShellSim, and Docker. The current Rotary Laser
Lock example requires Docker and asks for `/app/solution.py` or
`/app/solution.cpp`, with a `stdio` verifier. A chat-only agent could instead return
program text; the lowering can place that text into the isolated verifier workspace.
Compiling or running the submitted program still requires a verifier runtime.

General principle: distinguish tools needed to solve the task from the machinery
needed to submit and check an answer. Source packaging must not force an agent
filesystem when an assistant response can carry the same solution.

During the sweep, preserve language, input/output, and correctness requirements
across submission surfaces. The retained statement describes an interactive
problem, display feedback, and a rotation budget; audit that interaction contract
before claiming equivalent lowerings. Chat-only here means generating the solution
program without agent tools, not silently replacing the interactive problem with
a static answer question. This is a follow-up candidate, not an implemented or
validated conversion.


### 2026-09-13 — NeMo predicted-action comparator: inherited weak checks

The first-wave adapter preserves the comparator at NeMo Gym revision
`1e668906d2e69a9e8ee9aaafc60050a4025d9688`. A delegated source review confirmed
that a message target accepts any actual chat message, without checking its
content. Its Python type check also permits JSON `true` to match an expected
integer `1`. These are source-family limitations; the selected first-wave seed
is a function-call case, not evidence that message content is verified.

Regression cases in `test_nemo_predicted_action.py` record both outcomes. The
adapter retains them to make source fidelity explicit. A stricter comparator
must be a separately identified semantic profile, with wrong-message and
boolean-versus-number counterexamples, rather than an unrecorded source change.

 General principle: reproducing a source score does not establish that it checks
all nontrivial task requirements. Audit semantic coverage independently of format
validity and source fidelity; repair or reject weak cases during the cheap-model
sweep.

### 2026-09-14 — Clean `.9` structured-output samples

An independent review of the two sampled
`laion__nemotron-gym-instruction-following-structured-v3` archives found that
the required JSON fields are not recoverable from their documents. The
`if-structured-700bbfb7ae3d.tar.gz` document discusses *Everything Everywhere
All at Once*, but its schema requires a viewer rating, recommendation, viewing
context, platform, and duration that the document never states. The
`if-structured-a252a3bbe924.tar.gz` wearable-technology document does not state
the required `version` or `releaseYear`. Both instructions include an
evaluation-contract preamble that explicitly permits arbitrary schema-valid
values, while the JSON-schema verifier checks only structure. These sampled
tasks should be rejected unless an aligned source/schema or a semantic
extraction checker is supplied; do not invent the missing facts.

The sampled `laion__nemotron-gym-structured-outputs-v4` XML archive
`if-structured-v2-529e76d630c5.tar.gz` contains a long Indianapolis 500
extraction document and an inline schema in the instructions, but its private
verifier is only `required = ["output"]` with no schema resource or field
requirements. A well-formed empty `<output/>` therefore satisfies the declared
element-name check while bypassing every extraction and datatype requirement.
This task should be rejected pending a checker that validates the documented
schema and source-backed values. The sibling sampled YAML archive
`if-structured-v2-478906f3fa67.tar.gz` is a separate unsupported output format,
not evidence for a repair that changes YAML to JSON.

### 2026-09-14 — TaskTrove Clean `.9` shell checker does not enforce ordering

The `.9` mirror `open-athena/task-trove` was sampled at the exact paths
`task_1139` and `task_6529` from
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`. Both archives use
the same `tests/nl2bash_check.py`: `_score` compares `Counter` values, so it
accepts records in any order, and it rejects extras only when they contain one
of a small set of error words. The task instructions require sorted output and
the expected files are ordered (`4 apple`, `2 banana`, `1 orange` for
`task_1139`; `honeydew:5`, `fig:4`, `egg:3`, `carrot:2`, `apple:1` for
`task_6529`). This is a confirmed source-checker defect; no checker weakening
or invented repair was made. Keep both rows out pending a source-backed
ordered comparison repair.
