# TaskTrove `.8` representation sweep (historical)

**Superseded for current release work by the `.9` inventory and review.** This
file records the earlier checked-in fixture sweep; it is not a sample of the
current 1,449,686-row clean release.

Date: 2026-09-13

This bounded review samples two tasks from each requested TaskTrove set: MCQ
and exact answer tasks, two math tasks, two reference-answer judge tasks,
JSON/XML structured-output tasks, two pytest coding tasks, two stdio coding
tasks, and two shell tasks. The fixtures are under
`lib/taskcompendium/tests/fixtures`; the generated Harbor packages are under
`lib/taskcompendium/examples/poc/harbor`.

The checked-in release metadata identifies the source as
`s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8`, with producer
revision `ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2`, inspected converter
revision `bef70bb8584d7e0c1391d88c9eacb1605b551a90`, pinned verifier revision
`b2b68d8b0a770cdc0ab3903780172c4b3eea81b1`, and Harbor revision
`9551f376157d90104011107dcbf9ca3621228126`. The release manifest and ledger
were not readable from this host, so fixture provenance and the checked-in
manifest do not independently prove clean-release membership or row counts.

## Findings

| Set / task ID | Source and checked-in fixture | Representation and contract | Fidelity finding | Disposition |
| --- | --- | --- | --- | --- |
| MCQ `tasktrove-1972` | `laion__nemotron-gym-knowledge-mcqa-v2`; `tasktrove/answers/mcq-row-1972.tar.gz`; upstream row `595578`, UUID `634bfa88-f91f-5a99-a540-7f0ffdaa9b84` | The public prompt retains the question and ten options. The semantic verifier privately retains expected option `C` and option count `10`; requirements are empty and the lowering uses `NoToolEnvironment`. | All required facts are in the question, and the final-value contract is natural after removing the source delivery wrapper. The public prompt and `task.json` contain no verifier, judge, reward, gold, grader, hidden-test, or reference-answer text. | Keep. |
| Exact `tasktrove-119` | `laion__all-puzzles-v2`; `tasktrove/answers/exact-row-119.tar.gz`; path `all_puzzles-38270` | The public prompt retains the Blicket observations and answer choices. The private exact verifier expects `off`, with case/whitespace normalization and ordered matching; no environment is required. | The answer is determined directly by the supplied observations. The public text is evaluation-free and the exact output contract is natural. `puzzle_instructions` adds a generic coordinate-rounding sentence to this non-coordinate puzzle; it is irrelevant but does not change the checker. | Keep, with low-priority prompt cleanup tracked separately. |
| Math `tasktrove-114` | `laion__all-puzzles-v2`; `tasktrove/math/math-row-114.tar.gz`; path `all_puzzles-5399` | The public prompt supplies all triangle coordinates and rounding rules. The private math verifier expects `(-5.167, 4.693)` with scalar parsing; no environment is required. | Independent calculation gives orthocenter `(-1700/329, 1544/329)`, rounding to `(-5.167, 4.693)`. The coordinate and precision requirements are available publicly; wrong coordinate and empty extraction cases score zero/error. | Keep. |
| Math `tasktrove-115` | `laion__all-puzzles-v2`; `tasktrove/math/math-row-115.tar.gz`; path `all_puzzles-59580` | The public prompt contains the complete inclusive prime-count interval. The private math verifier expects `246`; no environment is required. | Independent enumeration gives `246`. The generic coordinate-rounding sentence is unrelated to this integer task, but no substantive requirement is lost and the verifier checks the answer. | Keep, with low-priority prompt cleanup tracked separately. |
| Judge `tasktrove-11676` | `laion__nemotron-gym-knowledge-openqa-v4`; `tasktrove/judge/judge-row-11676.tar.gz`; upstream row `120859`, UUID `57826cc2-4190-5013-b3ee-1c2196fdcbd8` | The public prompt retains the legal question and asks for an answer. The private judge verifier retains the reference answer, cleaned question, exact gate, and caller-supplied judge policy; no capabilities or tools are required. | The question supplies the material needed for the legal conclusion. Source `\boxed{}` and judge wording are removed from the canonical task and are available only through a selected rendering/policy. Exact reference text is private. | Keep. Live judge quality and endpoint availability remain unvalidated; exact-gate and fake-judge plumbing tests pass. |
| Judge `tasktrove-11677` | `laion__nemotron-gym-knowledge-openqa-v4`; `tasktrove/judge/judge-row-11677.tar.gz`; upstream row `38719`, UUID `4d41ae2b-0c86-5269-9214-ccd851ea0aee` | The prompt contains the resolutions, threshold, integration windows, measured ratios, and KMS comparison. The private verifier retains the reference explanation and exact gate plus the explicit judge policy. | The requested reason is answerable from the supplied numerical and method context. The output contract is plain by default; boxed output is a rendering choice. No private answer or judge machinery appears in the public task. | Keep, subject to live judge validation before production use. |
| Structured JSON `tasktrove-16636` | `laion__nemotron-gym-structured-outputs-v4`; `structured/json-row-16636.tar.gz`; upstream row `14056` | The public prompt includes the source document and complete inline schema; the schema is also stored privately as `schema.json`. Requirements are empty and the lowering uses `NoToolEnvironment`. | The schema checks only shape and types. It requires location, budget, inspection, mortgage, and closing facts absent from the document. A schema-valid answer containing `Atlantis`, invented inspection data, `99` interest, and an invented escrow company receives reward `1.0` (`detail.reason=valid`). The public prompt itself is evaluation-free, but the semantic specification is incomplete and the checker misses factual correctness. | Reject or redesign. Recover an aligned source/schema pair or redefine the task as explicit schema generation; do not invent facts or strengthen the checker without source evidence. |
| Structured XML `tasktrove-16634` | `laion__nemotron-gym-structured-outputs-v4`; `structured/xml-row-16634.tar.gz`; upstream row `45426` | The public prompt includes the Sikorsky document and XML schema definition. The private verifier requires six element names; no agent resource or environment is needed. | The source document supplies the aircraft facts, but the verifier checks names only. XML with invented values and `not a year` / `not a number` for integer fields receives reward `1.0` (`detail.reason=names_present`). It also ignores additional source-supported fields despite the instruction to extract all matching information. The importer now says “follows the schema in the task” instead of incorrectly calling the XML schema a “JSON Schema.” | Reject or redesign. Preserve the source only after adding source-grounded value/type/completeness checks, or exclude the row if no faithful checker can be recovered. |
| Pytest coding `tasktrove-1487` | `DCAgent__exp_rpt_curriculum-easy`; `coding/pytest-row-1487.tar.gz` | The public prompt describes the requested Python behavior. The private verifier retains the hidden pytest file and immutable Python runtime; requirements declare filesystem, shell, and process capabilities. | The task is a normal repository-editing task: the prompt contains the behavior and the hidden tests check it. Public Harbor input contains no hidden test or grading machinery. | Keep. |
| Pytest coding `tasktrove-1488` | `DCAgent__exp_rpt_curriculum-easy`; `coding/pytest-row-1488.tar.gz` | Same workspace and hidden-test contract as row 1487; the prompt includes the required `segno.py` behavior. | The source prompt/test path relationship needs runner confirmation: the prompt names `/app/segno.py`, while the test imports `app.segno`. This may be a packaging contradiction even though the task shape and capability declaration are otherwise natural. | Keep open for runner confirmation; repair or reject if the import path is genuinely inconsistent. |
| Stdio coding `tasktrove-0` | `laion__codeforces-v3`; `coding/stdio-row-0.tar.gz` | The prompt contains the complete batch input/output problem. Hidden input/output cases remain verifier-only resources, and the task requires filesystem, shell, and process capabilities in an immutable toolchain image. | Batch execution and submission semantics are represented faithfully; hidden cases are private and the public prompt contains no checker language. | Keep. |
| Stdio coding `tasktrove-2` | `laion__codeforces-v3`; `coding/stdio-row-2.tar.gz` | The source is marked interactive, but the importer’s static stdio contract only supports batch cases and a final workspace submission. | A static Harbor lowering cannot preserve the interactive protocol, feedback turns, or interaction budget. The importer now rejects the row instead of silently exporting a non-equivalent task. | Reject pending an interactive execution adapter. |
| Shell `tasktrove-16158` | `DCAgent2__nl2bash-tasks-cleaned-oracle-v2`; `shell/script-row-16158.tar.gz` | The public prompt asks for a shell command/script; ShellSim and Docker lowerings expose filesystem, shell, and process capabilities while keeping the checker private. | The source checker compares an order-insensitive multiset and permits extra non-error records, although the prompt requires descending frequency order. The representation preserves this source behavior but should not present it as a fully verified task. | Reject or repair the checker; tracked in the problematic-task ledger. |
| Shell `tasktrove-16159` | `DCAgent2__nl2bash-tasks-cleaned-oracle-v2`; `shell/script-row-16159.tar.gz` | Same ShellSim/Docker capability and private script-checker shape as row 16158. | The checker has the same permissive extra-output and ordering behavior. This row’s prompt does not independently establish an ordering requirement, so the concrete semantic defect is not confirmed. | Keep open; audit with the related rows before production use. |

## Lowering and Harbor checks

For every selected task, the generated `task.json` projection contains
instructions, public resources, submission, requirements, and provenance, but
no private verifier parameters. The corresponding `specification.json` keeps
the private verifier/oracle data, and `execution.json` selects the replay agent,
semantic verifier adapter, and `NoToolEnvironment` for these answer-only rows.
The manifests retain the pinned specification hash, lowering version `0.5`,
and Harbor revision. The selected package directories are:

- `harbor/tasktrove-1972-plain`
- `harbor/tasktrove-119-plain`
- `harbor/tasktrove-114-plain`
- `harbor/tasktrove-115-plain`
- `harbor/tasktrove-11676-plain`
- `harbor/tasktrove-11677-plain`
- `harbor/tasktrove-16636-plain`
- `harbor/tasktrove-16634-plain`
- `harbor/tasktrove-1487-workspace`
- `harbor/tasktrove-1488-workspace`
- `harbor/tasktrove-0-workspace`
- `harbor/tasktrove-2-workspace` (rejected after the sweep; retained only in the pre-sweep artifact)
- `harbor/tasktrove-16158-workspace`
- `harbor/tasktrove-16159-workspace`

The rejected stdio row is absent from the regenerated canonical POC exports;
the path above identifies the stale pre-sweep package only for provenance.

Coding and shell rows were also checked for public/private resource separation,
capability declarations, and forbidden verifier terms. The two pytest rows,
batch stdio row 0, and shell row 16159 remain representable; interactive stdio
row 2 is explicitly rejected.

MCQ, exact, and math fixtures were graded with correct, plausible-wrong, and
empty submissions: correct answers scored `1.0`, wrong answers scored `0.0`,
and empty output produced an extraction error. Judge row 11678's existing
focused test also confirms exact-gate success, judge success, and judge zero
paths; rows 11676 and 11677 share the same importer and private policy shape.
The structured counterexamples above were graded through the same public
`grade_attempt` boundary. Focused tests cover importer contracts, private
resource roles, rendering-only answer wrappers, and Harbor package shape.

Validation run:

- Ad-hoc `uv run --project lib/taskcompendium python` probes imported each
  fixture, rendered the public task, checked private/public field separation,
  independently checked the two math answers, and graded good/wrong/empty
  answer cases plus the structured counterexamples.
- `uv run --project lib/taskcompendium --group test pytest
  lib/taskcompendium/tests/test_tasktrove_answers.py
  lib/taskcompendium/tests/test_math_importers.py
  lib/taskcompendium/tests/test_judge_importer.py
  lib/taskcompendium/tests/test_structured_importer.py` passed 40 tests.
- `./infra/pre-commit.py --only ruff --only black
  lib/taskcompendium/src/taskcompendium/importers/tasktrove_structured.py
  lib/taskcompendium/tests/test_structured_importer.py` passed.

The XML wording regression is in
`lib/taskcompendium/tests/test_structured_importer.py`.
