# Static reimport findings for deferred NeMo answer rows

Reviewed 2026-09-14 against the two-row NeMo Gym sweep and the current
TaskCompendium answer, math, and judge models. This is an import plan; it does
not change code or claim that a new importer has been executed.

The safe public projection for every candidate below is the source question
and its answer-format instruction after removing delivery-only text. Gold
answers, reference answers, reward profiles, verifier metadata, and judge
instructions stay in verifier-only resources. They must never be copied into
`responses_create_params.input`, rendered prompts, public task JSON, or
model-visible recommendations.

## Pinned rows and disposition

| source revision and split | exact rows | source fields inspected | private expected value / shape | shared verifier plan | disposition |
| --- | --- | --- | --- | --- | --- |
| `nvidia/Nemotron-RL-knowledge-mcqa@62a1eec1f952723eab2ee3832222f533b8138067`, `default/train` | offsets `132738`, `441248` | `responses_create_params.input[0]`, `expected_answer`, ten-option `options`, `template_metadata.format_type=mcqa`, `output_regex`, `reward_profiles` | Both rows have one-letter gold `B` | MCQA exact-choice verifier: extract one option letter with the source-compatible answer format and compare privately. Ignore reward profiles. | **accept** |
| `nvidia/Nemotron-RL-math-OpenMathReasoning@5ef69384b65f13f08c35b73ddd5e7bf5e4621043`, `default/train` | offsets `0`, `1` (the two-row first-rows fallback used by the sweep) | `problem`, `expected_answer`, `responses_create_params.input[0]`, boxed-output instruction; no tool declaration is needed for these rows | Row `0` is scalar integer `32`; row `1` is a nonempty radical expression | Shared math exact-answer verifier with scalar/symbolic normalization. Preserve boxed final-answer extraction as a rendering/output convention; keep the gold private. Do not use the NeMo source evaluator. | **accept** |
| `nvidia/Nemotron-RL-math-stack_overflow@be489b25f36ef92864546a7ace22edec4e053ac3`, `default/train` | offsets `0`, `1` (first-rows fallback) | `problem`, `expected_answer`, `responses_create_params.input[0]`, boxed-output instruction; no external resource or tool is required | Row `0` is a scalar proportional-length result (approximately `1.159 R`); row `1` is scalar `0` | Same shared math exact-answer verifier. Numeric tolerance/normalization must be explicit enough to accept the source's approximate scalar representation; boxed output remains formatting, not gold text. | **accept** |
| `nvidia/Nemotron-RL-knowledge-openqa@3604d4119623f2961c9cd0a3a5365e0cff0dd393`, `default/train` | offsets `25704`, `86898` | `responses_create_params.input[0]`, `expected_answer`, boxed-answer instructions, model `reward_profiles`; source rows have no tool requirement | Row `25704` has a short factual reference (`19F`); row `86898` has a long legal explanation reference | Shared reference-answer judge with caller-supplied judge policy. Keep the cleaned question and boxed convention public; retain reference text and judge configuration privately. Treat the reward profiles as provenance only. | **accept** |
| `nvidia/Nemotron-RL-Science-v1@a7f55756f14bdd16c6469b94601d86be15e4c4fc`, `default/so_openq` | offsets `69602`, `144924` | `problem`, `expected_answer`, `responses_create_params.input[0]`, `template_metadata.output_regex`, `verifier_type`, and `tools` | Both have nonempty reference answers. Row `69602` also advertises `stateful_python_code_exec`; row `144924` has a format instruction and no source tool declaration | Row `144924`: shared reference-answer judge, preserving the source's bold final-answer extraction convention. Row `69602`: no safe static mapping while the advertised stateful Python tool is part of the contract. | **accept row 144924; defer row 69602** |
| `nvidia/Nemotron-RL-ReasoningGym-v1@ad2c929b2dfd64ca30afb4d60e2f69a5a1919c4d`, `default/train` | offsets `0`, `1` (first-rows fallback) | `responses_create_params.input[0]`, `question`, `answer`, `metadata`, `agent_ref`; no verifier field or tool declaration | Row `0` is a direct name-answer task with private gold `Richard`; row `1` is a self-contained scalar friends-count task with private gold `6` | Row `0`: reference-answer judge (exact/reference text is private). Row `1`: shared math exact-answer verifier for an integer scalar. Preserve the answer-only instructions supplied by each question. | **accept both, subject to verifier-contract tests** |

## Why these mappings are sound

The MCQA records already contain a fixed option alphabet, a single expected
letter, and a source output regex. Their model-facing input is only the
question/options and the delivery line; the answer and reward profiles are
private. This is the same semantic shape as the existing TaskCompendium MCQ
answer importer, although a Hub-row adapter is still needed because the
current implementation accepts TaskTrove archives rather than these raw Hub
records.

The four math records are answer-only, self-contained, and expose a single
nonempty expected answer. Their boxed instructions describe extraction, not a
separate environment. A shared math verifier can therefore replace the
`nemotron_math` source evaluator if it declares the accepted scalar/symbolic
normalization and tests approximate numeric answers. Do not import the source
`expected_answer` into the prompt.

The OpenQA and Science row `144924` records contain a question and a private
reference answer. They can use the existing reference-answer judge family with
a caller-supplied judge policy. The source's boxed, bold, or other final-answer
markers belong in rendering/extraction and must remain separate from the
reference and judge runtime. The source's model pass rates are not a verifier.

ReasoningGym row `0` is a direct answer request despite the corpus-wide
procedural generator metadata. Row `1` is a scalar arithmetic/social-graph
question. For these exact rows, the visible questions contain the required
facts and the answer fields are sufficient to seed private references. This is
an instance-level decision: do not generalize the whole ReasoningGym corpus to
one verifier without checking answer shape and metadata per row.

## Unsafe or deferred cases

* Science offset `69602` advertises `stateful_python_code_exec`. Removing the
  tool changes the advertised interaction contract; executing it would require
  a provider/runtime that is outside this static answer import. Keep it
  deferred unless a separate provider is pinned and the task is re-reviewed.
* ReasoningGym must not be bulk-imported from `answer` alone. Generated tasks
  with structured outputs, sequences, state transitions, or procedural scoring
  need their own verifier semantics. The two accepted rows are narrow
  answer-only instances and should be rejected if the row shape differs.
* Science's shared judge mapping is appropriate only for rows whose question is
  self-contained and has no required tools. Preserve `template_metadata` output
  extraction; stripping a format marker without retaining an equivalent
  renderer changes source behavior.
* OpenQA's long legal reference remains private and judge quality is not
  validated by the static inspection. A live judge check is a separate
  validation step; the import plan only establishes the semantic boundary.

## Small implementation plan

1. Add one Hub-row normalization boundary that records dataset revision, split,
   offset, UUID (when present), and a source-row digest in verifier-only
   provenance.
2. Route MCQA rows to one shared exact-choice verifier; route the four math rows
   plus ReasoningGym offset `1` to one shared math verifier; route OpenQA plus
   Science offset `144924` and ReasoningGym offset `0` to one shared
   reference-answer judge family.
3. Implement format extraction as rendering metadata (`boxed`, `bold`, or
   source regex), never as public gold or source evaluator text.
4. Add per-row contract tests for correct, wrong, empty, and malformed answers;
   verify that public task projections contain none of `expected_answer`,
   `answer`, `reference`, `reward_profiles`, `verifier_type`, or tool outputs.
5. Keep the two unsafe cases deferred and record any row-level rejection when
   a later sample violates the answer-only assumptions.

Evidence: `.agents/projects/taskcompendium-nemo-sweep/reviews/answer-corpora.json`,
`reviews/viewer-fallback.json`, `summary.json`, and current
`lib/taskcompendium/src/taskcompendium/importers/nemo.py` plus the shared
TaskTrove answer/math/judge importers.
