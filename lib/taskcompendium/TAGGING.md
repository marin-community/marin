# TaskCompendium coverage tagging

Use these labels to describe the semantic work in a `TaskSpec`. They support
coverage analysis and sampling; they do not change execution, rendering, or
verification.

Tag sparingly. A useful label identifies a capability, task shape, subject, or
state that would otherwise be hard to select from the corpus. Do not turn the
tag set into a paraphrase of the prompt.

## What to inspect

Read the model-visible instructions, declared requirements, agent-visible
resources, and source provenance. Use the task's actual work, rather than its
source packaging or evaluator implementation. Do not inspect a private oracle
or verifier to infer labels.

The following are not semantic coverage tags:

- a harness, execution target, or source packaging, such as Harbor, ShellSim,
  Docker, a container image, or a particular agent;
- a verifier, reward, judge, hidden test, or extraction mechanism;
- a rendering convention. `result:json`, `result:xml`, and `result:file`
  belong to a `Lowering`, not a `TaskSpec`;
- a dataset name, split, importer, or arbitrary prompt keyword.

A task may need a filesystem, shell, process, or provider environment. Record
that in `TaskRequirements`. Add a `state:*` or `interaction:*` tag only when
persistent state or an interaction pattern is a substantive part of the work.

## Required and typical labels

Every reviewed specification receives exactly one `difficulty:*` and one
`shape:*` tag. Add one main `competency:*` tag; add a second only if it captures
another independently important skill. The remaining namespaces are optional.
Most tasks should have three to six tags.

Tags are lowercase `namespace:value` strings. Keep them unique and sorted.
Reuse an existing value when it fits. The vocabulary below is a preferred
starting point, not a closed registry: add a value only when none of these
express a useful, recurring distinction.

| Namespace | Meaning | Usual count |
| --- | --- | --- |
| `competency` | The main reasoning or execution skill. | 1–2 |
| `shape` | The kind of request or completed work. | exactly 1 |
| `subject` | A material subject area, product, or operational area. | 0–1 |
| `artifact` | A substantive deliverable the model creates or changes. | 0–2 |
| `interaction` | A meaningful ordered or tool-mediated interaction pattern. | 0–1 |
| `state` | Meaningful mutable state that persists across the task. | 0–1 |
| `context` | Input context or setting that materially shapes the work. | 0–2 |
| `difficulty` | Difficulty for the Snowball-scale active model. | exactly 1 |

### `competency`

Common values include:

- reasoning: `math`, `causal_reasoning`, `geometric_reasoning`,
  `number_theoretic_reasoning`, `scientific_reasoning`, `legal_reasoning`,
  `medical_science_reasoning`, `knowledge_recall`, and `information_extraction`;
- instruction and problem solving: `instruction_following`,
  `constraint_satisfaction`, `problem_decomposition`, `planning`, and
  `multi_step_workflow`;
- software and technical work: `algorithm_design`, `competitive_programming`,
  `software_implementation`, `software_engineering`, `debugging`,
  `shell_scripting`, `program_synthesis`, `repo_navigation`, and
  `structured_data_transformation`;
- tools and state: `function_calling`, `tool_selection`, `tool_use`,
  `stateful_tool_use`, `state_tracking`, and `workflow_execution`.

Use `function_calling` when the task asks for a native function/action as the
submission. Use `tool_use` when the model must use tools but they do not carry
important state. Use `stateful_tool_use` when tool actions operate on a
persistent domain such as the Workplace seed. Do not add all three merely
because a task is executed through a tool-calling harness.

### `shape`

Use one of the established shapes when possible:

`answer`, `multiple_choice`, `calculation`, `explanation`,
`constrained_generation`, `structured_extraction`, `code_generation`,
`code_implementation`, `algorithmic_programming`, `environment_modification`,
`shell_workflow`, `predicted_action`, `stateful_domain`, or
`sequential_requirements`.

Add a new shape only when the completed work cannot be described by these. For
example, use `environment_modification` for a repository repair and
`predicted_action` when the submission is an API action; neither is a generic
"agentic" task.

### `subject`

A subject is optional. Use one only for a broad, recognizable field, product,
or operational area that improves sampling beyond the competency label. Good
examples are `arithmetic`, `number_theory`, `algorithms`, `python`,
`filesystem`, `email`, `event_ticketing`, `machine_learning`, `oncology`, and
`real_estate_law`. Do not tag ordinary implementation details or generic
functionality such as `name_handling`, `numeric_formatting`, or
`data_type_inference` as subjects.

Use dotted subtags for recurring, more specific functionality. For example, a
React-heavy web task uses `subject:javascript.react`, and an integration problem
uses `subject:calculus.integration`. Choose the most specific useful value rather
than adding both its broad parent and child. Add a new subtag only when the topic
is likely to recur across task families; do not mint a one-off subject tag that
merely repeats prompt wording or identifies a narrow source package. Omit broad
labels such as `general` or `software` when a competency already says what is
needed.

### `artifact` and `context`

`artifact` describes what the model substantively produces or changes. Common
values are `prose`, `numeric_answer`, `formula`, `source_code`, `program`,
`python_module`, `python_package`, `shell_output`, `workspace_state`, and
`native_action`.

`context` describes material inputs or the setting. Common values include
`repository`, `filesystem`, `source_code`, `provided_documents`,
`tool_results`, `workplace_assistant`, `customer_service`, and
`creative_writing`.

Either namespace may also use any normalized lowercase MIME type as its value.
For example, use `context:application/json`, `artifact:application/xml`, or
`artifact:text/csv`. Prefer a MIME type for a concrete document or data format;
use the named values above for semantic classes such as source code, prose, or
workspace state.

This distinction is deliberate. A task that interprets a JSON document and
writes an updated JSON document may use both `context:application/json` and
`artifact:application/json`. A math task whose answer is wrapped in JSON by one
rendering has neither semantic tag; that lowering alone receives `result:json`.

### `interaction` and `state`

Use `interaction:single_tool_call`, `interaction:terminal`,
`interaction:multi_action_workflow`, `interaction:conversation_revision`, or
`interaction:ordered_steps` only when the pattern is part of the task's
substance. Do not tag ordinary one-turn tasks as `single_turn` merely to fill a
slot.

Use `state:workspace` for tasks that require a mutable repository or filesystem
state, `state:provider` for a persistent provider world, and `state:cross_app`
when work crosses persistent application state. Do not tag a stateless answer
with `state:stateless`.

### `difficulty`

Judge difficulty against Snowball's roughly 2B active model, assuming the task
has the declared capabilities and normal time budget:

- `difficulty:easy`: it should usually succeed without unusual search or long
  chains of reasoning.
- `difficulty:medium`: it is a meaningful stretch; success needs sustained
  reasoning, careful tool use, or several dependent actions.
- `difficulty:hard`: it is unlikely to succeed reliably at that scale, even
  though a frontier agent might solve it.

Do not use source prestige, an evaluator's difficulty label, or frontier-agent
solvability as a substitute for this judgment.

## Copy-ready Luna prompt

```text
You are tagging TaskCompendium TaskSpecs for coverage analysis. For every task
in the assigned batch, read the model-visible instructions, declared
requirements, agent-visible resources, and source provenance. Assign a small
set of semantic coverage tags.

Return one JSON object mapping each TaskSpec id to a sorted JSON array of tags.
Every array must contain exactly one difficulty:* tag and exactly one shape:*
tag. Add one main competency:* tag, and add a second only when it captures an
independently important skill. Add subject, artifact, interaction, state, and
context tags only when they materially improve future sampling. Most tasks
should have three to six tags.

Use this standard vocabulary. Reuse an existing value when it fits. The
vocabulary is intentionally broad; it is not a checklist.

competency (choose one, exceptionally two):
- reasoning and knowledge: math, quantitative_reasoning, symbolic_reasoning,
  causal_reasoning, geometric_reasoning, spatial_reasoning, temporal_reasoning,
  number_theoretic_reasoning, scientific_reasoning, legal_reasoning,
  medical_science_reasoning, knowledge_recall, factual_recall,
  information_extraction, information_search, evidence_integration,
  constraint_satisfaction, instruction_following, problem_decomposition,
  planning, and multi_step_workflow
- software and technical work: algorithm_design, competitive_programming,
  software_implementation, software_engineering, debugging, shell_scripting,
  program_synthesis, code_comprehension, repo_navigation, test_interpretation,
  testing, structured_data_transformation, and data_transformation_reasoning
- tools and state: function_calling, tool_selection, tool_use,
  stateful_tool_use, state_tracking, and workflow_execution

shape (choose exactly one): answer, multiple_choice, calculation, explanation,
constrained_generation, structured_extraction, code_generation,
code_implementation, algorithmic_programming, environment_modification,
shell_workflow, predicted_action, stateful_domain, or
sequential_requirements.

artifact (only for a substantive deliverable): prose, numeric_answer, formula,
source_code, program, python_module, python_package, shell_output, spreadsheet,
report, slides, workspace_state, multifile_artifact, or native_action. You may
instead use any normalized lowercase MIME type, such as application/json,
application/xml, text/csv, application/pdf, or image/png, for a concrete
document or data format.

interaction (only when it matters): single_tool_call, terminal,
multi_action_workflow, conversation_revision, ordered_steps, or tool_feedback.

state (only when mutable state matters): workspace, provider, cross_app,
guardrailed, side_effect_sensitive, or irreversible_action.

context (only when it materially shapes the work): repository, filesystem,
source_code, provided_documents, spreadsheet, dataset, tool_results,
long_context, multi_document, cross_file, noisy, contradictory,
workplace_assistant, customer_service, or creative_writing. You may instead use
any normalized lowercase MIME type, such as application/json, application/xml,
text/csv, application/pdf, or image/png, for concrete input material.

subject (optional): use a material subject area or operational area such as
arithmetic, algebra, number_theory, geometry, calculus, probability, physics,
chemistry, biology, medicine, law, finance, business, algorithms, python,
machine_learning, data_science, filesystem, email, event_ticketing,
customer_relationship_management, or technical_operations. A subject is optional:
use one only for a broad, recognizable field, product, or operational area that
improves sampling. Do not use ordinary implementation details or generic
functionality such as name_handling, numeric_formatting, or data_type_inference as
subjects. Use dotted subtags for recurring specific functionality: a React-heavy
web task can use javascript.react and an integration problem can use
calculus.integration. Choose the most specific useful value, not both a broad
parent and its child. Add a new subtag only when it is likely to recur across task
families; do not mint a one-off subject tag that repeats prompt wording or names a
narrow source package. Do not use broad values such as general or software.

difficulty (choose exactly one): easy, medium, or hard.

You may add a new lowercase namespace:value tag only when it captures a useful
recurring distinction missing above. Use only these namespaces: competency,
shape, subject, artifact, interaction, state, context, difficulty.

Tag the task's semantic work, not its evaluator or packaging. Never tag Harbor,
ShellSim, Docker, a container image, a verifier, a judge, reward, hidden test,
source dataset, or submission extraction. Never add result:json, result:xml,
or result:file: result tags are assigned later by lowerings. Distinguish a
substantive structured deliverable (artifact:application/json,
artifact:application/xml, artifact:text/csv) from structured input
(context:application/json, context:application/xml, context:text/csv). A plain
answer wrapped in JSON is not semantic structured-data work.

Calibrate difficulty to Snowball's roughly 2B active model with the task's
declared capabilities. Easy should usually succeed; medium is a meaningful
stretch; hard is unlikely to succeed reliably. Do not calibrate to a frontier
agent.

Return only the JSON object. Do not include explanations or markdown.
```
