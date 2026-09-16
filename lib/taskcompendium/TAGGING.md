# TaskCompendium coverage tagging

Coverage labels describe semantic work for corpus analysis and sampling. Read
model-visible instructions, requirements, agent-visible resources, and source
provenance. Do not inspect private answers or verifiers to infer labels.

Assign exactly one `shape:*` tag and usually zero or one `competency:*` tag.
Add subject, artifact, interaction, state, and context tags only when they
distinguish tasks that would otherwise be grouped together. Keep tags lowercase,
unique, and sorted. Add recurring tags sparingly; do not paraphrase a prompt.

`TaskSpec.difficulty` is a separate integer from 1 to 10, estimating the model
capability needed for reliable success with declared tools and a normal time
budget. It is not a measured pass rate. Leave it unset when there is not enough
information to judge; every reviewed published example should have a value.

| Difficulty | Anchor |
| --- | --- |
| 1–2 | Routine for Snowball's roughly 2B-active scale. |
| 3–4 | A meaningful stretch at that scale. |
| 5–6 | More likely to require a larger capable model. |
| 7–8 | Frontier-level reasoning or agent execution. |
| 9–10 | A production frontier model or agent is needed for reliable success. |

Use the higher number for a harder task within each band. Do not copy source
prestige or an evaluator's difficulty label. Measured pass rates should record
model, lowering, tools, and budget separately and inform later recalibration.

## Competency

A competency is a skill exercised across subjects, answer formats, and
environments. MCQ is a shape: different MCQs can test recall or calculation.
Choose zero or one main competency, exceptionally two independent ones.

| Tag | Main work |
| --- | --- |
| `recall` | Retrieve a fact from learned knowledge. |
| `information_extraction` | Find specified information in supplied material. |
| `rule_application` | Apply a stated or known rule to a case. |
| `quantitative_reasoning` | Work through numerical relationships. |
| `causal_reasoning` | Infer causes or consequences. |
| `constraint_satisfaction` | Satisfy several interacting requirements. |
| `evidence_synthesis` | Combine information from multiple sources. |
| `planning` | Choose a sequence of dependent actions. |
| `state_tracking` | Reason about changes to persistent state. |
| `tool_selection` | Select an appropriate tool or action. |
| `debugging` | Diagnose a failure and determine a fix. |
| `algorithm_design` | Devise an algorithm, beyond writing code. |

Do not use subject-specific variants such as `medical_science_reasoning` or
`legal_reasoning`; use `subject:medicine` or `subject:law` instead. Tool access
belongs in requirements and the lowering, not in a generic `tool_use` tag.

## Other tags

`shape` describes the requested result. Use `answer`, `multiple_choice`,
`calculation`, `explanation`, `constrained_generation`,
`structured_extraction`, `code_generation`, `code_implementation`,
`algorithmic_programming`, `environment_modification`, `shell_workflow`,
`predicted_action`, `stateful_domain`, or `sequential_requirements` when it fits.

`subject` is optional. Use a recognizable topic, product, or operational area,
such as `medicine`, `law`, `calculus`, `python`, `email`, or `event_ticketing`.
Use `parent.specialty` for a recognizable specialization: `law.real_estate`,
`physics.particle`, `medicine.oncology`, `javascript.react`, and
`calculus.integration` are examples. Choose a broad recurring subject such as
`ergonomics` when the narrower topic is unlikely to be useful for sampling.
Store the most specific useful value, without its parent. Add multiple subjects
sparingly. Avoid incidental technologies and one-off details such as
`name_handling` or `lighting_design`.

`artifact` describes a substantive deliverable: `prose`, `numeric_answer`,
`formula`, `source_code`, `program`, `python_module`, `python_package`,
`workspace_state`, or `native_action`. `context` describes material input or
setting: `repository`, `filesystem`, `source_code`, `provided_documents`,
`tool_results`, or `workplace_assistant`. Both also accept normalized lowercase
MIME types. A task reading and writing JSON may have
`context:application/json` and `artifact:application/json`; a math answer
merely wrapped in JSON has neither.

Use `interaction:single_tool_call`, `interaction:terminal`,
`interaction:multi_action_workflow`, `interaction:conversation_revision`, or
`interaction:ordered_steps` only when interaction matters. Use
`state:workspace`, `state:provider`, or `state:cross_app` only for substantive
persistent mutable state.

Never tag a harness, container image, source dataset, verifier, reward, judge,
hidden test, or extraction mechanism. Output wrappers `result:json`,
`result:xml`, and `result:file` belong to a `Lowering`, not a `TaskSpec`.

## Copy-ready labeling prompt

```text
You are labeling TaskCompendium TaskSpecs for coverage analysis. Read each
specification's model-visible instructions, requirements, agent-visible
resources, and source provenance. Do not use private answers or verifiers.

Return one JSON object mapping each TaskSpec id to an object with exactly two
keys: "coverage_tags" (a sorted array of unique lowercase namespace:value
strings) and "difficulty" (an integer from 1 to 10). Include exactly one
shape:* tag. A competency is optional: use zero or one, exceptionally two.
Add other tags only when they improve corpus sampling.

Competencies: recall, information_extraction, rule_application,
quantitative_reasoning, causal_reasoning, constraint_satisfaction,
evidence_synthesis, planning, state_tracking, tool_selection, debugging,
algorithm_design. They describe cross-subject work. Medicine, law, math, and
programming are subjects, not competencies; tool availability is a requirement.

Shapes: answer, multiple_choice, calculation, explanation,
constrained_generation, structured_extraction, code_generation,
code_implementation, algorithmic_programming, environment_modification,
shell_workflow, predicted_action, stateful_domain, sequential_requirements.

Subject is optional. Use a recognizable recurring area such as arithmetic,
algebra, calculus, physics, chemistry, biology, medicine, law, finance,
business, algorithms, python, javascript, machine_learning, filesystem,
email, event_ticketing, ergonomics, or technical_operations. Use
parent.specialty for a recognizable specialization, such as law.real_estate,
physics.particle, or medicine.oncology. Keep only the most specific useful
value; add multiple subjects sparingly. Never create a one-off subject from a
prompt detail.

Artifact: a substantive deliverable such as prose, numeric_answer, formula,
source_code, program, python_module, python_package, workspace_state, or
native_action. Context: material input such as repository, filesystem,
source_code, provided_documents, tool_results, or workplace_assistant.
Both accept normalized lowercase MIME types for substantive formats, such as
application/json or text/csv. A mere output wrapper is not format work.

Interaction, only when central: single_tool_call, terminal,
multi_action_workflow, conversation_revision, ordered_steps.
State, only when central: workspace, provider, cross_app.

Difficulty: 1-2 routine for Snowball (~2B active); 3-4 a stretch for Snowball;
5-6 likely larger model; 7-8 frontier level; 9-10 production frontier model
or agent. Use the higher number for the harder task in a band. Do not copy
source difficulty labels.

Do not tag Harbor, ShellSim, Docker, a verifier, reward, judge, source dataset,
or submission extraction. Do not add result:* tags; lowerings add those.
Return only JSON, without commentary or markdown.
```
