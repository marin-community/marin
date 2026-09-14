# TaskCompendium

Read root AGENTS.md and TESTING.md. This is an independent Python package with a
separate uv workspace, so it can be installed into Harbor without Marin's JAX stack.

Use `uv run --project lib/taskcompendium --group test pytest lib/taskcompendium/tests`
for safe package tests. Harbor tests require `--extra harbor`. Preserve the default
marker exclusions. Use the repository `infra/pre-commit.py` entry point for lint.

Reuse the pinned tasktrove-verify ontology and graders. Submission extraction and
agent resource visibility belong here; source answer recovery does not. Never
run imported scripts, tests, or submitted programs on the host verifier runtime.

## Agent-facing task instructions

Write the task request directly. `StepSpecification.instructions`, rendered prompts,
and public task resources must not disclose evaluation machinery: judges, judgers,
verifiers, graders, rewards, evaluation scores, hidden tests, or reference answers.
Keep these in verifier/oracle metadata and resources, never in agent-facing text.

Preserve actual requirements when removing source evaluation commentary. Use
"Write your answer to `/app/answer.txt`" instead of describing what checks that file.
Use "Fix the bug while preserving existing behavior" instead of explaining that
pytest will grade the solution. Required formats, file paths, observable behavior,
and user-requested tests remain part of the task.

Importers should rewrite known source boilerplate explicitly, without deleting
unrelated domain language such as credit scores or requests to verify reproducibility.
Prompt generation and lowering must not add evaluation commentary back. Review both
canonical instructions and every rendered variant before publishing; add regressions
for source templates that leak evaluation details. Keep source archives unchanged.


## Ordered steps

Schema 0.6 stores all requests in `TaskSpecification.steps`. Keep step-specific
answer requirements, private resources, verifier dependencies, and context needs
on each step. A `TaskSpecification` is a pinned semantic instance; `TaskSpec` is the
source/family instantiation interface. Instantiate once before rendering variants.
`Rendering` controls submission conventions; `HarborTaskBinding`
in `execution.py` controls the required environment shape, public tools, and conversation mode; Harbor selects the agent at launch.
Pass one rendering per step to `lower_to_harbor`.

Use Harbor's native multi-step lifecycle. Reject unsupported success policies or
conversation modes before export. Preserve per-step extraction and infrastructure
failures; never report an aggregate that hides a missing or failed step.

Executable verifier modes require `TaskTroveVerifier.runtime`. Trusted answer checks
and model judges must not declare a runtime. Judge model/service configuration
belongs in `TaskTroveVerifier.judge`; extraction belongs in `Rendering`.


## Capabilities and public tasks

Task requirements declare operations and initial state. Use `TaskRequirements` and
`WorkspaceState`, preserving immutable resource/image identities where needed.
Do not put ShellSim, Docker provider configuration, an agent name, or harness tool
bindings in the semantic record. Source Docker packaging alone does not establish
native-process requirements.

Render public `Task`/`TaskStep` records with instructions, public resources,
requirements, and submission contracts only. Never serialize private verifiers or
oracle data into public tasks. Keep the pinned verifier unchanged across provider
and harness choices. Environment compatibility checks must cover both supplied
capabilities and required state.

Provider implementations and tool bindings belong in `execution.py`. Harbor owns
interaction and lifecycle; adapters may translate wire formats. Do not add a
universal tool schema or force reactive user simulators into static task steps.

## Tool bindings

Require explicit `HarborTaskBinding.interaction`. Do not infer model tool
access from Docker or ShellSim. Shell bindings specify the exposed function name
and backend; native/replay bindings specify their harness interface and backend.
Reject incompatible bindings. Keep tool availability separate from submission
location and final-state verification. `ProviderToolBinding` binds a declared `ActionInterface` to a provider adapter.
Provider configuration and mutable seed data stay private; match the interface
version and seed identity before execution. Verify authoritative provider state,
never a model-reported state snapshot. The initial domain provider is Workplace;
arbitrary MCP adapters and reactive user simulators remain deferred.

`FinalActionSubmission` is an output contract for predicting native actions. Its
advertised functions come from the public source request, never the expected
answer. Prediction does not grant action capabilities or dispatch tools. Preserve
source-specific batch/extra-call rules privately in `PredictedActionVerifier`.

Use the tagged verifier variants instead of encoding every source as a TaskTrove
mode: `ConstraintVerifier`, `CodeAnswerVerifier`, `PredictedActionVerifier`, and
`ProviderStateVerifier`. Code answers are materialized only inside the private
checker container; their tasks do not require agent filesystem or process access.
