# Spike: Unified TaskSpecification IR and Lowerings

Project name is TaskCompendium

## Goal

Design and prove a common semantic representation for Marin post-training tasks.

The spike should establish this pipeline:

```text
existing task source
(TaskTrove / R2E-Gym / RLVR / future generators)
        ↓
TaskSpecification
        ↓
Protocol + optional TaskContext
        ↓
lowering
        ↓
Runnable Task
        ↓
Harbor
        ↓
Validation trial / grading result
```

The important abstraction boundary is `TaskSpecification`: it should describe the task faithfully enough that we can choose different valid renderings and execution strategies later without reconstructing task semantics from a Harbor archive, grader script, or source-specific dataset.

The deliverable is a small dataset of semantic specifications with lowerings that Harbor can load and execute. Validate execution with known-good and known-bad attempts. Training integration and target-model rollout studies are deferred. The eventual training consumer is marinskyrl and the target checkpoint is Snowball; neither is a dependency of this spike.

## Design principles

### Preserve semantics; vary form

A specification should capture what the agent is being asked to accomplish, the resources or world state it requires, and how success can be measured.

A concrete task may then choose among compatible:

* instruction renderings;
* answer formats;
* answer extraction rules;
* agent interfaces;
* filesystem conventions;
* execution environments.

Not every `TaskSpecification` supports every lowering. Preserve output-format requirements when they are part of the skill being tested: a task requiring valid JSON cannot become an XML task through rendering alone. Allow alternate wrappers only when they preserve the task contract.

### Require only the environment the task actually needs

Support three environment classes in the POC:

```python
EnvironmentRequirement =
    NoEnvironment(...)
  | ShellSimEnvironment(...)
  | DockerEnvironment(...)
```

Interpret these as semantic requirements, not as descriptions of the source dataset.

In particular, **do not infer that a TaskTrove task requires Docker merely because the current TaskTrove representation contains a Dockerfile**. The cleanup pipeline currently carries a Dockerfile through its converted task representation and emits it into runnable Harbor tasks.

Use the existing TaskTrove audit as evidence when classifying environments. It already records ShellSim suitability and related fields such as `shellsim_now`, `shellsim_with`, and verifier mechanism.  Do not treat those labels as unquestionable ground truth.

### Distinguish validity from usefulness

A task can be valid and still be poor RL data.

Validity asks whether:

* the task is well specified;
* the oracle succeeds;
* null/empty answers fail;
* perturbations that should fail do fail;
* there is no answer leakage;
* the verifier represents the instruction reasonably;
* the environment is actually sufficient.

Training usefulness asks whether it is useful for **a roughly 2B-active target policy**, which is the relevant scale for Snowball.

Do not classify something as trivial merely because Sonnet, Codex, or another frontier model solves it easily.

Difficulty should ultimately be measured with target-sized model rollouts, outside this spike. Keep model-relative usefulness assessments separate from the canonical specification and importer rejection ledger.

### Bad tasks may be dropped

Conversion is not a preservation exercise.

The updated TaskTrove cleanup intentionally rejects tasks that require heuristic answer recovery, repository reconstruction, replacement tests, or invented subjective grading contracts, while also acknowledging that some weak or suspect retained tasks remain.

Importers may reject an entire source or individual rows.

## 1. Core abstractions

### `TaskSpecification`

The canonical semantic IR.

```python
@dataclass(frozen=True)
class TaskSpecification:
    schema_version: str
    id: str

    instructions: str
    answer_requirements: AnswerRequirements

    environment: EnvironmentRequirement
    resources: tuple[ResourceSpec, ...]

    verifier: VerifierSpec
    verifier_runtime: VerifierRuntime

    metadata: TaskMetadata
```

The exact field layout is part of the spike; this is illustrative. `AnswerRequirements` preserves constraints intrinsic to correctness, such as valid JSON under a specified schema, literal text constraints, or a final environment state. Incidental answer wrappers belong in `Protocol`.

A `TaskSpecification` should contain enough information to:

1. understand the semantic task;
2. determine compatible lowerings;
3. grade a successfully lowered task;
4. reproduce its resources.

It should **not** contain incidental conventions such as “write your math answer to `/app/answer.txt`” unless that convention is itself part of the semantic task.

### `EnvironmentRequirement`

This describes the agent-accessible world needed by the task. It does not describe the runtime needed to grade an attempt. POC variants:

```python
@dataclass(frozen=True)
class NoEnvironment:
    pass
```

For QA, math, MCQA, many instruction-following tasks, etc.

```python
@dataclass(frozen=True)
class ShellSimEnvironment:
    limits: ShellSimLimits | None = None
```

For tasks genuinely supported by ShellSim’s modeled shell/filesystem/Python environment. ShellSim is a deterministic in-process shell with persistent VFS/session state and explicit resource limits, so it is meaningfully different from both no environment and a real Docker container.

```python
@dataclass(frozen=True)
class DockerEnvironment:
    image: ResourceRef | ImageSpec
    requirements: DockerRequirements | None = None
```

For repository work, native compilers, real packages, system dependencies, etc.

### Verifier runtime and resource visibility

Declare verifier execution requirements separately: host Python, an isolated container, or an external judge endpoint as appropriate. A `NoEnvironment` task may still require a sandboxed verifier or an LLM judge.

Use one resource list with explicit role and placement. Distinguish agent-visible inputs, verifier-only resources, and validation-only oracle material. Resource visibility must be enforced by the lowering; hidden tests, private reference answers, and oracles must not be included in the agent-visible prompt or filesystem. Resources needed by multiple roles declare those roles explicitly. The full specification is trusted input to the lowering, never an agent prompt; derive an explicit agent-visible projection.

State-based verifiers receive the final candidate state or a specified snapshot/patch applied to a clean grading environment. The agent must not be able to modify the trusted verifier or forge its result. Keep this execution policy separate from what constitutes a correct answer.

### `Protocol`

How the task is presented and the submission is recovered.

```python
@dataclass(frozen=True)
class Protocol:
    interaction: Chat | ChatWithTools
    rendering: InstructionRendering
    submission: SubmissionProtocol
```

Examples of `interaction`:

```python
Chat()
ChatWithTools(tools=(TerminalTool(),))
```

The interaction declares the agent-visible interface. Tool contracts declare their required capabilities; ShellSim or Docker supplies those capabilities when supported. `ChatWithTools` does not imply Docker. Keep Terminus-2 and mini-SWE-agent as selectable agent implementations in the lowering configuration, with compatibility checked against the interaction and tool contracts.

Do not encode `Harbor` as the protocol. Harbor is the execution framework that can run multiple agent implementations; the current Harbor fork already contains Terminus-2 and many installed agents including mini-SWE-agent, Codex, Aider, Claude Code, etc.

### `TaskContext`

Keep this deliberately small in the POC.

It may eventually cover:

* prior state;
* distractors;
* multiple tasks;
* interruptions;
* task interleaving.

Do not spend meaningful spike time designing this. A single isolated task is sufficient for almost all POC examples.

### `Task`

The concrete executable output:

```python
task = lower(
    specification=spec,
    protocol=protocol,
    agent=agent_implementation,
    context=context,
)
```

A concrete task retains provenance:

```text
TaskSpecification hash/id
Protocol id/version
agent implementation/version
resolved verifier runtime and judge model/version
lowering implementation/version
source dataset/revision/row
```

Each supported POC lowering must produce a Harbor task package plus the agent/environment configuration needed to run it. Specify the Harbor fork revision and any required adapters in the export manifest.

## 2. Separate verifier semantics from submission extraction

This is a core design requirement.

The current TaskTrove verifier representation often includes an `output` path directly in verifier specs.  For the semantic IR, factor:

> **What is correct?**

from:

> **Where/how did the agent submit its answer?**

For example:

```python
MathVerifier(
    expected="3/4",
    math_type="scalar",
)
```

can be combined with:

```python
AssistantFinal(
    extractor=PlainText(),
)

AssistantFinal(
    extractor=BoxedLatex(),
)

AssistantFinal(
    extractor=JsonPath("$.answer"),
)

AssistantFinal(
    extractor=XmlPath("/answer"),
)

FileSubmission(
    path="/app/answer.txt",
    extractor=PlainText(),
)
```

Each extractor returns a canonical candidate value that is then passed to the same `MathVerifier`. Preserve the raw submission for verifiers that assess literal output constraints. Define ambiguous or malformed submissions as extraction errors; do not recover answers heuristically.

For state-based tasks:

```python
PytestVerifier(...)
```

the verifier directly examines final environment state and no textual answer extraction is required.

This separation is how the POC demonstrates that:

```text
same semantic problem
+ different answer conventions
→ same semantic grading
```

## 3. Serialization

### Canonical representation

A `TaskSpecification` must serialize to a **single JSON document**.

Example:

```json
{
  "schema_version": "0.1",
  "id": "r2egym/1234",

  "instructions": "Fix the bug described below...",
  "answer_requirements": {"kind": "final_state"},

  "environment": {
    "kind": "docker",
    "image": {
      "uri": "...",
      "digest": "sha256:..."
    }
  },

  "resources": [
    {
      "kind": "artifact",
      "roles": ["agent_input", "verifier_input"],
      "path": "/app/repository",
      "uri": "...",
      "sha256": "..."
    }
  ],

  "verifier": {
    "kind": "pytest",
    "must_pass": ["..."],
    "must_not_break": ["..."]
  },

  "verifier_runtime": {
    "kind": "isolated_container",
    "image": {"uri": "...", "digest": "sha256:..."}
  },

  "metadata": {
    "source": "r2e-gym",
    "competencies": ["software-engineering/debugging"],
    "task_shape": "environment-modification"
  }
}
```

We should use Parquet to store the dataset at scale.

### Dataset representation

At scale:

> **one `TaskSpecification` per Parquet row**

Use nested Arrow structs/lists where practical rather than putting an opaque JSON blob into a string column.

It is acceptable to denormalize common indexing fields into top-level columns for filtering/mixing:

```text
source
competency
task_shape
environment_kind
verifier_kind
...
```

but there should still be one logical self-contained task record.

### Resources

Small resources may be embedded.

Large resources should use immutable references:

```python
ResourceRef(
    uri="...",
    sha256="...",
)
```

These may be distinguished in the schema as different columns if that is easy.

Do not put repository tarballs or Docker images directly into every row. Docker images must be pinned by digest. Tags may be retained as descriptive metadata, but even specific tags can move.

Schema versioning is required from the beginning. JSON and Parquet must round-trip to the same logical specification; define canonical hashing independently of Parquet encoding and keep mutable rollout assessments outside that hash.

## 4. Verifier model

Reuse Russell’s verifier ontology in 9061 rather than inventing a parallel one.

The current cleanup already has typed modes for MCQ, math, numeric, exact match, JSON/XML/CSV structural checking, IFEval, stdio, pytest, junit, Go tests, LLM judge, and arbitrary scripts.

Use this ontology for the representative modes required by the selected families and section 14. Other existing modes do not require new implementations or execution evidence in this spike. Reject unsupported modes explicitly.

The ontology should remain lightweight. Implementations may simply call the existing verifier package.

## 5. LLM-as-judge POC

LLM judge should be a first-class verifier, not hidden inside a shell script.

The current verifier already supports:

* reference-answer judging;
* checklist/rubric judging;
* deterministic constraints before the LLM call;
* exact-match short circuits;
* OpenAI-compatible judge endpoints;
* an explicit model field.

The new representation should make judge configuration more explicit.

For example:

```python
@dataclass(frozen=True)
class LLMJudgeVerifier:
    rubric: JudgeRubric
    view: JudgeView
    judge: JudgeModelPolicy
```

```python
@dataclass(frozen=True)
class JudgeModelPolicy:
    provider: str | None
    family: str | None  # e.g. "claude"
    model: str | None  # e.g. "sonnet-4.6"
    size_class: Literal["small", "medium", "large"]
    samples: int = 1
    aggregation: str = "mean"
    # generation parameters
    temperature: float = 0.0
```

The **model size must be explicit**, because judge behavior is not invariant to judge capability.

Define what evidence the judge receives:

```python
JudgeView(
    transcript=True,
    files=("report.md",),
    reference_context="reference.txt",
)
```

Do not automatically expose the full environment or hidden gold data.

POC requirements:

* deterministic fake judge for unit tests;
* optional live OpenAI-compatible integration test; without it, label judge-family validation as plumbing-only and live grading quality as unvalidated;
* judge infrastructure failure must be represented separately from reward 0;
* preserve auxiliary judge metrics/reasons where available;
* record the resolved provider, model identifier/version where available, and generation settings for every judge invocation. A size class or moving alias alone is insufficient provenance.

## 6. Source importers

The POC should operate on real source data, not toy fixtures alone.

### A. Cleaned TaskTrove

Use the TaskTrove cleanup from [#9061](https://github.com/marin-community/marin/pull/9061) as the starting corpus, pinned to release `2026.09.10.8`:

```text
s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.8/
```

Read `manifest.json`, `ledger.parquet`, `report.md`, and representative rows from `tasks/part-00000.parquet`. Record the release, source row, and converter/verifier code revision in provenance. Resolve the code revision corresponding to this release before conversion; do not follow the PR head implicitly. Do not download or rewrite the full corpus for the spike.

The cleanup currently retains roughly 1.45M tasks across many sources/verifier types and already has source-specific converters plus a rejection ledger.

Do **not** rewrite the whole corpus during the spike.

Select task families covering:

* MCQA;
* exact QA;
* math;
* structured output;
* one judge-based family;
* one simple shell/filesystem family;
* two basic coding families. One with Python and one with a compiled language.

For each, rewrite the source into `TaskSpecification`.

Environment selection should be reassessed:

```text
existing Docker task
        ↓
does it semantically need:
    no environment?
    ShellSim?
    Docker?
```

### B. R2E-Gym

Import a representative slice, perhaps 20–100 tasks.

Purpose:

* demonstrate ability to force a real Docker/repository environment;
* exercise environment modification;
* exercise real test-based verification;
* provide a coding task family distinct from TaskTrove.

Do not attempt a full R2E-Gym migration in the spike.

### C. Existing non-agentic RLVR

Add one simple existing Snowball math source as a third independent importer.

This helps prove the IR is not “TaskTrove v2.” Importing existing math data does not require running Snowball or integrating marinskyrl.

## 7. Lowering/rendering matrix

The POC must show multiple concrete renderings of the same semantic specs.

At minimum:

| Semantic task     | Rendering / submission |
| ----------------- | ---------------------- |
| MCQA              | plain `B`              |
| MCQA              | `{"answer":"B"}`       |
| MCQA              | `<answer>B</answer>`   |
| exact QA          | plain chat             |
| exact QA          | JSON                   |
| math              | plain answer           |
| math              | `\boxed{...}`          |
| math              | JSON                   |
| math              | file `/app/answer.txt` |
| structured output | JSON                   |
| structured output | XML                    |

The structured-output rows apply to specifications whose semantic result permits both encodings. Format-specific tasks retain their required encoding. The goal is to prove the abstraction.

### Compatibility checking

Invalid combinations must fail during lowering.

For example:

```text
Chat()
+
FileSubmission("/app/answer.txt")
```

is invalid because the agent has no way to write the file. A valid file-output lowering selects `ChatWithTools` and a filesystem provider. A task with `NoEnvironment` has no intrinsic world requirement; its selected protocol may introduce one, such as ShellSim for file submission.

Likewise a repository-editing specification cannot lower to plain `Chat`. Check interaction capabilities, tool/environment compatibility, answer requirements, and verifier runtime availability separately.

## 8. Harbor integration

Harbor must load and execute each supported POC lowering, including no-tool and ShellSim paths. A package that merely serializes successfully is not sufficient.

Demonstrate at least:

### Direct/no-tool tasks

Find or implement a minimal Harbor path for:

```text
prompt → model response → extraction → verifier
```

Do not put MCQA/math into Docker solely because Harbor happens to support Docker.

### Terminus-2

Use as the primary terminal agent implementation for compatible `ChatWithTools` protocols.

### mini-SWE-agent

Use as a second coding agent implementation to demonstrate agent-interface variation.

### ShellSim

Determine the least-invasive Harbor integration.

Preferred architecture:

```text
TaskSpecification(ShellSim)
      ↓
ShellSim lowering
      ↓
Harbor environment/agent adapter
```

Standalone ShellSim execution is an intermediate milestone. Completion requires the smallest Harbor environment/agent adapter needed to execute the selected ShellSim examples. If that exceeds spike scope, report the missing adapter as an explicit gap and revisit scope before claiming completion.

Do not map ShellSim tasks to Docker just to make the POC superficially uniform.

## 9. Validation and round-trip tests

For every imported family:

```text
original source task
    ↓
TaskSpecification
    ↓
lowered task
```

should preserve intended grading semantics.

Require:

* oracle succeeds;
* empty/null attempt fails;
* at least one semantically meaningful perturbation fails;
* hidden gold is not agent-visible;
* compatible alternate answer renderings produce equivalent semantic scores;
* extraction errors are distinguishable from verifier failures;
* infrastructure failures are distinguishable from reward zero.

For LLM-judge families, the fake judge establishes execution and evidence-delivery behavior only. Review preservation of the source rubric and judge view separately. Known-good/bad semantic grading claims require a live judge check; absence of that check is recorded as unvalidated and does not block the Harbor integration milestone.

For code tasks, compare original and lowered grading on both:

* known-good solution;
* broken solution.

For existing TaskTrove families, compare against the cleaned verifier where possible. Load exported packages through the pinned Harbor entry point and execute known-good, empty, and meaningfully broken attempts. Scripted or replayed attempts are sufficient; target-model inference is not required. Preserve distinct outcomes for graded attempts, extraction errors, and verifier/infrastructure errors, with reward present only when grading produces one.

## 10. Deferred training and target-model suitability

The eventual training consumer is marinskyrl, using the Snowball checkpoint at roughly 2B active parameters. Training integration, checkpoint serving, target-model rollouts, and empirical difficulty classification are outside this spike.

When resumed, store suitability assessments separately, keyed by specification hash, checkpoint, protocol, and rollout settings. Measure pass@1, pass@k, reward statistics, error rates, and trajectory lengths. Use those results to select training data without changing a valid source specification or its import outcome.

Static review may flag suspected triviality or weak training signal for later measurement. Frontier-model success alone is not grounds for rejection.

## 11. Importer rejection policy

Every importer returns:

```python
TaskSpecification | Rejected
```

with structured reasons.

Examples:

```text
broken_grader
gold_leakage
underspecified
null_answer_passes
unrecoverable_source
unsupported_environment
duplicate
```

Importer rejection concerns task validity and faithful representation. Keep `trivial_for_target` and `poor_training_signal` in separate usefulness assessments, with supporting evidence when available. They are not static importer rejection reasons.

Do not invent new answer semantics, graders, or reconstruction heuristics solely to increase retention.

## 12. Agent execution model

The **lead agent** owns the architecture.

It is responsible for:

* core Python types;
* schema;
* JSON/Parquet serialization;
* verifier interfaces;
* protocol/lowering interfaces;
* compatibility checking;
* Harbor integration;
* end-to-end tests;
* design decisions;
* integration reviews.

### Delegate source rewrites

The lead agent should delegate actual source-family rewrites to **fresh GPT-5.6 Luna chats with Medium reasoning**.

Each rewrite gets:

* a fresh chat;
* a fresh self-contained prompt;
* one source family or narrowly related converter group.

Parallelize source rewrites after the initial schema and lowering contracts pass the vertical-slice checks in section 13.

Each delegated prompt must include:

1. the current `TaskSpecification` schema;
2. relevant original source/converter files;
3. representative task examples;
4. verifier definitions;
5. environment classification rules;
6. required tests;
7. rejection rules.

Explicitly instruct each Luna worker:

> Do not assume a task is good because TaskTrove cleanup retained it.

> Do not assume Docker is required because the source is packaged in Docker.

> Choose the least powerful environment that faithfully supports the task: none, then ShellSim where adequate, then Docker.

> Assess apparent triviality relative to a roughly 2B-active target model. Do not reject tasks merely because a frontier model solves them easily.

> If usefulness cannot be determined statically, preserve the valid task and mark it for target-model rollout measurement.

> Reject tasks with demonstrated validity defects: broken graders, underspecification, answer leakage, or missing semantics. Flag suspected weak training signal separately.

> Do not introduce heuristic answer recovery or new graders merely to save rows.

### Independent review

For a sample of converted families, use a **different fresh Luna/Medium chat** as reviewer, with fresh worker for each task.

Give it:

* original source examples;
* resulting `TaskSpecification`s;
* schema and environment rules.

Do not give it the author agent’s reasoning. Do not inherit prior conversation state.

Ask:

* Was semantic intent preserved?
* Is the verifier faithful?
* Is the environment stronger than necessary?
* Is anything leaked?
* Does the task look useful for a ~2B-active policy?
* If uncertain on difficulty, should it go to rollout measurement?

## 13. Suggested work split

First complete a vertical slice:

1. One math specification with two submission formats, separate verifier execution, and JSON/Parquet round trips.
2. One ShellSim task with explicit tool capabilities and resource visibility.
3. One Docker repository task with hidden tests and known-good/broken solutions.
4. Harbor loading and execution for those examples, with distinct extraction and infrastructure error results.

The lead establishes these interfaces and golden examples before broad source-family delegation. Then use this parallel decomposition:

```text
Lead:
  IR/schema + integration

Worker A:
  TaskTrove MCQA/exact

Worker B:
  TaskTrove math + independent existing Snowball math-source importer

Worker C:
  structured JSON/XML

Worker D:
  TaskTrove judge tasks

Worker E:
  simple shell → ShellSim candidates

Worker F:
  TaskTrove coding

Worker G:
  R2E-Gym importer

Worker H:
  Harbor direct-chat lowering

Worker I:
  Harbor Terminus-2 / mini-SWE lowerings

Worker J:
  serialization + Parquet

Worker K:
  verifier/extraction equivalence tests

Worker L:
  semantic review / rejection audit
```

The lead continuously merges only after schema and golden tests pass.

## 14. POC definition of done

The spike is successful when we have:

* a versioned `TaskSpecification` schema;
* JSON round-trip serialization;
* Parquet dataset serialization;
* TaskTrove importer for several representative families;
* R2E-Gym importer;
* one independent existing Snowball math-source importer;
* all three environment classes represented:

  * no required agent environment, exercised through `Chat`;
  * ShellSim;
  * Docker;
* verifier support for:

  * MCQA;
  * exact;
  * math;
  * basic coding tests;
  * LLM judge;
  * script fallback;
* answer/submission support for:

  * plain chat;
  * `\boxed{}`;
  * JSON;
  * XML;
  * file output;
* at least two different Harbor agent interfaces exercised for appropriate tasks;
* equivalent semantic grading across multiple renderings of the same specification;
* static validity checks, with plumbing-only LLM-judge evidence labeled explicitly;
* explicit agent/verifier/oracle resource visibility and separate verifier runtime requirements;
* rejection ledger with structured validity reasons;
* one small Parquet artifact of specifications and a reproducible export to Harbor task packages plus execution configuration;
* Harbor smoke executions of known-good and known-bad attempts for every supported lowering, including `Chat`, ShellSim, and Docker paths.

## Explicitly out of scope

For this spike, do not block on:

* marinskyrl integration or training;
* serving Snowball or running target-model suitability studies;
* migrating all of TaskTrove;
* migrating all of R2E-Gym;
* Mark’s generated-environment pipeline;
* broad competency coverage;
* preference optimization;
* sophisticated multi-task composition;
* interruption/resume;
* general long-horizon context machinery;
* a perfect universal verifier ontology;
* replacing Harbor.

The output of the spike is the **core semantic IR + several credible importers + several credible lowerings**, with real source tasks and Harbor execution evidence to assess whether semantics survive import and lowering.

## Implementation status (2026-09-11)

The working implementation is in `lib/taskcompendium` on `codex/taskcompendium`.
Its README documents installation, sample construction, execution, and limitations.
The package has an independent uv workspace and a dedicated safe CI workflow.

Implemented and locally exercised:

- Versioned semantic schema, canonical JSON/hash, nested Parquet, explicit resource
  visibility, protocol compatibility, and separate verifier runtimes.
- Real TaskTrove MCQ, exact, math, JSON/XML, reference-judge, shell, Python, and C++
  importers, plus pinned GSM8K import.
- Real Harbor chat, ShellSim, and Docker trials; Terminus-2 and mini-SWE-agent
  transport validation with scripted model responses.
- Source good/bad/empty checks, malformed extraction checks, independent Luna reviews,
  and structured rejection examples.
- Sample artifact: 41 specifications, including 20 R2E tasks, 81 generated exports,
  and two rejections. Run
  `examples/build_poc.py` to recreate the exports. Validation details are recorded in
  `examples/poc/validation.json`.

The `.8` cleanup producer was resolved from Iris job
`/power/iris-run-job-20260911-210153`: clean checkout
`ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2`. Its TaskTrove converter and verifier trees
match the inspected revision. The source release remains `.8`, including checks that
sampled archive Dockerfiles declare the pinned verifier revision.

R2E-Gym has 20 complete pinned source rows and observed-log parity checks. Compatible
verifier runtimes preserve the source `.venv` and use a separate Python 3.12
supervisor. Actual Harbor good/bad/empty trials pass for rows 0 and 1 with rewards
1/0/0. The remaining 18 rows have not been graded live. All 20 source-specific verifier
runtimes were built. The combined artifact passed JSON/Parquet hash checks, source
checksum verification, and all 81 export-manifest checks.

Local validation passes 184 safe Python tests, 36 Docker tests, one Rust test, type
checking, and required formatting checks. The new CI workflow has not run on GitHub;
changes remain uncommitted in the worktree.

Training, Snowball serving, target-sized rollouts, and live judge-quality assessment
remain deferred. The scoped implementation and local validation are complete.

## Agent-facing prompt policy

Canonical task instructions and every rendered lowering describe only the task,
required behavior, resources available to the agent, and the requested output.
They must never mention the existence of judges, verifiers, graders, rewards,
evaluation scores, hidden tests, or reference answers. Keep evaluation machinery
in the separate verifier/oracle fields and private resources. Rewrite source
boilerplate as direct requirements, preserving task semantics; do not delete
legitimate domain concepts merely because they contain words such as "score".
This rule applies to source-rewrite workers, prompt-generation templates, lowerings,
and independent prompt reviews. Audit regenerated prompts before publication.
