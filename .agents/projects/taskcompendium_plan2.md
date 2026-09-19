# TaskCompendium capabilities refactor

## Goal

Represent a task's requirements independently of the harness that runs it. Produce
versioned semantic records, public rendered tasks, and validated Harbor exports
from the bounded TaskTrove, GSM8K, R2E-Gym, and sequential-task sample.

Tasks declare requirements; environments provide them; harnesses orchestrate them;
traces record what happened.

```text
TaskSpec (source/family)
    │ instantiate a source row once
    ▼
TaskSpecification (pinned semantic instance + private verification)
    │ render with an answer convention
    ▼
Task (public instructions, requirements, resources, submission)
    │ requires capabilities and initial state
    ▼
Environment ◄──── Harness ────► Model
    │               │
    │               ▼
    │           Rollout / Trace
    ▼
Private verification result
```

Harbor is the first execution adapter. Training, marinskyrl integration, Snowball
serving, and target-model suitability studies remain deferred. The implementation
lives in [lib/taskcompendium](../../lib/taskcompendium/README.md).

## 1. Semantic family, instance, and rendered task

`TaskSpec` defines a source or task family and its instantiation contract. A source
row is instantiated once into a `TaskSpecification`. That concrete record fixes
its instructions, provenance, resource identities, answer semantics, success
policy, and private verifiers before rendering. The existing `TaskSpecification`
name continues to mean a concrete semantic instance; it must not become a factory
that resamples when an export is requested.

`Rendering` chooses an answer convention for each step. Rendering creates a public
`Task` containing ordered model-visible inputs, public resources, required
capabilities and state, and submission contracts. Plain, boxed, JSON, XML, and file
variants retain the same pinned semantic instance and private correctness rule.
Intrinsic output constraints still limit which renderings are valid.

The public task never names an agent or harness. Adapter-specific function schemas
and native terminal protocols belong to execution configuration. A task may require
shell actions without specifying an OpenAI function name, Terminus-2, or replay.

## 2. Capabilities and state

Replace the semantic `none | ShellSim | Docker` choice with `TaskRequirements`:
capabilities describe the required operations, and state describes the required
starting resources and workspace. Concrete provider configurations live downstream.

An ordinary MCQA task needs no action capability. A file answer convention adds
file-access requirements to the rendered task. A shell pipeline needs the supported
shell and filesystem operations. A repository repair needs filesystem and native
process capabilities plus the exact repository and dependency state.

ShellSim and Docker are environment implementations. ShellSim may satisfy a portable
shell task; Docker can also satisfy it when the relevant command semantics match.
A native-process task needs a provider that can execute its dependencies. An
immutable source image can identify required initial state without declaring
Docker to be the task's harness or semantic category.

Check state identity as well as capability names. Substituting an unrelated image
with a shell would invalidate R2E-Gym even if it exposed the same action interface.
Preserve immutable image/resource identities, workdir, setup, and dependency
boundaries. Source packaging is evidence to inspect; infer the minimum honest
requirements from the actual work and verification contract.

Do not introduce a universal canonical tool schema. Define enough capability
matching to reject incompatible execution. Environment adapters expose concrete
actions using the selected harness's protocol.

## 3. Environment, harness, and trace ownership

The environment owns backing state and supplies the task's required capabilities.
Private verification executes on the environment side, potentially in a separate
container. The pinned specification fixes correctness; changing providers or
harnesses cannot change the verifier, its evidence, or expected values.

The harness runs the model/environment loop. Harbor's direct chat, tool chat,
replay, and native agent adapters choose how to deliver instructions, expose tools,
collect submissions, and preserve context. A thin adapter may translate wire
formats while leaving the public task unchanged.

A rollout records messages, actions/results, model identity and version, available
tokens/logprobs, task and rendering identity, execution provenance, and verification
status. Responses-style records may be used as a wire or trace representation;
they do not define task or environment types. Do not build a competing lifecycle
or trace runtime for this spike; retain backend trajectories and result evidence.

## 4. Ordered requests and private verification

Every `TaskSpecification` has a nonempty tuple of `StepSpecification` records.
Each step has `instructions`, intrinsic `answer_requirements`, public/private
resources, a `context_requirement`, and its private tagged verifier. Shared resources
are available initially; step resources are released at their step. The public
rendered task strips verifier and oracle material.

The success policy is explicit: all required steps, final, or mean. Preserve
per-step outcomes and reject policies the selected backend cannot express. The
pinned Harbor supports final and mean; multi-step all-required-steps is rejected.
A missing step or extraction/infrastructure failure suppresses the aggregate;
a valid incorrect answer remains a score of zero.

Keep fixed sequential instructions. Current instruction plus persistent workspace
may suffice for coding extensions; a request to revise a previous answer requires
prior conversation. Native agent resumption is currently unsupported by the pin;
the direct chat and shell-tool adapters can retain history. Never disclose future
instructions to provide continuity.

Reactive user simulators remain deferred. They are semantically tasks, but require
a future conversational environment interface beyond a static task plus tools.
Harbor's pinned tau-bench integration is evidence for that later design, not a
reason to force simulated users into the initial capability interface.

---

# 5. Verifier semantics

Keep verification independent of answer representation.

For example:

```python
MathVerifier(
    expected="3/4",
)
```

means that the semantic answer is mathematically equivalent to `3/4`.

It should not contain:

```text
/app/answer.txt
\boxed{}
$.answer
<answer>...
```

Those belong to rendering/extraction.

For the POC, support at least:

```text
MCQ
Exact
Math
basic code tests / pytest / stdio
LLMJudge
Script
```

Reuse the TaskTrove verifier implementations pinned by the existing spike's
[package configuration](../../lib/taskcompendium/pyproject.toml)
wherever possible rather than designing a competing grading stack.

`Script` remains the escape hatch.

---

# 6. Rendering

`Rendering` selects a submission contract: `AssistantFinal`, `FileSubmission`, or
`FinalState`. The first two contain an extractor such as `PlainText`, `BoxedLatex`,
`JsonPath`, or `XmlPath`. The same answer may be returned as plain text, a boxed
expression, JSON, XML, or file contents when intrinsic answer requirements permit.

Rendering may add the filesystem capability needed for a file submission. It may
not change the objective, required knowledge, initial world, or private correctness
rule. A JSON-schema task still requires JSON even if an outer answer convention is
available elsewhere in the dataset.

Keep agent selection, native tool protocols, environment creation, retries,
persistent sessions, and trial configuration in the execution adapter. Record
rendering identity alongside the pinned specification hash.

---

# 7. Extraction

Extraction translates a concrete model submission into the canonical object the verifier expects.

Examples:

```python
PlainText()
BoxedLatex()
JsonPath("$.answer")
XmlPath("/answer")
FileSubmission(path="/app/answer.txt", extractor=PlainText())
```

So:

```text
model output
    ↓
extract
    ↓
canonical candidate
    ↓
semantic verifier
```

Malformed output should produce:

```text
extraction_error
reward = null
```

rather than silently becoming an incorrect answer with reward 0.

A semantically wrong but successfully extracted answer should produce reward 0.

That distinction should survive into rollout metadata.

The diagnostic `reward = null` does not make malformed agent output an
infrastructure failure or an attempt that can be discarded. If an agent violates
an explicit submission convention, a future training consumer must count it as
an unsuccessful attempt and apply an explicit penalty policy. Preserve the
extraction status separately. Infrastructure failures require separate handling;
training policy remains outside this spike.

For state-based verifiers such as repository tests, extraction may be absent; the verifier inspects final state.

---

# 8. Agent-facing instructions must not reveal grading machinery

As a general rule, prompts should never tell the model:

* that there is a verifier;
* how hidden tests work;
* what reward it receives;
* which judge evaluates it;
* hidden expected values;
* grader implementation details.

Agent-visible instructions should state only:

* the actual task;
* semantically necessary constraints;
* legitimate resources;
* genuinely required submission conventions.

Examples:

Good:

> Return the answer as JSON with an `answer` field.

Bad:

> The verifier parses the `answer` field from your JSON.

Good:

> Fix the bug while preserving existing behavior.

Bad:

> Hidden pytest tests will determine your reward.

If the task itself is naturally “make the test suite pass,” tests may of course be mentioned.

The invariant is:

> **Describe success, not how we detect success.**

Add tests/lints ensuring lowering does not accidentally interpolate hidden verifier data into prompts.

---

# 9. Multi-step / multiple user turns

Support scripted multi-step tasks directly in `TaskSpecification.steps`.

Example:

```python
TaskSpecification(
    requirements=TaskRequirements(...),
    steps=(
        StepSpecification(
            instructions="Implement support for X.",
            verifier=...,
        ),
        StepSpecification(
            instructions="Now extend it to support Y.",
            verifier=...,
        ),
        StepSpecification(
            instructions="Change the behavior so Z...",
            verifier=...,
        ),
    ),
)
```

This should lower naturally into Harbor's multi-step task representation.

## Context requirements and conversation transport

The same multi-step semantic task could potentially be run as:

```text
step 1 → fresh conversation
step 2 → fresh conversation, same workspace
step 3 → fresh conversation, same workspace
```

or:

```text
user step 1
assistant ...
user step 2
assistant ...
user step 3
assistant ...
```

when the selected adapter can retain the required context.

Conversation transport and native session resumption are execution choices.
Their compatibility depends on the step's context requirement. Fresh
conversations with the same workspace are valid only when the next instruction
and workspace contain all necessary information. A chat request such as
“revise your previous answer” requires the prior agent-visible exchange.

The lowering must either supply the required context through a supported Harbor
mechanism or reject the execution configuration. It must not silently drop
context, disclose future steps, or include hidden verifier information.

## Out of scope

Adaptive user simulation, where the next user message is generated dynamically based on the assistant's previous response, is out of scope for the spike.

If/when τ-style user simulators become important, extend from evidence rather than designing for them prematurely.

---

# 10. Serialization and Harbor export

Store one pinned `TaskSpecification` per nested Parquet row. Canonical JSON hashing
includes the schema version and is independent of Arrow encoding. Preserve source
revision, row identity, importer revision, checksums, and private verifier data.
Public rendered tasks and private specifications have separate schemas. Regenerate
older artifacts when the schema changes; do not add compatibility shims.

A rendered `Task` contains the model-visible instructions, public resources,
requirements, and submission contracts. It contains no verifier, oracle material,
selected environment implementation, or harness configuration. Keep private
verification information in its own dataset column and export-side files.

The Harbor adapter renders the semantic instance, checks its requirements against
the selected environment and action interface, then writes native Harbor artifacts.
`HarborExecutionConfig` owns the selected agent, provider configuration, explicit
tool bindings, timeouts, and conversation handling. These choices never become
semantic task fields. The current `lower_to_harbor` entry point accepts the pinned
specification, one rendering per step, execution configuration, and destination.

Reject incompatible exports before writing files: a file submission without file
access, a repository task without the required state and process capabilities,
a fresh conversation when prior exchanges are required, or an intrinsic JSON task
with an XML answer convention. More capable providers may satisfy the same task,
but must preserve its state identity and observable action semantics.

Harbor owns execution, step lifecycle, and trial aggregation. The adapter supplies
private verifier integration and translates submissions and results. It must not
create a second scheduler, trial runner, or generic episode framework.

---

# 12. Source importers

Use real sources.

## TaskTrove Clean

Import representative families covering:

* MCQA;
* exact;
* math;
* structured output;
* LLM judge;
* basic shell/filesystem;
* basic coding.

Do not migrate the full corpus during the spike.

Infer capability and state requirements from the source task. Source Docker packaging alone does not justify a native-process requirement.

Importers may reject rows.

Track confirmed defects and repair principles in the
[problematic-task ledger](../logbooks/taskcompendium-problematic-tasks.md) for the
next cheap-model sweep.

## R2E-Gym

Import representative repository-repair examples.

Keep representative Orange3 and SymPy examples to exercise distinct repository dependencies and test protocols.

Do not attempt full migration.

## Existing RLVR

Include at least a small non-agentic source such as GSM8K/math to prove this is not merely TaskTrove repackaging.

## Multi-step

Add at least one small multi-step source or synthetic/hand-transcribed example modeled after EvoCode-style sequential requirements.

The goal is to prove:

```text
TaskSpecification.steps
→ Harbor multi-step task
```

not to create a new multi-turn benchmark.

---

# 13. Rendering fan-out experiment

Demonstrate multiple Harbor tasks generated from the same specification hash.

Required examples should include some subset of:

```text
MCQA:
  plain
  JSON
  XML
  file

Math:
  plain
  boxed
  JSON
  XML
  file

Environment implementations where compatible:
  direct/no-env
  ShellSim
  Docker
```

The same verifier semantics should survive all compatible variants.

---

# 14. LLM judge

Keep LLM judge first-class enough to record materially relevant semantics:

```python
LLMJudgeVerifier(
    rubric=...,
    view=...,
    judge_policy=...
)
```

Record:

* provider;
* model/family;
* size/capability class;
* temperature;
* number of samples;
* aggregation;
* what evidence the judge sees.

Record returned model identity/fingerprint when available.

Infrastructure failure must remain distinct from a valid score of zero.

Use a deterministic fixture judge in unit tests.

One real production-caliber judge calibration experiment can follow the spike; do not block core IR work on it.

---

# 15. Validity versus training usefulness

These are separate.

## Validity

A task is valid when, for example:

* instructions are coherent;
* oracle passes;
* empty/null answer fails;
* meaningful perturbations fail;
* hidden data do not leak;
* verifier represents the requested work;
* environment suffices;
* source can be represented without inventing semantics.

## Training usefulness (deferred)

Target-model rollouts and training integration are follow-up work.

A valid task may still be poor training data.

Difficulty should be evaluated relative to the intended target, approximately **2B active parameters** for Snowball-like work.

Do not use frontier-agent solvability as the criterion for triviality.

In that follow-up, collect target-model rollouts and measure:

```text
pass@1
pass@k
mean reward
reward variance
extraction failure rate
environment/runtime failure rate
trajectory length where relevant
```

Avoid hard universal difficulty thresholds in v0.

Roughly:

* near-100% pass@1 may be too easy for capability RL;
* zero pass@k may be too hard or broken;
* meaningful pass@k with lower pass@1 is attractive;
* apparently easy tasks can still be useful for protocol adherence or preservation.

---

# 16. Rejection policy

Every importer may return:

```python
TaskSpecification | Rejected
```

with structured reasons such as:

```text
broken_grader
gold_leakage
underspecified
null_answer_passes
trivial_for_target
poor_training_signal
unsupported_environment
unrecoverable_source
duplicate
```

Keep import-validity rejections separate from downstream training-selection
exclusions. `trivial_for_target` and `poor_training_signal` require recorded
selection evidence and do not make the semantic specification invalid.
`trivial_for_target` should normally rely on target-model evidence rather than
a frontier model's intuition. Do not require that evidence during this spike.

Do not:

* invent missing answer semantics;
* author arbitrary new subjective grading contracts just to save data;
* reconstruct broken repositories unless justified;
* write increasingly source-specific answer heuristics merely to increase retention.

Some TaskTrove tasks are still useless. Preserve the ability to say so.

---

# 17. Provenance

Every specification should retain enough information to answer:

> Where did this semantic task come from?

At minimum:

```text
source dataset
source revision
source row/key
importer version
source checksum where practical
```

Every lowered task should additionally retain:

```text
TaskSpecification id/hash
Rendering id/version
lowering implementation/version
Harbor/runtime version in run provenance
```

This allows semantic identity to survive multiple concrete renderings.

---

# 18. Implementation and validation

Update the data model, importers, serialization, rendering, Harbor compatibility
checks, generated schemas, prompt-generation guidance, and examples together.
Remove semantic provider/harness fields and migrate all call sites. Preserve the
problematic-task ledger and previously rejected source rows; this refactor does
not authorize a source-repair or difficulty sweep.

Acceptance checks:

- A pinned semantic instance produces multiple answer conventions without changing
  its verifier or source identity.
- The same rendered public task can use compatible harness configurations without
  being regenerated or acquiring harness fields.
- Compatible environment implementations preserve required capabilities and initial
  state; incompatible state or capabilities fail before export.
- Public task serialization contains no verifier/oracle data or selected provider.
- Plain, boxed, JSON, XML, file, and final-state extraction preserve distinct
  correct, incorrect, malformed, and infrastructure outcomes.
- TaskTrove, GSM8K, Orange3, SymPy, and both sequential examples round-trip through
  JSON and nested Parquet with source provenance intact.
- Safe tests and appropriate native Harbor/Docker checks validate the adapters.
  Record which tasks were actually executed and which only imported/exported.
- Regenerated Hugging Face task/lowering tables retain a separate private verifier
  column and distinguish semantic requirements from execution choices.

Publish the updated bounded artifact to
[open-athena/taskcompendium-spike](https://huggingface.co/datasets/open-athena/taskcompendium-spike)
after regeneration and validation. Preserve historical checkpoint links and report
the new revision and validation limits.

## Environment coverage audit

Before adding source integrations, use the
[NeMo Gym and TaskTrove overlap audit](taskcompendium-coverage/research.md) to
distinguish existing source coverage, conversion fidelity, and missing execution
patterns. The [first-wave implementation](taskcompendium-coverage/implementation.md)
adds four diagnostic source cases: instruction constraints, a code answer with
isolated hidden execution, predicted native calls without dispatch, and Workplace
with actual seeded domain actions. Binary and fractional instruction scoring are
separate semantic instances. Rendering variants share each instance's verifier.

Schema 0.5 uses private tagged verifier contracts and a provider action interface.
Public tasks declare an interface version and seed identity; execution selects the
adapter and supplies its private configuration. The provider owns mutable state,
and private state verification reads it directly. A predicted-call submission is
an output contract, not an executable tool capability. Public schemas must never
be derived from the target call, its arguments, batch size, or order.

## Deferred work

Do not add training, model serving, target-scale difficulty rollouts, adaptive user
simulation, arbitrary MCP/domain tools, full source migration, a universal verifier
ontology, or a second agent/trial framework. Source-family repairs use the ledger
and a separately scoped cheap-model sweep. Any delegated source rewrites and fidelity
reviews should use fresh GPT-5.6 Luna workers at Medium reasoning as previously
specified for that sweep.
