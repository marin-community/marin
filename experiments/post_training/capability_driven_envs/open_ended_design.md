# Generic open-ended task authoring

## Scope

This package turns a fully hydrated curator input into an ordinary, calibrated
`TaskSpec`. It is domain-independent. Source acquisition, extraction, scientific
instructions, and runtime requirements are upstream inputs. Solver rollouts,
dataset lowering, comparative reward, correction, and reinforcement learning are
downstream consumers and are outside this package.

```text
upstream curator
  -> OpenEndedTaskBundle
  -> proposal planning and writing
  -> structured model authoring
  -> frozen TaskSpec construction
  -> pointwise control calibration
  -> accepted or rejected authoring record
```

A change of domain must require a new serialized bundle, not a production-code
change.

## Curator boundary

`OpenEndedTaskBundle` is fully hydrated. It contains:

- a human-authored brief describing intent, deliverable, family, and exclusions;
- exact solver-visible `TaskResource`s;
- compact private author evidence;
- environment requirements;
- source provenance, split identity, tags, and release status.

External references may appear in provenance but are not runtime dependencies.
The authoring campaign performs no joins or downloads. Source-specific extractors,
rights audits, row-selection rules, and domain policies run before this boundary.
They may encode task-specific requirements as bundle data, but Taskforge contains
no branches for a particular source or discipline.

Solver and private resources have disjoint paths. Only solver resources may be
copied into `TaskSpec.resources.worker`. Private evidence is internal authoring
input: it may ground the private reference, rubric, and controls, but it must not
be exposed in the solver instruction or workspace.

## Proposal boundary

Proposal generation uses the brief, provenance, and bounded previews of solver
resources. It first plans a coverage-guided portfolio and then writes one
canonical `TaskProposal` per supported slot. Locked fields preserve bundle
identity, resources, environment, solver-machine class, no-research policy, and
`llm_as_judge` verification. Unsupported or duplicate variants should become
explicit null slots.

A proposal is a blueprint, not a `TaskSpec` and not a completed task design.

## Generic author

The build program receives the immutable bundle input and proposal. A structured
model call produces one `OpenEndedTaskDesign` containing:

- final solver instruction;
- private reference answer;
- weighted rubric with gates or caps;
- exactly one reference, partial, severe-negative, evaluator-injection, and
  verbosity control;
- an output-token budget.

Task-specific content comes only from the bundle. The design schema and author
prompt are shared across domains. The structured design is a persisted build-step
output, so its digest, cache identity, validation, and resume behavior use normal
Taskforge machinery.

The design is frozen before calibration. Candidate control answers cannot modify
its instruction, reference, or rubric.

## Materialization and privacy

The generic sink constructs a text-answer `TaskSpec` with:

- the authored instruction;
- the bundle's environment, source, tags, and solver resources;
- a VerifyIt judge grader over the frozen reference and rubric.

Validation checks program provenance, source and tag preservation, exact worker
resource equality, exact environment equality, judge/reference identity, and the
absence of private resource paths from solver context. The complete `TaskSpec`
contains verifier-private grading material by design; downstream exporters are
responsible for preserving worker/verifier visibility boundaries.

## Calibration and review

A configured pointwise judge scores every frozen control with bounded
concurrency. Each control declares an expected reward interval. Any result outside
its interval rejects the workpiece. Structural-only validation is useful for
local plumbing tests but is not sufficient for release.

Accepted output includes the serialized `TaskSpec` and validation evidence.
Rejected and null outcomes remain first-class campaign records.

## Zephyr execution

One outer Zephyr `map_shard` owns a complete bundle lifecycle. A shard performs
proposal generation, Taskforge workpiece execution, calibration, review, and
output serialization. Workers process multiple rows sequentially; campaigns do
not create one Zephyr shard per workpiece. Model endpoint, credentials, relay,
concurrency, placement, and storage are campaign configuration rather than task
configuration.

Durable object-backed proposal, event, round, and release storage is required
before relying on preemption recovery across workers. Temporary local roots are
acceptable only for smoke tests.

## Explicit non-goals

This package does not:

- acquire, interpret, or license source datasets;
- contain source-specific extraction or scientific constants;
- execute solver rollouts or prove the declared solver environment works;
- lower tasks into a training framework;
- construct rollout groups, comparative rewards, or corrected trajectories;
- make an unreviewed curator bundle publishable.

Those boundaries keep the authoring machinery reusable and make accepted
`TaskSpec`s ordinary downstream artifacts rather than objects coupled to one
campaign.
