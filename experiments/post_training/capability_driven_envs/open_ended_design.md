# Open-ended scientific task generation and RL design

- **Status:** working design for review
- **Scope:** hydrated scientific-analysis tasks authored by Taskforge and consumed by beta's judge/corrector RL paths
- **First pilot:** bioinformatics analyses derived from expert-authored notebooks

This document consolidates the decisions behind the open-ended task work. It is
both a reminder of why the interfaces have their current shape and a checklist
for the remaining implementation. Statements marked **decided** are intended
invariants. Statements marked **proposed** still need review or evidence from the
local pilot.

## 1. The problem

We want science tasks whose useful answer is an analysis or argument rather than
a single objectively verifiable token. Examples include interpreting donor-level
immune-repertoire counts, assessing whether an annotation is supported, and
explaining limitations of an analysis.

A bare LLM score is too weak a reward contract. It can reward length, style,
unsupported confidence, evaluator-directed text, or conclusions that merely look
scientific. The intended system therefore combines:

1. curator-grounded private evidence;
2. a reference and rubric frozen before solver rollouts;
3. deterministic checks wherever possible;
4. model judgment for genuinely open-ended qualities;
5. comparative, group-relative reward across answers to the same task; and
6. durable provenance for replaying and auditing every reward.

Kimi's reported generative-reward approach is useful inspiration—especially
rubric-based product inspection, verifier grounding, comparative reward, and
verbosity control—but its training implementation is not public. This design is
ours and deliberately differs by freezing criteria before candidates are shown.

## 2. Goals and non-goals

### Goals

- Create contamination-conscious scientific tasks from expert-authored source
  material without exposing source conclusions to the solver.
- Preserve enough raw or sufficient data for the requested analysis to be real,
  including missingness, denominators, replication, and row-selection rules.
- Use the ordinary Taskforge proposal, workpiece, build, validation, and review
  lifecycle rather than a notebook-specific parallel framework.
- Produce canonical `TaskSpec` objects that can be exported through one generic
  Harbor/beta boundary.
- Generate auditable LLM reward and correction data for beta.
- Make retries idempotent and eventually runnable as one Zephyr `map_shard`
  lifecycle per input shard.

### Non-goals

- Taskforge is not a notebook-understanding or source-data curation system.
- Complete notebooks and original external datasets are not Taskforge runtime
  inputs.
- A source bundle is not itself a `TaskSpec`.
- Public benchmark questions are not training data. BixBench and
  `BixBench-Verified-50` remain held out.
- Model judgment is not called RLVR. Objective verifiers remain primary where
  they exist.
- Validation/adversary traces are not automatically exported as policy training
  data.
- The first pilot does not attempt process-level credit assignment over every
  tool call.

## 3. Terms and cardinalities

These objects must remain distinct:

| Object | Meaning | Cardinality |
|---|---|---|
| Source family | One underlying dataset/notebook family and split boundary | one family to many bundles |
| Curator packet | Human-reviewed extraction output before Taskforge loading | one packet to one bundle initially |
| `OpenEndedTaskBundle` | Fully hydrated source-specific Taskforge idea | one bundle to `N` proposal slots |
| `ProposalBatch` | Persisted plan and outcomes for all requested slots | one per bundle/config identity |
| `TaskProposal` | Canonical Taskforge construction blueprint | zero or one per successful slot |
| Workpiece | One proposal's resumable Taskforge lifecycle | one per non-null proposal |
| `TaskSpec` | One semantic task and private grading contract | zero or one accepted spec per workpiece |
| Trajectory | One policy attempt on one `TaskSpec` | many per accepted task |
| Cohort/prompt group | Comparable trajectories from one task and policy window | one group contains `K` trajectories |
| Scorepad | Judge inputs, pairwise outcomes, caps, and rewards for one cohort | one or more per cohort |
| Optimizer step | A training update over one or more prompt groups | unrelated to task count |

**Decided:** one bundle may yield several genuine task variants, but variant
slots must differ in scientific question or workflow rather than wording alone.

## 4. Trust and visibility boundaries

There are four relevant trust domains.

### 4.1 Curator-only source domain

May contain:

- the complete notebook;
- the original external dataset;
- notebook outputs and conclusions;
- source-system credentials; and
- manual audit notes.

None of these are automatically passed to Taskforge. A deterministic,
human-reviewed extractor emits a much smaller packet first.

### 4.2 Hydrated bundle domain

An `OpenEndedTaskBundle` contains every byte needed from this point onward:

```text
bundle_id
split_group
human-authored brief
solver_resources       # candidate solver-visible files
author_evidence        # private authoring/calibration files
source provenance
tags
```

**Decided invariants:**

- Hydrated means no unresolved downloads, joins, or source lookups.
- All runtime bytes are inline `TaskResource` objects.
- External URIs are provenance, not runtime dependencies.
- `solver_resources` and `author_evidence` cannot share paths.
- Private evidence participates in the bundle digest.
- All bundles from one source/dataset family remain in one train/dev/eval split.

### 4.3 Taskforge author/verifier domain

The source-specific author and validator may use private evidence to construct:

- a private reference answer;
- a weighted rubric;
- gates, caps, and disqualifiers;
- positive, partial, severe-negative, and adversarial controls; and
- narrowly scoped deterministic anchors.

Private evidence is not passed through generic `build_input`. Generic build
programs receive `OpenEndedBuildInput`, which contains public resources,
identity, provenance, split group, and tags only.

`Verdict.author_state` may persist compact identities and obligations. It must
not carry file bytes.

### 4.4 Solver domain

The solver receives only:

- the final instruction;
- `TaskSpec.resources.worker`;
- the declared no-network analysis environment; and
- ordinary tool observations produced in that environment.

It must not receive the source notebook, notebook conclusions, reference,
rubric, private controls, or author evidence. `TaskSpec` verifier resources stay
on the private grader side.

## 5. End-to-end architecture

```text
pinned source family + original data                 curator only
  ↓ license/runtime audit
  ↓ deterministic extraction + human review
curator packet
  ↓ load, hydrate, and verify stored content identity
OpenEndedTaskBundle                                  source-specific IdeaT
  ↓ OpenEndedProposalSource.propose(bundle, N)
  ↓ persist before any workpiece starts
ProposalBatch
  ├── null slot
  ├── failed slot
  └── TaskProposal
        ↓ stable workpiece ID
        ↓ triage
        ↓ bundle-aware author
        ↓ build with public OpenEndedBuildInput
        ↓ pointwise controls and solver trials
        ↓ review
accepted TaskSpec                                    canonical task boundary
  ↓ generic TaskSpec exporter
  ↓ K policy trajectories for the same task
trajectory cohort                                    beta runtime
  ↓ deterministic checks and caps
  ↓ blinded pairwise judge under frozen rubric
OpenEndedScorepad
  ↓ group-relative advantages
  ├── online policy update
  └── optional rubric-targeted correction rows
```

## 6. Proposal generation

### 6.1 Source type

**Decided:** `OpenEndedTaskBundle` is the source-specific `IdeaT` for
`ProposalSource[IdeaT]`. `TaskProposal` remains Taskforge's canonical proposal
type; there is no `HydratedTaskProposal` or notebook-specific proposal type.

Notebook, paper, or database origin belongs in provenance. “Open-ended” names
the grading contract.

### 6.2 Two-pass source

`OpenEndedProposalSource` will use two model passes:

1. **Portfolio planner:** allocate exactly `N` slots across distinct questions,
   analyses, or interpretation boundaries; mark unsupported slots null.
2. **Proposal writer:** write one canonical `TaskProposal` for each proposed
   slot, with one constrained repair on parse failure.

The planner/writer receive a safe projection of the bundle: brief, provenance,
resource paths and schemas, tags, split group, and fixed campaign policy. They do
not need private expected conclusions or raw private-evidence bytes.

The source locks rather than asks the model to choose:

- `SourceKind.OPEN_ENDED` and the bundle digest;
- allowed resource paths;
- no external research;
- split identity;
- open-ended model judgment;
- the campaign's solver environment; and
- privacy obligations.

A slot can be proposed, explicitly null, or independently fail. One malformed
slot must not discard its siblings.

### 6.3 Persist before workpieces

**Decided:** persist the complete `ProposalBatch` before creating workpieces. A
retry must reuse the same portfolio rather than resample tasks.

The batch cache identity should cover:

```text
bundle digest
proposal-source implementation version
planner model + policy
writer model + policy
slot count
campaign configuration
```

## 7. Workpiece identity and build input

A stable workpiece ID is derived from the campaign, bundle digest, and slot. It
must not depend on mutable display text.

The coordinator records the digest of immutable `build_input` in the `OPENED`
event. Resuming a workpiece with different input is an error. Any build step that
takes `build_input` includes its digest in the step-cache key. Child build
processes receive the same JSON value.

**Implemented:** this generic build-input seam and its local lifecycle tests are
in draft PR #10590.

Infrastructure failures propagate. They are not converted into task rejection.
A terminal workpiece is idempotent and runs no further stages.

## 8. Authoring and grading contract

### 8.1 Frozen criteria

**Decided:** reference and rubric are authored before solver rollouts and frozen
by content digest. Candidate answers are untrusted data and cannot add, remove,
or reinterpret criteria.

This intentionally differs from generating a rubric after inspecting a
candidate. A judge may explain how a frozen criterion applies, but it may not
invent a candidate-specific standard.

### 8.2 Three rubric layers

Every open-ended scientific rubric combines:

1. **Task-specific criteria:** the actual quantitative comparison, scientific
   question, and required limitations.
2. **Shared scientific-quality criteria:** numerical correctness,
   evidence-to-conclusion fit, reproducibility, experimental/statistical
   reasoning, calibration, and clarity where relevant.
3. **Prescriptive anti-hacking criteria:** gates or disqualifiers for fabricated
   work, evaluator-directed instructions,
   claims of success in place of evidence, cherry-picking, hidden denominator
   changes, unsupported causal claims, and verbosity substituting for substance.

Hard gates override weighted prose quality. A polished answer based on fabricated
results cannot receive a moderate passing reward.

### 8.3 Pointwise Taskforge validation

Taskforge validates one semantic task at a time. It runs controls and solver
trials against the ordinary `llm_as_judge` contract. This answers:

- Can a capable solver perform the task from visible resources?
- Does the reference earn the intended band?
- Are partial and severe-negative controls separated?
- Does evaluator-directed text fail?
- Does private evidence remain private?

Taskforge pointwise validation does not create an RL prompt group and does not
compute group-relative advantages.

### 8.4 Deterministic checks

**Decided principle:** deterministic evidence and model judgment are layered,
not alternatives.

Examples of hard checks include:

- required artifact/schema presence;
- recomputed numerical values and denominators;
- prohibited-row or prohibited-data use;
- citation existence and claim support where source passages are available;
- code execution and reproducibility; and
- budget compliance.

**Proposed interface:** analysis tasks that benefit from exact anchors emit a
machine-readable `results.json` beside a human-readable report. Whether this is
mandatory for every pilot task remains open.

## 9. Comparative reward in beta

### 9.1 Why comparison is downstream

A `TaskSpec` is one task, not a rollout group. Beta creates `K` trajectories from
one accepted task and a bounded policy-version window. Taskforge should not
encode `K` or an RL estimator into the task definition.

### 9.2 Candidate identity

Each candidate must resolve to an immutable trajectory artifact preserving:

- task and prompt-group identity;
- output content digest;
- rollout policy version;
- behavior log-probabilities needed by the optimizer;
- generation and tool budgets;
- model/tool transcript;
- hard-check results; and
- judge/verifier provenance.

Asynchronous work outside the accepted policy-staleness window is rejected, not
silently mixed into a current group.

### 9.3 Pairwise judgment

The judge receives the frozen task, reference, rubric, deterministic-check
summary, and blinded candidate products. Candidate order is randomized and
position bias is measured.

**Current experimental default:** a swapped directed ring. It provides two
ordered comparisons per candidate in linear rather than quadratic cost. It is a
starting point, not a claim that ring comparisons are statistically optimal.
Disagreement or low-margin cases may receive extra edges.

Judges may tie candidates. We should test swapped-order consistency and, where
cost permits, independent judges or repeated samples.

### 9.4 Scorepad

A durable scorepad records at least:

```text
task digest
rubric/reference digest
candidate IDs and content digests
blinded comparison order
hard-check outcomes and caps
judge endpoint/model identity
judge prompt/version digest
per-criterion scores and critiques
per-pair ratings and critiques
output/tool budget status
final per-candidate rewards
centered within-group rewards
```

The scorepad is audit evidence. Judge reasoning and adversarial controls do not
automatically become policy-training text.

### 9.5 Budget and reward shaping

An over-budget candidate loses to an otherwise eligible in-budget candidate.
Failed hard checks cap or zero the comparative reward. Scores are centered within
the task cohort; the RL implementation decides whether to variance-normalize or
apply another estimator.

The initial terminal reward is applied to policy-generated action tokens. More
ambitious process rewards should follow only after terminal reward is validated.

### 9.6 Corrector path

A corrector may receive the task, weak trajectory, and failed rubric criteria and
produce a corrected analysis. Correction rows are a separate supervised data
product from judge-reward trajectories. They preserve task, source policy,
corrector, rubric, and judge provenance.

## 10. Models and configuration

No model or endpoint belongs in a task proposal or bundle. Deployment supplies
role-specific clients for:

- portfolio planner;
- proposal writer;
- Taskforge author;
- solver trials;
- pointwise validation judge;
- comparative beta judge; and
- trajectory corrector.

The current Taskforge judge default of `glm-5.3` is temporary. Production routing
must be configured through the runtime/Zephyr port and persisted in provenance.

A judge should have an informational or computational advantage over the actor:
private reference material, deterministic results, stronger inference, execution
access, independent sampling, or some combination. A model grading its own prose
from the same information is not an adequate science reward by itself.

## 11. Persistence and distributed execution

### 11.1 Durable object layout

Taskforge source and evidence artifacts remain under Taskforge's namespace until
accepted tasks are explicitly lowered into beta data:

```text
s3://marin-us-east-02a/marin/taskforge/bioinformatics/notebook-pilot-v1/
  manifests/
  bundles/
  proposal-batches/
  workpieces/
  releases/
  scorepads/
```

The exact prefix is not created until schema and redistribution-rights review.
Post-training exports may later live under `marin/rl-data/`; copying them there is
a deliberate lowering step, not a visibility workaround.

### 11.2 Zephyr shape

The eventual campaign uses one `map_shard` invocation for the complete lifecycle.
Workers read Parquet ranges directly from object storage; the coordinator passes
specifications, not hydrated bulk rows.

Initial operating point:

- roughly 32 workers;
- 128–256 logical shards;
- one active workpiece per worker;
- at most two nested sandbox trials per workpiece;
- jitter and service rate limits; and
- terminal checkpointing after every slot.

Do not create one Zephyr shard per workpiece. Avoid `reshard()` unless source
layout provides inadequate parallelism, because it adds an intermediate
materialization boundary.

### 11.3 Current blocker

Taskforge event and round stores are local-`Path` based. Cross-worker Zephyr
retry is unsafe until proposal batches, workpiece events, rounds, and releases
have durable object-storage implementations.

## 12. Scientific-data rules

- Split source objects before task generation.
- Keep one underlying notebook/dataset family in one split.
- Preserve missingness, denominators, donor-level replication, and predeclared
  row-selection rules.
- Give raw-enough sufficient data when computation is the target; give summaries
  when interpretation is the target.
- Do not cherry-pick rows to force the notebook's conclusion.
- Record source URI/digest, extraction code/version, schema, row count, and
  output digest.
- Audit dataset redistribution rights separately from notebook repository
  licensing.
- Prefer sound negative results over incentives for novelty or positive findings.

The first MAIT packet intentionally excludes the notebook, expression matrix,
sequences, clinical metadata, original donor IDs, precomputed fractions, and
source conclusions. It remains `unreviewed_do_not_publish` until rights review.

## 13. Failure and retry semantics

| Failure | Required behavior |
|---|---|
| Portfolio cannot fill a genuine slot | Persist an explicit null slot |
| One proposal is malformed after repair | Persist a slot failure; continue siblings |
| Model/service unavailable | Propagate infrastructure failure for retry |
| Build machine creation fails | Leave resumable workpiece state; do not reject task |
| Bundle/build input changes | Reject resume under the old workpiece identity |
| Workpiece already terminal | Return terminal state without rerunning stages |
| Judge cohort output malformed/incomplete | Fail the whole cohort judgment; do not use partial rewards |
| Candidate is too stale | Reject it from the current prompt group |
| Private bytes appear solver-visible | Fail build/release |

All per-item operations should be idempotent and content-addressed.

## 14. Testing strategy

### Local deterministic tests

Use real Taskforge coordinator, event store, program loader, build runner, and
TaskSpec assembly with deterministic stage implementations. Cover:

- bundle JSON round-trip and content identity;
- public/private resource separation;
- changed-input resume rejection;
- proposal parse/repair/null outcomes;
- accepted lifecycle;
- infrastructure failure and resume;
- terminal idempotency;
- child-process build-input transfer;
- private-evidence leakage checks;
- scorepad wire round-trip;
- position swaps, ties, budgets, and hard-check caps; and
- malformed cohort failure.

Deterministic failure injection is preferable to relying on live models for retry
semantics.

### Model-interface integration

A small CPU model test is desirable for planner, writer, author, solver, and
judge interfaces. No intended TinyModels package or adapter has yet been located;
do not invent a repository-specific adapter merely to satisfy this test.

### Live smoke

Only after the local lifecycle works:

1. load the real MAIT bundle;
2. generate and persist a small proposal portfolio;
3. build and validate one task locally;
4. sample several real solver trajectories;
5. inspect pointwise controls and comparative scorepads manually; and
6. test a tiny Zephyr run before scaling.

## 15. Implementation status and sequence

### Implemented in draft PR #10590

- versioned curator-packet schema with explicit packet and TaskResource paths;
- strict packet inventory/digest verification and deterministic bundle loading;
- self-verifying canonical JSON records and deterministic Parquet bundle files;
- two-pass `OpenEndedProposalSource` with locked campaign fields, null slots, and one document repair;
- content-identified local `ProposalBatch` persistence that prevents retry resampling;
- pure bundle runner over persisted proposals and ordinary resumable Taskforge workpieces;
- `OpenEndedTaskBundle` and `OpenEndedBuildInput`;
- stable digest including private evidence and release status;
- generic immutable build-input transport and cache identity;
- accepted/retry/idempotency local lifecycle tests;
- notebook pilot inventory and MAIT extractor;
- initial scorepad models and swapped-ring reward computation; and
- frozen-rubric prompt-injection language for the pointwise judge.

### Next local vertical slice

1. Bundle-aware author and validator wrappers using private evidence.
2. Configurable role-specific model clients.
3. Real MAIT local lifecycle through accepted/rejected `TaskSpec`.
4. Pairwise judge request/response transport around the scorepad.
5. Generic accepted-`TaskSpec` export to beta.

### Distributed and training slice

1. Durable object-backed workpiece/event/round storage.
2. One-lifecycle Zephyr `map_shard` adapter.
3. Beta cohort rollout and scorepad persistence.
4. Judge-reward, corrector, and combined pilot runs.
5. Human/expert agreement and reward-hacking audit before larger online RL.

## 16. Open questions for review

These are not settled merely because an experimental implementation exists:

1. **Environment contract:** should the bundle contain an explicit locked solver
   environment/profile, or should campaign configuration supply it?
2. **Private-evidence resolver:** should author wrappers close over the bundle,
   or resolve evidence by digest from a durable bundle store?
3. **Structured deliverables:** should `results.json` be mandatory for all
   quantitative analysis tasks or only when useful deterministic anchors exist?
4. **Budget calibration:** reference-length multiplier, baseline-rollout
   percentile, or control-derived limit?
5. **Comparison graph:** swapped ring, sampled complete tournament, adaptive
   extra edges, or a mixture based on cohort size?
6. **Judge redundancy:** one strong judge for the smoke, repeated samples, or
   multiple independent model families from the start?
7. **Reward aggregation:** ordinal reward as currently prototyped versus binary
   wins/Elo/Bradley–Terry fitting before within-group centering?
8. **Private grader context:** how much derived private evidence beyond the
   reference and rubric should the comparative judge receive?
9. **Composite grading:** implement a first-class Taskforge composite grader, or
   keep deterministic checks as beta-side caps for the initial pilot?
10. **Release threshold:** what measured judge/expert agreement and adversarial
    robustness are required before online RL?

## 17. Decision recap

The most important decisions to retain are:

- `OpenEndedTaskBundle` is a source-specific idea, not a replacement proposal or
  task type.
- Hydration and human review happen before Taskforge.
- Complete notebooks and source datasets stay curator-side.
- Solver and private author evidence are separate resource groups.
- Private evidence changes bundle identity but never enters generic build input.
- One bundle may yield multiple proposals; one `TaskSpec` may yield many
  trajectories.
- Proposal portfolios are persisted before workpieces so retries do not
  resample them.
- References and rubrics are frozen before candidates are generated.
- Scientific rubrics combine task criteria, general quality, and anti-hacking
  gates.
- Objective checks and LLM judgment are layered.
- Taskforge performs pointwise task validation; beta performs comparative cohort
  reward.
- Pairwise/group-relative reward is preferred to trusting uncalibrated absolute
  judge scores.
- Output length and compute are explicit budgets, not proxies for quality.
- Every reward preserves task, rubric, candidate, judge, and policy provenance.
- Infrastructure failures propagate; terminal rejection is a scientific/task
  decision, not an operational catch-all.
- Source/evidence artifacts stay in Taskforge storage until an explicit beta
  export.
- Closed-book capability and tool-enabled/RAG performance are reported as
  different evaluation conditions.
- Local ordinary-lifecycle tests precede Zephyr and Iris execution.

## 18. Rejected shortcuts and why

| Shortcut | Why it was rejected |
|---|---|
| Pass complete notebooks or source datasets into Taskforge | Expands the leakage, licensing, context, and reproducibility surface; curation belongs upstream |
| Call the source `NotebookProposalSource` | Notebook is provenance, not the semantic task or grading contract |
| Introduce `HydratedTaskProposal` | Hydration is an input invariant; `TaskProposal` must remain canonical and source-independent |
| Put private evidence in `build_input` or `author_state` | Generic build programs and event logs are broader visibility domains and poor places for file bytes |
| Let each candidate induce its own rubric | Makes criteria candidate-dependent and invites post-hoc rationalization and evaluator manipulation |
| Use `DecisionGrader` merely to preserve a reference/rubric | Open scientific prose requires a generative model judge; decision grading is for narrower semantic comparisons |
| Ask one model for an unconstrained 1–10 score | Weak calibration, poor auditability, and high sensitivity to style and length |
| Treat deterministic and model rewards as alternatives | Scientific tasks often contain exact anchors and open interpretation at the same time |
| Generate hundreds of near-duplicate prompts | Genuine task diversity is limited; repeated trajectories per sound task are the intended first source of scale |
| Create one Zephyr shard per workpiece | Produces excessive scheduling/control-plane overhead and weak retry locality |
| Export every validation or adversarial trace for training | Validation evidence has different selection and leakage properties and requires an explicit data decision |
| Train on BixBench/LAB-Bench questions | Invalidates their use as held-out evaluations; borrow task contracts, not rows |
