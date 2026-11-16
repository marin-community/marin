# Curriculum generation and review workflow

This runbook describes the one-off agent workflow used to extend the catalog registered in `catalog_artifact.py`; no
API implements the workflow.
Keep the durable rubric, prompts, artifact declarations, and concise findings in the repository. Keep the subject
inventory, catalog payloads, raw task samples, agent transcripts, embeddings, mappings, and intermediate JSON in
immutable external experiment storage.

## Acceptance rule

Evaluate a subject in two independent ways:

1. A holistic reviewer scores the complete curriculum against `prompts/rubric.md`. A subject is structurally `pilot_ready`
   at 85/100 with no structural blockers. Evidence confidence is reported separately and guides later sampling.
2. A blind task generator creates `max(24, 2 * guidepost_count)` subject tasks without seeing the curriculum. A
   separate fit judge sees only the task instructions and curriculum and classifies each task as `exact`,
   `ambiguous`, `gap`, or `invalid`.

Report blind fit as `(exact + defensible ambiguous) / (total - invalid)`, together with all four counts. Do not add it
to the 100-point holistic score. The blind-fit test measures sampled coverage; the holistic review measures whether
the curriculum's boundaries and progression are useful for training. During calibration, any repeated or
operation-family-wide gap blocks promotion, but there is no permanent numeric blind-fit threshold. In one run,
`repeated` means at least two gap tasks with the same normalized central operation. The fit judge proposes the
grouping and the operator confirms it from the task text. A single gap tied to a guidepost with no capability also
blocks promotion; `validate_subject_promotion` detects that case from the hidden task metadata and holistic guidepost
accounting. Record observed rates across several subjects before setting a numeric threshold.

`pilot_ready` is the quality bar for treating a subject as reviewed. During full-catalog scale-out, a schema-valid
graph may enter the next published catalog artifact after one broad repair when every inventory guidepost has an explicit home. Its
latest non-passing review, blind-fit
misses, confirmed gaps, and evidence limitations must remain visible in the wave report. This provisional inclusion
does not satisfy the `pilot_ready` bar. Revisit those subjects using broader evidence instead of recursively
optimizing against one review or frozen blind sample.

Curriculum-derived Luna placement tasks remain an optional boundary diagnostic. They do not substitute for either
independent review: their wording tends to mirror the curriculum, and successful placement says little about mutual
self-confidence or epsilon continuity.

## Inputs and provenance

Create a run manifest before calling an agent. Record:

- subject ID, name, and guideposts from the `TASK_CURRICULUM_SUBJECT_INVENTORY` artifact;
- Git commit, `prompts/rubric.md` hash, prompt version, model, reasoning effort, provider, and agent/session ID for every role;
- sampling settings when the provider exposes them; record `provider controlled` instead of inventing a seed or
  temperature when it does not;
- discovery-task IDs and content hashes, with answers and verifiers excluded;
- evaluation names and immutable revisions used as held-out evidence; and
- output URIs, canonical SHA-256 hashes, and retention policy. Canonical JSON uses UTF-8, sorted keys, compact
  separators, and one trailing newline.

The [evaluation policy](https://github.com/marin-community/marin/issues/9193) permits inspection of in-distribution
evaluation questions as held-out coverage probes. The manifest lists their IDs and hashes, but only the holistic
reviewer receives their model-visible instructions. Neither generator receives a paraphrase or other content derived
from those questions, and no role sees answers, teacher rollouts, or verifier internals. For out-of-distribution
evaluations, use only policy-level metadata such as benchmark domain and format; never inspect questions. If there is
no applicable held-out evidence, say so and cap evidence confidence at medium.

The curriculum generator receives the subject inventory and model-visible discovery tasks. The holistic reviewer
receives those inputs plus held-out in-distribution evaluation instructions. The blind task generator receives the
subject name and guideposts. An optional domain brief must be frozen before curriculum generation and carry source
citations and a hash; it may not derive from discovery tasks, a candidate curriculum, or a review. The blind task
generator must not see the curriculum, its probes, discovery tasks, or earlier reviews. The fit judge must not see the
task generator's intended guideposts, operation families, or difficulty labels. The holistic reviewer must not see
blind-fit results or earlier reviews.

Use separate agent contexts for all four roles. A repair is a new curriculum version and receives a fresh complete
holistic review and blind-fit judgment against the same frozen blind tasks. Regenerate blind tasks only when they are
invalid, leaked the curriculum, or the subject definition changed.

### Curriculum source survey

After the first D-series catalog pass, audit each root against at least two public undergraduate course sequences
from independent institutions and one established introductory or survey textbook table of contents. Record the
source URL, institution or publisher, edition, access date, and a content hash when the source is stable. Normalize
topics and exercise families, then map each to an existing guidepost as `covered`, `cross-domain`, or
`candidate_gap`. Promote a candidate gap only when at least two independent sources support it and it describes a
learnable behavior that no existing guidepost covers.

For trades, public safety, clinical practice, and other fields poorly represented by college textbooks, substitute
authoritative licensing, accreditation, or professional competency standards. Store the source extraction and
topic mappings externally; check in only a versioned inventory revision and concise findings. Course order and
chapter structure are evidence for breadth and candidate progression, not capability boundaries, mutual
self-confidence, or learning-prerequisite edges. The normal generator, holistic review, blind-fit checks, and
separate learning-progression pass remain authoritative.

Store the manifest and frozen blind-task artifact at a durable URI before ending a wave that may need repair. `/tmp`
is acceptable only while a single wave is active. A concise checked-in or Loom report records the URI and hashes; it
does not copy task-level artifacts into this directory.

The reference role model is `gpt-5.6-sol` with high reasoning effort. Record a different pinned model explicitly when
availability or an experiment requires it. `Provider controlled` means the agent interface did not expose a sampling
parameter; it is not a request to leave an exposed parameter unset.

## Output contracts

The curriculum generator writes two files: a JSON object accepted by `Curriculum` in `models.py`, and a separate
design-audit JSON or Markdown file. Never wrap the curriculum and audit together. The audit is a non-authoritative
reasoning trace: hash and archive it, but derive promotion only from the canonical curriculum and validated reviews.
The curriculum JSON has this shape:

```json
{
  "version": "subject-version",
  "subject_id": "D00",
  "subject_name": "Subject name",
  "sections": [
    {
      "kind": "capability",
      "id": "d00.example",
      "parent_id": null,
      "name": "Observable capability",
      "outcome": "What a model can do",
      "includes": ["Included behavior"],
      "excludes": ["Neighboring behavior"],
      "prerequisites": [],
      "sampling_facets": [{"id": "facet_name", "description": "Reusable knowledge or data axis"}],
      "sample_tasks": [
        {"kind": "entry", "instruction": "Smallest new operation beyond prerequisites"},
        {"kind": "representative", "instruction": "Full outcome"}
      ]
    },
    {
      "kind": "group",
      "id": "d00.group",
      "parent_id": null,
      "name": "Organizational scope",
      "scope": "What its descendants cover",
      "includes": ["Included branch"],
      "excludes": ["Neighboring branch"]
    }
  ]
}
```

The blind task generator writes:

```json
{
  "subject_id": "D00",
  "prompt_version": "blind-tasks-v1",
  "tasks": [
    {
      "id": "d00-blind-001",
      "instruction": "Self-contained task instruction",
      "guidepost_basis": ["D00.1"],
      "operation_family": "Operation used to construct the sample",
      "difficulty_intent": "entry|representative|boundary"
    }
  ]
}
```

Before fit judging, strip every field except `id` and `instruction`. The fit judge writes:

```json
{
  "subject_id": "D00",
  "curriculum_version": "subject-version",
  "prompt_version": "blind-fit-v1",
  "judgments": [
    {
      "task_id": "d00-blind-001",
      "status": "exact|ambiguous|gap|invalid",
      "acceptable_capability_ids": ["d00.example"],
      "decisive_operation": "What determines placement",
      "explanation": "Short evidence-based rationale"
    }
  ],
  "counts": {"exact": 0, "ambiguous": 0, "gap": 0, "invalid": 0},
  "fit_numerator": 0,
  "fit_denominator": 0,
  "systematic_gaps": ["Repeated missing operation family, if any"]
}
```

An `ambiguous` judgment is defensible only when every listed capability covers the complete task, so every accepted
`ambiguous` row contributes to the numerator. A task that partly fits several sections is a `gap`, not an ambiguity.
`fit_numerator = exact + ambiguous` and `fit_denominator = total - invalid`. A zero denominator invalidates the run;
regenerate the blind sample.

The operator records one disposition for every proposed systematic gap:

```json
{
  "gap": "Normalized operation-family gap from the fit review",
  "status": "blocking|rejected",
  "rationale": "Why the task text confirms or rejects the proposed grouping"
}
```

A `blocking` disposition prevents promotion. A `rejected` disposition records why the proposed grouping does not
meet the repeated-gap rule. An empty `systematic_gaps` list requires an empty disposition list.

The holistic review writes:

```json
{
  "subject_id": "D00",
  "curriculum_version": "subject-version",
  "score": 0,
  "dimension_scores": {
    "coverage": 0,
    "mutual_self_confidence": 0,
    "local_progression": 0,
    "observable_boundaries": 0,
    "probe_quality_and_parsimony": 0
  },
  "status": "pilot_ready|revise|regenerate",
  "confidence": "low|medium|high",
  "blockers": ["Concrete blocking defect"],
  "highest_risk_sections": ["section.id"],
  "guidepost_accounting": [{"guidepost_id": "D00.1", "section_ids": ["d00.example"], "rationale": "..."}],
  "discovery_accounting": [{"item_id": "task-id", "status": "support|excluded|malformed|underdetermined", "section_ids": ["d00.example"], "rationale": "..."}],
  "evaluation_accounting": [{"item_id": "eval-item-id", "status": "support|excluded|malformed|underdetermined", "section_ids": ["d00.example"], "rationale": "..."}],
  "findings": ["Evidence-backed finding"],
  "recommended_changes": ["Minimal generalizable repair"],
  "proposed_rubric_changes": ["Only recurrent cross-subject changes"]
}
```

## Durable role prompts

The exact prompts are versioned separately so a run can hash and attach only the role it needs:

- [`prompts/generation.md`](prompts/generation.md) for curriculum generation;
- [`prompts/blind_tasks.md`](prompts/blind_tasks.md) for curriculum-blind task sampling;
- [`prompts/blind_fit.md`](prompts/blind_fit.md) for fit judgment;
- [`prompts/review.md`](prompts/review.md) for holistic review;
- [`prompts/learning_progression.md`](prompts/learning_progression.md) for catalog-level learning edges;
- [`prompts/learning_progression_review.md`](prompts/learning_progression_review.md) for their independent review; and
- [`prompts/luna_placement.md`](prompts/luna_placement.md) for the optional batched diagnostic.

The shared scoring and boundary definitions are in [`prompts/rubric.md`](prompts/rubric.md). Preserve the allowed-read
sets from Inputs and provenance when composing each role call. A prompt edit requires a new prompt version; do not
rewrite archived results to match it.


## Run sequence

1. Freeze the manifest and launch the curriculum generator and blind task generator independently. They may run in
   parallel.
2. Validate the generated curriculum:

   ```bash
   uv run python -c 'import json,sys; from experiments.post_training.task_curriculum.models import Curriculum; c=Curriculum.model_validate(json.load(open(sys.argv[1]))); c.check_generation_contract(maximum_depth=4)' /tmp/curriculum.json
   ```

   Also verify that `subject_id` matches the inventory object and every section ID starts with the lowercase subject
   ID followed by a dot; these are run-level invariants rather than `Curriculum` schema rules.
3. Validate unique blind-task IDs, then produce the judge input deterministically with
   `jq -S '{subject_id, tasks: [.tasks[] | {id, instruction}]}'`. Canonicalize it with compact JSON plus one newline,
   scan it for leaked sampling fields, record its SHA-256 hash, and launch the fit judge and holistic reviewer
   independently. They may run in parallel.
4. Parse the source task set, fit result, gap dispositions, and holistic review with `BlindTaskSet`, `BlindFitReview`,
   `SystematicGapDisposition`, and `HolisticReview` from `models.py`. Call `validate_subject_run` from `validation.py`
   with the inventory guidepost IDs and evidence-manifest item IDs. It enforces sample size and guidepost
   representation, subject and version identity, frozen task IDs, capability-only references, resolved systematic
   gaps, and complete accounting ledgers. Initial `revise` and `regenerate` results still receive this full validation.
5. Repair concrete structural blockers once. Revalidate and, when the repair changes broad boundaries, rerun the
   complete holistic review and fit judgment on the frozen blind tasks. Stop after one repair pass unless a guidepost
   remains uncovered or the schema/generation contract fails. Record isolated misses and sparse evidence for the next
   version instead of adding task-specific sections or repeatedly rereviewing the graph.
6. Call `validate_subject_promotion` before marking a subject `pilot_ready`. It requires a passing holistic review,
   no confirmed systematic gap, and no blind-task gap tied to an uncovered guidepost. A breadth-first scale-out may
   still add a provisional subject after the one-repair stopping rule above; do not call that subject `pilot_ready`,
   and preserve its exact last review and fit result in the wave report. Add the selected curriculum to
   a locally materialized catalog and declare `routing_facet` from the intended
   membership semantics and record whether a labeled member/near-neighbor fixture calibrated that choice. A missing
   fixture leaves routing provisional but does not block a structurally sound curriculum. Give `catalog_version` a
   date-based immutable identifier such as `YYYY.MM.DD-cross-domain-vN`. Validate the entire file with
   `CurriculumCatalog.model_validate_json`, which checks routing values and global subject/section uniqueness.
   Recompute catalog counts from the parsed object. Serialize the complete catalog as canonical JSON (UTF-8, sorted
   keys, compact separators, and one trailing newline), upload it to a new immutable S3 directory, verify its SHA-256
   by reading it back,
   and update the source and version of `TASK_CURRICULUM` in `catalog_artifact.py`. Preserve the prior immutable handle
   for comparison and rollback. Do not check the catalog payload into git. Task mapping remains a diagnostic and is
   not a promotion gate.

   ```bash
   uv run python -c 'from pathlib import Path; from experiments.post_training.task_curriculum.models import CurriculumCatalog; c=CurriculumCatalog.model_validate_json(Path("/tmp/task-curriculum-next.json").read_bytes()); print(len(c.curricula), sum(len(x.curriculum.sections) for x in c.curricula), sum(len(x.curriculum.capability_sections()) for x in c.curricula))'
   ```
7. Record a concise result: models and prompt versions, subject/version, node counts, holistic score/status/confidence,
   blind-fit X/Y and counts, systematic gaps and dispositions, repairs, optional Luna results, evidence limitations,
   and catalog version. Archive raw one-off artifacts at the manifest URI, or delete them only after the wave is
   accepted and no repair needs the frozen task set. Do not check them into this directory.
8. Run focused tests, type checking, and the repository lint workflow before publishing the change.

## Learning-progression pass

Generate learning prerequisites after capability boundaries are stable. The production pass is subject-local: it
does not send the whole catalog to an agent and it does not regenerate capabilities. The reviewed graph lives once at
the catalog root in `learning_progression`; keep every capability's embedded `prerequisites` array empty.

For each subject:

1. Materialize a compact packet containing its groups and each capability's ID, parent, name, outcome, includes,
   excludes, and entry/representative probes. Omit facets, evidence, scores, task mappings, previous prerequisites,
   other subjects, and generation transcripts.
2. Give the packet and `prompts/learning_progression.md` to one Sol/high proposer. Bind structured output to
   `LearningProgression`, record catalog/prompt/model hashes, and require every endpoint to be a capability in that
   subject.
3. Parse and validate capability references, unique edge pairs, distinct witness-family pairs, and acyclicity.
4. Give a fresh Sol/high reviewer the same packet, proposal, and `prompts/learning_progression_review.md`. The reviewer
   judges every proposal and returns complete witness-backed objects for clear omissions.
5. Validate exact review accounting, then combine accepted proposals with reviewer omissions. Do not run a routine
   repair call: rejected edges stay rejected, and supplied omissions are already complete edge objects.

After all subjects finish, combine their edges, validate the global graph, and merge it into a new immutable catalog
version. Cross-subject prerequisites are a separate follow-up because discovering them would require a different
routing or retrieval procedure; the subject-local pass must not infer them from a partial catalog view.

The edge contract implements learning enablement. Mastery of A must materially improve the chance of success on a
recurring family of entry-level B tasks. A may cover only a declared stratum of B. Pure artifact handoff, course
order, general sophistication, and domain relabeling do not qualify. Each edge has two concise task-family sketches
that reuse the same upstream foundation and add one main dependent operation. These are structural hypotheses; the
workflow does not claim causal transfer without training evidence. Exact duplicate family pairs are rejected by the
model contract. Semantic family diversity is reviewer-enforced because paraphrase equivalence is not a reliable
string-level invariant. Run exactly one proposer and one reviewer per subject, never one call per capability or edge.

## Catalog-wide bounded repair

For a production-scale revision, freeze the current catalog and attempt at most one repair per selected subject. The
repair role receives the complete subject graph, its latest review, the inventory guideposts, and a compact evidence
brief; it does not receive blind tasks or alternative generated graphs. Require an exact unchanged/changed section
partition and flag repairs that replace more than 35% of baseline sections for manual review. Regeneration is reserved
for a graph whose current review explicitly says `regenerate`.

Compare the complete baseline and repaired graphs anonymously in one high-reasoning review call. Select the repair
only when it is preferred and scores at least 70; retain the established graph on ties or regressions. A separate
fit call may reuse frozen tasks to compare the two candidates, but label that result as a reused diagnostic whenever
an earlier baseline repair saw those tasks or their summary. It is not a fresh holdout claim and does not override
mutual-confidence, epsilon-progression, boundary, or probe findings.

Before publication, merge selected subjects mechanically and validate global identifiers, parents, prerequisites,
probes, inventory coverage, and serialized round-trip equality. Run a cross-root audit on a small stratified set of
real tasks plus evaluation metadata. Around 70% reasonable initial task placement is sufficient; repeated operation-
family gaps require follow-up, while isolated misses remain recorded. Publish the catalog, summary, comparison,
routing audit, exact prompts and schemas, raw role outputs, runner source, repository commit, and file manifest under
one new immutable prefix, then read every object back and verify its hash.

## Resume checklist

A new operator can resume from the catalog artifact, inventory, rubric, this runbook, and the latest concise report.
Before starting another wave, confirm that all promoted subjects validate, list inventory subjects absent from the
catalog, inspect the last wave's recurrent findings, and select a varied group of subjects rather than adjacent
specialties. Change a prompt only when a finding recurs across subjects; record a new prompt version and retain the
old result's provenance. Do not rewrite prior results to match a newer rubric.
