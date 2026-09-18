# Curriculum generation and review workflow

This runbook describes the one-off agent workflow used to extend `curriculum.yaml`. It is intentionally a review
procedure, not an API. Keep the durable rubric, prompts, canonical YAML, and concise findings in the repository. Keep
raw task samples, agent transcripts, embeddings, mappings, and intermediate JSON in external experiment storage.

## Acceptance rule

Evaluate a subject in two independent ways:

1. A holistic reviewer scores the complete curriculum against `rubric.md`. Promotion requires at least 85/100, no
   blockers, and medium or high evidence confidence.
2. A blind task generator creates 24 subject tasks without seeing the curriculum. A separate fit judge sees only the
   task instructions and curriculum and classifies each task as `exact`, `ambiguous`, `gap`, or `invalid`.

Report blind fit as `(exact + defensible ambiguous) / (total - invalid)`, together with all four counts. Do not add it
to the 100-point holistic score. The blind-fit test measures sampled coverage; the holistic review measures whether
the curriculum's boundaries and progression are useful for training. During calibration, any repeated or
operation-family-wide gap blocks promotion, but there is no permanent numeric blind-fit threshold. In one run,
`repeated` means at least two gap tasks with the same normalized central operation, or one gap tied to a guidepost for
which no capability exists. The fit judge proposes the grouping and the operator confirms it from the task text.
Record observed rates across several subjects before setting a numeric threshold.

Curriculum-derived Luna placement tasks remain an optional boundary diagnostic. They do not substitute for either
independent review: their wording tends to mirror the curriculum, and successful placement says little about mutual
self-confidence or epsilon continuity.

## Inputs and provenance

Create a run manifest before calling an agent. Record:

- subject ID, name, and guideposts from `subject_inventory.json`;
- Git commit, `rubric.md` hash, prompt version, model, reasoning effort, provider, and agent/session ID for every role;
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

Store the manifest and frozen blind-task artifact at a durable URI before ending a wave that may need repair. `/tmp`
is acceptable only while a single wave is active. A concise checked-in or Loom report records the URI and hashes; it
does not copy task-level artifacts into this directory.

## Output contracts

The curriculum generator writes two files: a JSON object accepted by `Curriculum` in `models.py`, and a separate
design-audit JSON or Markdown file. Never wrap the curriculum and audit together. The curriculum JSON has this shape:

```json
{
  "version": "subject-version",
  "subject_id": "C00",
  "subject_name": "Subject name",
  "sections": [
    {
      "kind": "capability",
      "id": "c00.example",
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
      "id": "c00.group",
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
  "subject_id": "C00",
  "prompt_version": "blind-tasks-v1",
  "tasks": [
    {
      "id": "c00-blind-001",
      "instruction": "Self-contained task instruction",
      "guidepost_basis": ["C00.1"],
      "operation_family": "Operation used to construct the sample",
      "difficulty_intent": "entry|representative|boundary"
    }
  ]
}
```

Before fit judging, strip every field except `id` and `instruction`. The fit judge writes:

```json
{
  "subject_id": "C00",
  "prompt_version": "blind-fit-v1",
  "judgments": [
    {
      "task_id": "c00-blind-001",
      "status": "exact|ambiguous|gap|invalid",
      "acceptable_capability_ids": ["c00.example"],
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

The holistic review writes:

```json
{
  "subject_id": "C00",
  "curriculum_version": "subject-version",
  "score": 0,
  "dimension_scores": {
    "coverage": 0,
    "mutual_self_confidence": 0,
    "progression_and_epsilon_continuity": 0,
    "observable_boundaries": 0,
    "probe_quality_and_parsimony": 0
  },
  "status": "pilot_ready|revise|regenerate",
  "confidence": "low|medium|high",
  "blockers": ["Concrete blocking defect"],
  "highest_risk_sections": ["section.id"],
  "guidepost_accounting": [{"guidepost_id": "C00.1", "section_ids": ["c00.example"], "rationale": "..."}],
  "discovery_accounting": [{"item_id": "task-id", "status": "support|excluded|malformed|underdetermined", "section_ids": ["c00.example"], "rationale": "..."}],
  "evaluation_accounting": [{"item_id": "eval-item-id", "status": "support|excluded|malformed|underdetermined", "section_ids": ["c00.example"], "rationale": "..."}],
  "findings": ["Evidence-backed finding"],
  "recommended_changes": ["Minimal generalizable repair"],
  "proposed_rubric_changes": ["Only recurrent cross-subject changes"]
}
```

## Prompt 1: curriculum generator

Use a high-reasoning model. Replace bracketed fields and attach `rubric.md`, the relevant inventory object, the
model-visible discovery evidence, and policy-level evaluation metadata. Do not attach held-out evaluation questions
or paraphrases derived from them.

```text
You are designing one subject graph for a training curriculum. Produce Curriculum JSON and a concise design audit.

SUBJECT
[subject inventory object]

EVIDENCE
[manifest, model-visible discovery evidence, and policy-level evaluation metadata]

CONSTRAINTS
- Follow the attached rubric. Optimize for useful training boundaries, not encyclopedic coverage.
- Maximum hierarchy depth is 4. Section IDs begin with [lowercase subject ID].
- First enumerate the subject's distinct central operation families. Do not infer one section per guidepost.
- A capability is an observable outcome and a task-assignment target. A group is hierarchy only. Convert routing
  menus and omnibus parents to groups; retain an internal capability only when one natural representative task
  requires coherent cross-child synthesis.
- Every capability has exactly one entry probe followed by one representative probe. Both are concrete,
  self-contained task instructions. Do not specify solutions, graders, harnesses, or verifier behavior.
- For every proposed capability, instantiate the most operationally distant permitted pair. Split it only when
  mastery does not transfer because the central operation, tool interaction, or evaluation contract differs. A
  different topic or tool name alone is insufficient.
- Use a sampling facet only when the central operation and evaluation contract remain stable across its values.
  Apply the same distant-pair test to the facet's most distant values. Split instead when values change the solver
  loop, state transition, or evaluation contract.
- Prerequisites connect capabilities only. For each edge, identify the full upstream outcome used by every dependent
  representative, the single new operation in the dependent entry, and whether supplying the completed upstream
  artifact would eliminate the dependency. Reject workflow-order edges.
- The entry probe must give a learner who mastered all prerequisites a non-trivial chance of success. The
  representative probe must exercise the full outcome.
- Account for every guidepost and evidence item with a section or an exclusion rationale. Malformed tasks are not
  positive evidence.

DESIGN AUDIT
Return as a separate file: operation families; guidepost/evidence accounting; the distant-pair result
for every capability and facet; the role decision for every node; accepted and rejected prerequisite edges with the
completed-artifact counterfactual; and known evidence limitations. Do not mention or infer a target section count.
```

## Prompt 2: curriculum-blind task generator

Use a separate high-reasoning context. Do not attach the curriculum, rubric, curriculum probes, reviews, or fit
results.

```text
Generate exactly 24 diverse, concrete tasks that a capable practitioner could reasonably be asked to perform in the
subject below. You are sampling the subject, not designing or reverse-engineering a curriculum.

SUBJECT
[subject ID, name, guideposts, and optional pre-curriculum domain brief with sources and hash]

REQUIREMENTS
- Do not ask for a taxonomy, syllabus, curriculum, or discussion of these guideposts.
- Cover every guidepost with at least two tasks. This 24-task procedure supports at most 12 guideposts; for a larger
  inventory use `max(24, 2 * guidepost_count)` tasks. Use the remaining tasks for cross-guidepost work, boundary
  cases, and underrepresented central operations.
- Vary central operations, artifacts, contexts, and difficulty. Include entry, representative, and boundary tasks.
- Each instruction must be self-contained. Supply the data, code, measurements, source excerpts, legal text, or
  interface facts needed to act. Do not rely on an unspecified file, hidden source, or private verifier.
- Keep each task genuinely in the named subject. Narrative subject matter alone is insufficient.
- Avoid near-duplicates and simple surface rewrites. Do not include answers or solution sketches.
- Output strict JSON using the blind-task contract. `guidepost_basis`, `operation_family`, and `difficulty_intent`
  document how you sampled; they will be hidden from the fit judge.
```

## Prompt 3: blind-fit judge

Use a separate high-reasoning context. Attach the candidate curriculum and only the blind task IDs and instructions.
Do not attach generation metadata, holistic reviews, prior fit judgments, or answers.

```text
Judge whether each independently generated task has a useful home in this subject curriculum. Inspect the complete
task, not its nouns or output format. Groups are never task targets.

For each task:
- `exact`: one capability covers the complete task and its decisive operation.
- `ambiguous`: two or more capabilities each plausibly cover the complete task; list all acceptable capabilities.
- `gap`: no capability covers the complete task. Partial coverage by several capabilities is still a gap.
- `invalid`: the task is malformed, materially underdetermined, or not genuinely in the subject.

Do not reward a capability merely because its wording resembles the task. Do not penalize benign changes in data,
surface form, or sampling-facet value. Name the decisive operation and explain the result briefly. Then return the
strict blind-fit JSON contract, verify that every input task appears exactly once, compute the four counts, set the
numerator to exact plus defensible ambiguous, set the denominator to total minus invalid, and identify repeated gaps
that indicate a missing operation family.
```

## Prompt 4: holistic curriculum reviewer

Use a separate high-reasoning context. Attach `rubric.md`, the candidate curriculum, inventory object, evidence
manifest, model-visible discovery tasks, and permitted held-out in-distribution evaluation instructions. Do not attach
answers, verifiers, earlier reviews, or blind-fit results.

```text
Perform a fresh complete-subject curriculum review. Return strict JSON using the holistic-review contract.

Read every capability, group, sampling facet, prerequisite edge, entry probe, representative probe, guidepost, and
evidence item. Score: coverage 25; mutual self-confidence 25; progression and epsilon continuity 25; observable
boundaries 15; probe quality and parsimony 10.

Required checks:
- Account for every guidepost and evidence item with a primary capability or exclusion rationale.
- For every capability, instantiate its most operationally distant permitted task pair. A blocking split finding
  must name both tasks, their different central operations, and how splitting changes sampling or evaluation.
- For every sampling facet, compare its most distant values and reject it if the solver loop, state transition, or
  evaluation contract changes.
- For every prerequisite edge, identify where every dependent representative uses the full prerequisite outcome,
  the one new operation in the entry, and the completed-artifact counterfactual.
- Check every internal capability for one natural cross-child synthesis. A concatenated set of child deliverables is
  not synthesis.
- Check all entry-to-representative orderings for a plausible epsilon step. Structural plausibility is not evidence
  of actual training transfer.
- Distinguish evidence confidence from structural quality. High confidence requires direct or held-out evidence
  reaching every leaf and sampled edges; sparse task evidence caps confidence at medium.

`pilot_ready` requires at least 85 points, no blockers, and medium or high confidence. Any blocker or score 70--84 is
`revise`; pervasive structural failure or a score below 70 is `regenerate`. Recommend the smallest repair that fixes
each concrete defect. Propose a rubric change only for a recurrent issue that generalizes beyond this subject.
```

## Optional prompt: blinded Luna placement

Use this only when a boundary or difficulty uncertainty could change a finding. Batch 8--16 tasks. Hide intended
targets and expose capability sections only. Record model, prompt version, task IDs, chosen target, rationale, and the
artifact hash. This is advisory evidence: the approximate 70% reasonable-placement rate used during early pilots is
not a promotion threshold.

```text
For each task, choose one capability ID from the supplied curriculum or `out_of_scope`. Groups are not valid
targets. Base placement on the decisive operation, not narrative nouns or answer format. Return each task ID exactly
once with the chosen target and one-sentence rationale. If two capabilities are genuinely equivalent, choose the one
whose representative probe requires more of the task's central operation and name the alternative.
```

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
4. Check outputs mechanically. The holistic dimensions must remain within 25/25/25/15/10 and sum to `score`; score
   is in 0--100. Fit judgments must contain every frozen task ID exactly once and no other ID. Counts must equal the
   judgments; section references must exist and target capabilities; numerator is exact plus ambiguous; denominator
   is total minus invalid and must be nonzero. Guidepost, discovery, and evaluation ledgers must cover every supplied
   item exactly once.
5. Repair concrete blockers. Revalidate, rerun the complete holistic review, and rerun fit judgment on the frozen
   blind tasks. Do not accept a local blocker-only check.
6. Promote only the final reviewed curriculum into `curriculum.yaml`. Declare `routing_facet` from the intended
   membership semantics and record whether a labeled member/near-neighbor fixture calibrated that choice. A missing
   fixture leaves routing provisional but does not block a structurally sound curriculum. Increment `catalog_version`
   as `waveN-R`, where `N` is the wave and `R` is the catalog revision within it. Validate the entire file with
   `load_catalog`, which checks routing values and global subject/section uniqueness. Recompute catalog counts from
   the parsed object. Task mapping remains a diagnostic and is not a promotion gate.

   ```bash
   uv run python -c 'from pathlib import Path; from experiments.post_training.task_curriculum.catalog import load_catalog; c=load_catalog(Path("experiments/post_training/task_curriculum/curriculum.yaml")); print(len(c.curricula), sum(len(x.curriculum.sections) for x in c.curricula), sum(len(x.curriculum.capability_sections()) for x in c.curricula))'
   ```
7. Record a concise result: models and prompt versions, subject/version, node counts, holistic score/status/confidence,
   blind-fit X/Y and counts, systematic gaps, repairs, optional Luna results, evidence limitations, and catalog
   version. Archive raw one-off artifacts at the manifest URI, or delete them only after the wave is accepted and no
   repair needs the frozen task set. Do not check them into this directory.
8. Run focused tests, type checking, and the repository lint workflow before publishing the change.

## Resume checklist

A new operator can resume from the canonical YAML, inventory, rubric, this runbook, and the latest concise report.
Before starting another wave, confirm that all promoted subjects validate, list inventory subjects absent from the
catalog, inspect the last wave's recurrent findings, and select a varied group of subjects rather than adjacent
specialties. Change a prompt only when a finding recurs across subjects; record a new prompt version and retain the
old result's provenance. Do not rewrite prior results to match a newer rubric.
