# Task curriculum

This experiment turns a broad subject inventory into trainable curricula and maps tasks onto reviewed sections. It
keeps curriculum design separate from task correctness:
TaskCompendium owns model-visible task semantics and private verifier contracts; a curriculum describes observable
capabilities, boundaries, and examples. A separate catalog-level pass proposes learning prerequisites after those
capability boundaries are stable.

The canonical cross-domain v3 catalog is the immutable object registered by `catalog_artifact.py`. Canonical means
versioned and addressable. It contains all 45 D-series subject roots and 2,405 globally unique nodes: 1,999 trainable
capabilities and 406 organizational groups. Its catalog-level learning graph adds 1,812 reviewed same-subject edges;
the capability definitions otherwise retain production v2. Twenty-four subjects are `pilot_ready`; 21 remain
explicitly provisional after the stopping rule. `HISTORY.md` records the experiments and tradeoffs. The 7.6 MB
canonical JSON payload lives in CoreWeave S3.

- Artifact handle: `TASK_CURRICULUM`, version `2026.09.20.1`
- Catalog: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json`
- Progression summary: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/summary.json`
- Capability scores: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/comparison.json`
- Routing audit: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/routing_audit.json`
- Evidence: `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/evidence.tar.gz`
- Human viewer: [Task curriculum](https://applets.marina.oa.dev/a/67f69132-2ef4-4c9e-b8b5-77cabd126442/)

Materialize the JSON before local validation or mapping:

```bash
uv run fsutil cp \
  s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json \
  /tmp/task-curriculum-cross-domain-v3.json
```

`fsutil` reads `CW_KEY_ID` and `CW_KEY_SECRET` for this bucket. CoreWeave Iris tasks receive the same credentials from
their task environment.

## Inputs

- `TASK_CURRICULUM_SUBJECT_INVENTORY` is the `2026.09.18-cross-domain-v1` inventory: 45 D-series subject roots and
  356 guideposts. Its typed artifact loads `subject_inventory.json` from
  `s3://marin-us-east-02a/marin/task-curriculum/inventories/2026.09.18-cross-domain-v1-ef10281c71ad/`. ISCED-F, CIP,
  Frascati, evaluation inventories, and occupational taxonomies are coverage checks rather than curriculum node
  definitions.
- `prompts/rubric.md` defines the cross-subject review criteria. The other files in `prompts/` are the exact durable
  role prompts.
- `workflow.md` is the resumable operating procedure and defines role isolation, artifact contracts, and promotion.
- `HISTORY.md` summarizes each completed generation, review, source-audit, and mapping experiment and links its
  external evidence artifact.
- In the general workflow, a small set of model-visible TaskTrove tasks supplies concrete discovery evidence. In-distribution examples listed
  by the evaluation policy are held-out coverage probes. Out-of-distribution evaluations contribute domain and task
  format metadata only. Source names, solutions, and verifier implementations are excluded from generation. The
  breadth-first D-series v1 run did not add direct TaskTrove or evaluation tasks: it used the frozen inventory and
  optional matching C-series graphs, which is why its evidence confidence is mostly low.
- TaskCompendium's subject, competency, task-shape, artifact, context, interaction, and state facets are optional
  vocabulary for describing task semantics.

The five-subject source survey tested public college course sequences, textbook exercise families, and professional
standards as generator inputs. The subsequent v3 experiment applied its bounded-repair recommendation to six roots.
Production v2 then used the method across the remaining catalog. `HISTORY.md` records both experiments. V1 and
experimental v3 remain immutable as `TASK_CURRICULUM_V1` and `TASK_CURRICULUM_SOURCE_AUDIT_V3`. The semantically
identical YAML serialization of production v2 remains available as `TASK_CURRICULUM_YAML_V2` for rollback and
provenance, but active consumers use JSON.

Production v2 used anonymous same-call comparisons rather than absolute score deltas from different reviewers. A
repair was selected only when the reviewer preferred it and it scored at least 70. The frozen blind tasks were reused
as a diagnostic: the production repair could not see them, but the older baseline had been shaped by an earlier fit
summary, so the aggregate 1,001/1,080 fit is not a fresh holdout claim. A separate practical routing audit placed all
24 sampled shell, repository-workspace, and bug-repair tasks in D02; 21/24 then had one complete capability home. The
three misses were unrelated operation families, not evidence for another omnibus capability. Evaluation-policy and
AAII metadata found subject homes across all 45 roots; ten benchmark entries were classified as cross-domain task
mechanics rather than subjects.

The embedded prerequisite arrays in production v2 were a conservative hard-dependency experiment: only 17 edges in
four subjects survived it, so they were not a useful easy-to-hard ordering. Production v3 replaces them with one
catalog-level `learning_progression` graph under the original learning-enablement rule: mastery of A should materially
improve the chance of some success on a recurring family of entry-level B tasks. It was generated subject by subject
from compact capability packets, with one Sol/high proposer and one independent reviewer per subject. Capability
definitions are unchanged from v2 except that the 17 obsolete embedded prerequisites are cleared. Cross-subject
progression is deliberately deferred rather than inferred from incomplete context.

## Curriculum iteration

One-off agent runs produce curricula and reviews. Their reports record the evidence, and reviewed changes are promoted
into a new immutable catalog version. For each selected macro area:

1. Give a high-reasoning generator the inventory area, rubric, a maximum tree depth, and representative tasks. Before
   emitting the curriculum, it enumerates operation families, tests the most distant permitted pair for every
   capability, and leaves embedded prerequisite arrays empty. Every section includes an `entry` probe for the
   smallest self-contained exercise of its outcome and a `representative` probe for the full outcome.
2. Validate the JSON with `Curriculum` and `Curriculum.check_generation_contract`.
3. Give the curriculum and rubric to an independent high-reasoning reviewer. The reviewer returns a score, verdict,
   concrete findings, and proposed rubric changes in one call for the complete subject curriculum.
4. When an uncertain boundary or difficulty claim could change the review, the reviewer may generate focused tasks
   and send them to a blinded Luna model in batches of 8–16. These diagnostics are optional and do not replace the
   whole-subject judgment.
5. Compare findings across subjects. Mutual self-confidence and local entry-to-representative continuity are
   blocking criteria even when the numeric score is high. Revise the rubric or generation instructions only for
   problems that recur, then generate another version.
6. After capability boundaries are stable, run one Sol/high learning-progression proposer and one independent
   reviewer per compact subject packet. Each accepted edge has two witness-family sketches. `workflow.md` defines the
   subject-local procedure.

The holistic score controls structural readiness: 85 or higher with no structural blockers is `pilot_ready`.
Evidence confidence is tracked independently. Low confidence marks branches that need better discovery or held-out
tasks; it does not prevent a coherent graph from entering the versioned catalog. Later evidence, blind-fit failures,
and task mappings can trigger a new curriculum version.

In parallel with curriculum generation, a curriculum-blind Sol/high agent generates
`max(24, 2 * guidepost_count)` diverse tasks from the subject definition. A separate judge reports how many have an
exact home, an ambiguous but complete home, a coverage gap, or are invalid. This blind-fit X/Y is a sampled coverage
diagnostic reported separately from the holistic score. Freeze the task set across curriculum repairs.
During initial calibration, a systematic operation-family gap blocks promotion; `workflow.md` defines the complete
independence and reporting rules.

The hierarchy has two node kinds. A `capability` is a trainable outcome and the only kind that may receive task
assignments or carry entry and representative probes. A `group` is an organizational scope; it has no outcome or
probes. Catalog-level learning edges connect capabilities and may cross subject roots. Use a group when a parent
would otherwise be a routing menu or an artificial bundle of child deliverables. Capabilities may still contain
narrower capabilities when their representative probe exercises a natural cross-child synthesis.

Reusable knowledge dimensions belong in a capability's `sampling_facets`. A facet declares an axis such as language
direction, jurisdiction, or scientific-model family; its concrete value is recorded with task annotations outside
the catalog. The effective task set is a capability plus fixed facet values. This makes mutual self-confidence
testable under one knowledge regime without creating one curriculum node per language pair or jurisdiction.

Publish a reviewed catalog as a new immutable object and update `catalog_artifact.py`; do not check in another
catalog payload or task-level experiment artifacts.

The detailed rubric is a development tool for a small subject sample. Routine scale-out uses its compact five-part
score in one high-reasoning call per curriculum version. Sampled Luna placements and failures provide optional
boundary and difficulty evidence; this workflow does not run model-training experiments.

## Task mapping

Task mapping uses a curriculum-independent semantic key and two routing stages:

```text
model-visible TaskSpec -> semantic key -> cached domain, mechanic, and operation embeddings
curriculum graph       -> declared membership facet -> zero or more graph candidates
selected graph         -> operation embedding       -> capability candidates within that graph
```

The key records `subject_domain`, `task_mechanic`, requested result, hardest operation, required operations, and
answer form. `subject_domain` is the broad knowledge or tool environment required by the decisive operation.
Narrative content and output format do not belong in it. `task_mechanic` is a domain-neutral operation family such as
implementation, repair, supplied-context extraction, calculation, explanation, recall, or tool-state mutation. The
hardest and required operations rank capabilities after graph selection. Groups never receive assignment anchors or
task mappings.

Annotate each task once in model batches of 8–16. Persist the task-content hash, annotation prompt version, and model.
Do not include curriculum labels in the prompt. `task_mapping/cli.py` caches each embedding by text and model, derives bootstrap
anchors from curriculum outcomes and includes, accepts separate assignment anchors, and ranks with NumPy in bounded
row batches. A curriculum edit re-embeds changed anchors and reuses task vectors. A semantic-key schema change requires
reannotation; an embedding-model change requires re-embedding.

Run a pilot mapping with:

```bash
uv run python -m experiments.post_training.task_curriculum.task_mapping.cli \
  --annotations /tmp/task-annotations-000.jsonl \
  --catalog /tmp/task-curriculum-cross-domain-v2.json \
  --assignment-anchors /tmp/assignment-anchors.jsonl \
  --cache /tmp/curriculum-embeddings.sqlite \
  --embedding-model text-embedding-3-small \
  --top-k 3 \
  --output /tmp/C14-mappings.jsonl
```

The output contains a membership score and section candidates for every graph. Thresholds are calibrated independently
per graph on a frozen member, close-neighbor, overlap, and out-of-scope sample. Compare scores only within one
projection. The mapper has no source-specific rules or per-task exceptions. Mapping is a diagnostic signal and is
excluded from the curriculum promotion gate. Roughly 70% reasonable placement is adequate for initial scale-out.

For the full TaskTrove run, materialize the approximately 4 GB clean release and shard annotation and embedding work.
Keep these artifacts outside the curriculum JSON:

- semantic keys by task-content hash, annotation schema, prompt, and model;
- task vectors by semantic-key hash and embedding model;
- assignment anchors and anchor vectors by catalog and embedding model;
- graph calibration policies by catalog, anchor set, and labeled fixture; and
- assignments by task vector, catalog, anchor set, and calibration policy.

Use Parquet shards in GCS for the internal working set. A curated public snapshot can be published to Hugging Face.
The local SQLite cache is only a single-node development cache.

## Experiment history

[`HISTORY.md`](HISTORY.md) summarizes the generation, review, blind-fit, source-survey, and production-repair
experiments that led to the cross-domain v2 catalog. Full one-off inputs and outputs remain in the external evidence
artifacts named there.

## Provenance

- [UNESCO ISCED Fields of Education and Training](https://uis.unesco.org/sites/default/files/documents/isced-fields-of-education-and-training-2013-en.pdf)
- [NCES Classification of Instructional Programs 2020](https://nces.ed.gov/ipeds/cipcode/browse.aspx?y=56)
- [OECD Frascati Manual](https://www.oecd.org/en/publications/frascati-manual-2015_9789264239012-en.html)
- [O*NET Content Model](https://www.onetcenter.org/content.html)
- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [Evaluation policy #9193](https://github.com/marin-community/marin/issues/9193)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
