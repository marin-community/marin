# Task curriculum

This experiment turns a broad subject inventory into trainable curricula and maps tasks onto reviewed sections. It
keeps curriculum design separate from task correctness:
TaskCompendium owns model-visible task semantics and private verifier contracts; a curriculum describes observable
capabilities, boundaries, examples, and prerequisites.

`curriculum.yaml` is the canonical catalog. It contains nine reviewed curriculum graphs and 292 globally unique
sections. The Markdown reports under `pilot/` and `wave2/` preserve the experimental evidence; runtime assignment
loads only the YAML catalog.

## Inputs

- `subject_inventory.json` is the September 3, 2026 TaskTrove competency chart reduced to 34 macro areas and 151
  micro guideposts. Its labels help find omissions; generated sections must still satisfy the curriculum rubric.
- `rubric.md` defines the cross-subject review criteria.
- A small set of model-visible TaskTrove tasks supplies concrete discovery evidence. In-distribution examples listed
  by the evaluation policy are held-out coverage probes. Out-of-distribution evaluations contribute domain and task
  format metadata only. Source names, solutions, and verifier implementations are excluded from generation.
- TaskCompendium's subject, competency, task-shape, artifact, context, interaction, and state facets are optional
  vocabulary for describing task semantics.

## Curriculum iteration

One-off agent runs produce curricula and reviews. Their reports record the evidence, and reviewed changes are promoted
into the canonical YAML. For each selected macro area:

1. Give the generator the inventory area, rubric, a maximum tree depth, and representative tasks. The generator uses
   as many levels and sections as the learning criteria require. Every section includes an `entry` probe that isolates
   the smallest prerequisite delta and a `representative` probe for the full outcome.
2. Validate the JSON with `Curriculum` and `Curriculum.check_generation_contract`.
3. Give the curriculum and rubric to an independent reviewer. The reviewer returns a score, verdict, concrete
   findings, and proposed rubric changes.
4. Compare findings across subjects. Mutual self-confidence and prerequisite continuity are blocking criteria even
   when the numeric score is high. Revise the rubric or generation instructions only for problems that recur, then
   generate another version.

The checked-in `pilot/` and `wave2/` reports summarize the one-off results. Promote reviewed changes into
`curriculum.yaml`; do not add another runtime curriculum or check in task-level experiment artifacts.

## Task mapping

Task mapping uses a curriculum-independent semantic key and two routing stages:

```text
model-visible TaskSpec -> semantic key -> cached domain, mechanic, and operation embeddings
curriculum graph       -> declared membership facet -> zero or more graph candidates
selected graph         -> operation embedding       -> section candidates within that graph
```

The key records `subject_domain`, `task_mechanic`, requested result, hardest operation, required operations, and
answer form. `subject_domain` is the broad knowledge or tool environment required by the decisive operation.
Narrative content and output format do not belong in it. `task_mechanic` is a domain-neutral operation family such as
implementation, repair, supplied-context extraction, calculation, explanation, recall, or tool-state mutation. The
hardest and required operations rank sections after graph selection.

Annotate each task once in model batches of 8–16. Persist the task-content hash, annotation prompt version, and model.
Do not include curriculum labels in the prompt. `cli.py` caches each embedding by text and model, derives bootstrap
anchors from curriculum outcomes and includes, accepts separate assignment anchors, and ranks with NumPy in bounded
row batches. A curriculum edit re-embeds changed anchors and reuses task vectors. A semantic-key schema change requires
reannotation; an embedding-model change requires re-embedding.

Run a pilot mapping with:

```bash
uv run python -m experiments.post_training.task_curriculum.cli \
  --annotations /tmp/task-annotations-000.jsonl \
  --catalog experiments/post_training/task_curriculum/curriculum.yaml \
  --assignment-anchors /tmp/assignment-anchors.jsonl \
  --cache /tmp/curriculum-embeddings.sqlite \
  --embedding-model text-embedding-3-small \
  --top-k 3 \
  --output /tmp/C14-mappings.jsonl
```

The output contains a membership score and section candidates for every graph. Thresholds are calibrated independently
per graph on a frozen member, close-neighbor, overlap, and out-of-scope sample; scores from distinct projections are
not comparable. The mapper has no source-specific rules or per-task exceptions.

For the full TaskTrove run, materialize the approximately 4 GB clean release and shard annotation and embedding work.
Keep these artifacts outside the curriculum YAML:

- semantic keys by task-content hash, annotation schema, prompt, and model;
- task vectors by semantic-key hash and embedding model;
- assignment anchors and anchor vectors by catalog and embedding model;
- graph calibration policies by catalog, anchor set, and labeled fixture; and
- assignments by task vector, catalog, anchor set, and calibration policy.

Use Parquet shards in GCS for the internal working set. A curated public snapshot can be published to Hugging Face.
The local SQLite cache is only a single-node development cache.

## Wave-three findings

The frozen regression fixture contains 63 tasks and 81 independently labeled graph memberships. Fourteen tasks have
two valid graph labels and two have three, confirming that a single primary curriculum is ill posed. With corrected
facet annotations, the declared per-graph projection reached 0.627 micro average precision for membership. Applying
`subject_domain` to every graph reached 0.574; applying `task_mechanic` to every graph reached 0.421. The declared
projection retrieved at least one valid graph first for 51/63 tasks and retrieved all valid graphs in the top three
for 45/63. C03 and C21 had no positive examples, so this is a regression fixture rather than a threshold-calibration
set. Eight blind mapping anchors for each of the two practice graphs raised micro average precision to 0.832. With
those graph anchors, at least one valid graph ranked first for 56/63 tasks and all valid graphs appeared in the top
three for 57/63.

Section-local retrieval remains the larger weakness. With bootstrap outcome/include anchors and the correct graph
given, an acceptable section appeared first for 30/81 memberships and in the top three for 47/81. Assignment anchors
generated blind for all C23 and C25 sections reduced top-three retrieval to 43/81, so they were rejected. Section
anchors must be derived or selected against a calibration split and evaluated on a fresh holdout before full-corpus
mapping.

A six-edge prerequisite audit generated two prerequisite representatives, two dependent entries, and two dependent
representatives per edge. Batched Luna review passed the transfer and added-capability checks on all six. It rejected
two edges under the completed-artifact counterfactual: requirements-to-design-proposal and journal-entry-to-ledger-
posting. The canonical catalog removes those edges and supplies the upstream artifacts in the dependent probes.

## Provenance

- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
