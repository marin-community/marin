# Task curriculum pilot

This experiment tests a small, repeatable loop for turning a broad subject inventory into trainable curricula and
then mapping tasks onto the reviewed sections. It keeps curriculum design separate from task correctness:
TaskCompendium owns model-visible task semantics and private verifier contracts; a curriculum describes observable
capabilities, boundaries, examples, and prerequisites.

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

One-off agent runs produce curricula and reviews. The repository retains their inputs and results. For each selected
macro area:

1. Give the generator the inventory area, rubric, a maximum tree depth, and representative tasks. The generator uses
   as many levels and sections as the learning criteria require. Every section includes an `entry` probe that isolates
   the smallest prerequisite delta and a `representative` probe for the full outcome.
2. Validate the JSON with `Curriculum` and `Curriculum.check_generation_contract`.
3. Give the curriculum and rubric to an independent reviewer. The reviewer returns a score, verdict, concrete
   findings, and proposed rubric changes.
4. Compare findings across subjects. Mutual self-confidence and prerequisite continuity are blocking criteria even
   when the numeric score is high. Revise the rubric or generation instructions only for problems that recur, then
   generate another version.

The checked-in `pilot/` directory contains the resulting curricula and reviews. The artifacts are evidence for
changing the rubric; they are not canonical training taxonomies.

## Task mapping

Task mapping uses a curriculum-independent semantic key:

```text
model-visible TaskSpec -> semantic key -> cached task embedding
reviewed curriculum    -> section anchors -> cached anchor embeddings
cached vectors         -> batched cosine ranking -> top section candidates
```

The semantic key records the operational subject, requested result, hardest operation, required operations, and
answer form. The subject is the domain of work the solver performs. Narrative details are omitted. For example, a
calendar request about a nursing workshop has the subject calendar scheduling.
Annotate it once in model batches of 8–16 and persist it with the task-content hash, annotation prompt version, and
model. Do not include a curriculum label in the annotation prompt. `cli.py` embeds those reviewed keys, caches each
vector by text and embedding model, derives section anchors from outcomes and sample tasks, and ranks sections with
NumPy in bounded row batches. Repeating `--annotations` and `--curriculum` builds one diagnostic index over several
shards and curriculum graphs. A curriculum edit therefore re-embeds changed anchors and reuses all task vectors.
Changing the semantic-key schema or annotation model requires reannotation; changing the embedding model requires
re-embedding.

Run a pilot mapping with:

```bash
uv run python -m experiments.post_training.task_curriculum.cli \
  --annotations experiments/post_training/task_curriculum/pilot/C14/task_annotations.jsonl \
  --curriculum experiments/post_training/task_curriculum/pilot/C14/curriculum_v4.json \
  --cache /tmp/curriculum-embeddings.sqlite \
  --embedding-model text-embedding-3-small \
  --top-k 3 \
  --output /tmp/C14-mappings.jsonl
```

The mapper has no source-specific rules. Use the flat multi-curriculum index as a diagnostic baseline. The
[six-area scale-out](wave2/README.md) shows that the catalog mixes overlapping domain and practice graphs. Review
low-similarity tasks, unstable top candidates, and repeated out-of-scope examples as possible anchor, boundary, or
coverage failures. Do not add per-source exceptions.

For the full TaskTrove run, materialize the approximately 4 GB clean release, render only model-visible task text,
and shard annotation and embedding work. Store annotation and vector artifacts under content and model identities;
use the same ranking computation per task shard. The pilot uses SQLite on one node; a distributed run needs sharded
artifact storage under the same content identities.

## Provenance

- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
