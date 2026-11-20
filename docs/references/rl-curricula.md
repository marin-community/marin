# RL curricula

In Marin RL, a task mix is called a **curriculum**: which task sources a run trains on, at what
weights or in what order. A curriculum is the RL analog of an SFT data mix. Sources are the catalog;
the curriculum is the policy an experiment declares over them, and the versioned curriculum object
(`RlCurriculum` in
[`marin/rl/curriculum.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/rl/curriculum.py))
records that policy so any run's task mix is recoverable from its artifact record.

## Source

A source is an `RlDataSource` entry
([source.py](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/source.py));
the catalog `runnable_sources()`
([sources.py](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py))
lists the sources a training run may sample, keyed by invocation name. Each runnable source carries
its pinned upstream dataset (name, revision, URL), its recipe version, and the curation pipeline
that produced its tasks. Curation receipts live in the [RL Data Atlas](rl-data-atlas.md).

## Curriculum

Build a curriculum with `weighted_curriculum(...)` or `ordered_curriculum(...)`
([curriculum.py](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/curriculum.py)).
Both take the pool steps the run trains on, so the curriculum's entries pin the same records the
spec's `train_data` carries. `weighted_curriculum` is the RL analog of
`mixture()`: the caller maps pool steps to weights, and each entry records the pool's artifact
identity — the provenance chain that ends in the source declaration, its pinned release and its
receipts. Two pools sharing a label are rejected because the label is the trainer's sampling key.

`weighted_curriculum(name, version, train={pool_step: weight}, seed=...)` declares a weight-random
mix. The trainer composes every training window into the exact weighted mixture, the behavior of
`weighted_quotas` in
[domain_sampling.py](https://github.com/marin-community/marin/blob/main/vendor/skyrl/skyrl-train/skyrl_train/domain_sampling.py).

`ordered_curriculum(name, version, stages=(CurriculumStage(...), ...), pools={label: pool_step},
seed=...)` declares the schedule as `CurriculumStage` objects in the order the trainer consumes
them, each stage carrying its own draw quota (an absent quota runs one exact pass over that pool's
rows, reshuffled on entry). After the last stage the sequence restarts, so a longer run cycles the
declared order. Every stage label must resolve to a pool, and every pool must be visited. The
trainer's `OrderedStages` sampler draws this sequence deterministically from the seed, and rejects
a stage sequence whose per-label demand inside one training window exceeds a pool's row count.

Both modes require an explicit `seed`. The curriculum validates itself at construction: an unpinned
pool, an empty stage order, and a non-positive weight or quota are errors.

The curriculum's `fingerprint` hashes the canonical pool entries: each one's trainer label, pool
artifact identity, and weight. It also hashes the mode, the stage order and quotas, and the seed.
The same inputs fingerprint identically; changing a pool, a weight, the stage order, or the seed
produces a different fingerprint. The curriculum's own `version` is identity, not content:
re-badging the same mix keeps the fingerprint, as with every artifact.

## Launch with a curriculum

Pass the curriculum to `SkyRLSpec` (`curriculum=...`). Construction rejects a curriculum whose
entries name pools the run does not train on, pin a different pool identity than `train_data`
carries, or pair with a recipe that declares `data.kind` other than `parquet` — curricula sample
labeled prompt pools. The input recipe stays unchanged; the curriculum's sampling projection
combines with it when the launch config is built: `kind=domain-weighted` with `domain_weights`, or
`kind=ordered` with `stages`. Rows are matched to sources by the pool's `teacher_route` column, the
same key both modes resolve.

The launch document records the full curriculum under `inputs.curriculum`: its identity, mode,
seed, sampling projection, per-pool identities, and stage order.

## Recover a run's curriculum

Every step output carries a `.artifact.json` record
([artifact.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/artifact.py)).
A SkyRL run record names its curriculum in two places:

- `result.curriculum` — the curriculum's `name@version:fingerprint` identity.
- `config` → the resolved launch YAML → `inputs.curriculum` — the full detail: sources with pins,
  weights, stage order, and seed.

```python
import yaml

from marin.execution.artifact import read_record

run = read_record("<run output path>")
print(run.result["curriculum"])  # users/<name>/curricula/<mix>@2026.11.18:<fingerprint>
launch = yaml.safe_load(run.config["launch_config_yaml"])
print(launch["inputs"]["curriculum"]["sources"])
```

The identity recovers the exact mix: rebuild the curriculum with the matching builder using the recorded
pool identities, weights, stages, and seed, and its fingerprint must equal the recorded one. A matching
fingerprint means the rebuilt mix samples identically. The pool identities tie the record to the
artifact graph, so the pinned source records behind each pool — its declaration, release pin, and
receipts — are recoverable from the run's dependency chain.
