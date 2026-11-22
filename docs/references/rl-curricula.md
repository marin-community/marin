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
artifact identity, and weight. It also hashes the mode, the stage order and quotas, the seed, and
the validation split. The same inputs fingerprint identically; changing a pool, a weight, the stage
order, the seed, or the held-out rows produces a different fingerprint. The curriculum's own
`version` is identity, not content: re-badging the same mix keeps the fingerprint, as with every
artifact.

## Validation split

Fixing a curriculum fixes its validation split: a random holdout drawn from the curriculum's own
training distribution, recorded in the curriculum's metadata, and removed from the pools the run
trains on. The split is not a hand-composed file — the held-out signal is a random subset of what
training samples, so its mix matches the curriculum's mixture.

`weighted_curriculum` and `ordered_curriculum` therefore require an explicit `validation_size`
(there is no default) and each pool's row identities (`pool_rows`, keyed by pool label, rows as
`PoolRow(id, sha256)` — a `task_spec.id` plus the canonical SHA-256 of its task document). The
draw (`draw_validation_split` in
[curriculum.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/rl/curriculum.py))
partitions every pool row:

- each label receives a largest-remainder allocation of `validation_size`, proportional to the
  curriculum's own mixture (weighted mode: entry weights; ordered mode: each pool's consumed
  prompts — its stage quotas, capped by the pool's row count);
- within a label, rows are selected uniformly without replacement by the
  `sha256(f"{seed}:{label}:{id}")` order, so the draw is deterministic for a seed and independent
  of enumeration order;
- the result is a `ValidationSplit`: the recorded `CurriculumValidation` (size, seed, and the held-out
  `ValidationRow`s) plus the disjoint training remainder. A pool builder writes the training pools
  from the remainder and extracts the held-out rows into the validation data, so the trainer never
  sees them and the holdout is true.

The per-label counts match the mixture exactly up to integer apportionment — a 0.7/0.3 mix with
`validation_size=200` holds out 140/60 rows. The recorded split travels with the curriculum: the
fingerprint covers it, so two curricula that differ only in their split are different objects.

## Launch with a curriculum

Pass the curriculum to `SkyRLSpec` (`curriculum=...`). Construction rejects a curriculum whose
entries name pools the run does not train on, pin a different pool identity than `train_data`
carries, or pair with a recipe that declares `data.kind` other than `parquet` — curricula sample
labeled prompt pools. The input recipe stays unchanged; the curriculum's sampling projection
combines with it when the launch config is built: `kind=domain-weighted` with `domain_weights`, or
`kind=ordered` with `stages`. Rows are matched to sources by the pool's `teacher_route` column, the
same key both modes resolve.

The launch document records the full curriculum under `inputs.curriculum`: its identity, mode,
seed, sampling projection, per-pool identities, stage order, and the validation split (size, seed,
held-out rows). The run's validation data is the materialization of that record — the held-out rows
extracted from the pinned pools — and a run's best-checkpoint gating derives its expected scored
tasks from the split's size.

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
pool identities, weights, stages, seed, and the recorded `pool_rows` and `validation_size`, and its
fingerprint must equal the recorded one — the held-out rows must reproduce too. A matching
fingerprint means the rebuilt mix samples identically and holds out the same prompts. The pool identities tie
the record to the artifact graph, so the pinned source records behind each pool — its declaration, release pin,
and receipts — are recoverable from the run's dependency chain.
