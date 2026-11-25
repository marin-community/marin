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
The builder resolves each named source against the catalog and records, per source: the catalog
name, the source's version, its pinned dataset, and the pipeline identity. The builder rejects an
unknown source name and a source without a pinned dataset, so a curriculum always names pinned
inputs.

`weighted_curriculum(name, version, weights={source: weight}, seed=...)` declares a weight-random
mix. The trainer composes every training window into the exact weighted mixture, the behavior of
`weighted_quotas` in
[domain_sampling.py](https://github.com/marin-community/marin/blob/main/vendor/skyrl/skyrl-train/skyrl_train/domain_sampling.py).

`ordered_curriculum(name, version, stages=(...), quotas={source: quota}, seed=...)` declares the
sequence the trainer consumes in order, and `quotas` bounds each stage visit in prompts (an absent
quota runs one full pass over that source's rows). After the last stage the sequence restarts, so a
longer run cycles the declared order. The trainer's `OrderedStages` sampler draws this sequence
deterministically from the seed.

Both modes take a `seed`. The curriculum validates itself at construction: an unknown source, an
unpinned source, an empty stage order, and a non-positive weight or quota are errors.

The curriculum's `fingerprint` hashes the canonical source entries: each one's trainer label,
catalog `name@version`, pinned dataset (name, revision, URL), and pipeline identity. It also hashes
the mode, the weights, the stage order and quotas, and the seed. The same inputs fingerprint
identically; bumping a source's version, changing a weight, reordering stages, or changing the seed
produces a different fingerprint. The curriculum's own `version` is identity, not content:
re-badging the same mix keeps the fingerprint, as with every artifact.

## Launch with a curriculum

Pass the curriculum to `SkyRLSpec` (`curriculum=...`). The spec requires `data.kind=parquet` —
curricula sample labeled prompt pools — and renders the curriculum's projection into the recipe's
`data.sampling` subtree: `kind=domain-weighted` with `domain_weights`, or `kind=ordered` with
`stages`. A recipe that already declares disagreeing sampling values is rejected at construction.
Rows are matched to sources by the pool's `teacher_route` column, the same key both modes resolve.

The launch document records the full curriculum under `inputs.curriculum`: its identity, mode,
seed, sampling projection, per-source provenance, and stage order.

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
sources, weights, stages, and seed, and its fingerprint must equal the recorded one. A matching
fingerprint means the rebuilt mix samples identically.


## Derive a controlled holdout from recorded inputs

For a historical comparison, pin the original training parquets and curriculum
record, then draw the new holdout from those exact task documents. Give the
derived curriculum a new identity and record its parent. Remove all selected
holdout prompts from training, preserving the original task documents and
graders in both outputs. A changed validation
split changes the training population; do not reuse the parent fingerprint.

A published pool manifest may carry the complete typed `curriculum_definition`
when the parent source recipe versions differ from the current catalog. The
reader reconstructs `RlCurriculum` and checks both its identity and its holdout
against the manifest. The round may pin `pools.manifest_sha256`; the native
preflight refuses changed metadata before constructing its data dependencies.
No scorer regeneration or dataset conversion is implied by a new split.
