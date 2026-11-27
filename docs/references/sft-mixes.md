# SFT data mixes

An SFT mix is [`mixture()`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/experiment/data.py)
over versioned artifact handles. Every component carries a `name@version` identity
and a recipe fingerprint. The run's `.artifact.json` record identifies its mix;
recovering the source bytes also requires immutable inputs and retained provenance.
This page defines the three layers of a mix (source, variant, mix) and the recovery
procedure.

RL task sources are cataloged as `RlDataSource` entries
([source.py](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/source.py);
`all_sources()` in
[sources.py](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py))
with pinned inputs and a
[`SourceRecipe`](https://github.com/marin-community/marin/blob/main/lib/taskcompendium/src/taskcompendium/pipeline/models.py),
and their curation receipts live in the Atlas registry
([RL Data Atlas](rl-data-atlas.md)). RL task mixes (which sources, route weights
via `weighted_quotas` in
[domain_sampling.py](https://github.com/marin-community/marin/blob/main/vendor/skyrl/skyrl-train/skyrl_train/domain_sampling.py),
input versions) currently have no versioned mix object and are out of scope for
this doc.

## Source

A source is a pinned
[`DatakitChatSource`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sft_sources.py).
`all_sft_sources()` returns the registry as `{name: DatakitChatSource}`. Each
source holds its `StepSpec` chain: download and chat-normalize steps that pin the
upstream Hugging Face revision, then `chat_normalized` (validated structured
chat), `rendered` (Marin chat template), and `normalized` (deduplicated text).
[`DatakitSource`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sources.py)
is the registry row for text and pretraining sources; its `normalize_steps` chain
ends at the normalized output that downstream steps tokenize.

Datakit sources expose `StepSpec`s, not `ArtifactStep` handles. Build the chosen
source's normalization graph first, then adopt the completed output in the
experiment that defines the mix. `apply()` does not turn a `StepSpec` into a
dependency, and `lower()` converts only from `ArtifactStep` to `StepSpec`.

For an already-built `source: DatakitChatSource`, retain its upstream revision
pins in `upstream_pins` and register the normalized output:

```python
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import tokenized

normalized = source.normalized
raw = ArtifactStep.adopt(
    f"sft/{source.name}/normalized",
    "2026.11.27",
    source=normalized.output_path,
    config={"step": normalized.name_with_hash, "upstream_pins": upstream_pins},
)
cache = tokenized(
    f"sft/{source.name}/tokenized",
    version="2026.11.27",
    tokenizer=marin_tokenizer,
    raw=raw,
    glob="outputs/main/*.parquet",
)
```

Adoption records the selected location and supplied metadata; it neither runs
Datakit's graph nor reconstructs its upstream provenance. Keep the source's
receipts and pins, and keep the adopted data immutable. The glob above selects
normalized main output and excludes the duplicate side output. A variant can
consume `raw` through `apply()` before tokenization.

Rollout-transcript corpora in
[`experiments/rollout_data/`](https://github.com/marin-community/marin/tree/main/experiments/rollout_data)
are ordinary tokenized sources: pre-recorded transcripts used like SFT or
midtraining text. They follow the same rules.

## Filtered or repaired variant

A filtered or repaired variant is an artifact step that consumes the source
handle plus the transform config: filters, label tables, decontamination drops.
Build it with
[`apply()`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/lazy.py)
or a custom `ArtifactStep` builder in `experiments/**`. A handle input
contributes its `name@version` to the step identity, and every other input is a
literal that bears identity. The variant's fingerprint covers its dependency's
`name@version` and the transform config; it does not recursively hash the
dependency's config. Record source pins in the adoption config and bump the
adopted handle's version when those pins change.

Bump the variant's explicit version whenever its recipe changes, including a
filter threshold. A changed config changes the fingerprint, but a fixed
`name@version` still resolves to `{prefix}/{name}/{version}`. If that output
already exists, the default drift check warns and reuses it. A new version gives
the changed recipe a distinct output location. Bump downstream tokenized-cache
and training versions when their dependency identities change too.

## Mix

`mixture(ctx, train={handle: weight}, shuffle=...)`
([data.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/experiment/data.py))
assembles the training data config from `ArtifactStep[TokenizedCache]` handles.
`validation=` adds handles at weight 0.
[`train_lm(datasets={handle: weight})`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/experiment/train.py)
calls `mixture()` internally and derives the step's dependencies from the same
handles. The stage weights and the shuffle policy (`shuffle`, `permutation_type`)
are fields of the resulting `LmDataConfig`, so they run with the mix and land in
the record.

At fingerprint time the mix contributes the sorted `{name@version: weight}` map.
At run time each component is built from its cache's `TokenizedCache` record
(tokenizer, format, path), and all components must share one tokenizer.

Adoption covers data that already exists:

- `pretokenized(name, repo_id=..., tokenizer=..., version=...)` adopts a prebuilt
  Levanter cache hosted on Hugging Face.
- `ArtifactStep.adopt(name, version, source=..., kind=..., config=...)` registers
  a pre-existing output as a managed `name@version` handle without moving or
  recomputing it.
- `pin=` on the builders references existing data at a fixed location.

The chat-SFT launcher (`SFTSpec` in
[experiments/sft/launcher.py](https://github.com/marin-community/marin/blob/main/experiments/sft/launcher.py))
addresses each component as a `DatasetSpec` (Hugging Face id plus revision pin)
or an `ArtifactDatasetSpec` (artifact handle). Keep the same rule there: pin the
revision or name the artifact.

## Recover a run's mix

Produced artifact handles use `{prefix}/{name}/{version}` output paths and carry
a `.artifact.json` record
([artifact.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/artifact.py),
`ArtifactRecord`). `read_record(output_path)` reads it back. The record carries:

| Field | Meaning |
|---|---|
| `name`, `version` | The handle identity `name@version`. |
| `fingerprint`, `fingerprint_payload` | The recipe fingerprint and the canonical config JSON it hashes. |
| `deps`, `dep_paths` | Dependency identities as `name@version`, and where each dependency's record lives. |
| `config` | The materialized config that ran. |
| `provenance` | Who, when, which commit, and which argv produced the output. |

For a training run or checkpoint, read the record at the run's output path. The
`config` holds the materialized config that ran. For a `train_lm` run it is the
`TrainLmOnPodConfig`, and the data config is `config["train_config"]["data"]`:

- `train_weights` — `{component: weight}`.
- `components` — keyed by component name; each component's `cache_dir` points at
  the component's tokenized cache. A produced handle resolves to a
  `{name}/{version}` path; an adopted or pinned handle resolves to its
  pre-existing location, which may carry no record.
- `shuffle`, `permutation_type`, `tokenizer`.

Produced component paths carry records with `name@version`, `fingerprint`, and
`deps`. For adopted data, read the registration record at its canonical
`{prefix}/{name}/{version}` path; the source location may have no record. Trace
the dependency records and adoption config back to the source pins and transform
configs. Missing receipts or mutable adopted data prevent exact byte recovery.

```python
from marin.execution.artifact import read_record

run = read_record("<run output path>")
data = run.config["train_config"]["data"]
print(data["train_weights"])  # {"fineweb-edu-10M": 1.0}
for name, component in data["components"].items():
    cache = read_record(component["cache_dir"])
    if cache is None:
        print(name, "no record at the resolved path")
    else:
        print(name, cache.name, cache.version, cache.fingerprint)
```

## Example

`experiments/tutorials/train_tiny_model.py` defines a mix of one adopted cache
and trains on it:

```python
handle = pretokenized(
    "fineweb-edu-10M",
    repo_id="marin-community/fineweb-edu-pretokenized-10M",
    tokenizer=marin_tokenizer,
    version="2026.06.28",
)
train_lm(
    name="checkpoints/tiny-fineweb-edu-h100x8",
    datasets={handle: 1.0},
    ...
)
```

`train_lm` derives the checkpoint step's dependencies from `datasets`, so the
cache materializes first and the data config cannot drift from the dependencies.

## Raw path components

A training component addressed by a literal `s3://` or `gs://` cache prefix, or
by a bare directory prefix, has no `name@version` identity. The path's contents
can change under the run, the handle-based metadata and drift checks do not apply,
and the run's record alone cannot identify the source bytes. Pass handles
instead, and adopt existing data with `pretokenized()` or `adopt()`. The advisory
lint rule `ml-raw-path-training-data`
([infra/lint/rules](https://github.com/marin-community/marin/tree/main/infra/lint/rules/robustness))
flags raw-path training components.
