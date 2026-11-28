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

For an already-built `source: DatakitChatSource`, register its normalized output.
The example also expects the tokenizer identifier in `marin_tokenizer`:

```python
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import tokenized

normalized = source.normalized
raw = ArtifactStep.adopt(
    f"sft/{source.name}/normalized",
    "2026.11.27",
    source=normalized.output_path,
    config={"step": normalized.name_with_hash},
)
cache = tokenized(
    f"sft/{source.name}/tokenized",
    version="2026.11.27",
    tokenizer=marin_tokenizer,
    raw=raw,
    glob="outputs/main/*.parquet",
)
```

The recorded `step` identifies the normalization chain: `StepSpec` hashes include
its dependency identities and hash attributes, including upstream revisions when
the source declares them. Retain the source definitions and receipts to interpret
that identity, and keep the adopted data immutable. The glob selects normalized
main output and excludes the duplicate side output. A variant can consume `raw`
through `apply()` before tokenization. See [adoption semantics](../explanations/lazy-artifacts.md#adopting-pre-existing-data).

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
dependency's config. Bump the adopted source version when its normalization
chain changes.

Bump the variant's explicit version whenever its recipe changes, including a
filter threshold. Bump downstream tokenized-cache and training versions when
their dependency identities change too. Follow the canonical
[drift and versioning rules](../explanations/lazy-artifacts.md#advisory-drift);
a fingerprint change alone does not schedule a fresh build of a fixed version.

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

For data that already exists:

- `pretokenized(name, repo_id=..., tokenizer=..., version=...)` adopts a prebuilt
  Levanter cache hosted on Hugging Face.
- `ArtifactStep.adopt(name, version, source=..., kind=..., config=...)` registers
  a pre-existing output as a managed `name@version` handle without moving or
  recomputing it.
- Seal steps produce records that pin external cache catalogs and source-entry
  digests, without recomputing the data. [Exp3 V5](#production-sft-example) uses
  this pattern for caches built outside its artifact graph.
- `pin=` on the builders references existing data at a fixed location and writes
  no record. `read_only=True` adoption also writes no registration record.

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
| `deps` | Dependency identities as `name@version`. |
| `dep_paths` | Optional resolved dependency paths. Lazy handle records currently leave this empty; do not rely on it for recovery. |
| `config` | The materialized config that ran. |
| `source` | The original data location for a writable adopted handle. |
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

Walk the run record's `deps` to the canonical records for produced, writable
adopted, and sealed handles. A component's `cache_dir` is a data location: it may
point at an adopted source with no record, or a shared cache root used by many
seals. It is not a reliable address for the component's record.

The following walk assumes every dependency uses the known `record_prefix`.
For graphs with per-handle `output_path_prefix` overrides, recover those prefixes
from the experiment definition and use each handle's lowered output path.
A `pin=` or read-only adoption contributes an identity but writes no record;
retain its producer's receipt or external pin separately.

```python
from marin.execution.artifact import read_record
from rigging.filesystem.storage_path import prefix_join

record_prefix = "<artifact prefix used by this run>"
run = read_record("<run output path>")
if run is None:
    raise FileNotFoundError("run has no artifact record")
data = run.config["train_config"]["data"]
print(data["train_weights"])
pending = list(run.deps)
seen = set()
while pending:
    identity = pending.pop()
    if identity in seen:
        continue
    seen.add(identity)
    name, version = identity.rsplit("@", 1)
    record = read_record(prefix_join(record_prefix, f"{name}/{version}"))
    if record is None:
        print(identity, "no canonical record; inspect the producer receipt")
        continue
    print(identity, record.fingerprint, record.source, record.config)
    pending.extend(record.deps)
```

## Production SFT example

[Exp3 V5](https://github.com/marin-community/marin/blob/main/experiments/sft_new_mix/exp3_v5_mix.py)
expresses the production SFT release as adopted release/catalog documents and
434 `TokenizedCache` seal steps. Each seal records its source's catalog-entry
SHA-256, row/token/shard counts, loss mode, and packing configuration. The catalog
pins the content-addressed shard paths; plain adoption of a mutable root does
not provide those content pins.

A consumer calls `exp3_v5_data_config(ctx)` in its `build_config` and declares
`exp3_v5_mix_deps()` as dependencies. Those dependencies include the release,
cache catalog, every source seal, and `exp3_v5_seal_check()`. The check verifies
the catalog digest, source ledgers and shard existence before use. The data
config retains the release's scheduled weights, shuffle settings and validation
reservations.

The seal records live at each source's `name@version` address, while their
`config.cache_dir` points at a shared cache root. Follow dependency identities
as above, then use the recorded catalog digest and source-entry digests to
recover the data provenance. The module's default CLI prints the graph without
reading remote data; `--run` builds it and performs the seal check.

## Raw path components

A training component addressed by a literal `s3://` or `gs://` cache prefix, or
by a bare directory prefix, has no `name@version` identity. The path's contents
can change under the run, the handle-based metadata and drift checks do not apply,
and the run's record alone cannot identify the source bytes. Pass handles
instead, and adopt existing data with `pretokenized()` or `adopt()`. The advisory
lint rule `ml-raw-path-training-data`
([infra/lint/rules](https://github.com/marin-community/marin/tree/main/infra/lint/rules/robustness))
flags raw-path training components.
