# Lazy artifacts

Marin's execution model is built around **lazy artifacts**: typed, identity-keyed handles
that describe what to build without doing any work at definition time. An experiment script
constructs `ArtifactStep[T]` handles, lowers them to a runnable step graph, and hands the
graph to `StepRunner`. Tokenization, training, and every other step execute exactly once and
are cached for future runs.

## Artifacts: `name@version` handles

A **handle** is an `ArtifactStep[T]` — a frozen object whose identity is its `name` and
`version`. Constructing one does not run anything. The type parameter `T` is the
materialized result type produced when the step runs:

- `ArtifactStep[TokenizedCache]` — a handle whose step produces a tokenized dataset
  (consumed by `train_lm` and other data assemblers).
- `ArtifactStep[LevanterCheckpoint]` — a handle whose step produces a Levanter model
  checkpoint (consumed by `train_lm`'s `init_from` and by eval steps).

The result types are `Artifact` subclasses that live with their producers:
`TokenizedCache` in `marin.processing.tokenize.tokenize`, `LevanterCheckpoint` in
`marin.training.training`. They describe the materialized result, not the handle itself.
Import the handle from `marin.execution.lazy`:

```python
from marin.execution.lazy import ArtifactStep              # the handle
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint     # materialized result types
```

A handle's storage address is its explicit `{prefix}/{name}/{version}` path. There is no
content hash in the path. The `version` is a calendar version, `YYYY.MM.DD` (with an
optional `.N` suffix), or a `dev` string during development. Changing the name or version
creates a distinct artifact; changing how an existing `name@version` is built produces an
advisory drift warning and serves the cached output (see [Advisory drift](#advisory-drift)).

In practice, experiment scripts rarely construct `ArtifactStep(...)` directly. Helpers such
as `tokenized`, `hf_download`, and `train_lm` return the appropriate `ArtifactStep[T]` handle
from high-level arguments.

## Steps and StepContext

An `ArtifactStep` is flat — there is no separate recipe object. Its fields split cleanly
into identity (`name`, `version`, `artifact_type`) and build instructions:

```python
ArtifactStep(
    name="checkpoints/my-run",        # identity: the {name}/{version} address
    version="2026.06.28",             # identity: calendar version
    artifact_type=LevanterCheckpoint, # the materialized result type
    run=_train_job,                   # the step fn, or remote(fn, resources=…)
    build_config=build_config,        # assembles the config from the StepContext
    deps=(dataset,),                  # upstream handles this step reads
    runtime_args={"train_resources": resources},  # execution choices, excluded from identity
)
```

`build_config` is a pure function of a `StepContext`. The context is the dividing line
between what an artifact *is* and *where/how* it runs:

```python
@dataclass(frozen=True)
class StepContext:
    output_path: str    # this artifact's output path (excluded from fingerprint)
    prefix: str         # the live storage prefix (excluded)
    region: str | None  # the GCP region, resolved at run time (excluded)
    is_fingerprint: bool # True only while computing the fingerprint

    def artifact_path(self, dep: ArtifactStep) -> str: ...  # a dependency's output path
    def runtime_arg(self, key: str) -> Any: ...             # a step-declared runtime arg
```

Values written as literals in `build_config` — model architecture, hyperparameters, dep
versions — define the artifact and enter its fingerprint. Values pulled from `ctx` —
output paths, the prefix, the region, compute resources — are execution choices and are
excluded from the fingerprint.

A concrete example from `experiments/tutorials/exp1078_reproduce_dclm_7b1x.py`:

```python
return train_lm(
    name="checkpoints/dclm_7b_1x_how_to",
    version="2026.06.28",
    model=llama_7b_dclm,         # literal → bears identity
    optimizer=AdamConfig(
        learning_rate=2e-3,      # literal → bears identity
        ...
    ),
    datasets=weighted,           # literal → bears identity
    validation=validation,
    batch_size=BATCH_SIZE,       # literal → bears identity
    ...
    resources=TRAIN_RESOURCES,   # runtime arg → excluded from fingerprint
)
```

`train_lm` constructs the `ArtifactStep` internally and passes `resources` as a
`runtime_arg`, so changing the TPU never re-fingerprints the checkpoint.

## Lowering and running

`lower(handle)` converts an `ArtifactStep[T]` graph into a `StepSpec` graph that
`StepRunner` can execute; it traverses dependencies recursively. `run(*handles)` lowers and
builds them (independent steps in parallel) and **returns each handle's loaded, typed
artifact**, in argument order — so the driver gets the resolved values directly, no separate
read. `resolve(handle)` is `run` for a single handle, returning its one artifact.

```python
from marin.execution.lazy import run

checkpoint = build()                      # returns ArtifactStep[LevanterCheckpoint]; nothing runs yet
result = resolve(checkpoint)              # materializes deps, runs training, returns LevanterCheckpoint
print(result.training_metrics().eval_loss)
```

Inside a step you never read a dependency this way — use `ctx.resolved(dep)`, which the runner
guarantees is materialized first. `run`/`resolve` are the driver-level entry points.

`StepRunner.run` applies two guards before executing each step:

1. **Cache check** — if the output path already contains a completed artifact record, skip.
2. **Lock** — prevents concurrent processes from building the same step twice.

The standard `main` block in a Marin experiment:

```python
if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

## Advisory drift

Each computed artifact records a **fingerprint** — a hash of its config, built with
dependency versions substituted in place of paths and placeholders for every value drawn
from `ctx`. The fingerprint captures hyperparameters and dep identities but is independent
of the storage prefix, region, or hardware.

When the runner encounters an existing artifact record, it compares the current fingerprint
against the recorded one. A mismatch is **advisory**: the runner logs a warning and serves
the cached output rather than raising an error. To produce a new result from an updated
build, bump the `version`:

```python
# bump version to force a rebuild when the build changes
return train_lm(name="checkpoints/dclm_7b_1x", version="2026.07.01", ...)
```

During development, use a `dev` version string (e.g. `"dev"` or `"mylabel-dev"`) to opt out
of the cache entirely and always rebuild.

## Adopting pre-existing data

`adopt` brings data that already exists on disk into the lazy artifact graph without moving
or recomputing it:

```python
from marin.execution.lazy import adopt
from marin.processing.tokenize.tokenize import TokenizedCache

dclm_tokenized = adopt(
    "tokenized/dclm-baseline",
    "2026.06.28",
    source="gs://marin-us-central1/tokenized/dclm/...",
    kind=TokenizedCache,
)
```

A consumer that depends on `dclm_tokenized` resolves to the source path. Lowering writes
a provenance record at the canonical `{prefix}/{name}/{version}` path so the advisory drift
check governs the alias — re-adopting the same `name@version` from a different source logs
a warning.

Use `adopt` for datasets tokenized outside the current experiment graph, for checkpoints
produced by a separate run, or for any pre-existing artifact that needs to appear in the
dependency graph with full provenance tracking.

## Data builders

`marin.experiment.data` provides concise builders that return `ArtifactStep[TokenizedCache]`
handles:

- `tokenized(name, *, tokenizer, version, source=…, paths=…, raw=…, glob=…)` — tokenize a
  HuggingFace dataset or a raw path glob into a Levanter cache. Provide exactly one of
  `source`, `paths`, or `raw+glob`.
- `hf_download(name, *, hf_id, revision, version, urls_glob=…)` — download a HuggingFace
  repo to GCS as a raw-data `ArtifactStep[Artifact]` for `tokenized` to consume.
- `raw_download(name, *, fn, build_config, version)` — a raw-data download driven by a custom
  `fn`/`build_config`, for sources `hf_download` does not cover.
- `pretokenized(name, *, repo_id, tokenizer, version)` — a pre-built Levanter cache hosted on
  HuggingFace (downloads rather than re-tokenizes).

For a custom single step (a filter, conversion, or format change), construct an
`ArtifactStep(...)` directly, or use `apply` for the common keyword-input shape:

```python
from marin.execution.lazy import OUT, apply

# fn receives the resolved inputs as keyword arguments; OUT resolves to the output path,
# and any ArtifactStep input is resolved to its output path automatically.
shards = apply("data/sharded", shard_parquet, version="2026.06.28", source=raw, output_path=OUT)
```

## How this differs from the old content-addressed executor

The previous executor assigned each step an output path containing a hash of the step's
config and all its transitive dependencies. A hyperparameter change silently re-addressed
every downstream step, and the path gave no clue about what version of the data it held.

Lazy artifacts address data by an explicit `{name}/{version}`. The fingerprint still
catches accidental config drift (the advisory warning) and can be enforced by bumping the
version at review time, but the storage path is stable and human-readable. Bumping a version
is an explicit author decision, not an automatic consequence of any config change.
