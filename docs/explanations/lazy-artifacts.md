# Lazy artifacts

Marin's execution model is built around **lazy artifacts**: typed, identity-keyed handles
that describe what to build without doing any work at definition time. An experiment script
constructs `Lazy[T]` handles, lowers them to a runnable step graph, and hands the graph to
`StepRunner`. Tokenization, training, and every other step execute exactly once and are
cached for future runs.

## Artifacts: `name@version` handles

A **handle** is a `Lazy[T]` — a frozen object whose identity is its `name` and `version`.
Constructing one does not run anything. The type parameter `T` is the materialized result
type produced when the step runs:

- `Lazy[Dataset]` — a handle whose step produces a tokenized dataset (consumed by
  `train_lm` and other data assemblers).
- `Lazy[Checkpoint]` — a handle whose step produces a Levanter model checkpoint (consumed
  by `train_lm`, `init_from`, and eval steps).

`Dataset`, `Checkpoint`, and `Artifact` are pydantic value types in
`marin.execution.artifact`; they describe the materialized result, not the handle itself.
Import the handle wrapper from `marin.execution.lazy`:

```python
from marin.execution.artifact import Dataset, Checkpoint   # materialized result types
from marin.execution.lazy import Lazy                       # handle wrapper
```

A handle's storage address is its explicit `{prefix}/{name}/{version}` path. There is no
content hash in the path. Changing the name or version creates a distinct artifact;
changing the recipe for an existing `name@version` produces an advisory drift warning and
serves the cached output (see [Advisory drift](#advisory-drift)).

In practice, experiment scripts rarely construct `Lazy(...)` directly. Helpers such as
`tokenized`, `hf_download`, `derived`, and `train_lm` return the appropriate `Lazy[T]`
handle from high-level arguments.

## Recipes and RunContext

A `Recipe` is the build specification attached to a handle:

```python
@dataclass(frozen=True)
class Recipe:
    fn: Callable                              # step function, or remote(fn, resources=…)
    build_config: Callable[[RunContext], Any] # assembles the config from the run context
    deps: tuple[Lazy[...], ...]               # upstream handles this step reads
    run_args: Mapping[str, Any]               # execution choices excluded from fingerprint
```

`build_config` is a pure function of a `RunContext`. The context is the dividing line
between what an artifact *is* and *where/how* it runs:

```python
@dataclass(frozen=True)
class RunContext:
    out: str           # this artifact's output path (excluded from fingerprint)
    prefix: str        # the live storage prefix (excluded)
    region: str | None # the GCP region, resolved at run time (excluded)

    def path(self, dep: Lazy[...]) -> str: ...   # dependency's output path (excluded)
    def run_arg(self, key: str) -> Any: ...      # recipe-declared run-arg (excluded)
```

Values written as literals in `build_config` — model architecture, hyperparameters, dep
versions — define the artifact and enter its fingerprint. Values pulled from `ctx` —
output paths, the prefix, the region, compute resources — are execution choices and are
excluded from the fingerprint.

A concrete example from `experiments/tutorials/exp1078_reproduce_dclm_7b1x.py`:

```python
return train_lm(
    name="checkpoints/dclm_7b_1x_how_to",
    version="v1",
    model=llama_7b_dclm,         # literal → bears identity
    optimizer=AdamConfig(
        learning_rate=2e-3,      # literal → bears identity
        ...
    ),
    datasets=weighted,           # literal → bears identity
    validation=validation,
    batch_size=BATCH_SIZE,       # literal → bears identity
    ...
    resources=TRAIN_RESOURCES,   # run-arg → excluded from fingerprint
)
```

`train_lm` builds the `Recipe` internally. The `resources` argument goes into `run_args`
so changing the TPU never re-fingerprints the checkpoint.

## Lowering and running

`lower(handle)` converts a `Lazy[T]` graph into a `StepSpec` graph that `StepRunner` can
execute. It traverses dependencies recursively.

```python
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

checkpoint = build()                      # returns Lazy[Checkpoint]; nothing runs yet
StepRunner().run([lower(checkpoint)])     # materializes deps, then runs training
```

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
recipe, bump the `version`:

```python
# bump version to force a rebuild when the recipe changes
return train_lm(name="checkpoints/dclm_7b_1x", version="v2", ...)
```

During development, use a `dev` version string (e.g. `"v1-dev"`) to opt out of the cache
entirely and always rebuild.

## Adopting pre-existing data

`adopt` brings data that already exists on disk into the lazy artifact graph without moving
or recomputing it:

```python
from marin.execution.artifact import Dataset
from marin.execution.lazy import adopt

dclm_tokenized = adopt(
    "tokenized/dclm-baseline",
    "v1",
    source="gs://marin-us-central1/tokenized/dclm/...",
    kind=Dataset,
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

`marin.experiment.data` provides concise builders that return `Lazy[Dataset]` handles:

- `tokenized(name, *, tokenizer, source=…, paths=…, raw=…, glob=…)` — tokenize a HuggingFace
  dataset or a raw path glob into a Levanter cache. Provide exactly one of `source`, `paths`,
  or `raw+glob`.
- `hf_download(name, *, hf_id, revision, urls_glob=…)` — download a HuggingFace repo to GCS
  as a raw-data handle for `tokenized` to consume.
- `pretokenized(…)` — a pre-built Levanter cache hosted on HuggingFace (downloads rather
  than re-tokenizes).

For custom single-step transforms (filter, conversion, format change), `marin.execution.lazy`
provides two generic builders:

- `derived(name, *, fn, build_config, deps=…, kind=Artifact)` — the general form; `fn`
  receives a typed config assembled by `build_config(ctx)`.
- `apply(name, fn, *, version="v1", result_type=Artifact, **inputs)` — a simpler form
  where `fn` receives keyword arguments directly; bare `Lazy` inputs are resolved to paths
  automatically.

## How this differs from the old content-addressed executor

The previous executor assigned each step an output path containing a hash of the step's
config and all its transitive dependencies. A hyperparameter change silently re-addressed
every downstream step, and the path gave no clue about what version of the data it held.

Lazy artifacts address data by an explicit `{name}/{version}`. The fingerprint still
catches accidental config drift (the advisory warning) and can be enforced by bumping the
version at review time, but the storage path is stable and human-readable. Bumping a version
is an explicit author decision, not an automatic consequence of any config change.
