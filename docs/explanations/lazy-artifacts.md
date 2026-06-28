# Lazy artifacts

Marin's execution model is built around **lazy artifacts**: typed, identity-keyed handles
that describe what to build without doing any work at definition time. An experiment script
constructs artifact handles, lowers them to a runnable step graph, and hands them to
`StepRunner`. Tokenization, training, and every other step execute exactly once and are
cached for future runs.

## Artifacts: `name@version` handles

An `Artifact` is a frozen Python dataclass whose identity is its `name` and `version`.
Constructing one does not run anything. Two concrete subtypes route handles to their
consumers:

- `Dataset` — a tokenized dataset (consumed by `mixture` and other data assemblers).
- `Checkpoint` — a Levanter model checkpoint (consumed by `train_lm`, `init_from`, and
  eval steps).

An artifact's storage address is its explicit `{prefix}/{name}/{version}` path. There is
no content hash in the path. Changing the name or version creates a distinct artifact;
changing the recipe for an existing `name@version` is a guarded conflict (see
[Build-once immutability](#build-once-immutability)).

In practice, experiment scripts rarely construct `Artifact` objects directly. Helpers such
as `tokenized`, `hf_download`, `derived`, and `train_lm` build the appropriate handle and
recipe from high-level arguments.

## Recipes and RunContext

A `Recipe` is the build specification attached to an artifact:

```python
@dataclass(frozen=True)
class Recipe:
    fn: Callable                              # step function, or remote(fn, resources=…)
    build_config: Callable[[RunContext], Any] # assembles the config from the run context
    deps: tuple[Artifact, ...]                # upstream handles this step reads
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

    def path(self, dep: Artifact) -> str: ...   # dependency's output path (excluded)
    def run_arg(self, key: str) -> Any: ...     # recipe-declared run-arg (excluded)
```

Values written as literals in `build_config` — model architecture, hyperparameters, dep
versions — define the artifact and enter its fingerprint. Values pulled from `ctx` —
output paths, the prefix, the region, compute resources — are execution choices and are
excluded from the fingerprint.

A concrete example from `experiments/tutorials/dclm_1b_1x_inline.py`:

```python
return train_lm(
    name="checkpoints/dclm_1b_1x_how_to",
    version="v1",
    model=llama_1_4b_dclm,      # literal → bears identity
    optimizer=AdamConfig(
        learning_rate=3e-3,      # literal → bears identity
        ...
    ),
    data=lambda ctx: mixture(ctx, weighted, validation=validation),
    deps=(*weighted, *validation),
    batch_size=BATCH_SIZE,       # literal → bears identity
    ...
    resources=TRAIN_RESOURCES,   # run-arg → excluded from fingerprint
)
```

`train_lm` builds the `Recipe` internally. The `resources` argument goes into `run_args`
so changing the TPU never re-fingerprints the checkpoint.

## Lowering and running

`lower(artifact)` converts a handle graph into a `StepSpec` graph that `StepRunner` can
execute. It traverses dependencies recursively.

```python
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

checkpoint = build()                      # returns a lazy Checkpoint; nothing runs yet
StepRunner().run([lower(checkpoint)])     # materializes deps, then runs training
```

`StepRunner.run` applies three guards before executing each step:

1. **Cache check** — if the output path already contains a completed artifact record, skip.
2. **Lock** — prevents concurrent processes from building the same step twice.
3. **Build-once guard** — if the artifact has been built before, verifies the fingerprint
   matches before running again.

The standard `main` block in a Marin experiment:

```python
if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

## Build-once immutability

Each computed artifact records a **fingerprint** — a hash of its config, built with
dependency versions substituted in place of paths and placeholders for every value drawn
from `ctx`. The fingerprint captures hyperparameters and dep identities but is independent
of the storage prefix, region, or hardware.

When the runner encounters an existing artifact record, it compares the current fingerprint
against the recorded one. A mismatch is a guarded conflict: the artifact was built with
different config and its output is immutable. To produce a new result, bump the `version`.

To pin the fingerprint at authoring time, set `expected_fingerprint` on the artifact:

```python
checkpoint = Checkpoint(
    name="checkpoints/dclm_1b_1x",
    version="v2",
    recipe=...,
    expected_fingerprint="abc123...",  # lower() raises if config no longer matches
)
```

This makes config changes review-visible: a reviewer sees the pin update and understands
that the artifact's identity changed.

## Adopting pre-existing data

`adopt` brings data that already exists on disk into the lazy artifact graph without moving
or recomputing it:

```python
from marin.execution.lazy import adopt, Dataset

dclm_tokenized = adopt(
    "tokenized/dclm-baseline",
    "v1",
    source="gs://marin-us-central1/tokenized/dclm/...",
    kind=Dataset,
)
```

A consumer that depends on `dclm_tokenized` resolves to the source path. Lowering writes
a provenance record at the canonical `{prefix}/{name}/{version}` path so the build-once
guard governs the alias — re-adopting the same `name@version` from a different source is a
guarded conflict.

Use `adopt` for datasets tokenized outside the current experiment graph, for checkpoints
produced by a separate run, or for any pre-existing artifact that needs to appear in the
dependency graph with full provenance tracking.

## Data builders

`marin.experiment.data` provides concise builders that return `Dataset` handles:

- `tokenized(name, *, tokenizer, source=…, paths=…, raw=…, glob=…)` — tokenize a HuggingFace
  dataset or a raw path glob into a Levanter cache. Provide exactly one of `source`, `paths`,
  or `raw+glob`.
- `hf_download(name, *, hf_id, revision, urls_glob=…)` — download a HuggingFace repo to GCS
  as a raw-data handle for `tokenized` to consume.
- `derived(name, *, fn, build_config, deps=…)` — a generic single-step transform (filter,
  conversion, format change).
- `pretokenized(…)` — a pre-built Levanter cache hosted on HuggingFace (downloads rather
  than re-tokenizes).
- `mixture(ctx, {handle: weight}, validation=…)` — assembles a Levanter `LmDataConfig`
  from a handle-to-weight mapping. Call inside the `data=lambda ctx: …` argument of
  `train_lm`.

## How this differs from the old content-addressed executor

The previous executor assigned each step an output path containing a hash of the step's
config and all its transitive dependencies. A hyperparameter change silently re-addressed
every downstream step, and the path gave no clue about what version of the data it held.

Lazy artifacts address data by an explicit `{name}/{version}`. The fingerprint still
catches accidental config drift (the build-once guard) and can be pinned for review-time
verification, but the storage path is stable and human-readable. Bumping a version is an
explicit author decision, not an automatic consequence of any config change.
