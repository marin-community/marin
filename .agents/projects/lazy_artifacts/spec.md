# Spec: Lazy Artifacts

The contract a reviewer is agreeing to: each symbol this design introduces or relies on, with full
signature, parameter types, defaults, and behavior; the persisted on-disk shapes; the errors;
the concurrency and idempotency guarantees. On branch `weaver/adopt-executor-context-for-remai`.

Two layers. **Core API** (`marin.execution.*`) is the low-level contract — the artifact model,
lowering, registry, fingerprint, compute dispatch, and the runner entry point. **Authoring layer**
(`marin.experiment.*`) is convenience built on the core; it could be replaced without touching it.
The lower engine *reused as-is* (`StepRunner` internals `run_step`/`check_cache`/`step_lock`/
`StatusFile`, and `StepSpec.as_executor_step`) is existing infrastructure, not new contract — only
its observable entry point (`StepRunner.run`) and the guarantees it provides are pinned here.

---

## Core: handle, recipe, context (`marin.execution.lazy`)

```python
@dataclass(frozen=True, eq=False)
class Artifact:
    name: str
    version: str
    recipe: "Recipe"
    override_path: str | None = None
    adopt_source: str | None = None
    expected_fingerprint: str | None = None

    def __post_init__(self) -> None: ...           # ValueError if adopt_source and override_path both set
    def fingerprint_payload(self) -> str: ...
    def fingerprint(self) -> str: ...
    def path(self, prefix: str | None = None) -> str: ...

@dataclass(frozen=True, eq=False)
class Dataset(Artifact): ...        # tokenized-dataset handle; routes to dataset consumers
@dataclass(frozen=True, eq=False)
class Checkpoint(Artifact): ...     # Levanter checkpoint handle; routes to initialize_from_checkpoint_path
```

- `name` — logical path segment; the address is `{prefix}/{name}/{version}`. May contain slashes.
- `version` — explicit. Immutable unless it equals `"dev"` or ends with `"-dev"` (then mutable: always rebuilds, guard skipped).
- `recipe` — how to build it (below).
- `override_path` — a **pin**: resolve this artifact (and consumers of it) to existing data at this location instead of `{name}/{version}`. Relative → resolved against the prefix; absolute/URL → used as-is. Writes **no** record (no provenance, no guard). Mutually exclusive with `adopt_source`.
- `adopt_source` — register pre-existing data as this `name@version` (set via `adopt()`, below). Mutually exclusive with `override_path`.
- `expected_fingerprint` — if set, `lower()` raises `FingerprintMismatchError` when the computed fingerprint differs. Checked before any build.

`eq=False`: two handles are equal iff identical objects (so a handle can key a dict despite an unhashable `ResourceConfig` in its recipe); value-equality lives in `name@version` + the fingerprint.

- `fingerprint_payload() -> str` — the canonical config bytes the fingerprint hashes. For a computed artifact: `canonical_json(recipe.build_config(RunContext.for_fingerprint(recipe.run_args.keys())))`. For an adopted artifact: `json.dumps({"adopt_source": adopt_source}, sort_keys=True)`.
- `fingerprint() -> str` — `fingerprint_hash(fingerprint_payload())` (8 hex chars).
- `path(prefix=None) -> str` — resolved location: `adopt_source` if adopted, else `override_path` if pinned, else `{prefix or marin_prefix()}/{name}/{version}`. Relative pins/sources are joined to the prefix; absolute ones returned as-is.

```python
@dataclass(frozen=True)
class Recipe:
    fn: Callable[[Any], Any]
    build_config: Callable[["RunContext"], Any]
    deps: tuple[Artifact, ...] = ()
    run_args: Mapping[str, Any] = field(default_factory=dict)
```

- `fn` — `fn(config)`. A plain callable runs inline in the runner; a `RemoteCallable` (`remote(fn, resources=…)`) dispatches a Fray job and returns `None` to its caller. The runner persists the value returned to it as the artifact's `.artifact.json` payload either way: an inline `fn`'s return value is saved; a `remote` `fn` returns `None`, so the sidecar holds `null` and the real output is whatever the Fray job wrote to `ctx.out`. So a step that must produce a readable payload (e.g. a sweep trial's metrics) should run inline; a step that produces data at `ctx.out` (tokenize, train) is the remote case.
- `build_config` — `build_config(ctx) -> config`. Pure function of the context. Literals in it bear identity (enter the fingerprint); values pulled from `ctx` do not.
- `deps` — artifacts that must materialize first; their **versions** enter the fingerprint, their paths are resolved at run time via `ctx.path(dep)`.
- `run_args` — execution choices the config pulls via `ctx.run_arg(key)` (e.g. the accelerator). Excluded from the fingerprint. The declared keys are what `for_fingerprint` substitutes as placeholders.

```python
@dataclass(frozen=True)
class RunContext:
    out: str
    prefix: str
    region: str | None
    _dep_ref: Callable[[Artifact], str]      # private resolution hooks; not for direct construction
    _run_args: Mapping[str, Any]

    def path(self, dep: Artifact) -> str: ...
    def run_arg(self, key: str) -> Any: ...                    # KeyError if key not in recipe.run_args
    @staticmethod
    def for_run(out: str, prefix: str, *, region: str | None = None,
                run_args: Mapping[str, Any] | None = None) -> "RunContext": ...
    @staticmethod
    def for_fingerprint(run_arg_keys: Iterable[str] = ()) -> "RunContext": ...
```

Construct a `RunContext` **only** through `for_run` / `for_fingerprint`; `_dep_ref` and `_run_args`
are private resolution hooks (the direct constructor is not a public API).

- `for_run` binds the context to the live environment (real `out`/`prefix`/`region`, `path(dep)` → the dep's region-local output path, `run_arg(key)` → its real value).
- `for_fingerprint` substitutes placeholders for every pull (`out="<out>"`, `prefix="<prefix>"`, `region="<region>"`, `path(dep)="{dep.name}@{dep.version}"`, `run_arg(key)="<key>"`) — so nothing pulled affects the fingerprint, and dep identity enters as `name@version`.

```python
def lower(artifact: Artifact) -> StepSpec: ...
def adopt(name: str, version: str, source: str, *, kind: type[Artifact] = Dataset) -> Artifact: ...
def materialized_config(artifact: Artifact, prefix: str) -> Any: ...
```

- `lower(artifact)` — recurse `deps` into a `StepSpec` graph addressed by `{name}/{version}` (or the pin); compute the fingerprint once; raise `FingerprintMismatchError` if `expected_fingerprint` is set and differs; carry `{fingerprint, version, deps}` in `hash_attrs` and the canonical payload in `StepSpec.fingerprint_payload`; the step fn writes an `ArtifactRecord` on success (unless pinned). Pure transform — never inspects `recipe.fn`.
- `adopt(name, version, source, *, kind=Dataset)` — return `kind(name, version, recipe=<noop>, adopt_source=source)`. Consumers resolve to `source` (no move, no recompute); lowering writes a record at the canonical address with `source` recorded; re-adopting `name@version` from a different `source` re-fingerprints → `ImmutableArtifactError`.
- `materialized_config(artifact, prefix)` — `build_config` under `for_run(out=artifact.path(prefix), prefix=prefix)` with **`region=None`** and no run-args, for inspection / golden tests. Runs nothing. This is *not* identical to the run-time config: real lowering passes `region=marin_region()` and the recipe's `run_args`, so a `build_config` that reads `ctx.region`/`ctx.run_arg(...)` will see placeholders/`None` here.

## Core: compute dispatch (`marin.execution.remote`)

```python
@dataclass(frozen=True)
class RemoteCallable(Generic[P, R]):
    fn: Callable[P, R]
    resources: ResourceConfig
    env_vars: dict[str, str] = field(default_factory=dict)
    pip_dependency_groups: list[str] | None = None
    name: str | None = None

    def named(self, name: str) -> "RemoteCallable": ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...   # submits a Fray job; returns None

def remote(fn: Callable[P, R] | None = None, *, name: str | None = None,
           resources: ResourceConfig | None = None, env_vars: dict[str, str] | None = None,
           pip_dependency_groups: list[str] | None = None) -> RemoteCallable | Callable[..., RemoteCallable]: ...
```

`remote(fn, resources=…)` wraps a callable for Fray dispatch (also usable as a decorator
`@remote(resources=…)`). `__call__` submits the job and **returns `None`** — a remote step
communicates only by writing `ctx.out`. Resources ride on the callable, never on the `StepSpec`
node, so they never enter the fingerprint.

## Core: the lowered step (`marin.execution.step_spec`)

```python
@dataclass(frozen=True)
class StepSpec:
    name: str
    output_path_prefix: str | None = None
    deps: list["StepSpec"] = field(default_factory=list)
    hash_attrs: dict[str, Any] = field(default_factory=dict)
    override_output_path: str | None = None
    fingerprint_payload: str | None = None
    fn: Callable[[str], Any] | None = None
    resources: ResourceConfig | None = None

    @cached_property
    def output_path(self) -> str: ...        # override_output_path (prefix-joined if relative) else {prefix}/{name_with_hash}
    @cached_property
    def name_with_hash(self) -> str: ...      # f"{name}_{hash_id}"
    @cached_property
    def hash_id(self) -> str: ...             # sha256({name, attrs: hash_attrs, deps})[:8] — note: NOT fingerprint_payload
```

`lower()` sets `override_output_path={name}/{version}` (or the pin), `hash_attrs={"fingerprint","version","deps"}`, and `fingerprint_payload=<canonical json>`. `fingerprint_payload` is **not** part of `hash_id` and is not logged — it exists only as provenance and for conflict diffs.

## Core: runner entry point + payload IO (`marin.execution.step_runner`, `marin.execution.artifact`)

```python
class StepRunner:
    def run(self, steps: Iterable[StepSpec], *, dry_run: bool = False,
            force_run_failed: bool = True, max_concurrent: int | None = None) -> None: ...

class Artifact:   # marin.execution.artifact — the output-payload sidecar, distinct from the registry record
    @classmethod
    def save(cls, artifact: T, base_path: str) -> None: ...
    @classmethod
    def from_path(cls, base_path: str | StepSpec, artifact_type: type[T] = ...) -> "T | PathMetadata | dict": ...

class PathMetadata(BaseModel): ...   # returned by from_path when a step SUCCEEDED but wrote no payload sidecar
```

`StepRunner().run([lower(artifact)])` is the entry point: it post-order schedules deps (deduped by
output path), runs each through the guard → cache-check → lock → run → status protocol, bounded by
`max_concurrent` (default 8). `dry_run` logs without touching remote status; `force_run_failed`
(default `True`) reruns a previously-FAILED step instead of raising. `Artifact.save`/`from_path`
read and write the `.artifact.json` payload sidecar (below); `from_path` returns `PathMetadata`
when a step succeeded but produced no payload (a remote step that only wrote `ctx.out`).

## Core: registry + fingerprint (`marin.execution.registry`, `marin.execution.fingerprint`)

```python
@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    version: str
    fingerprint: str
    output_path: str
    git_commit: str | None
    user: str | None
    created_at: str
    deps: list[str]                      # dependency identities as "name@version"
    source: str | None = None            # adopted-source location; None for a computed artifact
    fingerprint_payload: str | None = None

class ImmutableArtifactError(Exception): ...
class FingerprintMismatchError(Exception): ...

def is_mutable_version(version: str) -> bool: ...                 # version == "dev" or endswith("-dev")
def read_record(output_path: str) -> ArtifactRecord | None: ...  # parses {output_path}/.artifact_record.json
def write_record(record: ArtifactRecord) -> None: ...            # writes it (indent=2)
def guard_immutable(output_path: str, name: str, version: str,
                    fingerprint: str, payload: str | None = None) -> None: ...
def enforce_immutability(step: StepSpec) -> bool: ...            # returns True iff mutable (caller rebuilds)

def canonical_json(config: object) -> str: ...
def fingerprint_hash(payload: str) -> str: ...                   # md5(payload.encode()).hexdigest()[:8]
```

- `guard_immutable` — no-op when the version is mutable, when no record exists, or when the recorded fingerprint matches; otherwise raises `ImmutableArtifactError`. When both the recorded `fingerprint_payload` and the current `payload` are present, the message appends a field-level diff (`learning_rate: 3e-3 -> 4e-3`), capped at 20 lines.
- `enforce_immutability(step)` — reads `fingerprint`/`version` from `step.hash_attrs`; returns `False` for a non-lazy step (no fingerprint); returns `True` for a mutable version; otherwise calls `guard_immutable(step.output_path, step.name, version, fingerprint, step.fingerprint_payload)` and returns `False`. Run by `StepRunner` before serving a cached SUCCESS. Known limitation: today it runs *before* `step_lock` and isn't re-checked on the already-done path, so concurrent first-builds with different fingerprints can both pass (design.md Open Questions).
- `canonical_json` — deterministic JSON. Canonicalizes: dataclasses (`asdict`), `Enum` (`.value`), `timedelta` (`{days,seconds,microseconds}`), `Path` (str), `set`/`frozenset` (`{"__set__": sorted-by-canonical-form}`), numpy `dtype`/scalar-type (`{"__dtype__": name}`), numpy scalar (`.item()`), numpy/jax arrays (`{"__array__": tolist, "dtype", "shape"}`), other type objects (`{"__type__": "mod.qualname"}`). **Raises `TypeError`** on callables, `functools.partial`, and objects using the default `object.__repr__` — anything with no reproducible representation. Dict keys must be strings (`sort_keys=True`). The same config always yields identical bytes across processes.

---

## Authoring: sweep / select (`marin.experiment.sweep`)

```python
def read_replicated_metrics(output_path: str) -> Mapping[str, Any]: ...
class AnnotatedCheckpoint(Checkpoint):  # adds: metrics_reader = read_replicated_metrics
def annotate(checkpoint: Checkpoint, *, metrics_reader=read_replicated_metrics) -> AnnotatedCheckpoint: ...
def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]: ...
def sweep(trial: Callable[..., AnnotatedCheckpoint], **axes: Sequence[Any]) -> list[AnnotatedCheckpoint]: ...
def select(name: str, version: str, trials: Sequence[AnnotatedCheckpoint], *,
           metric: str, mode: str = "min") -> Checkpoint: ...
```

- A trial is just a `Checkpoint`-producing function (e.g. `lambda **p: train_lm(...)`); there is no metrics payload to return. Selection reads each trial's metric from where the trial *wrote* it — its output path.
- `AnnotatedCheckpoint` — a `Checkpoint` paired with `metrics_reader(output_path) -> Mapping`, "how to read my own recorded metrics". The reader is read at run time, not built into the fingerprint. `annotate(checkpoint)` wraps a plain `Checkpoint` (e.g. from `train_lm`) without changing its identity.
- `read_replicated_metrics(output_path)` — the default reader: reads `<output>/tracker_metrics.jsonl` (the WandB `replicate_path` that `train_lm` mirrors next to its checkpoints) and returns its `summary` mapping. Selecting on `train/loss` or any logged `eval/.../loss` needs no per-trial wiring.
- `grid(**axes)` — Cartesian product of the named axes as `{axis: value}` dicts, row-major (last axis fastest).
- `sweep(trial, **axes)` — `[trial(**params) for params in grid(**axes)]`. The `trial` builder must fold each grid point's values into both its config (distinct fingerprint) and its `name` (distinct address).
- `select(name, version, trials, *, metric, mode="min")` — a reducer depending on every trial; returns the winner as a `Checkpoint` addressed at `name@version`. At run time it reads each trial's metrics through that trial's own reader, ranks by `metric`, and writes `{"winner", "score", "winner_path", "metrics", "scores"}`. `metric`/`mode` bear identity; trial *values* are read at run time. Trials are keyed by full `name@version`. Raises `ValueError` if `mode not in {"min","max"}`, if `trials` is empty, or if two trials share a `name@version`; `KeyError` if a trial's recorded metrics lack `metric` (at run time).

## Authoring: data builders (`marin.experiment.data`)

`DEFAULT_VERSION = "v1"`. Each builder returns a `Dataset` whose recipe runs a tokenize/download
step; `resources` (when given) dispatches that step to Fray via `remote`, otherwise it runs inline.

```python
def tokenized(name: str, *, tokenizer: str,
              source: str | None = None,
              paths: Sequence[str] | None = None,
              raw: Dataset | None = None,
              glob: str | None = None,
              validation: bool = False,
              pin: str | None = None,
              text_key: str = "text",
              version: str = "v1",
              tags: Sequence[str] = (),
              resources: ResourceConfig | None = None) -> Dataset: ...
```

Produce a tokenized cache. **Exactly one** raw-input mode (else `ValueError`):
- `source` — a HuggingFace id `org/name` (detected: one slash, no `://`, not absolute) tokenized via `HfTokenizeConfig`; otherwise a single raw path tokenized via `TokenizeConfig`.
- `paths` — raw globs, each resolved against `ctx.prefix`, tokenized via `TokenizeConfig`.
- `raw` + `glob` — depend on a download handle and tokenize `f"{ctx.path(raw)}/{glob}"`. `raw` and `glob` must be given together (else `ValueError`); `raw` becomes a recipe dep.

Other parameters: `tokenizer` (required) — tokenizer id; `validation` — route the data to the cache's validation split (`validation_paths`) instead of train; `pin` — reference already-tokenized data at an existing location (no recompute); `text_key` — the document text field (`TextLmDatasetFormat`); `version`; `tags` — passed through to the tokenize config; `resources` — Fray resources for the tokenize job.

```python
def pretokenized(name: str, *, repo_id: str, tokenizer: str,
                 revision: str | None = None, version: str = "v1",
                 pin: str | None = None, tags: Sequence[str] = (),
                 resources: ResourceConfig | None = None) -> Dataset: ...
```

Download an already-tokenized Levanter cache from HuggingFace `repo_id` (optional `revision`) into
`ctx.out` via `PretokenizedCacheDownloadConfig` / `fetch_pretokenized_cache` — no re-tokenization.
Consumed as a normal tokenized `Dataset`. `pin` references an already-downloaded cache.

```python
def raw_download(name: str, *, fn: Callable[[Any], Any],
                 build_config: Callable[[RunContext], Any],
                 version: str = "v1", pin: str | None = None,
                 resources: ResourceConfig | None = None) -> Dataset: ...
```

A raw-data handle: `build_config(ctx)` produces a config and `fn(config)` writes the download to
`ctx.out`. Returned as a `Dataset` so `tokenized(raw=…)` can depend on it; it is not itself a
tokenized cache.

```python
def mixture(ctx: RunContext, train: Mapping[Dataset, float], *,
            validation: Sequence[Dataset] = (),
            shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE) -> LmDataConfig: ...
```

Assemble a Levanter `LmDataConfig`. `train` maps each handle to its weight; `validation` handles
are added at weight `0.0`. Each component is keyed by the handle's `name` and rooted at
`ctx.path(handle)` (so the caller must pass the same handles as the recipe's `deps`).
`permutation_type` is `"feistel"`. Raises `ValueError` when: there are no components (`train` and
`validation` both empty); two handles share a `name` (they would collide in the component dict);
or the components span more than one tokenizer. Each component handle's `build_config` must produce
a tokenize config (`TokenizeConfigBase`) else `TypeError`.

---

## Persisted shapes

Two paths, which coincide for a computed artifact but diverge for an adopted one:

- **Registry-record path** — always `{prefix}/{name}/{version}/` (the `StepSpec` output path). The
  build record is written here even for an adopted artifact.
- **Consumer-data path** — where consumers resolve the artifact's bytes: the registry-record path
  for a computed artifact, `adopt_source` for an adopted one, `override_path` for a pin.

Sidecar files at the **registry-record path** (a pin writes none of them):

- **`.artifact_record.json`** — the build-once record. JSON object with exactly the `ArtifactRecord`
  fields: `name`, `version`, `fingerprint` (8 hex), `output_path`, `git_commit` (nullable),
  `user` (nullable), `created_at` (ISO-8601), `deps` (`["name@version", …]`), `source` (the adopted
  source location, else `null`), `fingerprint_payload` (the canonical config JSON string, or
  `null`). Written `indent=2`.
- **`.artifact.json`** — the output-payload sidecar holding the value the step fn returned to the
  runner (`marin.execution.artifact.Artifact.save`/`from_path`). An inline fn's return value is
  saved; a `remote` fn returns `None`, so the value is `null` and the real output is at `ctx.out`.
  Legacy names `.artifact`, `artifact.json` are read but not written.
- **`.executor_status`** — the status marker (`SUCCESS`/`FAILED`/…) written by `StatusFile`; the
  cache check serves a cached artifact only when this reads `SUCCESS`.

The **fingerprint payload** format is the `canonical_json` string described above; the same string
appears in `ArtifactRecord.fingerprint_payload` and `StepSpec.fingerprint_payload`.

## Errors

| Error | Raised when |
|---|---|
| `ImmutableArtifactError` | rebuild of a fixed `name@version` from a changed recipe (message carries a field-level diff); re-adopt of `name@version` from a different `source` |
| `FingerprintMismatchError` | `expected_fingerprint` is set and the computed fingerprint differs (at `lower()`, before any build) |
| `FileNotFoundError` | an adopted step runs and no data exists at its source; `Artifact.from_path` finds no payload sidecar and either the typed caller wanted a non-`PathMetadata` type or `.executor_status` is not `SUCCESS` |
| `TypeError` (encoder) | a config value has no reproducible serialization (callable, `partial`, default-`repr` object) |
| `TypeError` (payload IO) | `Artifact.from_path` is given an `artifact_type` that isn't a pydantic `BaseModel` subclass |
| `TypeError` (mixture) | a `mixture` component config isn't a tokenize config |
| `ValueError` | `Artifact` with both `adopt_source` and `override_path`; `tokenized` not given exactly one source mode, or `raw`/`glob` not paired; `mixture` is empty or its components span >1 tokenizer; `select` bad `mode`, empty sweep, or colliding trial `name@version` |
| `KeyError` | `ctx.run_arg(key)` for a key not declared in `recipe.run_args`; a `select` trial's recorded metrics missing `metric` |
| `FileNotFoundError` (select) | `read_replicated_metrics` finds no `tracker_metrics.jsonl` for a trial |
| *(Fray failure)* | a `remote` step's Fray job fails to submit or fails at run time — the underlying error propagates from `StepRunner.run` (the runner blocks on the job and re-raises) |

## Invariants & preconditions

What the contract requires of callers, and what it does **not** check (so the value is "accepted but
undefined" — passing it is the caller's bug, not a guarded error):

- **`name`** — used verbatim as a path segment; slashes are allowed (it nests). Empty strings, `..`,
  leading/trailing slashes, and URL-like names are **not checked** and will produce malformed paths.
- **`version`** — any string. Mutable iff it equals `"dev"` or ends with `"-dev"`; every other value
  is treated as immutable. No grammar is enforced.
- **`expected_fingerprint`** — compared as an opaque string against the 8-hex computed value; format
  is **not validated** (a malformed pin simply never matches → `FingerprintMismatchError`).
- **`deps`** — may contain duplicate artifacts; the runner dedupes scheduling by output path, so
  duplicates are harmless but **not rejected**. Dep identity in the fingerprint is `name@version`, so
  two deps with the same `name@version` but different recipes are indistinguishable to a consumer.
- **`run_args` keys** — expected to be strings (they become `ctx.run_arg` lookups and
  `for_fingerprint` placeholders); non-string keys are **not checked**.
- **`build_config`** — must be a deterministic pure function of its `RunContext` for the fingerprint
  to be stable across processes; I/O or env/time reads are **not prevented** but make the fingerprint
  (and therefore the build-once guard) unreliable. This is a contract on the caller, not enforced.
- **`recipe.fn` source** — **not** part of the fingerprint. Changing the function body while keeping
  the config identical serves cached output; bump the version to force a rebuild.

## Concurrency & idempotency

- **`lower()` is repeatable but not pure.** Re-lowering the same handle yields the same `StepSpec`
  and fingerprint, but the step fn it produces closes over `get_git_commit()`/`get_user()` read in
  the launching process, so the `ArtifactRecord` it later writes embeds the launch commit/user.
- **Fixed versions are guarded only by the recorded fingerprint.** The build-once guard compares the
  current fingerprint against `{output_path}/.artifact_record.json`; with no record (first build) it
  is a no-op. A pin (`override_path`) writes no record and is therefore unguarded.
- **`dev` versions always rebuild,** even after a successful run — the guard and the SUCCESS
  cache-check are both skipped.
- **Concurrent first-builds can converge silently.** `enforce_immutability` runs *before* `step_lock`
  and isn't re-checked on the already-done path, so two processes lowering different recipes to the
  same unbuilt `name@version` can both pass the guard; one wins the lock and writes the record, the
  other serves it as done. Tightening this (guard-under-lock) is an Open Question, not part of this
  contract.

## File layout

| Piece | Path |
|---|---|
| Handles, recipe, context, `lower`, `adopt`, `materialized_config` | `lib/marin/src/marin/execution/lazy.py` |
| Build-once guard, `ArtifactRecord`, conflict diff, errors | `lib/marin/src/marin/execution/registry.py` |
| Strict deterministic encoder | `lib/marin/src/marin/execution/fingerprint.py` |
| `StepSpec` (+ `fingerprint_payload`) | `lib/marin/src/marin/execution/step_spec.py` |
| Runner (guard, lock, cache, dispatch) | `lib/marin/src/marin/execution/step_runner.py` |
| Compute-on-the-fn (`remote`, `RemoteCallable`) | `lib/marin/src/marin/execution/remote.py` |
| Payload sidecar (`Artifact.save`/`from_path`) | `lib/marin/src/marin/execution/artifact.py` |
| Sweep / select | `lib/marin/src/marin/experiment/sweep.py` |
| Data builders + mixture | `lib/marin/src/marin/experiment/data.py` |
| Catalogs (lazy) | `experiments/pretraining_datasets/{dclm,simple,nemotron}_lazy.py`, `experiments/paloma_lazy.py`, `experiments/evals/uncheatable_lazy.py` |

## Pins and adopted data — three tiers of guarantee

| Mechanism | Guarantees | Does NOT guarantee |
|---|---|---|
| `name@version` (normal) | build-once: content recorded + fingerprint-guarded | — |
| `adopt(name, version, source)` | **pointer** immutability: the alias→source mapping is recorded and guarded (re-adopt from a different `source` raises) | **content** immutability — it fingerprints the `source` *string*, so external data mutating in place, or a relative source under a moved prefix, is invisible |
| `override_path` pin | nothing — resolves to existing data | no record: no provenance, no guard. An explicit, unguarded escape hatch |

## Out of scope

These are contracts this spec does **not** pin (the migration plan and sequencing live in
`design.md` / `research.md`):

- **Deleting the Executor.** `Executor`/`executor_main`/`versioned`/`InputName`/`this_output_path`
  remain importable; this spec defines the target surface, not their removal.
- **The import-time guard's default.** `MARIN_EXECUTOR_STRICT` stays opt-in; flipping it default-on
  is a sequencing decision, not an API change.
- **Concurrency-tightening the build-once guard** (guard-under-lock) — see Concurrency & idempotency
  above; the current contract is what's pinned.
- **A scheme-tagged `tokenized(source=…)`** (`hf://`, `gs://`, …) collapsing the source-mode
  heuristic — a proposed authoring-layer refinement (`research.md`), not committed here.
- **Fingerprint attribution / tracing** and **auto-versioning** (fingerprint-suffixed paths).
