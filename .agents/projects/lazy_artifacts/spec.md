# Spec: Lazy Artifacts

The contract a reviewer is agreeing to: each symbol this design introduces or relies on, with full
signature, parameter types, defaults, and behavior; the persisted on-disk shapes; the errors; the
concurrency and idempotency guarantees. On branch `weaver/adopt-executor-context-for-remai`.

Two layers. **Core** (`marin.execution.*`) is the low-level contract — the handle, the artifact,
lowering, the record, the fingerprint, compute dispatch, and the run/resolve entry points.
**Authoring** (`marin.experiment.*`) is convenience built on the core; it could be replaced without
touching it. The lower engine *reused as-is* (`StepRunner` internals, `step_lock`, `StatusFile`) is
existing infrastructure, not new contract — only its observable entry points and guarantees are
pinned here.

Naming note: the lazy **handle** is `Lazy[T]`; the realized **output** is an `Artifact`. There is no
longer a class named `Artifact` that is a handle, and no separate output *serializer* class — the
old `marin.execution.artifact.Artifact.save/from_path` payload role and the old
`registry.ArtifactRecord` are unified into one `ArtifactRecord` (below).

---

## Core: handle, recipe, context (`marin.execution.lazy`)

```python
T = TypeVar("T", bound="Artifact")

@dataclass(frozen=True, eq=False)
class Lazy(Generic[T]):
    name: str
    version: str
    recipe: "Recipe"
    result_type: type[T]
    override_path: str | None = None
    adopt_source: str | None = None
    expected_fingerprint: str | None = None

    def __post_init__(self) -> None: ...           # ValueError if adopt_source and override_path both set
    def fingerprint_payload(self) -> str: ...
    def fingerprint(self) -> str: ...
    def path(self, prefix: str | None = None) -> str: ...
```

- `name` — logical path segment; the address is `{prefix}/{name}/{version}`. May contain slashes.
- `version` — explicit. Immutable unless it equals `"dev"` or ends with `"-dev"` (then mutable: always rebuilds, drift check skipped).
- `recipe` — how to build it (below).
- `result_type` — the `Artifact` subclass this handle produces. Used by `resolve` to reload the realized value (`result_type.load(path)`) and by consumers for static typing (`Lazy[Dataset]` vs `Lazy[Checkpoint]`).
- `override_path` — a **pin**: resolve this handle (and consumers of it) to existing data at this location instead of `{name}/{version}`. Relative → resolved against the prefix; absolute/URL → used as-is. Writes **no** record. Mutually exclusive with `adopt_source`.
- `adopt_source` — register pre-existing data as this `name@version` (set via `adopt()`). Mutually exclusive with `override_path`.
- `expected_fingerprint` — if set, `lower()` raises `FingerprintMismatchError` when the computed fingerprint differs. The **only** hard identity gate, and it is opt-in.

`eq=False`: two handles are equal iff identical objects (so a handle can key a dict despite an unhashable `ResourceConfig` in its recipe); value-equality lives in `name@version` + the fingerprint.

- `fingerprint_payload() -> str` — the canonical config bytes the fingerprint hashes. Computed: `canonical_json(recipe.build_config(RunContext.for_fingerprint(recipe.run_args.keys())))`. Adopted: `json.dumps({"adopt_source": adopt_source}, sort_keys=True)`.
- `fingerprint() -> str` — `fingerprint_hash(fingerprint_payload())` (8 hex chars).
- `path(prefix=None) -> str` — resolved location: `adopt_source` if adopted, else `override_path` if pinned, else `{prefix or marin_prefix()}/{name}/{version}`. Relative pins/sources are joined to the prefix; absolute ones returned as-is.

```python
@dataclass(frozen=True)
class Recipe:
    fn: Callable[[Any], Any]
    build_config: Callable[["RunContext"], Any]
    deps: tuple[Lazy, ...] = ()
    run_args: Mapping[str, Any] = field(default_factory=dict)
```

- `fn` — `fn(config)`. A plain callable runs inline; a `RemoteCallable` (`remote(fn, resources=…)`) dispatches a Fray job and **returns `None`** to the runner. So the two step shapes are: a **data** step writes its bytes to `ctx.out` (inline or remote; `result` is `None`); a **value** step `return`s a `JsonArtifact` the runner serializes into the record's `result` — and **must run inline**, since a remote job returns nothing to the caller. `lower()` raises `ValueError` if a `JsonArtifact` `result_type` is paired with a `RemoteCallable` fn.
- `build_config` — `build_config(ctx) -> config`. Pure function of the context. Literals bear identity; `ctx` pulls do not.
- `deps` — handles that must materialize first; their **versions** enter the fingerprint, their paths resolve at run time via `ctx.path(dep)`.
- `run_args` — execution choices the config pulls via `ctx.run_arg(key)`. Excluded from the fingerprint.

```python
@dataclass(frozen=True)
class RunContext:
    out: str
    prefix: str
    region: str | None
    _dep_ref: Callable[[Lazy], str]          # private resolution hooks; not for direct construction
    _run_args: Mapping[str, Any]

    def path(self, dep: Lazy) -> str: ...
    def run_arg(self, key: str) -> Any: ...                    # KeyError if key not in recipe.run_args
    @staticmethod
    def for_run(out, prefix, *, region=None, run_args=None) -> "RunContext": ...
    @staticmethod
    def for_fingerprint(run_arg_keys: Iterable[str] = ()) -> "RunContext": ...
```

Construct a `RunContext` **only** through `for_run` / `for_fingerprint`.

- `for_run` binds the live environment (real `out`/`prefix`/`region`, `path(dep)` → the dep's region-local output path, `run_arg(key)` → its real value).
- `for_fingerprint` substitutes placeholders for every pull (`out="<out>"`, `prefix="<prefix>"`, `region="<region>"`, `path(dep)="{dep.name}@{dep.version}"`, `run_arg(key)="<key>"`).

## Core: the realized artifact (`marin.execution.artifact`)

`Artifact` is a single pydantic base (no multiple inheritance). `path` is a field set on load;
`record` is a cached property that reads the sidecar at `path`. `Artifact.load` is **concrete** (a
path ref), so the default `result_type=Artifact` is resolvable.

```python
class Artifact(pydantic.BaseModel):       # not frozen; path is set on load
    path: str = ""                        # output location; populated by load()
    @functools.cached_property
    def record(self) -> "ArtifactRecord | None": ...   # reads {path}/artifact.json lazily
    @classmethod
    def load(cls, source: str) -> Self:
        return cls(path=source)           # data ref: nothing read into the process

class Dataset(Artifact): ...              # a tokenized Levanter cache at .path  (inherits path-ref load)
class Checkpoint(Artifact): ...           # a Levanter checkpoint dir at .path   (inherits path-ref load)

class JsonArtifact(Artifact):             # a computed value persisted in the record's `result`
    @classmethod
    def load(cls, source: str) -> Self:
        rec = read_record(source)         # FileNotFoundError if absent; ValidationError if corrupt
        obj = cls.model_validate(rec.result)   # validates the payload against THIS schema
        obj.path = source
        return obj
```

- `Artifact` — the produced, persisted thing. `load(source)` reconstructs the typed value from an output path; it pairs with the recipe `fn` that wrote that location. The default `load` (and `Dataset`/`Checkpoint`) is a **data ref**: it returns `cls(path=source)` and reads nothing — weights/caches never enter the launcher.
- `JsonArtifact` — base for **computed values**: `load` reads `ArtifactRecord.result` (one file, the unified record) and `model_validate`s it against the subclass schema. A step that produces a value returns a `JsonArtifact` instance; the runner serializes it into the record's `result`. Authors subclass this instead of writing `load`. **Schema-drift safety:** because the record stores `result_type` (below), `resolve` first checks the recorded type matches the requested `result_type` and raises `ArtifactTypeMismatchError` *before* attempting `model_validate` — a renamed/reshaped value type is a hard error, not a silent mis-load.

`Lazy[T]` requires `T` to be an `Artifact` subclass, so `resolve` is total and typed. Because
`Artifact` itself is concrete, a handle that only needs a path (the `apply` default) resolves to a
plain `Artifact`.

## Core: lowering, run, resolve, adopt (`marin.execution.lazy`)

```python
def lower(handle: Lazy) -> StepSpec: ...
def run(*handles: Lazy, max_concurrent: int = 8, dry_run: bool = False,
        force_run_failed: bool = True) -> None: ...
def resolve(handle: Lazy[T], *, max_concurrent: int = 8) -> T: ...
def apply(name: str, fn: Callable[..., Any], *, version: str = "v1",
          result_type: type[T] = Artifact, resources: ResourceConfig | None = None,
          pin: str | None = None, **inputs: Any) -> Lazy[T]: ...
def derived(name: str, *, fn: Callable[[Any], Any],
            build_config: Callable[[RunContext], Any], deps: Iterable[Lazy] = (),
            version: str = "v1", pin: str | None = None,
            resources: ResourceConfig | None = None, kind: type[T] = Artifact) -> Lazy[T]: ...
def adopt(name: str, version: str, source: str, *, kind: type[T] = Dataset) -> Lazy[T]: ...
def materialized_config(handle: Lazy, prefix: str) -> Any: ...

OUT: Final = ...        # sentinel: in apply(**inputs), resolves to ctx.out
```

- `lower(handle)` — recurse `deps` into a `StepSpec` graph addressed by `{name}/{version}` (or the pin); compute the fingerprint once; carry `{fingerprint, version, result_type, deps}` in `hash_attrs` and the canonical payload in `StepSpec.fingerprint_payload`; the step fn writes an `ArtifactRecord` on success (unless pinned). Pure transform of *structure* — never inspects `recipe.fn`. Raises at `lower()` time: `FingerprintMismatchError` if `expected_fingerprint` is set and differs; `ValueError` if a fixed (non-`dev`) handle has a `dev`/`-dev` dep (a mutable dep would rebuild while the fixed parent stays cached → silent staleness; make the parent `dev` or pin the dep); `ValueError` if a `JsonArtifact` `result_type` is paired with a `RemoteCallable` fn.
- `run(*handles)` — `StepRunner().run([lower(h) for h in handles], …)`. The everyday entry point; executes for side effects. `dry_run` logs without touching remote status; `force_run_failed` reruns a previously-FAILED step instead of raising.
- `resolve(handle) -> T` — `run(handle)`, then load via `handle.result_type.load(handle.path())`. Before loading, it checks the served record's `result_type` matches `handle.result_type` and raises `ArtifactTypeMismatchError` on a mismatch (guards against a drifted value artifact whose schema changed). For a value artifact returns the typed value; for a data artifact a path-bearing ref. No build on a cache hit (just `load`).
- `apply(name, fn, **inputs)` — the generic single-step builder, **direct-call form**. Each value in `inputs` is classified, **recursing into `list`/`tuple`/`dict`**: a `Lazy` handle becomes a recipe dep and resolves to `ctx.path(handle)` at run time (its `name@version` enters identity); the `OUT` sentinel resolves to `ctx.out`; anything else is a literal that bears identity. The recipe's `fn(**resolved_inputs)` is called directly (no config object, no wrapper). A `Lazy`/`OUT` nested inside an *opaque* object (a dataclass/config instance, not a builtin container) is **not** resolved — use the `Lazy`+`Recipe` tier for that. `result_type` selects the produced `Artifact` type (default `Artifact`, a path ref); `resources` dispatches via `remote`; `pin` references existing data. Raises `TypeError` if `fn`'s signature can't bind the resolved kwargs (checked at lower via `inspect.signature`).
- `adopt(name, version, source, *, kind=Dataset)` — return `kind`-typed `Lazy` with `adopt_source=source`. **Consumers always resolve to the handle's `source`** (a pure pointer — no move, no recompute); the record at the canonical address is provenance only. Re-adopting `name@version` from a different `source` is a drift case: it warns (the recorded source differs from the live one), and consumers follow the live handle. Pin the source or bump the version to make a source change unambiguous.
- `derived(name, *, fn, build_config, deps=(), kind=Artifact, …)` — the generic single-step builder, **config-object form**: `fn(build_config(ctx))`. The tier beneath `apply` for the common case `apply` *can't* express — a step fn that takes a typed config object (a dataclass/pydantic config), or inputs derived from a dep path (`f"{ctx.path(dep)}/sub"`). `build_config(ctx)` pulls dep paths and `ctx.out` itself; pass the same handles in `deps`. `kind` selects the produced `Artifact` type (consumer routing); `resources`/`pin` as for `apply`. This is the workhorse for the dataset catalogs (transforms, conversions, filters) — it is the thin, named wrapper over the `Lazy`+`Recipe` construction, kept so a catalog author writes one call, not five lines of plumbing.
- `materialized_config(handle, prefix)` — `build_config` under `for_run(out=handle.path(prefix), prefix=prefix, run_args=handle.recipe.run_args)` with `region=None`, for inspection / golden tests. Runs nothing. Passes the recipe's real `run_args` (so a `build_config` that reads `ctx.run_arg(...)` — e.g. `train_lm` — does not `KeyError`); still not identical to the run-time config, which also binds a real `region`.

Two researcher helpers, by fn shape: `apply` when the step fn takes keyword inputs (a bare `Lazy`
resolves to its path, `OUT` to `ctx.out`); `derived` when it takes a typed config object or needs
`build_config` to compose paths. Both return `Lazy[T]` and sit over the same `Lazy`+`Recipe` tier the
typed builders (`tokenized`, `train_lm`, `mixture`) use directly for full `RunContext` control.

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

def remote(fn=None, *, name=None, resources=None, env_vars=None,
           pip_dependency_groups=None) -> RemoteCallable | Callable[..., RemoteCallable]: ...
```

`remote(fn, resources=…)` wraps a callable for Fray dispatch (also usable as `@remote(resources=…)`).
Resources ride on the callable, never on the `StepSpec` node, so they never enter the fingerprint.

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
    def hash_id(self) -> str: ...             # sha256({name, attrs: hash_attrs, deps})[:8] — NOT fingerprint_payload
```

`lower()` sets `override_output_path={name}/{version}` (or the pin), `hash_attrs={"fingerprint", "version", "result_type", "deps", "expected_fingerprint"?}`, and `fingerprint_payload=<canonical json>`. `fingerprint_payload` is **not** part of `hash_id` and is not logged.

**Caching for lazy artifacts is by output path + status, not `hash_id`.** Because `lower()` sets an explicit `override_output_path` (`{name}/{version}`), the `name_with_hash`/`hash_id` machinery does not pick the address — the runner serves a cached output iff `.executor_status` at that path reads `SUCCESS` (gated by `check_drift`). `hash_attrs` exists so the record-bearing fields (fingerprint/version/result_type/deps) travel to `check_drift` and the record, not to select the path.

## Core: the artifact record + drift check (`marin.execution.artifact`)

```python
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

class ArtifactRecord(pydantic.BaseModel):
    name: str
    version: str
    fingerprint: str                     # 8 hex, identity hash (placeholders for pulled values)
    result_type: str                     # "module.QualName" of the produced Artifact subclass
    output_path: str
    deps: list[str]                      # dependency identities as "name@version"
    config: dict[str, JSONValue] | None = None   # materialized config that ran (canonical-encoded), for humans
    command_line: list[str] | None = None        # sys.argv of the launching process
    git_commit: str | None = None
    user: str | None = None
    created_at: str                      # ISO-8601
    source: str | None = None            # adopted-source location; None for a computed artifact
    result: dict[str, JSONValue] | None = None   # JsonArtifact.model_dump() for a value artifact; None for data
    fingerprint_payload: str | None = None       # canonical-JSON identity basis, for the drift diff

class FingerprintMismatchError(Exception): ...      # opt-in hard pin (lower) / pinned drift (cache)
class ArtifactTypeMismatchError(Exception): ...     # resolve: recorded result_type != requested

def is_mutable_version(version: str) -> bool: ...                 # version == "dev" or endswith("-dev")
def read_record(output_path: str) -> ArtifactRecord | None: ...  # the full record at {path}/artifact.json
def write_record(record: ArtifactRecord) -> None: ...
def read_artifact(output_path: str, schema: type[M]) -> M: ...   # manual: record.result validated as M
def write_artifact(value: M, output_path: str) -> None: ...      # manual: minimal record(result=value)
def check_drift(step: StepSpec) -> bool: ...                     # returns True iff mutable (caller rebuilds)
```

- `ArtifactRecord` — the single output descriptor (pydantic; `model_dump_json`/`model_validate_json`). **All fields except `result` are optional**, so a minimal *manual* record (`write_artifact`) and pre-existing legacy files load without error (missing fields → `None`); the lazy runner fills them all. Subsumes the old `.artifact.json` payload sidecar (now `result`) and the old `.artifact_record.json` (provenance). **Serialization:** `config` is the materialized config through the canonical encoder (`json.loads(canonical_json(cfg))` — JSON of real values, for humans); `result` is `JsonArtifact.model_dump(mode="json")` for a value artifact, `None` for data; a value type that isn't JSON-encodable surfaces at write time.
- `read_record` / `write_record` — read/write the full record at `{output_path}/artifact.json`. A corrupt/partial file raises `pydantic.ValidationError`. Legacy names (`.artifact_record.json`, `.artifact`) are read but not written.
- `read_artifact` / `write_artifact` — the **manual** typed-payload API datakit uses (the replacement for `Artifact.from_path`/`Artifact.save`). `read_artifact(p, T)` returns `read_record(p).result` validated as `T` (or, for a legacy bare-payload file, the file validated as `T`); `write_artifact(v, p)` writes a minimal record. The automatic layer is the same `read_record`/`write_record` with full provenance.
- `check_drift(step)` — reads `fingerprint`/`version` from `step.hash_attrs`. Returns `False` for a non-lazy step (no fingerprint). Returns `True` for a mutable (`dev`) version so the caller rebuilds. Otherwise, if a record exists and its `fingerprint` differs from the step's: normally **logs a warning** with a field-level diff (current vs recorded `fingerprint_payload`, capped at 20 lines) and returns `False` so the cached output is served (advisory). **But** if the step carries an `expected_fingerprint` (the opt-in pin), the same mismatch **raises `FingerprintMismatchError`** instead — the pin makes drift a hard gate on the cache hit too, not just at `lower()`. Run by `StepRunner` before serving a cached SUCCESS. (Replaces `enforce_immutability` + `ImmutableArtifactError`.)
- `FingerprintMismatchError` — raised by `lower()` when `expected_fingerprint` is set and differs from the computed fingerprint, and by `check_drift` when a pinned artifact's recorded fingerprint differs from the computed one. The opt-in hard gate, both before build and before serving cache.
- `ArtifactTypeMismatchError` — raised by `resolve` when the served record's `result_type` differs from the handle's `result_type` (a value artifact whose schema/type changed under a reused version).

## Core: fingerprint (`marin.execution.fingerprint`)

```python
def canonical_json(config: object) -> str: ...
def fingerprint_hash(payload: str) -> str: ...                   # md5(payload.encode()).hexdigest()[:8]
def register_fingerprint(tp: type, encode: Callable[[Any], object]) -> None: ...
```

- `canonical_json` — deterministic JSON. Canonicalizes: dataclasses (`asdict`), `Enum` (`.value`), `timedelta`, `Path`, `set`/`frozenset` (sorted members), numpy `dtype`/scalar/array, type objects, and any type registered via `register_fingerprint`. For an unknown type with no canonical form it falls back to a **stable representation and logs once** (advisory mode) rather than raising — a fingerprint misfire is a noisy warning, not a blocked build. Dict keys must be strings (`sort_keys=True`).
- `register_fingerprint(tp, encode)` — teach the encoder a canonical form for an identity-bearing custom type (e.g. a project's config carrying a non-serializable field), so it fingerprints precisely instead of via the best-effort fallback.

---

## Authoring: sweep / select (`marin.experiment.sweep`)

```python
class Selection(JsonArtifact):
    winner: str                          # winning trial "name@version"
    score: float
    winner_path: str
    scores: dict[str, float]
    metrics: dict[str, Any]

def read_replicated_metrics(output_path: str) -> Mapping[str, Any]: ...
def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]: ...
def sweep(trial: Callable[..., Lazy[Checkpoint]], **axes: Sequence[Any]) -> list[Lazy[Checkpoint]]: ...
def select(name: str, version: str, trials: Sequence[Lazy[Checkpoint]], *,
           metric: str, mode: str = "min",
           reader: Callable[[str], Mapping[str, Any]] = read_replicated_metrics) -> Lazy[Selection]: ...
```

There is **no** `AnnotatedCheckpoint` / `annotate`. A trial is a plain `Lazy[Checkpoint]`-producing
function (e.g. `lambda **p: train_lm(...)`). `select` reads each trial's metrics with a single
`reader` (default `read_replicated_metrics`), since a sweep is homogeneous — every trial writes its
metrics the same way. A custom metric source is a `reader=` argument to `select`, not a per-trial
wrapper.

- `read_replicated_metrics(output_path)` — reads `<output>/tracker_metrics.jsonl` (the WandB `replicate_path` that `train_lm` mirrors) and returns its `summary` mapping. Raises `FileNotFoundError` if absent.
- `grid(**axes)` — Cartesian product of the named axes as `{axis: value}` dicts, row-major (last axis fastest).
- `sweep(trial, **axes)` — `[trial(**params) for params in grid(**axes)]`. The `trial` builder must fold each grid point's values into both its config (distinct fingerprint) and its `name` (distinct address).
- `select(name, version, trials, *, metric, mode="min", reader=…)` — a reducer depending on every trial; returns a `Lazy[Selection]` at `name@version`. At run time it reads each trial's metrics with `reader`, ranks by `metric`, and produces a `Selection` (`winner`/`score`/`winner_path`/`scores`/`metrics`). `metric`/`mode` bear identity; the **`reader` does not** — like a run-arg, it is a read-time concern (a callable has no stable fingerprint), so swapping readers does not re-identify the selection; bump `version` if a reader change should be a new artifact. Trial *values* are read at run time; trials are keyed by full `name@version`. Raises `ValueError` if `mode not in {"min","max"}`, if `trials` is empty, or if two trials share `name@version`; `KeyError` if a trial's metrics lack `metric` (run time).

## Authoring: training (`marin.experiment.train`)

```python
DEFAULT_VERSION = "v1"
MARIN_PRECISION = "p=f32,c=bfloat16"

def train_lm(*, name: str, model: LmConfig, optimizer: OptimizerConfig,
             datasets: Mapping[Lazy[Dataset], float],
             batch_size: int, seq_len: int, num_train_steps: int,
             z_loss_weight: float | None, evals: EvalSuite | None, resources: ResourceConfig,
             validation: Sequence[Lazy[Dataset]] = (),
             init_from: Lazy[Checkpoint] | None = None,
             version: str = DEFAULT_VERSION, mp: str = MARIN_PRECISION,
             tensor_parallel_size: int = 1, steps_per_eval: int = 1000,
             wandb_project: str = "marin", wandb_group: str | None = None,
             run_id: str | None = None, tags: Sequence[str] = (),
             env_vars: dict[str, str] | None = None) -> Lazy[Checkpoint]: ...
```

The opaque `data: Callable[[RunContext], LmDataConfig]` + `deps` pair is replaced by
`datasets: Mapping[Lazy[Dataset], float]` (handle → mixture weight) plus `validation:
Sequence[Lazy[Dataset]]` (added at weight 0). `train_lm` assembles the `mixture` internally and
derives the recipe's `deps` from the keys — so the caller can no longer desync `data` from `deps`,
and the signature shows a reader exactly what to pass. A non-mixture data config is still reachable
via the lower-level `Lazy`+`Recipe` tier. `init_from` chains onto another checkpoint (a dep, seeds
`initialize_from_checkpoint_path`). `resources` is a run-arg (never in the fingerprint).

## Authoring: data builders (`marin.experiment.data`)

`apply` is **no longer here** (it moved to the lazy core). The data module keeps the typed,
domain-specific builders, each returning a `Lazy[Dataset]`; `resources` (when given) dispatches via
`remote`, else inline.

```python
def tokenized(name: str, *, tokenizer: str, source: str | None = None,
              paths: Sequence[str] | None = None, raw: Lazy[Dataset] | None = None,
              glob: str | None = None, validation: bool = False, pin: str | None = None,
              text_key: str = "text", sample_count: int | None = None,
              version: str = "v1", tags: Sequence[str] = (),
              resources: ResourceConfig | None = None) -> Lazy[Dataset]: ...

def hf_download(name: str, *, hf_id: str, revision: str, urls_glob: Sequence[str] = (),
                version: str = "v1", pin: str | None = None,
                resources: ResourceConfig | None = None) -> Lazy[Dataset]: ...

def pretokenized(name: str, *, repo_id: str, tokenizer: str, revision: str | None = None,
                 version: str = "v1", pin: str | None = None, tags: Sequence[str] = (),
                 resources: ResourceConfig | None = None) -> Lazy[Dataset]: ...

def mixture(ctx: RunContext, train: Mapping[Lazy[Dataset], float], *,
            validation: Sequence[Lazy[Dataset]] = (),
            shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE) -> LmDataConfig: ...
```

- `tokenized` — exactly one raw-input mode (else `ValueError`): `source` (HF id `org/name`, else a single path), `paths` (globs resolved against `ctx.prefix`), or `raw` + `glob` (a download handle + subpath; both required together). `validation` routes to the validation split; `sample_count` bears identity; `pin` references existing tokenized data.
- `hf_download` — a HuggingFace-Hub download as a raw-data `Lazy[Dataset]` that `tokenized(raw=…)` or `apply` can depend on. (Replaces the old `raw_download`+`hf_download` pair; a generic raw download is now just `apply(name, fn, output_path=OUT, …)`.)
- `pretokenized` — download an already-tokenized Levanter cache from HF `repo_id`; consumed as a normal tokenized `Dataset`.
- `mixture` — assemble a Levanter `LmDataConfig` from handles. `train` maps each to its weight; `validation` handles added at weight 0; each component keyed by the handle's `name` (collisions rejected) and rooted at `ctx.path(handle)`. Raises `ValueError` if empty or if components span >1 tokenizer; `TypeError` if a component's config isn't a tokenize config. Called inside a consumer's `build_config` with the same handles as `deps`.

---

## Persisted shapes

Two paths, which coincide for a computed artifact but diverge for an adopted one:

- **Record path** — always `{prefix}/{name}/{version}/` (the `StepSpec` output path). The record is written here even for an adopted artifact.
- **Consumer-data path** — where consumers resolve the bytes: the record path for a computed artifact, `adopt_source` for an adopted one, `override_path` for a pin.

Sidecar files at the **record path** (a pin writes none):

- **`artifact.json`** — the unified `ArtifactRecord` (pydantic JSON): identity (`name`/`version`/`fingerprint`), `output_path`, `deps`, the materialized `config`, `command_line`, `git_commit`/`user`/`created_at`, `source` (adopted) and `result` (value artifacts), and `fingerprint_payload` (the drift-diff basis). One file replaces the former `.artifact.json` + `.artifact_record.json`.
- **`.executor_status`** — the status marker (`SUCCESS`/`FAILED`/…) written by `StatusFile`; the cache check serves a cached artifact only when this reads `SUCCESS`. (Filename unchanged; `executor_step_status.py` is reused as-is.)

## Errors

| Error | Raised when |
|---|---|
| `FingerprintMismatchError` | `expected_fingerprint` is set and the computed fingerprint differs — at `lower()` (before build) and in `check_drift` (before serving cache). The opt-in hard identity gate. |
| `ArtifactTypeMismatchError` | `resolve` finds the served record's `result_type` differs from the handle's `result_type` |
| `FileNotFoundError` | an adopted step runs and no data exists at its source; `read_record`/`Artifact.load` finds no `artifact.json`; `read_replicated_metrics` finds no `tracker_metrics.jsonl` |
| `pydantic.ValidationError` | a record (`artifact.json`) is corrupt/partial; a `JsonArtifact.load` payload fails to validate against its schema |
| `ValueError` (lower) | a fixed handle depends on a `dev`/`-dev` dep; a `JsonArtifact` `result_type` is paired with a `RemoteCallable` fn; a step `fn` returns a non-`Artifact` for a `JsonArtifact` `result_type` (run time) |
| `ValueError` (name/version) | `name` or `version` fails the grammar (empty, contains `..`, leading/trailing slash, or a URL scheme) — checked at `Lazy` construction |
| `TypeError` (apply) | `fn`'s signature cannot bind the resolved kwargs |
| `TypeError` (mixture) | a `mixture` component config isn't a tokenize config |
| `ValueError` | `Lazy` with both `adopt_source` and `override_path`; `tokenized` not given exactly one source mode, or `raw`/`glob` not paired; `mixture` empty or spanning >1 tokenizer; `select` bad `mode`, empty sweep, or colliding trial `name@version` |
| `KeyError` | `ctx.run_arg(key)` for an undeclared key; a `select` trial's metrics missing `metric` |
| *(lock/write failure)* | the distributed `step_lock` cannot be acquired/refreshed, or a record/status write fails — propagates from `StepRunner` (see `executor_step_status`) |
| *(Fray failure)* | a `remote` step's Fray job fails — the underlying error propagates from `run`/`resolve` |

**No `ImmutableArtifactError`.** A fixed `name@version` rebuilt from a changed recipe (without an
`expected_fingerprint` pin) is an advisory **warning** (field-level diff) and serves the existing
output; it is not an error.

## Invariants & preconditions

What the contract requires of callers, and what it does **not** check:

- **`name`** — a path segment that may nest with `/`. Validated at construction: non-empty, no `..`, no leading/trailing slash, no URL scheme — else `ValueError`. (Tightened from the prior "not checked" stance; malformed names are a caller bug, not silent malformed paths.)
- **`version`** — non-empty, same grammar as `name` (no slashes needed but allowed-free). Mutable iff `"dev"` or `*-dev`; otherwise immutable.
- **`result_type`** — an `Artifact` subclass. The handle's `result_type` is recorded and re-checked at `resolve` (→ `ArtifactTypeMismatchError` on drift); a `fn` that returns the wrong type for a `JsonArtifact` `result_type` raises at run time.
- **`expected_fingerprint`** — opaque 8-hex compare; format not validated (a malformed pin never matches → `FingerprintMismatchError`).
- **`deps`** — may contain duplicates; the runner dedupes by output path. Two deps sharing `name@version` are indistinguishable to a consumer's fingerprint.
- **`build_config`** — must be a deterministic pure function for the fingerprint to be stable; I/O/env/time reads are not prevented but make the (advisory) fingerprint noisy.
- **`recipe.fn` source** — not part of the fingerprint. Changing the body while keeping the config identical serves cached output; bump the version.

## Concurrency & idempotency

- **`lower()` is structurally deterministic, not side-effect-free.** Re-lowering the same handle yields the same `StepSpec`/fingerprint/address, but the step fn it produces closes over `get_git_commit()`/`get_user()`/`sys.argv` captured in the launching process, so the record it later writes embeds the launch provenance. (So "pure transform" above means *of structure* — same inputs, same graph — not "no captured state".)
- **Concurrent builds are serialized by the existing `step_lock`.** This is unchanged infrastructure (`executor_step_status`): the first builder of an output path takes the distributed lock, runs, writes the output + record + `SUCCESS`; a racing builder waits, then sees `SUCCESS` and serves it. "First-build-wins" is that existing serialization — it is *not* a new guarantee this design adds, and there is no per-fingerprint build-once gate.
- **The drift check is advisory and read-only.** Run before serving a cached SUCCESS, it compares the computed fingerprint against the record and *warns* (or, if `expected_fingerprint` is pinned, raises). It never blocks a build, overwrites an output, or rewrites a record. It does not participate in the lock.
- **`dev` versions always rebuild,** even after a successful run, and a fixed artifact may not depend on one (rejected at `lower()`).
- **Drift detection is best-effort.** A non-deterministic `build_config` or an unregistered custom type can make the fingerprint noisy → a spurious "recipe changed" warning, never a wrong build. `result_type`/schema mismatch, by contrast, is a *hard* `ArtifactTypeMismatchError` at `resolve` — type identity is enforced even though config drift is not.

## File layout

| Piece | Path |
|---|---|
| Handle, recipe, context, `apply`, `run`, `resolve`, `lower`, `adopt`, `materialized_config`, `OUT` | `lib/marin/src/marin/execution/lazy.py` |
| `Artifact` base + `Dataset`/`Checkpoint`/`JsonArtifact`, `ArtifactRecord`, `read/write_record`, `check_drift` | `lib/marin/src/marin/execution/artifact.py` |
| Deterministic encoder + `register_fingerprint` | `lib/marin/src/marin/execution/fingerprint.py` |
| Provenance (git/user/argv/now) | `lib/marin/src/marin/execution/provenance.py` |
| `StepSpec` (+ `fingerprint_payload`) | `lib/marin/src/marin/execution/step_spec.py` |
| Runner (drift check, lock, cache, dispatch) | `lib/marin/src/marin/execution/step_runner.py` |
| Compute-on-the-fn (`remote`, `RemoteCallable`) | `lib/marin/src/marin/execution/remote.py` |
| Sweep / select | `lib/marin/src/marin/experiment/sweep.py` |
| Training assembler | `lib/marin/src/marin/experiment/train.py` |
| Data builders + mixture | `lib/marin/src/marin/experiment/data.py` |

`marin.execution.registry` is **removed** — its `ArtifactRecord` and guard fold into
`marin.execution.artifact`, which is also where the old payload serializer lived. This is the
concrete fix for the "two modules named `artifact`/`registry` doing overlapping things" confusion.

**Manual vs. automatic serialization (one scheme).** There are not two systems: the lazy
("automatic") layer sits on the datakit ("manual") layer. Both read/write the same `ArtifactRecord`
JSON at an output path; the only difference is who fills it and how much:

```python
def read_record(path: str) -> ArtifactRecord | None: ...        # the full record (or None)
def write_record(record: ArtifactRecord) -> None: ...           # write a full record
def read_artifact(path: str, schema: type[M]) -> M: ...         # manual: typed payload = record.result as M
def write_artifact(value: M, path: str) -> None: ...            # manual: write a minimal record (result=value)
```

- **Manual** (datakit StepSpec pipelines): a step calls `write_artifact(value, path)` — a minimal
  `ArtifactRecord(result=value, name/version/... best-effort or omitted)` — and a consumer calls
  `read_artifact(dep_path, T)`.
- **Automatic** (lazy runner): on success the runner calls `write_record` with the *full* record
  (config, fingerprint, git, argv, deps, result_type, result). `Artifact.load` / `resolve` read it.

Because every new field on `ArtifactRecord` is optional, a minimal manual record and a pre-existing
legacy file (`.artifact.json`, `.artifact`) both load without error — missing fields read as `None`.

**Blast radius.** The old `Artifact.from_path(path, T)` / `Artifact.save(value, path)` classmethods
(~60+ sites, almost all `experiments/datakit/**`) migrate **in this PR**: `Artifact.from_path(p, T)`
→ `read_artifact(p, T)` (or `T.load(p)` for an `Artifact` subclass); `Artifact.save(v, p)` →
`write_artifact(v, p)`; the synthetic `PathMetadata` return → a plain `Artifact` path ref. Mechanical
and wide, delegated to sub-agents.

## Out of scope

- **Deleting the remaining Executor symbols** beyond what this branch already removed.
- **`resolve` loading large data into the process** — it returns a path-bearing ref; a lazy `Checkpoint.load_model(...)` is a possible later addition (Open Questions).
- **Surfacing drift beyond a log** (run-summary, record flag) — Open Questions.
- **The `Dataset = Lazy[ConcreteDataset]` ergonomic alias** — deferred.
- **A scheme-tagged `tokenized(source="hf://…")`** collapsing the source-mode heuristic.
