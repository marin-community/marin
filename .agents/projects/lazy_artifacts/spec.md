# Spec: Lazy Artifacts

The public surface a reviewer is agreeing to, plus the canonical authoring patterns. Signatures
are pinned to branch `weaver/adopt-executor-context-for-remai` @ `4221dde`.

## Core handle + recipe (`marin.execution.lazy`)

```python
@dataclass(frozen=True, eq=False)
class Artifact:
    name: str
    version: str                      # "vN"/CalVer immutable; "dev"/"*-dev" mutable
    recipe: "Recipe"
    override_path: str | None = None          # pin to existing data; no identity effect
    adopt_source: str | None = None           # register pre-existing data (set via adopt())
    expected_fingerprint: str | None = None   # optional pin; lower() raises on mismatch

    def fingerprint_payload(self) -> str: ...  # canonical config JSON (or {"adopt_source": …})
    def fingerprint(self) -> str: ...          # md5(payload)[:8]
    def path(self, prefix: str | None = None) -> str: ...

@dataclass(frozen=True, eq=False)
class Dataset(Artifact): ...      # routes to dataset consumers
@dataclass(frozen=True, eq=False)
class Checkpoint(Artifact): ...   # routes to initialize_from_checkpoint_path

@dataclass(frozen=True)
class Recipe:
    fn: Callable[[Any], Any]                       # fn(config); RemoteCallable to run on Fray
    build_config: Callable[["RunContext"], Any]    # pure function of the context
    deps: tuple[Artifact, ...] = ()
    run_args: Mapping[str, Any] = field(default_factory=dict)
```

**Contract.** An `Artifact` is inert — constructing it runs nothing. `eq=False`: two handles are
the same iff they are the same object (so a handle can key a `mixture` dict despite an unhashable
`ResourceConfig` in its recipe); value-equality lives in `name@version` + the fingerprint. A step
produces its artifact by writing serialized output to `ctx.out`; an inline `fn` may instead
`return` a value (the runner persists it), but a `remote` fn cannot (a Fray job returns nothing).

## The identity line (`RunContext`)

```python
@dataclass(frozen=True)
class RunContext:
    out: str
    prefix: str
    region: str | None
    def path(self, dep: Artifact) -> str: ...      # dep's resolved (region-local) output path
    def run_arg(self, key: str) -> Any: ...         # a recipe-declared run-arg's value

    @staticmethod
    def for_run(out, prefix, *, region=None, run_args=None) -> "RunContext": ...
    @staticmethod
    def for_fingerprint(run_arg_keys=()) -> "RunContext": ...   # placeholders for all pulls
```

**Contract — the rule that defines identity.** Values written as **literals** in `build_config`
(model, hyperparameters, dep *versions*) enter the fingerprint. Values **pulled from `ctx`**
(`ctx.out`, `ctx.path(dep)`, `ctx.prefix`, `ctx.region`, `ctx.run_arg(key)`) never do.
`for_fingerprint()` substitutes placeholders for every pull, so a prefix/region/accelerator change
cannot re-fingerprint. `run_arg(key)` raises `KeyError` if `key` is not declared in `recipe.run_args`.

## Lowering + adoption

```python
def lower(artifact: Artifact) -> StepSpec: ...
def adopt(name: str, version: str, source: str, *, kind: type[Artifact] = Dataset) -> Artifact: ...
def materialized_config(artifact: Artifact, prefix: str) -> Any: ...   # for inspection / golden tests
```

**`lower` contract.** Pure transform: recurses deps into a `StepSpec` graph addressed by explicit
`{name}/{version}` (or the pin), carries `{fingerprint, version, deps}` in `hash_attrs` and the
canonical payload in `StepSpec.fingerprint_payload`, and writes an `ArtifactRecord` on success.
Never inspects `recipe.fn` (no Iris/Fray awareness). Raises `FingerprintMismatchError` when
`expected_fingerprint` is set and differs from the computed fingerprint — *before* any build.

**`adopt` contract.** Registers `name@version` pointing at `source`; consumers resolve to `source`
(no move, no recompute); a provenance record is written at the canonical address; re-adopting the
same `name@version` from a different source raises `ImmutableArtifactError`.

## Registry + fingerprint

```python
# marin.execution.registry
@dataclass(frozen=True)
class ArtifactRecord:
    name: str; version: str; fingerprint: str; output_path: str
    git_commit: str | None; user: str | None; created_at: str; deps: list[str]
    source: str | None = None
    fingerprint_payload: str | None = None     # enables field-level conflict diffs

def is_mutable_version(version: str) -> bool: ...        # "dev" or endswith("-dev")
def guard_immutable(output_path, name, version, fingerprint, payload=None) -> None: ...
def enforce_immutability(step: StepSpec) -> bool: ...    # True iff mutable (rebuild)

# marin.execution.fingerprint
def canonical_json(config: object) -> str: ...           # strict, deterministic
def fingerprint_hash(payload: str) -> str: ...           # md5(payload)[:8]
```

**Guard contract.** `guard_immutable` is a no-op when unbuilt, when the recorded fingerprint
matches, or when the version is mutable; otherwise it raises `ImmutableArtifactError`. When both
the recorded and current payloads are present it appends a field-level diff
(`learning_rate: 3e-3 -> 4e-3`, capped at `_MAX_DIFF_LINES = 20`). The fingerprint covers the
config `build_config` produces, **not** the source of `recipe.fn` — a behavior change inside a step
fn that leaves the config unchanged is not detected. `enforce_immutability` runs in `StepRunner`
*before* a cached SUCCESS is served; today it runs *before* `step_lock` and isn't re-checked on the
already-done path, so concurrent first-builds with different fingerprints are not yet guarded
(design.md Open Questions). The intended contract is guard-under-lock.

**Encoder contract.** `canonical_json` canonicalizes dataclasses (`asdict`), enums (value),
`timedelta`/`Path`, numpy/jax dtypes (name), sets/frozensets (sorted members), and numpy/jax arrays
(content + dtype + shape). It **raises `TypeError`** on callables and objects using the default
`object.__repr__` — anything with no reproducible serialization — rather than falling back to
`str(o)`. The same config always produces identical bytes across processes.

## Authoring APIs (`marin.experiment`)

```python
# sweep.py
def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]: ...
def sweep(trial: Callable[..., Artifact], **axes: Sequence[Any]) -> list[Artifact]: ...
def select(name: str, version: str, trials: Sequence[Artifact], *,
           metric: str, mode: str = "min") -> Artifact: ...   # mode in {"min","max"} else ValueError

# data.py  (DEFAULT_VERSION is the catalog default)
def tokenized(name: str, *, tokenizer: str, source: str | None = None,
              paths: Sequence[str] | None = None, raw: Dataset | None = None,
              glob: str | None = None, validation: bool = False, pin: str | None = None,
              text_key: str = "text", version: str = DEFAULT_VERSION,
              tags: Sequence[str] = (), resources: ResourceConfig | None = None) -> Dataset: ...
def pretokenized(name: str, *, repo_id: str, tokenizer: str, revision: str | None = None,
                 version: str = DEFAULT_VERSION, pin: str | None = None,
                 tags: Sequence[str] = (), resources: ResourceConfig | None = None) -> Dataset: ...
def raw_download(name: str, *, fn, build_config, version: str = DEFAULT_VERSION,
                 pin: str | None = None, resources: ResourceConfig | None = None) -> Dataset: ...
def mixture(ctx: RunContext, train: Mapping[Dataset, float], *,
            validation: Sequence[Dataset] = (), shuffle=DEFAULT_LM_DATA_SHUFFLE) -> LmDataConfig: ...
```

**Contracts.** `select` depends on every trial, reads each trial's metrics payload at run time, and
writes `{"winner", "score", "winner_path", "metrics", "scores"}`; `metric`/`mode` bear identity,
trial *values* do not. It keys trials by full `name@version` and **raises `ValueError` if two
trials share an identity** (so a same-name/different-version collision can't silently drop a trial);
a `trial` builder must therefore fold its swept values into both the config (distinct fingerprint)
and the `name` (distinct address). `tokenized` covers pinned / fresh-from-HF (`source`) / raw-glob
(`paths`) / download-dependent (`raw` + `glob`) caches; exactly one source mode. `mixture` resolves
each component's cache path via `ctx.path(handle)` and emits weight-0 entries for `validation` sets.

### Identity-bearing vs run-only

The line `build_config` draws (Open Question in `design.md`); when in doubt, make it a literal.

| Value | Identity-bearing (literal → fingerprint) | Run-only (pulled from `ctx`) |
|---|---|---|
| Model / hyperparameters / token budget | ✔ literal | |
| Logical mesh (`MeshConfig` axes, partitioning) | ✔ literal | |
| Dependency *identity* (`name@version`) | ✔ (dep versions enter the fingerprint) | |
| Dependency *path* | | `ctx.path(dep)` (region-local) |
| Output path / storage prefix / GCP region | | `ctx.out`, `ctx.prefix`, `ctx.region` |
| Physical accelerator (`ResourceConfig.with_tpu`) | | `ctx.run_arg("train_resources")` |
| Random seed, data source revision | ✔ literal (they change the result) | |

Rule: anything that changes *what is computed* is a literal; `run_args` are reserved for placement
that leaves the result equivalent. Cross-hardware numeric drift is the known soft spot (design.md).

## Canonical patterns

**Single run** — `build()` returns a `Checkpoint`; every decision is plain config inline:

```python
def build(*, version: str = "v1") -> Checkpoint:
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [*paloma_validation(...), *uncheatable_validation(...)]
    weighted = {train[n]: DCLM_MIXTURE_WEIGHTS[n] for n in train}
    def build_config(ctx: RunContext) -> TrainLmOnPodConfig:
        inner = TrainLmConfig(data=mixture(ctx, weighted, validation=validation),
                              optimizer=AdamConfig(learning_rate=3e-3, ...), model=..., ...)
        return TrainLmOnPodConfig(train_config=inner, resources=ctx.run_arg("train_resources"),
                                  output_path=ctx.out, env_vars=None)
    return Checkpoint(name="checkpoints/dclm_1b_1x_how_to", version=version,
                      recipe=Recipe(fn=_train, build_config=build_config,
                                    deps=(*train.values(), *validation),
                                    run_args={"train_resources": TRAIN_RESOURCES}))
# _train(cfg): remote(run_levanter_train_lm, resources=cfg.resources)(cfg)
# run: StepRunner().run([lower(build())])
```

**Sweep + select** — fan out over a grid, pick by metric:

```python
trials = sweep(_train_and_eval, learning_rate=[3e-4, 6e-4, 1e-3], weight_decay=[0.0, 0.1, 0.2])
best = select("sweeps/tiny/best", "v1", trials, metric="train/loss", mode="min")
```

**Multi-phase chain** — each phase initializes from its parent's checkpoint:

```python
def _phase(name, *, fn, steps, parent, ...) -> Checkpoint:
    def build_config(ctx):
        init_from = f"{ctx.path(parent)}/checkpoints" if parent is not None else None
        return ...
    deps = () if parent is None else (parent,)
    return Checkpoint(name=name, version="v1", recipe=Recipe(fn=fn, build_config=build_config, deps=deps, ...))
pretrain = _phase("…/pretrain", parent=None, ...)
mid      = _phase("…/midtrain", parent=pretrain, ...)
```

**Adopt pre-existing data** — register, don't recompute:

```python
cache = adopt("tokenized/dclm_baseline", "v1", "gs://…/dclm_baseline-0206f1/", kind=Dataset)
```

## Pins and adopted data — three tiers of guarantee

| Mechanism | What it guarantees | What it does NOT |
|---|---|---|
| `name@version` (normal) | build-once: content recorded + fingerprint-guarded | — |
| `adopt(name, version, source)` | **pointer** immutability: the alias→source mapping is recorded and guarded (re-adopt from a different `source` raises) | **content** immutability — it fingerprints the `source` *string*, so external data mutating in place is invisible; a *relative* source under a moved prefix points the same fingerprint at different bytes |
| `override_path` pin | nothing — resolves to existing data | writes no record: no provenance, no guard. An explicit, unguarded escape hatch |

## File layout

| Piece | Path |
|---|---|
| Handles, recipe, context, `lower`, `adopt` | `lib/marin/src/marin/execution/lazy.py` |
| Build-once guard, record, conflict diff | `lib/marin/src/marin/execution/registry.py` |
| Strict deterministic encoder | `lib/marin/src/marin/execution/fingerprint.py` |
| `StepSpec` (+ `fingerprint_payload` field) | `lib/marin/src/marin/execution/step_spec.py` |
| Runner (guard before cache-skip) | `lib/marin/src/marin/execution/step_runner.py` |
| Compute-on-the-fn | `lib/marin/src/marin/execution/remote.py` |
| Sweep / select | `lib/marin/src/marin/experiment/sweep.py` |
| Data builders + mixture | `lib/marin/src/marin/experiment/data.py` |
| Catalogs (lazy) | `experiments/pretraining_datasets/{dclm,simple,nemotron}_lazy.py`, `experiments/paloma_lazy.py`, `experiments/evals/uncheatable_lazy.py` |

## Errors

| Error | Raised when |
|---|---|
| `ImmutableArtifactError` | rebuild of a fixed `name@version` from a changed recipe (with field-level diff); re-adopt from a different source |
| `FingerprintMismatchError` | `expected_fingerprint` set and the computed fingerprint differs (at `lower()` time, before build) |
| `TypeError` (encoder) | a config value has no reproducible serialization (callable, default-`repr` object) |
| `ValueError` | `select` mode not in `{"min","max"}`, an empty sweep, or trials with colliding `name@version` |
| `KeyError` | `ctx.run_arg(key)` for a key not declared in `recipe.run_args` |

## Out of scope

- **Deleting the Executor.** `Executor`/`executor_main`/`versioned`/`InputName`/`this_output_path`
  stay until catalogs migrate; this design defines the target, not the deletion PR.
- **Migrating the 113 remaining import-time sites.** Tracked separately; `defaults.py` + dataset
  catalogs first.
- **Flipping `MARIN_EXECUTOR_STRICT` default-on.** A sequencing decision (Open Question 1).
- **Fingerprint attribution / tracing.** The field-level diff ships; per-field *why* (Open
  Question 4) is deferred.
- **Auto-versioning** (fingerprint-suffixed paths). Versions are author-chosen (Open Question 3).
