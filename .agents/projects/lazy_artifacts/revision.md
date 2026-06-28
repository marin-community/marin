# Lazy Artifacts — Revision: artifacts-with-producer, a pure lazy core

This revises the design now on PR #6649 after a second review pass. The first cut landed
`Lazy[T]` + advisory drift + a unified `ArtifactRecord`, but the review surfaced three
structural problems and a cleanup list. This doc states the changes; it does not re-derive
the parts that stand (advisory drift, the record schema, `name@version` addressing).

_Updated after a codex review pass; the resolutions of its P1/P2/P3 findings are folded into
the sections below and noted in §"Review-comment coverage"._

## What the review is telling us

Three themes, each from several comments:

1. **Artifact types belong with their producer, not in a central `artifact.py`.**
   > "Why isn't `Checkpoint` defined here? … it feels this could simply be a
   > `LevanterCheckpoint` artifact, defined here" (train.py:38)

   A `Checkpoint`'s on-disk shape, its `load`, and the `checkpoints/` subdir are all known by
   `run_levanter_train_lm` — the thing that *writes* it. Splitting the type from its producer
   means the format is described in two places. `Selection` already lives next to `select`
   (sweep.py:31); that is the pattern, applied everywhere.

2. **`Lazy` is only the produce-via-step-runner model — it must not know concrete types.**
   > "`Lazy` shouldn't know anything about Datasets?" (lazy.py:48)

   `lazy.py` imports `Dataset` and defaults `adopt(kind=Dataset)`. The lazy core should know
   exactly one artifact concept: the `Artifact` *base* (and the record/IO framework around it).
   It must never name a concrete producer type.

3. **No implicit remote execution; `ResourceConfig` is not a lazy primitive.**
   > "where did `resources` come from, this shouldn't be here? this should be part of a
   > `RunContext` at best?" (lazy.py:497)

   `apply`/`derived` take `resources=` and wrap the fn in `remote(...)`. Remote dispatch is an
   *opt-in wrapper the caller applies to its own function*, never a parameter baked into the
   generic builders. Resources that a step's *config* genuinely needs travel through
   `RunContext.run_arg` (already how `train_lm` does it) — that is the only place "where it
   runs" touches the lazy layer.

Plus a cleanup list (versions, provenance, sweep coupling, naming, file placement) in §"Smaller
cleanups".

## Module layout: before → after

| Module | Before | After |
|---|---|---|
| `execution/lazy.py` | `Lazy`, `Recipe`, `RunContext`, `apply`, `derived`, `adopt`, free `lower`/`run`/`resolve`; imports `Dataset`, `remote`, `ResourceConfig` | `Lazy`, `Recipe`, `RunContext`, `apply`, `adopt`; `Lazy.lower/run/resolve` methods; imports only artifact **framework** (`Artifact` base, `ArtifactRecord`, IO, errors, hash-key constants) — never a concrete type, `remote`, or `ResourceConfig`. No `derived` |
| `execution/artifact.py` | `Artifact`, `Dataset`, `Checkpoint`, `JsonArtifact`, `ArtifactRecord`, IO, `check_drift`, `_diff_json` | a **single** `Artifact` base, `ArtifactRecord`, IO, `check_drift`. **No `JsonArtifact`, no concrete types.** Diff helper moves to `fingerprint.py` |
| `execution/fingerprint.py` | canonical/hash/strict | + `describe_drift(old_payload, new_payload)` (pure diff, moved from artifact) |
| `execution/provenance.py` | free `get_git_commit`/`get_user`/… | a `Provenance` **BaseModel** + `Provenance.capture()` |
| `execution/remote.py` | `remote`/`RemoteCallable` | unchanged — this *is* the "run a fn on iris easily" helper |
| `execution/executor_step_status.py` | per-output status + lock | renamed `execution/step_status.py` |
| `training/training.py` | producer fn only | + `LevanterCheckpoint(Artifact)` and the inline `training_metrics` result step |
| `processing/tokenize/tokenize.py` | producer fn only | + `TokenizedCache(Artifact)` (was `Dataset`), exposing the metadata `mixture` needs |
| `experiment/sweep.py` | `select(metric=, reader=)` | `select(key=, mode=)` over typed result artifacts; `Selection` stays |

Dependency edges (verified against current imports): `lazy → artifact → {fingerprint,
provenance, step_spec}` — note `artifact.py` already imports `StepSpec`/`_is_relative_path`
(artifact.py:30), so `artifact → step_spec` is a real, kept edge. Producers (`training`,
`processing.tokenize`) and `experiment/*` import the `Artifact` base and define their own
concrete types. No reverse edges, no new cycles.

## The lazy core, made pure (`lazy.py`)

`Recipe.fn` becomes a plain `Callable[[Config], Artifact | None]`. The lazy layer never wraps,
inspects, or knows about remote dispatch:

```python
@dataclass(frozen=True)
class Recipe:
    fn: Callable[[Any], "Artifact | None"]   # plain callable; wrap in remote() yourself for Fray
    build_config: Callable[[RunContext], Any]
    deps: tuple["Lazy", ...] = ()
    run_args: Mapping[str, Any] = field(default_factory=dict)
```

Removed from `lazy.py`: the `from fray.types import ResourceConfig` and
`from marin.execution.remote import …` imports; the `resources=` parameter on every builder;
the `RemoteCallable` value-vs-remote guard in `lower`. What `lazy.py` keeps importing from
`artifact.py` is **framework only** — the `Artifact` base, `ArtifactRecord`, `read_record`/
`write_record`, the errors, and the `hash_attrs` key constants. It still builds and writes the
record (that is the runner's job); it just never names a concrete artifact type. (This is the
precise reading of "imports only `Artifact`": the *handle/runner* layer owns record IO; the
*producer* layer owns concrete types.)

**Persistence is uniform** (see §"One `Artifact`, no `JsonArtifact`"): the runner calls
`result.result_payload()` on whatever the step fn returns and stores it as the record's
`result` — no `issubclass` branch, no concrete type named:

```python
result = fn(config)                              # an Artifact (data ref or value) or None
payload = result.result_payload() if isinstance(result, Artifact) else None
write_record(ArtifactRecord(..., result=payload))
```

A data step writes bytes to `ctx.out` and returns `None` (or a fields-less `Artifact`, whose
`result_payload()` is `None`); a value step returns its populated `Artifact` subclass. `resolve`
loads with `result_type.load(path)`; a value subclass with required fields that finds no
`result` fails with a clear pydantic error naming the missing fields (a value fn wrongly wrapped
in `remote` returns `None`, so this is exactly where it surfaces). `lazy.py` imports only the
`Artifact` base.

**How remote works now.** A caller that wants Fray dispatch wraps its own function. There are
two valid patterns, and they do not duplicate the `ResourceConfig`:

```python
# A. Resources are only an execution choice (not in the config): capture in the closure.
recipe = Recipe(fn=remote(my_step, resources=R), build_config=build_config)

# B. The config itself must carry the resources (e.g. TrainLmOnPodConfig.resources).
#    The ResourceConfig flows once, as a run_arg, and the dispatch reads it from the config —
#    the config is the single source of truth (this is exactly what train_lm does today):
def build_config(ctx):
    return TrainLmOnPodConfig(resources=ctx.run_arg("train_resources"), ...)
def _train_job(cfg):                      # the recipe fn
    remote(run_levanter_train_lm, resources=cfg.resources)(cfg)
```

A `run_arg` is excluded from the fingerprint by construction, so the resources never bear on
identity in either pattern. `data.py`'s builders keep their convenience `resources=` parameter
but implement it locally by wrapping their producer in `remote()` (the existing `_on` helper),
*not* by passing `resources` into the lazy core.

**Verbs become methods on the handle** (the researcher-facing surface):

```python
ckpt = train_lm(...)        # Lazy[LevanterCheckpoint]
ckpt.run()                  # lower + run for side effects
best = ckpt.resolve()       # run, then load the typed artifact -> LevanterCheckpoint
spec = ckpt.lower()         # handle -> StepSpec (rarely needed directly)
run(a, b, c)                # module-level: run several handles together
```

The module-level `lower`/`resolve` free functions are **deleted** (no aliases — no back-compat);
only the methods remain. `run(*handles)` stays a free function (it is variadic over many
handles). Tutorials become `build().run()` / `build().resolve()`, dropping the
`StepRunner().run([lower(build())])` boilerplate everywhere.

**`derived` is removed.** `apply(name, fn, **inputs)` stays as the one generic single-step
helper (the readable common case). The cases `apply` can't express — a fn taking a typed config
object, or inputs built from a dep path like `f"{ctx.path(dep)}/sub"` — are written by
constructing a `Lazy(Recipe(...))` directly, which is what every library builder already does
and reads more clearly than a third helper. Migration example — `raw_download` (data.py:122)
stops delegating to `derived` and constructs the handle itself:

```python
# before:  return derived(name, fn=fn, build_config=build_config, ..., kind=Dataset)
# after:
return Lazy(
    name=name, version=version, result_type=TokenizedCache, override_path=pin,
    recipe=Recipe(fn=_on(fn, resources), build_config=build_config),
)
```

## One `Artifact`, no `JsonArtifact`

`JsonArtifact` earns its keep over the base only by a four-line `load` that reads `record.result`
into typed fields — a second *behavior*, not a second concept. Collapse it. An artifact is a
directory with a record (provenance + an optional JSON payload) and a `load` that reads the dir;
the default is "just the path and its record." The base round-trips a subclass's **own declared
fields** through `record.result` generically, so the data-vs-value distinction dissolves into
"what does the subclass declare":

```python
class Artifact(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    path: str = ""

    @functools.cached_property
    def record(self) -> "ArtifactRecord | None":
        return read_record(self.path)

    def result_payload(self) -> dict | None:
        """What the record stores as `result`: this artifact's declared fields (minus path),
        or None for a pure data ref that declares none. Override to persist something else."""
        return self.model_dump(mode="json", exclude={"path"}) or None

    @classmethod
    def load(cls, path: str) -> Self:
        """Default: a handle into `path`; `.record` carries provenance + the run's config.
        A subclass with value-fields repopulates them from `record.result`."""
        rec = read_record(path)
        return cls(path=path, **((rec.result if rec is not None else None) or {}))
```

- **Data refs declare no fields, only properties.** `LevanterCheckpoint(Artifact)` has a
  `checkpoint_dir` property; `result_payload()` → `None`; consumers read the attribute.
- **Value artifacts declare fields.** `Selection(Artifact)` / `TrainMetrics(Artifact)` declare
  `winner`/`score`/`eval_loss`; the base persists and reloads them with no override.
- **Either can expose record-derived attributes.** `TokenizedCache(Artifact)` declares no value
  fields but exposes `tokenizer` / `as_component()` that read `self.record.config`.

The runner never branches on artifact type (it calls `result_payload()` polymorphically); the
*producer* decides shape by what its subclass declares. This is the mechanism the rest of the
revision leans on: richer subclasses give consumers what they need as attributes, instead of
anyone reaching into a `recipe`.

## Artifacts live with their producers

`artifact.py` keeps only the single `Artifact` base, `ArtifactRecord`, the IO functions, and
`check_drift`. Concrete types move next to the code that writes them and become *rich* — they
own their layout and accessors, so callers stop hard-coding subpaths.

```python
# marin/training/training.py — next to run_levanter_train_lm
class LevanterCheckpoint(Artifact):
    """A Levanter training run's output dir: the rolling checkpoints, config, and metrics."""
    @property
    def checkpoint_dir(self) -> str:
        return f"{self.path}/checkpoints"          # was train.py's _CHECKPOINTS_SUBDIR literal
```

`train_lm` returns `Lazy[LevanterCheckpoint]`. Inside `build_config` only a resolved *path* is
available (`ctx.path(init_from)`), not a loaded artifact, so the resume path is computed by
instantiating the path-wrapper — the subdir knowledge stays on the type, not in the assembler:

```python
init_path = LevanterCheckpoint(path=ctx.path(init_from)).checkpoint_dir   # not init_from.checkpoint_dir
```

### `TokenizedCache` must expose what `mixture` needs (the central-coupling fix)

Codex's biggest-risk point: moving types to producers is cosmetic if consumers still reach into
`dep.recipe`. Today `mixture` reconstructs each dataset's `TokenizeConfig` by re-running
`dataset.recipe.build_config(...)` (data.py:227,238) — which couples the consumer to producer
internals **and breaks for adopted/pinned caches** (their recipe builds no `TokenizeConfig`).
The revision makes this a hard contract, not a follow-up:

- `TokenizedCache` (the producer type for `tokenize`) exposes the metadata `mixture` needs as
  **attributes that read its record** — `TokenizedCache.load(path).tokenizer`, `.format`, and
  `.as_component()` → Levanter `DatasetComponent` — sourced from the record's materialized
  `config` (already written by `tokenize`/`pretokenized`). Dataset `adopt` must record the same
  `config` so adopted/pinned caches are first-class. No new fields, no `dep.recipe`.
- **At run time** `mixture` builds each component from `TokenizedCache.load(ctx.path(handle))` —
  reading the record — uniformly for tokenized, pretokenized, and adopted caches. No
  `dep.recipe` introspection anywhere.
- **At fingerprint time** there is no record. Dataset *identity* still enters the training
  fingerprint through the placeholder cache paths (`ctx.path(handle)` renders to `name@version`)
  keyed by weight: the data contribution is the sorted `{name@version: weight}` map. The
  tokenizer/format are **not** separately in the training fingerprint — they are determined by
  the chosen datasets and verified (one tokenizer across components) at run time. If a dataset's
  tokenizer changes, its content changed, so its `name@version` must change (author bump) — which
  already moves the training fingerprint.
- `mixture` branches on `ctx.is_fingerprint` (a new explicit flag on `RunContext`, set by
  `for_fingerprint`) instead of sniffing placeholder strings: fingerprint → emit the
  `{name@version: weight}` summary with placeholder components; run → assemble the full
  `LmDataConfig` from records.

This is the load-bearing decision in the revision (see Open Questions #1): it changes the
training fingerprint's data contribution from "the reserialized tokenize configs" to "the
dataset `name@version`s + weights," and requires dataset `adopt` to record tokenizer/format.

## Provenance as one object

Replace the four loose `ArtifactRecord` fields and the four free getters with a captured object:

```python
# execution/provenance.py
class Provenance(BaseModel):           # BaseModel: it nests inside the pydantic ArtifactRecord
    git_remote: str | None
    git_commit: str | None
    git_branch: str | None
    user: str | None
    created_at: str          # ISO-8601, the launch time
    command_line: list[str]

    @classmethod
    def capture(cls) -> "Provenance": ...
```

`ArtifactRecord.provenance: Provenance` replaces `command_line`/`git_commit`/`user`/
`created_at`. **Capture lifetime:** `Provenance.capture()` is called **once** at the lowering
entry point (in the launcher, which holds the git checkout) and the same object is threaded
through recursive `lower` into every step's record — so a remote step still records the *launch*
commit/argv, and all steps of one invocation share one launch timestamp. (No back-compat:
nothing has written real records yet.)

## Sweep/select over typed results

Today `select(metric="eval/loss", reader=read_replicated_metrics)` bakes in *how a `train_lm`
trial stores metrics* (`tracker_metrics.jsonl`) — the framework knows too much about the
producer (sweep.py:28). Invert it: the producer emits a **typed result**, and `select` is a
trivial, generic reduction the user parameterizes.

**The producer-side metrics step** (in `training/`, next to `train_lm`):

```python
class TrainMetrics(Artifact):           # value fields -> persisted to record.result by the base
    eval_loss: float
    train_loss: float
    summary: dict[str, float]          # the full final summary, for ad-hoc keys

def training_metrics(ckpt: Lazy[LevanterCheckpoint], *, version: str) -> Lazy[TrainMetrics]:
    """Read a finished run's final metrics into a typed value artifact.

    Runs INLINE (it returns a value), depends only on `ckpt` (so it runs after training), and
    reads `<ckpt path>/tracker_metrics.jsonl` from storage. Raises FileNotFoundError if the run
    wrote no metrics. Owns the 'where train_lm puts metrics' knowledge, co-located with train_lm.
    """
```

**`select` reduces typed results with a user key:**

```python
def select(name, version, results: Sequence[Lazy[T]], *,
           key: Callable[[T], float], mode: Literal["min", "max"]) -> Lazy[Selection]: ...

best = select("sweeps/lr", "20260628",
              [training_metrics(t, version="20260628") for t in sweep(trial, learning_rate=[...])],
              key=lambda m: m.eval_loss, mode="min")
```

Defined behaviors (codex P2): **ties** → first trial in input order wins (strict `<`/`>`
comparison keeps the earlier one); **missing key** → the user's `key` raising `KeyError`/
`AttributeError` propagates with the trial id in context; **NaN** → `select` raises
`ValueError` (a NaN score is never silently "best" or "worst"). **Identity:** the selection's
identity is its `name@version` + the trial `name@version`s (deps) + `mode`. The `key` callable
is **not** fingerprinted (callables have no stable identity — same stance as the old `reader`);
changing the ranking logic requires a version bump. This is documented on `select`; an alternate
design (a fingerprintable `key_name`/policy enum) is noted in Open Questions #2.

## Versions: calendar, not `v1`

> "'v1' isn't a version, artifacts are calversioned" (train.py:45)

Remove every `DEFAULT_VERSION = "v1"` and the `version="v1"` defaults. **`version` is required**
and validated by `Lazy.__post_init__` to one of:

- a calendar version `YYYYMMDD` or `YYYYMMDD.N` (the `.N` disambiguates two immutable revisions
  on the same day — codex P2),
- a semantic version `MAJOR.MINOR.PATCH` (the review's "or at least major.minor.patch"),
- the mutable forms `dev` / `<label>-dev` (always rebuild, drift check skipped).

`v1`-style strings are rejected. Validation is by shape (8 digits `[.N]`, or `N.N.N`, or the
`dev` forms); the date is the authoring date (local; no timezone ceremony). Tutorials/examples
pass real values (`version="20260628"`).

## Smaller cleanups

- **`EvalSuite` slop** (evals.py:24): drop the hand-written `__init__` doing `object.__setattr__`;
  a plain `@dataclass(frozen=True)` with `tasks: tuple[...]` (coerced once in `__post_init__` if
  a sequence is allowed).
- **`_diff_json`** → move to `fingerprint.py` as `describe_drift(old_payload, new_payload)`
  (pure); `check_drift` stays in `artifact.py` (it does record IO + logging) and calls it. Note:
  `check_drift` keeps its `artifact → step_spec` dependency; if we want to drop that edge,
  `check_drift` can take `(output_path, fingerprint, version, payload, expected)` instead of a
  whole `StepSpec` (optional, P3).
- **`executor_step_status.py`** → `step_status.py`; on-disk markers `.executor_status[.lock]` →
  `.step_status[.lock]` (no real outputs to migrate).
- **`Lazy` annotations** (data.py:227/238): `_component_for`/`_tokenizer_of` (folded into
  `TokenizedCache.as_component`) take `Lazy[TokenizedCache]`, not bare `Lazy`.
- **`adopt`**: default `kind=Artifact` (the base), since `lazy.py` no longer imports `Dataset`.

## Review-comment coverage

| Comment / codex finding | Resolution |
|---|---|
| train.py:38 — `LevanterCheckpoint` with producer | §"Artifacts live with producers" |
| lazy.py:48 — Lazy shouldn't know Datasets | §"lazy core" — imports framework only, no concrete types |
| lazy.py:497 — `resources` doesn't belong | §"lazy core" — removed; via `remote()`/`run_arg` |
| lazy.py:460 / data.py:122 — `derived` weird/unneeded | §"lazy core" — removed; `apply` + direct `Lazy` (raw_download before/after) |
| lazy.py:495 / train.py:45 — `v1` not a version | §"Versions" |
| lazy.py:258 — free fn vs method | §"lazy core" — methods; free `lower`/`resolve` deleted |
| provenance.py:21 — Provenance dataclass | §"Provenance as one object" (BaseModel, capture-once) |
| sweep.py:28 — embeds sweep knowledge | §"Sweep/select over typed results" |
| artifact.py:220 — `_diff_json` in fingerprint? | §"Smaller cleanups" |
| executor_step_status.py:5 — rename | §"Smaller cleanups" |
| data.py:227/238 — Lazy subtype | §"Smaller cleanups" + `TokenizedCache.as_component` |
| evals.py:24 — slop | §"Smaller cleanups" |
| codex P1 — lazy "purity" wording | §"lazy core" — framework-vs-concrete clarified |
| codex P1 — `checkpoint_dir` in build_config | §"Artifacts live with producers" — `LevanterCheckpoint(path=…).checkpoint_dir` |
| codex P1 / biggest risk — mixture `dep.recipe` leak | §"TokenizedCache must expose what mixture needs" |
| codex P2 — `result_payload()` None ambiguity | §"lazy core" — value-vs-data contract + `MissingResultError` |
| codex P2 — select key/ties/NaN/missing | §"Sweep/select" — defined behaviors |
| codex P2 — version too rigid | §"Versions" — `YYYYMMDD[.N]` + semver |
| user — why `Artifact` **and** `JsonArtifact`? | §"One `Artifact`, no `JsonArtifact`" — collapsed to one base; rich subclasses expose attributes |
| npm_registry_metadata.py:40 — file issue | external write blocked by sandbox; needs the user to file or grant permission |

## Open questions (for review)

1. **Mixture / training-fingerprint contract** (the load-bearing one). The revision makes the
   training fingerprint's data contribution `{dataset name@version: weight}` and reads
   tokenizer/format from `TokenizedCache` records at run time (fixing adopted caches and the
   `dep.recipe` leak). The alternative keeps a build-time `CacheMetadata` (tokenizer/format) on
   the handle so the tokenizer stays *in* the fingerprint. Recommendation: the record-based
   contract — confirm before implementing.
2. **`select` key identity.** Accept that `key` is not fingerprintable (consistent with the old
   `reader`; bump version to re-rank), or require a fingerprintable `key_name`/policy enum so a
   changed ranking forks identity automatically? Recommendation: not fingerprinted + document.
3. **`TokenizedCache` name** (replacing `Dataset`) — `LmDataset` / `LevanterCache` are
   alternatives. Bikeshed; pick one.
4. **Version format** — calver `YYYYMMDD[.N]` + semver covers the review's ask; is semver worth
   keeping, or calver-only for one convention?
