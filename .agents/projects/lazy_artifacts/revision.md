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
   means the format is described in two places. The rule: an artifact type lives with the
   function that writes its bytes, applied everywhere.

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
| `training/training.py` | producer fn only | + `LevanterCheckpoint(Artifact)` with a `training_metrics()` method (+ `TrainMetrics` value model) |
| `processing/tokenize/tokenize.py` | producer fn only | + `TokenizedCache(Artifact)` (was `Dataset`), exposing the metadata `mixture` needs |
| `experiment/sweep.py` | `grid`/`sweep`/`select`/`Selection`/`read_*` | collapses to an optional `grid(**axes)`; `select`/`Selection`/readers **deleted** — selection is user code |

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
- **Value artifacts declare fields.** A user's `EvalResult(Artifact)` with `accuracy: float`
  (or a promoted standalone `TrainMetrics` — see the deferred item) persists and reloads its
  fields with no override; the base round-trips them through `record.result`.
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

This changes the training fingerprint's data contribution from "the reserialized tokenize
configs" to "the dataset `name@version`s + weights" (the dataset version encodes its tokenizer,
so identity is covered — Decision 1), and requires dataset `adopt` to record its `config`.

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

## Sweep/select: a semantic checkpoint, user-written reduction

Today `select(metric="eval/loss", reader=read_replicated_metrics)` bakes *how a `train_lm` trial
stores metrics* (`tracker_metrics.jsonl`) into the framework (sweep.py:28), and ships a
`grid`/`sweep`/`select`/`Selection`/`read_*` stack. Most of it isn't pulling its weight: fan-out
is a list comprehension, and selection is one-liner user code over the concrete outputs.

**Make the checkpoint semantic instead** (the right approach for today). `LevanterCheckpoint`
carries a method that reads its own final metrics — the artifact is the place to attach whatever
semantics the output has:

```python
class TrainMetrics(BaseModel):                 # a plain value read on demand — not a build step
    eval_loss: float
    train_loss: float
    summary: dict[str, float]                  # full final summary, for ad-hoc keys

class LevanterCheckpoint(Artifact):
    @property
    def checkpoint_dir(self) -> str: ...
    def training_metrics(self) -> TrainMetrics:
        """This run's final metrics, read from <path>/tracker_metrics.jsonl.
        Raises FileNotFoundError if the run wrote none."""
```

**A sweep is then fan-out + an ordinary reduction over the resolved, typed outputs** — no
`select`, no `Selection`, no metric-string or reader in the framework:

```python
trials = [trial(learning_rate=lr) for lr in (3e-4, 6e-4, 1e-3)]   # list[Lazy[LevanterCheckpoint]]
run(*trials)
scored = [(t, t.resolve().training_metrics().eval_loss) for t in trials]
best, best_loss = min(scored, key=lambda s: s[1])
# persist the outcome if you want it addressable later, with the low-level API:
write_artifact({"winner": best.name, "eval_loss": best_loss}, "sweeps/lr/20260628")
```

`sweep.py` collapses to an optional `grid(**axes)` axis-expansion helper; `select`, `Selection`,
`read_replicated_metrics`, and `read_eval_records` are **deleted**. Selection logic is the
user's, written over typed outputs and persisted (when wanted) with the low-level
`write_artifact`.

**Deferred — multiple artifacts per step.** `TrainMetrics` *could* instead be its own
addressable `Lazy[TrainMetrics]` (useful when a result is reused across experiments, or produced
by an expensive standalone eval). That raises "does one step write several artifacts?" — a real
question (a `train_lm` emitting both a checkpoint and a metrics artifact) we are **not** resolving
now. The semantic-method approach needs no multi-artifact step, so it ships first; the
own-artifact form is added later when something needs it.

## Versions: calendar, not `v1`

> "'v1' isn't a version, artifacts are calversioned" (train.py:45)

Remove every `DEFAULT_VERSION = "v1"` and the `version="v1"` defaults. **`version` is required**
and validated by `Lazy.__post_init__` to one of:

- a calendar version `YYYYMMDD` (optionally `YYYYMMDD.N` to disambiguate two immutable revisions
  on the same day),
- the mutable forms `dev` / `<label>-dev` (always rebuild, drift check skipped).

**Calver only to start** — semver is deliberately not accepted yet, to keep one convention
(Decision 3). `v1`/`llama3`-style strings are rejected. Validation is by shape (8 digits `[.N]`,
or the `dev` forms); the date is the authoring date (local). The ~8 existing `v1`/`llama3` call
sites and the tutorials migrate to real dates (`version="20260628"`).

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
| sweep.py:28 — embeds sweep knowledge | §"Sweep/select" — `select`/`Selection`/readers deleted; selection is user code over outputs |
| artifact.py:220 — `_diff_json` in fingerprint? | §"Smaller cleanups" |
| executor_step_status.py:5 — rename | §"Smaller cleanups" |
| data.py:227/238 — Lazy subtype | §"Smaller cleanups" + `TokenizedCache.as_component` |
| evals.py:24 — slop | §"Smaller cleanups" |
| codex P1 — lazy "purity" wording | §"lazy core" — framework-vs-concrete clarified |
| codex P1 — `checkpoint_dir` in build_config | §"Artifacts live with producers" — `LevanterCheckpoint(path=…).checkpoint_dir` |
| codex P1 / biggest risk — mixture `dep.recipe` leak | §"TokenizedCache must expose what mixture needs" |
| codex P2 — `result_payload()` None ambiguity | §"lazy core" — uniform payload; a value subclass missing its `result` fails with a clear pydantic error |
| codex P2 — select key/ties/NaN/missing | moot — `select` removed; reduction is user code |
| codex P2 — version too rigid | §"Versions" — `YYYYMMDD[.N]` (calver only) |
| user — why `Artifact` **and** `JsonArtifact`? | §"One `Artifact`, no `JsonArtifact`" — collapsed to one base; rich subclasses expose attributes |
| user — `training_metrics` as a method | §"Sweep/select" — `LevanterCheckpoint.training_metrics()`; own-artifact form deferred |
| npm_registry_metadata.py:40 — file issue | external write blocked by sandbox; needs the user to file or grant permission |

## Decisions (settled in review)

1. **Mixture / training fingerprint** = sorted `{dataset name@version: weight}`. The dataset
   version encodes its tokenizer, so identity is covered; tokenizer/format are read from
   `TokenizedCache` records at run time (fixing adopted caches and the `dep.recipe` leak).
2. **No `select`.** Selection is ordinary user code over resolved, typed outputs
   (`ckpt.resolve().training_metrics()`), persisted when wanted with low-level `write_artifact`.
   `select`/`Selection`/`read_replicated_metrics`/`read_eval_records` are deleted; `key` was never
   fingerprintable anyway.
3. **Versions: calver only** — `YYYYMMDD` (`.N` for same-day), `dev`/`-dev`; no semver yet.
4. **`TokenizedCache`** name kept.
5. **One `Artifact` base** (no `JsonArtifact`); richer subclasses expose what consumers need as
   attributes/methods (`LevanterCheckpoint.training_metrics()`, `TokenizedCache.as_component()`).

### Deferred

- **Multiple artifacts per step.** Letting one step emit several independently-addressable
  artifacts (e.g. a checkpoint *and* a standalone `Lazy[TrainMetrics]`). The semantic-method
  approach covers today's need without it; revisit when an output must be addressed on its own.

## Codex review #2 resolutions (implementation spec)

The settled answers to codex's implementation-readiness gaps; these are the contracts to build.

- **`RunContext.is_fingerprint: bool`** — a real field. `for_run(...)` sets `False`,
  `for_fingerprint(...)` sets `True`. `mixture` branches on it.
- **`Artifact.result_payload()`** uses **declared fields** — `type(self).model_fields` minus
  `path` — not `model_dump()` (which would sweep in `extra="allow"` extras). Data refs declare
  none → `None`. Schema drift within a reused version is the author's call (bump the version).
- **`mixture`, run time** — `TokenizedCache.as_component()` builds, uniformly for tokenized / HF /
  pretokenized / adopted caches (a *built* cache is just a dir + format; no per-type
  discrimination):
  ```python
  dataset_component(UrlDatasetSourceConfig(
      cache_dir=self.cache_dir, format=self.format, train_urls=[], validation_urls=[], tags=self.tags))
  ```
  with `self.cache_dir = self.record.source or self.path` (adopted → its source dir),
  `tokenizer = self.record.config["tokenizer"]`, and `self.format` draccus-decoded from
  `self.record.config["format"]` (default `TextLmDatasetFormat()`). These come from
  `record.config`, which the launcher writes — `build_config` runs launcher-side even when
  `recipe.fn` dispatches remote — so they exist for every produced cache.
- **`mixture`, fingerprint time** (`ctx.is_fingerprint`) — return an `LmDataConfig` whose
  components are keyed by `handle.name`, each a `DatasetComponent` with
  `cache_dir=ctx.path(handle)` (renders to `name@version`), a placeholder
  `format=TextLmDatasetFormat()` and empty urls; `train_weights = {name: weight}` for train and
  `{name: 0.0}` for validation; `tokenizer="<tokenizer>"` placeholder. Identity is thus the sorted
  `name@version` + weights, nothing from records.
- **Adopted tokenized caches** — `data.py` gains `adopt_tokenized_cache(name, version, source, *,
  tokenizer, format=TextLmDatasetFormat())`, which adopts and records a synthetic
  `config={"tokenizer", "format"}` so `as_component()` reads it the same way. (Generic `adopt`
  stays metadata-free for non-dataset dirs.)
- **`ArtifactRecord.provenance: Provenance | None = None`** — optional, so low-level
  `write_artifact(...)` writes a minimal record; the lazy runner fills it via a private
  `_lower(handle, provenance)` that both `Lazy.lower()` and `run(*handles)` call after one
  `Provenance.capture()` per entry (single launch timestamp).
- **`LevanterCheckpoint.training_metrics()`** parses `<path>/tracker_metrics.jsonl`'s `summary`
  (levanter keys: `train/loss`, `eval/loss`, `eval/<name>/loss`): returns
  `TrainMetrics(summary=<full dict>, train_loss=summary.get("train/loss"),
  eval_loss=summary.get("eval/loss"))` with `eval_loss` optional; `summary` is the escape hatch
  for per-task keys. `FileNotFoundError` if the file is absent.
- **`raw_download` → `Lazy[Artifact]`** (raw shards are not a `TokenizedCache` and must not feed
  `mixture`); `tokenized(raw=...)` accepts `Lazy[Artifact]`.
- **`TokenizedCache` stays self-contained** — it does not import `data_configs`,
  `download_pretokenized`, or `experiment.data` (would cycle); it constructs
  `UrlDatasetSourceConfig` directly.

## Migration plan (call-site sweep)

Implement in codex's order (core → lazy → TokenizedCache → adopt → mixture/train_lm →
training_metrics → deletions+migration). The mechanical call-site sweep (sonnet, disjoint
directories) follows the stable core, per the inventory:

| Change | Rule | Scale |
|---|---|---|
| `Dataset` (from artifact) | → `TokenizedCache` from `marin.processing.tokenize.tokenize`; `Lazy[Dataset]`→`Lazy[TokenizedCache]` | ~30 files |
| `Checkpoint` (from artifact) | → `LevanterCheckpoint` from `marin.training.training`; `Lazy[Checkpoint]`→`Lazy[LevanterCheckpoint]` | ~19 files |
| `JsonArtifact` | → `Artifact` (tests' `Tokens`/`Ckpt`/`Toy` subclass `Artifact`) | tests + sweep |
| `derived(name, fn=F, build_config=BC, deps=D, kind=K, version=V, pin=P)` | → `Lazy(name, version=V, result_type=K, override_path=P, recipe=Recipe(fn=F, build_config=BC, deps=tuple(D)))` (wrap `F` in `remote(...)` only if it passed `resources=`) | ~18 files |
| free `lower(x)` / `resolve(x)` | → `x.lower()` / `x.resolve()`; `StepRunner().run([lower(x)])` → `x.run()` | ~16 files |
| `version="v1"`/`"llama3"`, `DEFAULT_VERSION` | → calver `"20260628"`; remove the `DEFAULT_VERSION` constants and builder defaults | ~all builders |
| `select`/`Selection`/`read_replicated_metrics` | deleted — tutorial + `test_sweep_select.py` rewrite to `run(*trials)` + `min(...)` over `t.resolve().training_metrics()` | tutorial + 1 test |
| `marin.execution.executor_step_status` | → `marin.execution.step_status` | ~6 files |
