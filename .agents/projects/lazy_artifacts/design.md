# Lazy Artifacts

_Why are we doing this? What's the benefit?_

Marin experiments build their step DAG at **import time**: module-level `ExecutorStep(...)`
definitions, `versioned()` fields, `this_output_path()`, and `InputName`/`.cd()`
cross-references. The `Executor` then content-addresses every step as `{name}-{md5}`, so any
code change silently forks a new output path, the build environment's region/prefix is baked
into the graph, and the resulting paths are unreadable. This design replaces that with **lazy
typed artifacts** addressed by an explicit `name@version`, computed on demand. Authoring an
experiment stops being coupled to a heavyweight engine; paths become human-readable and stable;
and the choice of when an artifact is "new" moves from an automatic hash to the author's hand,
with a drift *warning* as the guardrail rather than a content hash as the law.

## Challenges

The hard part is **separating identity from execution** without a content hash. The Executor
fused them: the md5 of the whole config tree *was* the identity, which is why region, prefix,
and the specific TPU all leaked into the address. We need a model where "what an artifact is"
(model, hyperparameters, dependency identities) is distinguished from "where/how it runs"
(output path, prefix, region, accelerator), and where that line is drawn by construction. Second,
two modules both spelled their central type `Artifact` — the lazy *handle* and the output
*serializer* — which is genuinely confusing to read. Third, the authoring surface leaked: the
generic "build any artifact" helper lived in the *data* module, every config change tripped a
hard error, and a sweep needed an `AnnotatedCheckpoint` wrapper. The redesign below addresses all
three: one name for the handle (`Lazy[T]`), one name for the output (`Artifact`), a small generic
helper set in the core, and a drift warning in place of the hard guard.

## Design

### The handle and the artifact — one name each

A **lazy handle** is an inert `Lazy[T](name, version, recipe)`; constructing one runs nothing.
`T` is the *type the recipe produces*, and it is bounded: every `T` is an `Artifact`. An
**`Artifact`** is the produced, persisted thing — it carries its `path` and its `record`
(provenance), and it knows how to reload itself:

```python
class Artifact:
    path: str
    record: ArtifactRecord          # config + fingerprint + git + argv + deps, set on load
    @classmethod
    def load(cls, source: str) -> Self: ...
```

This is the whole resolution of the two-`Artifact` confusion: `Lazy` is the **recipe/promise**,
`Artifact` is the **realized output**. There is no longer a handle class and a serializer class
sharing a name. `T.load` pairs with the recipe's producing function — the function writes bytes to
`ctx.out` in its format, and `load` reads that location back — so each artifact type owns both
halves of its own format. Two base flavors keep `load` from being boilerplate:

- **Data refs** — `Dataset`, `Checkpoint`. `load(source)` returns a lightweight handle into the
  cache/checkpoint directory (`cls(path=source)`), never pulling weights into the launcher.
- **Computed values** — a `JsonArtifact` base (pydantic). `load(source)` reads the artifact's JSON
  payload and `model_validate`s it. A sweep's selection is a typed `Selection(JsonArtifact)`.

Authors almost never write `load`; they pick a base. The realized `Artifact` (with `.path` and
`.record`) is what makes outputs **complete and inspectable** — `resolve(x).record` is the entry
point to "what config, which commit, which command line produced this."

### Identity vs. execution

`build_config(ctx)` is a pure function of a `RunContext`, and that context draws the line.
Values written as **literals** (model, hyperparameters, dependency *versions*) bear identity and
enter the fingerprint. Values **pulled from `ctx`** never do: `ctx.out`, `ctx.path(dep)`,
`ctx.prefix`, `ctx.region`, and `ctx.run_arg(key)` (e.g. the TPU a dispatched job runs on). The
fingerprint is computed by re-running `build_config` under `RunContext.for_fingerprint()`, which
substitutes placeholders for everything pulled — so a prefix move, a region move, or a different
accelerator never re-fingerprints. Compute likewise rides on the fn via `remote(fn, resources=…)`,
never on the graph node.

### Authoring: two tiers, and the `apply` story

The core distinguishes **library authors** from **researchers**:

- **Library** primitives — `Recipe(fn, build_config, deps, run_args)` and the typed builders
  (`tokenized`, `train_lm`, `mixture`). Full `RunContext` control; this is where the marin-on-TPU
  plumbing lives.
- **Researcher** sugar — two generic single-step builders, both **in the lazy core** (they are not
  about data, so they no longer live in the data module), picked by the step function's shape:

  - `apply` — the *direct-call* form, when the fn takes keyword inputs. It removes the
    `build_config` lambda and dict-wrapper entirely:

    ```python
    staged = apply("raw/massive", stage_massive_raw, output_path=OUT)
    parts  = apply("data/massive", transform_staged_massive, input_path=staged, output_path=OUT)
    ```

    A bare `Lazy` argument becomes a dependency and resolves to its path at run time; the `OUT`
    sentinel resolves to `ctx.out`; every other value is a literal that bears identity. `apply`
    calls the underlying function directly — *name an output, say which function makes it, pass its
    inputs.*

  - `derived` — the *config-object* form, `fn(build_config(ctx))`, for the (common) case `apply`
    can't express: a fn that takes a typed config dataclass, or inputs composed from a dep path
    (`f"{ctx.path(dep)}/sub"`). It is the named one-liner over the `Lazy`+`Recipe` construction, and
    the workhorse the dataset catalogs (transforms, conversions, filters) actually use. Keeping it —
    rather than forcing every such site to spell out `Lazy`+`Recipe` inline — is the point: *a basic
    set of helpers to make common actions simple.*

The set is completed by `run(*handles)` (execute for side effects — the everyday entry point) and
`resolve(handle) -> T` (run, then return the realized, typed `Artifact` via `T.load`). For a value
artifact, `resolve(select(...)) -> Selection` hands back the winner in-process; for a data
artifact it hands back a path-bearing ref.

### The complete artifact record

A successful step writes **one** record next to its output — an `ArtifactRecord` (pydantic),
subsuming the old payload sidecar and the registry record into a single, complete descriptor:

```
name, version, fingerprint, output_path, deps[name@version], source?,
config (materialized, real values), command_line (argv), git_commit, user, created_at, result?
```

`config` is the human-readable materialized config (what actually ran); `fingerprint` is the
identity hash (placeholders for pulled values); `result` holds the returned value for a value
artifact. This is what "tagged with the artifact for someone to look at later" means concretely.

**One serialization scheme, two entry points.** This is deliberately *not* a second system bolted
beside datakit's existing `Artifact.from_path`/`save`. There is one `ArtifactRecord` and one pair of
record IO functions; the difference is only who fills the record:

- **Manual** (datakit StepSpec pipelines) — a step calls `write_artifact(value, path)` and a consumer
  `read_artifact(path, T)`. The record is minimal (just the `result`); the author manages paths.
- **Automatic** (the lazy runner) — on success the runner writes the *full* record (config, git,
  argv, deps, fingerprint, result_type, result) and `resolve`/`Artifact.load` read it.

The automatic path is literally the manual one plus provenance, so every new field is optional and an
old or minimal record still loads (missing fields read as `None`). The old payload-IO `Artifact`
class is renamed to this `ArtifactRecord` scheme; its ~60+ datakit call sites migrate in this PR
(`Artifact.from_path` → `read_artifact`, `Artifact.save` → `write_artifact`).

### Versioning is by convention; drift is advisory

Identity is the explicit `{prefix}/{name}/{version}`, and the address is **first-build-wins**. The
fingerprint is recorded, but it is a **guardrail, not a gate**. If a fixed `name@version` already
has a record and the current recipe fingerprints differently, the runner **logs a warning with a
field-level diff** (`learning_rate: 3e-3 -> 4e-3`) and serves the existing output — it never
raises. The contract is the one you asked for: *trust the author to bump the version (or use
`dev`) when they mean to produce something new; reuse a version and you get what's already there,
loudly noted.* `ImmutableArtifactError` leaves the default path entirely. Two escape hatches
remain: `dev`/`-dev` versions always rebuild (iteration), and `expected_fingerprint` is an
**opt-in** hard pin that raises `FingerprintMismatchError` at `lower()` for the few artifacts that
genuinely want a mandate.

### Fingerprint: best-effort and extensible

Because the fingerprint is now advisory, the encoder no longer needs to be perfectly total to be
*correct* — a misfire is a noisy warning, not a blocked build. So an unknown type degrades to a
**defined** stable fallback — `{"__repr__": f"{type.__module__}.{type.__qualname__}", "vars":
sorted(vars(o))}` when it has a `__dict__`, else its `repr()` — with a one-time log, instead of
raising. `register_fingerprint(type, fn)` lets a project teach the encoder a precise canonical form
for an identity-bearing custom type, and an opt-in **strict mode** (env/flag, default off) restores
the old raise-on-unknown for CI that wants drift to be exact. This retires workarounds like grug's —
reconstituting `PartitionSpec`-bearing configs at run time purely so the strict encoder wouldn't
reject them.

Drift is advisory, but **type identity is not**: the produced `result_type` is recorded, and
`resolve` hard-errors (`ArtifactTypeMismatchError`) if a served artifact's recorded type differs
from what the handle now declares. So a value artifact whose schema changed under a reused version
fails loudly at load instead of silently mis-validating — the one place where "serve the old output"
would be unsafe is closed without re-introducing the hard config guard.

### Migration path

`lower()` is a pure transform with no Iris/Fray awareness, so catalogs migrate independently: a
module that exposed module-level `ExecutorStep`s becomes a *function* returning `Lazy` handles
(mechanism stays in the library, policy moves to the experiment). A catalog's `derived(...)` calls
keep their shape — only the import moves (`marin.experiment.data` → `marin.execution.lazy`) and
`Dataset`/`Checkpoint` annotations become `Lazy[Dataset]`/`Lazy[Checkpoint]`; new simple steps use
`apply`'s direct-call form; the tutorial tree collapses to fewer, parameterized scripts
(`--device`, `--dataset`). The `Executor` content-addressing layer is already deleted on this
branch; this redesign is the cleanup of the surface that replaced it.

## Testing

The anti-drift gate is **materialized-config equality** (`materialized_config(handle, prefix)`):
golden tests assert the readable inline code produces the intended training decisions. The
registry, fingerprint, adopt, sweep, and phase-chain tests drive the **real** `StepRunner` against
a `tmp_path` prefix (no mocks). The behavior changes here add: drift **warns** (assert the log and
that the cached output is served, *not* a raise); `expected_fingerprint` still raises;
`resolve` round-trips a value artifact through `T.load`; and a `register_fingerprint` converter
makes a custom type fingerprint stably.

## Costs / Risks

- **A reused version silently serves stale output.** This is the deliberate cost of advisory drift:
  change a recipe but keep the version and you get the old artifact, mitigated only by a warning. The
  trade is intentional — trust plus a guardrail, not a mandate. `expected_fingerprint` is there for
  anyone who wants the old strictness on a specific artifact.
- **`recipe.fn` source is not fingerprinted.** Editing logic inside a step fn without touching its
  config serves cached output. Bump the version (or use `dev`) when behavior the config doesn't
  capture changes. (Advisory drift makes this lower-stakes than under the hard guard.)
- **Every `T` needs a `load`.** Mitigated by the `Dataset`/`Checkpoint`/`JsonArtifact` bases; a
  genuinely novel format is the only place an author writes one.
- **Pins are an unguarded escape hatch.** `override_path` resolves to existing data and writes no
  record; `adopt()` guarantees *pointer* immutability, not *content* immutability.

## Open Questions

- **Should drift be surfaced beyond a log?** A warning is easy to miss in a long run. Worth
  recording "served despite drift" in the record, or emitting a run-summary of drifted artifacts?
- **`resolve` for very large data artifacts.** Settled as a lightweight path-bearing ref (no
  in-process load); confirm, and whether `Checkpoint` should grow a lazy `.load_model(...)` later.
- **The `Dataset = Lazy[ConcreteDataset]` alias — now or later?** Signatures read `Lazy[Dataset]`
  today; the alias would let them read `Dataset` again while meaning the handle. Pure convenience,
  deferrable.
