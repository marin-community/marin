# Artifact.from_id + ArtifactRegistry

_Why are we doing this? What's the benefit?_

Today an artifact's identity is its GCS path. `Artifact.from_path(...)` resolves the metadata sidecar under that path, which is fine until the path needs to change — and we can't easily restructure GCS. Downstream code can't refer to a dataset or model checkpoint by a stable name; it has to know the exact computed path that produced it.

We want a path-independent way to refer to artifacts: `Artifact.from_id("datasets/fineweb-resiliparse", "v3", NormalizedData)` should resolve to the same payload regardless of where the bytes live, today or after a re-org. This requires a small registry that maps `(id, version) → path + metadata`, with a default backed by GCS / local filesystem and a user-supplied override hook.

## Background

Current state, prior art, and pitfalls survey: see [`research.md`](./research.md). Short version:

- `Artifact` is a JSON metadata envelope (`artifact.json` sidecar) over Pydantic payloads like `PathMetadata`, `TokenizedAttrData`. `from_path(base, type)` requires `type` to subclass `BaseModel`. ([`lib/marin/src/marin/execution/artifact.py:32`](https://github.com/marin-community/marin/blob/f07122bbd20376b97e8951ac75f0eb8a7ebee7e3/lib/marin/src/marin/execution/artifact.py#L32))
- Marin already has *content-addressed* path hashes from the executor ([`step_spec.py:90`](https://github.com/marin-community/marin/blob/f07122bbd20376b97e8951ac75f0eb8a7ebee7e3/lib/marin/src/marin/execution/step_spec.py#L90)) and *schema* versions inside payloads. This proposal adds a third axis: a **logical name + user-supplied version** that is independent of both.
- No existing registry in `marin/execution/`. Prior art (MLflow, W&B, DVC, HF Hub) converges on `(name, version) → metadata + pointer`, with mutable aliases layered over immutable versions. A manifest-file-on-storage registry is firmly in the DVC / HF-as-git camp.

## Challenges

_What's hard?_

The core mechanics are easy; the contentious bits are policy choices.

1. **Concurrent writes on object storage.** GCS has no native locking. We avoid this at v1 with a layout where concurrent registrations of distinct `(id, version)` pairs never touch the same key; same-pair races are last-writer-wins. CAS (`if-generation-match`) is a clean follow-up.
2. **Version flavor is a one-way door.** Once callers pin `"v3"`, changing the version model (content-hash, monotonic-int) costs a migration. We pick *user-supplied string* and accept that immutability discipline now lives with the caller.

## Costs / Risks

- New surface area in `lib/marin/src/marin/execution/` that downstream code will start depending on.
- Two ways to load the same artifact — `from_path` (legacy) and `from_id` (new). `from_id` delegates to `from_path` so the typed-loader behavior stays in one place; the duplication is API-only.
- Users must *register* artifacts before `from_id` works. Existing pipelines keep working unchanged; only consumers who want id-based lookup pay the one-line `registry.register(...)` cost.

## Design

_How are we doing this?_

Two new pieces in `lib/marin/src/marin/execution/`:

**`ArtifactRegistry`** — an interface plus a default `FilesystemArtifactRegistry` backed by GCS or local FS via `rigging.filesystem` / `url_to_fs` (the abstractions `artifact.py` already uses, [artifact.py:9](https://github.com/marin-community/marin/blob/f07122bbd20376b97e8951ac75f0eb8a7ebee7e3/lib/marin/src/marin/execution/artifact.py#L9)). Two methods:

- `register(id, version, uri) -> ArtifactEntry` — fails loudly if `(id, version)` already exists. No overwrite, no upsert at v1.
- `lookup(id, version) -> ArtifactEntry` — returns the stored entry or raises `ArtifactNotFoundError`.

`ArtifactEntry` is a minimal Pydantic record: `id`, `version`, `uri`. It serves both as the persisted file schema and as the return type of `lookup` / `register`. Extra fields (`created_at`, `metadata`, `artifact_type`) are deferred until there's a consumer for them — easier to add later than to deprecate.

**Manifest layout** is one file per `(id, version)`:

```
<registry-root>/<namespace>/<name>/<version>.json
```

Example: `gs://marin-us-central2/artifact_registry/datasets/fineweb-resiliparse/v3.json`. The hot path is id+version → known key (O(1) GET); directory size doesn't matter for reads. Concurrent writers of distinct `(id, version)` pairs never contend; same-pair races are last-writer-wins (no CAS at v1).

**Immutability is registry-level only.** A registered `(id, version)` always points at the same `uri`, but the bytes at that `uri` are protected by GCS IAM, not by this design. Anyone with write access can delete or repoint the underlying artifact; the open content-hash field would let `from_id` detect this.

**Decisions / considered alternatives.**

- *Sharded prefix* (`<name[-2:]>/<name>/<version[-2:]>/<version>.json`) — rejected: reads go straight to a known key, and "list all versions" is a debugging operation. Adding sharding later is a one-time migration.
- *Single global `registry.json`* — rejected (worst concurrency, worst diff-ability).
- *SQL-backed service* — out of scope; the manifest-file shape parallels DVC and HF-as-git, the right reference class for a library component.
- *Content-addressable storage (content-hash as identity)* — **considered and rejected.** Marin already content-addresses one layer down ([`step_spec.py:90`](https://github.com/marin-community/marin/blob/f07122bbd20376b97e8951ac75f0eb8a7ebee7e3/lib/marin/src/marin/execution/step_spec.py#L90)), so the registry hash would mostly duplicate that. More importantly, humans say "give me `fineweb-resiliparse` v3", not "`sha256:e3b0c44…`" — content-hash naming reintroduces the path-coupling problem we're trying to leave. Callers pick the version string; `register` fails on collision. A content-hash field for verification (not identity) remains an Open Question.
- *Mutable aliases (`latest`, `prod`, …)* — **intentionally not supported, non-goal not deferral.** The registry is `(id, version) → uri` and nothing else. Callers needing a "latest" pointer maintain it themselves. Rationale: aliases drift implicitly on every registration, breaking reproducibility of old run logs without anyone touching them — exactly the failure mode the registry exists to prevent.

**`Artifact.from_id`** delegates to existing machinery after resolving:

```python
@classmethod
def from_id(
    cls,
    id: str,                          # "<namespace>/<name>", e.g. "datasets/fineweb-resiliparse"
    version: str,                      # user-supplied, immutable, e.g. "v3"
    artifact_type: type[T] | None = None,  # Pydantic schema, same role as in from_path
    *,
    registry: ArtifactRegistry | None = None,
) -> T | PathMetadata | dict[str, Any]:
    reg = registry or get_default_registry()
    entry = reg.lookup(id, version)
    return cls.from_path(entry.uri, artifact_type)
```

`get_default_registry()` returns a module-level singleton built from `MARIN_ARTIFACT_REGISTRY` (default `{marin_prefix()}/artifact_registry`). Tests and notebook users pass `registry=` to override.

**IDs** are strict `<namespace>/<name>` (one slash, both segments non-empty, restricted character set). `register` and `lookup` validate this up-front; ambiguous inputs raise immediately.

**Out of scope** at v1: explicit overwrite / upsert, content hashing in entries, multi-writer transactional semantics, the auto-registration shortcut from step outputs, deletion / garbage collection. Each is a clean follow-up; see [`spec.md`](./spec.md) for the explicit out-of-scope list. Mutable aliases (`latest`, `prod`, …) are a non-goal, not a deferral — see Decisions.

## Testing

_Agents make mistakes — how do we catch them?_

Integration tests against the local-FS backend cover the load-bearing behaviors:

- Round-trip: `register(id, "v1", uri, MyModel)` then `from_id(id, "v1", MyModel)` returns the same payload via the existing `from_path` deserialization.
- Immutability: second `register(id, "v1", ...)` raises a typed error; the original entry is untouched.
- ID validation: malformed ids (`"foo"`, `"foo/"`, `"a/b/c"`, empty version) raise `ValueError` before any FS access.
- Manifest layout invariant: a registered entry lands at the expected path. (Pins the on-disk contract.)
- Override: `from_id(..., registry=other)` ignores the default singleton.

We will also run one smoke test against a GCS sandbox bucket to confirm the same suite passes with a `gs://` registry root — this is the only test that exercises the real object-store semantics we care about.

## Open Questions

- **Content hash on `ArtifactEntry` for verification (not identity)?** Adding the field later is cheap (`extra="ignore"` on the model), but writing it only on new registrations leaves the registry mixed-mode forever. If reviewers want it, we add it at v1.
- **Per-region registries vs. one canonical region?** Marin runs in multiple regions and we don't want cross-region reads on the read path. Leaning per-region. Interacts with `uri` portability: a `us-east1` registry recording a `gs://marin-us-central2/...` uri is a cross-region pointer that may want to be a registration-time error.
- **Should `Artifact.save` auto-publish?** Tempting but couples the registry to every step output. Leaning *no* — explicit `registry.register(...)` — but reviewers may have a better instinct.
