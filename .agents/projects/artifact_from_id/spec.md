# Spec: `Artifact.from_id` + `ArtifactRegistry`

Concrete contracts for the design in [`design.md`](./design.md). Implementation lives elsewhere; this file pins the public surface reviewers are asked to agree to.

## Files

| Purpose | Path |
|---|---|
| New: registry protocol, default impl, errors, singleton helpers | `lib/marin/src/marin/execution/artifact_registry.py` |
| Modified: add `Artifact.from_id` classmethod | `lib/marin/src/marin/execution/artifact.py` |
| New: tests | `tests/execution/test_artifact_registry.py` |

No proto, no schema-registry tables, no migrations. The registry's persisted format is JSON on the filesystem (described below).

## Constants

```python
# lib/marin/src/marin/execution/artifact_registry.py

DEFAULT_REGISTRY_ENV: str = "MARIN_ARTIFACT_REGISTRY"
"""Environment variable read by `get_default_registry()` to locate the default registry root.
   When unset, the default is `f"{marin_prefix()}/artifact_registry"`."""

ID_SEPARATOR: str = "/"
"""Separator between namespace and name in an artifact id."""
```

## `ArtifactEntry`

```python
class ArtifactEntry(pydantic.BaseModel):
    """Persisted record for a single (id, version) artifact registration.

    This is both the wire format of `<root>/<namespace>/<name>/<version>.json` and the
    return type of `ArtifactRegistry.register` / `ArtifactRegistry.lookup`. Entries are
    immutable once written; the on-disk file is overwritten by a future explicit-upsert
    feature only (out of scope at v1).

    Frozen (`model_config = ConfigDict(frozen=True, extra="ignore")`): callers cannot mutate
    a returned entry. Unknown fields read from disk are ignored, so adding optional fields
    (`created_at`, `content_hash`, `metadata`, `artifact_type`) in a future revision will
    not break readers running older code.
    """

    id: str
    """Artifact id in the form `<namespace>/<name>`. Validated by `validate_id`."""

    version: str
    """User-supplied version string. Validated by `validate_version`."""

    uri: str
    """Location of the artifact bytes — the `base_path` argument to `Artifact.from_path`.
       MUST be absolute: either a URI with scheme (`gs://...`, `file://...`) or an absolute
       local path (`/...`). Relative paths are rejected by `register` to avoid having the
       same registered entry resolve differently in different processes."""
```

## `ArtifactRegistry` protocol

```python
class ArtifactRegistry(typing.Protocol):
    """A `(id, version) → uri` map. Append-only at v1; existing entries cannot be
    overwritten or removed. Concrete impls back the protocol with filesystem,
    in-memory, or test-double storage."""

    def register(self, id: str, version: str, uri: str) -> ArtifactEntry:
        """Record `(id, version) → uri`.

        Raises `ArtifactAlreadyExistsError` if an entry for `(id, version)` already
        exists; the existing entry is not modified. Raises `InvalidArtifactIdError`
        on a malformed id, version, or non-absolute uri. Returns the newly-written
        entry on success.

        Implementations MUST write to storage before returning, and MUST do so atomically
        — the manifest file is either fully present and valid, or absent. On local
        filesystems this is `tempfile + os.replace`; on GCS, per-object PUT is naturally
        atomic. Implementations MUST treat the existence check + write as best-effort
        against concurrent writers — two processes racing to register the *same*
        `(id, version)` may both believe they succeeded (last-writer-wins on the file).
        v1 does not provide CAS; callers needing it should coordinate externally."""

    def lookup(self, id: str, version: str) -> ArtifactEntry:
        """Look up the entry for `(id, version)`.

        Raises `ArtifactNotFoundError` if no entry exists. Raises `InvalidArtifactIdError`
        on malformed inputs. Raises `ArtifactRegistryError` if the manifest file exists
        but is unreadable, not valid UTF-8, not valid JSON, or fails `ArtifactEntry`
        validation (the underlying exception is chained via `raise ... from`). Performs
        an O(1) GET against storage — never lists."""
```

## `FilesystemArtifactRegistry`

```python
class FilesystemArtifactRegistry(ArtifactRegistry):
    """The default `ArtifactRegistry`. Backed by GCS or local filesystem via
    `rigging.filesystem.open_url` / `fsspec`. Stores one JSON file per entry at:

        {root}/{namespace}/{name}/{version}.json

    where the file contents are `ArtifactEntry.model_dump_json()`.

    `root` may be any URI accepted by `rigging.filesystem` (e.g. `gs://bucket/prefix`,
    `/local/path`). Trailing slashes are normalized away. The registry does not create
    the root directory eagerly; the first `register` call creates intermediate
    directories as needed.

    The instance is cheap to construct and holds no open file handles. Multiple
    instances pointing at the same root are interchangeable."""

    def __init__(self, root: str) -> None:
        """Store the normalized `root`. Performs NO I/O and does NOT validate that `root`
           is reachable, exists, or is writable. The first `register` / `lookup` call
           surfaces I/O errors. Empty string and non-string inputs raise `TypeError` /
           `ValueError`."""

    @property
    def root(self) -> str:
        """The normalized registry root URI (trailing slashes stripped)."""

    def register(self, id: str, version: str, uri: str) -> ArtifactEntry: ...

    def lookup(self, id: str, version: str) -> ArtifactEntry: ...

    def entry_path(self, id: str, version: str) -> str:
        """Return the storage path for a given entry. Public: pins the on-disk layout
           contract for tooling, tests, and human introspection (`gsutil cat $(...)`)."""
```

## Singleton helpers

```python
def get_default_registry() -> ArtifactRegistry:
    """Return the process-wide default registry, constructing it on first call from
    `os.environ[DEFAULT_REGISTRY_ENV]` (fallback: `f"{marin_prefix()}/artifact_registry"`).
    Raises `RuntimeError` if both the env var is unset AND `marin_prefix()` is unset /
    raises — the registry has no sensible default in that case and the caller must set
    one explicitly via `set_default_registry` or the env var.

    The returned instance is cached at module scope. Re-reading the env var requires
    `set_default_registry(None)` to clear the cache; this is intentional — env-var
    flips mid-process are not a supported use case.

    Thread-safety: the first call is NOT guarded by a lock. In multi-threaded entrypoints,
    call `get_default_registry()` once during startup before forking threads so the cache
    is populated. Concurrent first-call races may construct multiple instances, of which
    one wins the cache slot; the others are GC'd. This is safe but wasteful."""


def set_default_registry(registry: ArtifactRegistry | None) -> None:
    """Override (or clear) the module-level default. Passing `None` clears the cache
    so the next `get_default_registry()` call re-reads the environment.

    In-flight callers retain whatever reference `get_default_registry()` already returned
    to them — swapping the default does not invalidate already-resolved registries. Tests
    that flip the default should use a fixture that restores the prior value in teardown.

    Primary consumer: tests, notebooks, and any caller that needs a non-default registry
    without threading it through every `Artifact.from_id` call site."""
```

## `Artifact.from_id`

Added to the existing `Artifact` class in `lib/marin/src/marin/execution/artifact.py`.

```python
@classmethod
def from_id(
    cls,
    id: str,
    version: str,
    /,
    artifact_type: type[T] | None = None,
    *,
    registry: ArtifactRegistry | None = None,
) -> T | PathMetadata | dict[str, Any]:
    """Load an artifact by registry id + version.

    `id` and `version` are positional-only — callers always write `Artifact.from_id("ns/n",
    "v3", ...)`, never `Artifact.from_id(version="v3", id="ns/n")`. This keeps every call
    site readable and aligns with the `from_path` shape.

    Resolves `(id, version)` against `registry` (or the module-level default if
    `registry is None`) to obtain a URI, then delegates to `Artifact.from_path(uri,
    artifact_type)` — meaning the return value, the `PathMetadata` fallback (synthesized
    when the sidecar is missing but `.executor_status` reads `SUCCESS`), and the
    `artifact_type` deserialization semantics are identical to the path-based loader.

    `artifact_type`, if provided, MUST be a Pydantic `BaseModel` subclass (the existing
    contract on `from_path` — see `artifact.py` today). The registry does not record
    the type; the caller asserts it on read.

    Raises whatever `ArtifactRegistry.lookup` raises (`ArtifactNotFoundError`,
    `InvalidArtifactIdError`, `ArtifactRegistryError`), plus whatever `Artifact.from_path`
    raises on the resolved URI."""
```

## ID and version validation

Both `register` and `lookup` validate their `id` and `version` arguments before any I/O. Validation rules:

```python
ID_PATTERN: typing.Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
VERSION_PATTERN: typing.Final = re.compile(r"^[A-Za-z0-9_.-]+$")

def validate_id(id: str) -> tuple[str, str]:
    """Returns `(namespace, name)`. Raises `InvalidArtifactIdError` if the id is not
    exactly one `/`-separated pair of non-empty segments matching `ID_PATTERN`."""

def validate_version(version: str) -> str:
    """Returns the version unchanged. Raises `InvalidArtifactIdError` if it does not
    match `VERSION_PATTERN`."""
```

The pattern is intentionally narrow — anything we allow becomes part of a GCS path, so we disallow `/`, whitespace, and shell metacharacters by construction. The validators are exposed (no leading underscore) so callers generating ids can pre-check.

## Errors

```python
class ArtifactRegistryError(Exception):
    """Base class for all registry errors. Catch this to handle registry failures
    generically; catch the subclasses for specific cases."""


class ArtifactNotFoundError(ArtifactRegistryError, KeyError):
    """No entry exists for `(id, version)`. Subclasses `KeyError` so existing
    `dict`-style consumers keep working.

    `__init__` calls `super().__init__((id, version))` so `KeyError`'s `args[0]` is the
    `(id, version)` tuple — preserving the `KeyError` contract while keeping both fields
    accessible via `.id` and `.version` attributes."""

    id: str
    version: str

    def __init__(self, id: str, version: str) -> None: ...


class ArtifactAlreadyExistsError(ArtifactRegistryError):
    """An entry for `(id, version)` already exists; `register` refuses to overwrite.
    The existing entry is exposed on `.existing` so callers can compare."""

    existing: ArtifactEntry

    def __init__(self, existing: ArtifactEntry) -> None: ...


class InvalidArtifactIdError(ArtifactRegistryError, ValueError):
    """Malformed id or version. Subclasses `ValueError` for ergonomics."""
```

## Persisted shape

One JSON file per `(id, version)`:

- Path: `{root}/{namespace}/{name}/{version}.json`
- Contents: `ArtifactEntry.model_dump_json()` (UTF-8, no enclosing array).
- Encoding: UTF-8 with no BOM. No trailing newline guarantee.

Example (`gs://marin-us-central2/artifact_registry/datasets/fineweb-resiliparse/v3.json`):

```json
{"id": "datasets/fineweb-resiliparse", "version": "v3", "uri": "gs://marin-us-central2/documents/fineweb-resiliparse-8c2f3a"}
```

The `uri` is typically itself a content-addressed executor output (the `_8c2f3a` suffix is the executor step hash from `step_spec.py`). The registry layer adds logical naming on top of that existing content-addressed storage.

The file is the source of truth. There is no index file, no companion `_meta`, no schema-registry registration. Listing all versions of an id is a recursive prefix scan under `{root}/{namespace}/{name}/`.

## Out of scope (v1)

Listed explicitly so reviewers know what *not* to push back on. Each is a clean follow-up; none are foreclosed:

- **Mutable aliases** (`latest`, `prod`, `@champion`). Non-goal, not a deferral — see Decisions in `design.md`.
- **Explicit overwrite / upsert.** `register(..., overwrite=True)` was scoped out for v1.
- **Content-addressable storage** (content-hash as identity). Considered and rejected — see Decisions in `design.md`.
- **Content-hash field on `ArtifactEntry`** for verification (not identity). Open question.
- **Auto-registration from `Artifact.save`.** Steps that want registry coverage call `registry.register(...)` explicitly.
- **Deletion / garbage collection.** No `delete(id, version)` at v1. The registry is append-only by convention.
- **Multi-writer transactional semantics** (GCS CAS via `if-generation-match`, distributed locking, etc.).
- **Listing APIs** (`list_ids()`, `list_versions(id)`). When a consumer needs one, we add it.
- **Test helper context manager** (`use_registry(reg)`). Deferred — callers can write a local pytest fixture; we'll formalize one when a consumer asks.
- **Per-region federation.** One registry root per process. Cross-region resolution is out of scope.
- **Migrating existing `from_path` call sites.** `from_id` is purely additive; `from_path` stays.
- **Path sharding in the manifest layout.** Considered and rejected at v1 — see Decisions in `design.md`.
