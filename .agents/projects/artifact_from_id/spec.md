# Spec: `Artifact.from_id` + `ArtifactRegistry`

Concrete contracts for the design in [`design.md`](./design.md). Implementation lives elsewhere; this file pins the public surface reviewers are asked to agree to.

## Files

| Purpose | Path |
|---|---|
| New: registry protocol, default impl, errors, singleton helpers | `lib/marin/src/marin/execution/artifact_registry.py` |
| Modified: add `Artifact.from_id` classmethod | `lib/marin/src/marin/execution/artifact.py` |
| Modified: add `is_remote_path` helper (used by `register`'s local-in-remote guard) | `lib/rigging/src/rigging/filesystem.py` |
| Modified: drop local `is_remote_path`, import from `rigging.filesystem` | `lib/marin/src/marin/evaluation/utils.py` + 3 evaluator call sites |
| New: tests | `tests/execution/test_artifact_registry.py` |

No proto, no schema-registry tables, no migrations. The registry's persisted format is JSON on the filesystem (described below).

## Constants

```python
# lib/marin/src/marin/execution/artifact_registry.py

DEFAULT_REGISTRY_ENV: str = "MARIN_ARTIFACT_REGISTRY"
"""Environment variable read by `get_default_registry()` to override the registry root."""

DEFAULT_REGISTRY_ROOT: str = "gs://marin-us-central1/artifact_registry"
"""Canonical default root used when `DEFAULT_REGISTRY_ENV` is unset. One global location so an id
   resolves identically from every region/cloud. Tests are kept off it by an autouse fixture in
   `tests/conftest.py` that redirects the default to a temp dir."""

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
    """Artifact id in the form `<namespace>/<name>`. Validated by `_validate_id`."""

    version: str
    """CalVer version string `YYYY.MM.DD` with optional `-<modifier>` suffix
       (e.g. `"2026.05.29"`, `"2026.10.01-fall-hero"`). Validated by `_validate_version`,
       which both pattern-matches AND constructs a `datetime.date` to reject impossible
       calendar dates."""

    uri: str
    """Absolute location of the artifact bytes at registration time — the `base_path` argument to
       `Artifact.from_path`. MUST be absolute: either a URI with scheme (`gs://...`, `file://...`)
       or an absolute local path (`/...`). Relative paths are rejected by `register` to avoid having
       the same registered entry resolve differently in different processes. A local-filesystem uri
       (bare `/...` path or `file://`) is also rejected by `register` when the registry root is
       remote (e.g. `gs://`) — it would be an unresolvable pointer for other readers of the shared
       registry. This is the cross-region-stable fallback; readers prefer `relative_path`."""

    relative_path: str | None = None
    """Path of the artifact relative to `marin_prefix()` at registration time, or `None` when `uri`
       was not under `marin_prefix()`. Marin replicates data across regional buckets under a
       region-specific `marin_prefix()` (`gs://marin-{region}`); recording the relative path lets a
       reader resolve the region-local replica via its OWN `marin_prefix()` (avoiding a cross-region
       read) before falling back to `uri`. Optional + defaulted so manifests written before this
       field still load (`extra="ignore"` + default)."""
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
        on a malformed id, version, non-absolute uri, or a local-filesystem uri
        registered in a remote (shared) registry. Returns the newly-written entry on
        success.

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

    def register(self, id: str, version: str, uri: str) -> ArtifactEntry: ...

    def lookup(self, id: str, version: str) -> ArtifactEntry: ...

    # Internal. The normalized root (`self._root`, trailing slashes stripped) and the on-disk
    # layout (`self._entry_path(id, version)` → `{root}/{ns}/{name}/{version}.json`) are
    # implementation details, not part of the `ArtifactRegistry` contract. No external consumer
    # today; promote to a public accessor only when one appears.
```

## Singleton helpers

The default is held in a module-level `contextvars.ContextVar`, so overrides are context-local
and async-safe (a `with`-block or `set_default_registry` override is visible to the current
context and tasks derived from it, isolated from other contexts).

```python
def get_default_registry() -> ArtifactRegistry:
    """Return the default registry for the current context, constructing it on first use from
    `os.environ[DEFAULT_REGISTRY_ENV]`, falling back to `DEFAULT_REGISTRY_ROOT`
    (`gs://marin-us-central1/artifact_registry`). Tests must not hit the canonical
    root: an autouse fixture in `tests/conftest.py` installs a temp-dir registry as the
    context-local default for every test (via `use_default_registry`).

    The constructed instance is cached in the context var; `set_default_registry(None)` clears it
    so the next call re-reads the environment.

    Context semantics: a new OS thread starts with a fresh context and so falls back to the
    env-var / `DEFAULT_REGISTRY_ROOT` default — explicit overrides do not cross thread boundaries,
    but the configured default does, so every thread resolves the same root in the absence of an
    override."""


def set_default_registry(registry: ArtifactRegistry | None) -> None:
    """Install (or clear) the context-local default. Passing `None` clears it so the next
    `get_default_registry()` call re-reads the environment. For scoped overrides that restore
    automatically, prefer `use_default_registry`."""


@contextlib.contextmanager
def use_default_registry(registry: ArtifactRegistry) -> Iterator[ArtifactRegistry]:
    """Install `registry` as the context-local default for the duration of the `with` block,
    restoring the previous default on exit (even on exception). Primary consumers are tests and
    notebooks that need a non-default registry without threading `registry=` through every
    `Artifact.from_id` call site."""
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
    "2026.05.29", ...)`, never `Artifact.from_id(version="2026.05.29", id="ns/n")`. This keeps every call
    site readable and aligns with the `from_path` shape.

    Resolves `(id, version)` against `registry` (or the module-level default if
    `registry is None`) to an `ArtifactEntry`, then delegates to `Artifact.from_path(...,
    artifact_type)` — meaning the return value, the `PathMetadata` fallback (synthesized
    when the sidecar is missing but `.executor_status` reads `SUCCESS`), and the
    `artifact_type` deserialization semantics are identical to the path-based loader.

    Region-aware resolution: when the entry has a `relative_path` (its uri was under
    `marin_prefix()` at registration), it resolves that path against THIS process's
    `marin_prefix()` first, so a reader loads the region-local replica instead of reading
    across regions. If that copy is absent (`from_path` raises `FileNotFoundError`), it
    falls back to the absolute `entry.uri`, logging a warning when the fallback is
    cross-region (the absolute uri is not under the current `marin_prefix()`).

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
VERSION_PATTERN: typing.Final = re.compile(
    r"^(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})(?:-(?P<suffix>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)

def _validate_id(id: str) -> tuple[str, str]:
    """Returns `(namespace, name)`. Raises `InvalidArtifactIdError` if the id is not
    exactly one `/`-separated pair of non-empty segments matching `ID_PATTERN`."""

def _validate_version(version: str) -> str:
    """Returns the version unchanged on success. The version MUST be CalVer:
    `YYYY.MM.DD` with optional `-<modifier>` suffix where the modifier begins with
    an alphanumeric and contains only `[A-Za-z0-9._-]`.

    Validation is two-step: first `VERSION_PATTERN` rejects shape mismatches; then the
    `(year, month, day)` groups are passed to `datetime.date(...)` to reject impossible
    calendar dates (e.g. `2026.02.30`, `2026.13.01`). Either check failing raises
    `InvalidArtifactIdError`.

    The modifier MUST NOT contain `/` — the whole version becomes a single path segment
    in `{root}/{namespace}/{name}/{version}.json`, and embedded slashes would silently
    create directory levels and break `lookup`. `/` is excluded by `VERSION_PATTERN`'s
    suffix alphabet; this is called out here so it isn't lost in the regex.

    Examples accepted: `"2026.05.29"`, `"2026.10.01-fall-hero"`, `"2026.10.01-rc1"`.
    Examples rejected: `"v3"`, `"2026.5.29"` (no zero-pad), `"2026.02.30"` (impossible
    date), `"2026.10.01-"` (empty suffix), `"2026.10.01--foo"` (suffix must start with
    alphanumeric), `"2026.10.01-foo/bar"` (slash in modifier)."""
```

The id pattern is intentionally narrow — anything we allow becomes part of a GCS path, so the alphabet excludes `/`, whitespace, and shell metacharacters. CalVer for versions gives lexical sort = chronological sort (zero-padded date prefix) and embeds provenance. The validators are module-private (`_validate_id` / `_validate_version`); `register` / `lookup` call them at the boundary.

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

Example (`gs://marin-us-central2/artifact_registry/datasets/fineweb-resiliparse/2026.05.29.json`):

```json
{"id": "datasets/fineweb-resiliparse", "version": "2026.05.29", "uri": "gs://marin-us-central2/documents/fineweb-resiliparse-8c2f3a", "relative_path": "documents/fineweb-resiliparse-8c2f3a"}
```

The `uri` is typically itself a content-addressed executor output (the `_8c2f3a` suffix is the executor step hash from `step_spec.py`). The registry layer adds logical naming on top of that existing content-addressed storage. `relative_path` is the `marin_prefix()`-relative form of `uri` recorded at registration time (`null` when the uri was not under `marin_prefix()`); readers resolve it against their own `marin_prefix()` for a region-local replica before falling back to `uri`.

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
- **Per-region federation.** One registry root per process. Cross-region resolution is out of scope.
- **Migrating existing `from_path` call sites.** `from_id` is purely additive; `from_path` stays.
- **Path sharding in the manifest layout.** Considered and rejected at v1 — see Decisions in `design.md`.
