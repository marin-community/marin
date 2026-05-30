# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A `(id, version) → uri` map for artifacts, decoupled from where the bytes live.

`Artifact.from_path` ties an artifact's identity to its storage path. This registry adds a
logical name + user-supplied CalVer version on top, so downstream code can refer to an
artifact by `(id, version)` regardless of where it lives today or after a storage re-org.

The default backend stores one JSON file per entry at `{root}/{namespace}/{name}/{version}.json`,
backed by GCS or local filesystem via `rigging.filesystem`. See `.agents/projects/artifact_from_id/`
for the design and spec.
"""

import contextlib
import contextvars
import datetime
import os
import re
import typing
from collections.abc import Iterator

import pydantic
from rigging.filesystem import is_remote_path, marin_prefix, open_url, url_to_fs

DEFAULT_REGISTRY_ENV = "MARIN_ARTIFACT_REGISTRY"
"""Environment variable read by `get_default_registry` to override the registry root."""

DEFAULT_REGISTRY_ROOT = "gs://marin-us-central1/artifact_registry"
"""Canonical registry root used by `get_default_registry` when `DEFAULT_REGISTRY_ENV` is unset.

A single global location so an artifact id resolves identically from every region and cloud
(the manifests are tiny JSON, so cross-region/cross-cloud reads are cheap). Tests must NOT hit
this — `tests/conftest.py` installs an autouse fixture that redirects the default to a temp dir.
"""

ID_SEPARATOR = "/"
"""Separator between namespace and name in an artifact id."""

ID_PATTERN: typing.Final = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
VERSION_PATTERN: typing.Final = re.compile(
    r"^(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})(?:-(?P<suffix>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)


class ArtifactRegistryError(Exception):
    """Base class for all registry errors.

    Catch this to handle registry failures generically; catch the subclasses for specific cases.
    """


class ArtifactNotFoundError(ArtifactRegistryError, KeyError):
    """No entry exists for `(id, version)`.

    Subclasses `KeyError` so existing `dict`-style consumers keep working. The `(id, version)`
    tuple is the `KeyError` payload (`args[0]`); both fields are also exposed as attributes.
    """

    id: str
    version: str

    def __init__(self, id: str, version: str) -> None:
        super().__init__((id, version))
        self.id = id
        self.version = version


class ArtifactAlreadyExistsError(ArtifactRegistryError):
    """An entry for `(id, version)` already exists; `register` refuses to overwrite.

    The existing entry is exposed on `.existing` so callers can compare.
    """

    existing: "ArtifactEntry"

    def __init__(self, existing: "ArtifactEntry") -> None:
        super().__init__(f"artifact {existing.id!r} version {existing.version!r} already registered at {existing.uri!r}")
        self.existing = existing


class InvalidArtifactIdError(ArtifactRegistryError, ValueError):
    """Malformed id or version. Subclasses `ValueError` for ergonomics."""


def _validate_id(id: str) -> tuple[str, str]:
    """Return `(namespace, name)` for a well-formed artifact id.

    Raises `InvalidArtifactIdError` if the id is not exactly one `/`-separated pair of
    non-empty segments matching `ID_PATTERN`.
    """
    if not ID_PATTERN.match(id):
        raise InvalidArtifactIdError(
            f"invalid artifact id {id!r}: expected '<namespace>/<name>' with segments matching "
            f"{ID_PATTERN.pattern!r}"
        )
    namespace, name = id.split(ID_SEPARATOR)
    return namespace, name


def _validate_version(version: str) -> str:
    """Return `version` unchanged if it is a valid CalVer string.

    The version MUST be `YYYY.MM.DD` with an optional `-<modifier>` suffix where the modifier
    begins with an alphanumeric and contains only `[A-Za-z0-9._-]` (no `/`, since the whole
    version becomes a single path segment). The `(year, month, day)` are passed to
    `datetime.date` to reject impossible calendar dates. Either check failing raises
    `InvalidArtifactIdError`.
    """
    match = VERSION_PATTERN.match(version)
    if not match:
        raise InvalidArtifactIdError(f"invalid artifact version {version!r}: expected CalVer 'YYYY.MM.DD[-modifier]'")
    year, month, day = int(match["year"]), int(match["month"]), int(match["day"])
    try:
        datetime.date(year, month, day)
    except ValueError as e:
        raise InvalidArtifactIdError(f"invalid artifact version {version!r}: {e}") from e
    return version


class ArtifactEntry(pydantic.BaseModel):
    """Persisted record for a single `(id, version)` artifact registration.

    This is both the wire format of `<root>/<namespace>/<name>/<version>.json` and the return
    type of `ArtifactRegistry.register` / `ArtifactRegistry.lookup`. Entries are immutable once
    written. Frozen so callers cannot mutate a returned entry; unknown fields read from disk are
    ignored, so adding optional fields in a future revision will not break older readers.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    id: str
    """Artifact id in the form `<namespace>/<name>`. Validated by `_validate_id`."""

    version: str
    """CalVer version string `YYYY.MM.DD` with optional `-<modifier>` suffix. Validated by `_validate_version`."""

    uri: str
    """Absolute location of the artifact bytes at registration time — the `base_path` argument to
    `Artifact.from_path`.

    MUST be absolute: either a URI with scheme (`gs://...`, `file://...`) or an absolute local
    path (`/...`). Relative paths are rejected by `register` so the same entry resolves identically
    across processes. This is the cross-region-stable fallback; readers prefer `relative_path`.
    """

    relative_path: str | None = None
    """Path of the artifact relative to `marin_prefix()` at registration time, or `None` when `uri`
    was not under `marin_prefix()`.

    Marin replicates data across regional buckets under a region-specific `marin_prefix()`
    (`gs://marin-{region}`). Recording the relative path lets a reader resolve the region-local
    replica via its OWN `marin_prefix()` — avoiding a cross-region read — before falling back to the
    absolute `uri`. Optional + defaulted so manifests written before this field still load.
    """


_SCHEME_PATTERN: typing.Final = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")


def _uri_scheme(uri: str) -> str | None:
    """Return the lowercased URL scheme of `uri` (e.g. `gs`, `s3`, `file`), or `None` for a bare path."""
    match = _SCHEME_PATTERN.match(uri)
    return match.group(1).lower() if match else None


def _is_absolute_uri(uri: str) -> bool:
    """True if `uri` has a URL scheme (`gs://`, `file://`, ...) or is an absolute local path."""
    return uri.startswith("/") or _uri_scheme(uri) is not None


def _relative_to_marin_prefix(uri: str) -> str | None:
    """Return `uri` as a path relative to the current `marin_prefix()`, or `None` if not under it.

    Used by `register` to record a region-agnostic path alongside the absolute uri, so readers in
    other regions can resolve a region-local replica via their own `marin_prefix()`.
    """
    prefix = marin_prefix().rstrip("/")
    if uri.startswith(prefix + "/"):
        return uri[len(prefix) + 1 :]
    return None


@typing.runtime_checkable
class ArtifactRegistry(typing.Protocol):
    """A `(id, version) → uri` map. Append-only at v1; existing entries cannot be overwritten."""

    def register(self, id: str, version: str, uri: str) -> ArtifactEntry:
        """Record `(id, version) → uri`.

        Raises `ArtifactAlreadyExistsError` if an entry for `(id, version)` already exists (the
        existing entry is not modified), `InvalidArtifactIdError` on a malformed id, version,
        non-absolute uri, or a local-filesystem uri registered in a remote (shared) registry.
        Returns the newly-written entry on success.
        """
        ...

    def lookup(self, id: str, version: str) -> ArtifactEntry:
        """Look up the entry for `(id, version)`.

        Raises `ArtifactNotFoundError` if no entry exists, `InvalidArtifactIdError` on malformed
        inputs, `ArtifactRegistryError` if the manifest file exists but is unreadable or invalid.
        """
        ...


class FilesystemArtifactRegistry(ArtifactRegistry):
    """The default `ArtifactRegistry`, backed by GCS or local filesystem via `rigging.filesystem`.

    Stores one JSON file per entry at `{root}/{namespace}/{name}/{version}.json`, where the file
    contents are `ArtifactEntry.model_dump_json()`. The instance is cheap to construct and holds no
    open file handles; multiple instances pointing at the same root are interchangeable.
    """

    def __init__(self, root: str) -> None:
        """Store the normalized `root`. Performs NO I/O and does not validate reachability.

        Empty string raises `ValueError`; non-string input raises `TypeError`.
        """
        if not isinstance(root, str):
            raise TypeError(f"root must be a str, got {type(root).__name__}")
        if not root:
            raise ValueError("root must be a non-empty string")
        self._root = root.rstrip("/")

    def _entry_path(self, id: str, version: str) -> str:
        """Return the storage path for a given entry — the on-disk layout `{root}/{ns}/{name}/{version}.json`."""
        namespace, name = _validate_id(id)
        _validate_version(version)
        return f"{self._root}/{namespace}/{name}/{version}.json"

    def register(self, id: str, version: str, uri: str) -> ArtifactEntry:
        if not _is_absolute_uri(uri):
            raise InvalidArtifactIdError(f"uri must be absolute (URI with scheme or '/'-rooted path), got {uri!r}")
        # A local-filesystem uri recorded in a remote (shared) registry is a broken pointer for every
        # other reader, so refuse it. The reverse — a remote uri in a local registry — is a valid,
        # resolvable pointer and is allowed.
        if is_remote_path(self._root) and not is_remote_path(uri):
            raise InvalidArtifactIdError(
                f"cannot register local-filesystem uri {uri!r} in the remote registry at {self._root!r}: "
                f"a local path is not resolvable by other readers of a shared registry"
            )
        path = self._entry_path(id, version)
        entry = ArtifactEntry(id=id, version=version, uri=uri, relative_path=_relative_to_marin_prefix(uri))

        fs, fs_path = url_to_fs(path)
        if fs.exists(fs_path):
            raise ArtifactAlreadyExistsError(self.lookup(id, version))

        # Publish the manifest with a single write: an object-store upload is one atomic PUT (no
        # partial-read window), and `open_url` auto-creates parent dirs on local FS. The registry is
        # append-only, so a same-pair race is last-writer-wins (no CAS at v1).
        with open_url(path, "wb") as fd:
            fd.write(entry.model_dump_json().encode("utf-8"))
        return entry

    def lookup(self, id: str, version: str) -> ArtifactEntry:
        path = self._entry_path(id, version)
        try:
            with open_url(path, "rb") as fd:
                data = fd.read()
        except FileNotFoundError as e:
            raise ArtifactNotFoundError(id, version) from e
        try:
            return ArtifactEntry.model_validate_json(data)
        except (UnicodeDecodeError, ValueError) as e:
            raise ArtifactRegistryError(f"manifest at {path!r} is unreadable or invalid: {e}") from e


_default_registry: contextvars.ContextVar[ArtifactRegistry | None] = contextvars.ContextVar(
    "marin_default_artifact_registry", default=None
)


def get_default_registry() -> ArtifactRegistry:
    """Return the default registry for the current context, constructing it on first use.

    The root comes from `os.environ[DEFAULT_REGISTRY_ENV]`, falling back to `DEFAULT_REGISTRY_ROOT`.
    The constructed instance is cached in the context var, so repeated calls in the same context
    return the same object; `set_default_registry(None)` clears it so the next call re-reads the
    environment.

    Context semantics: the default lives in a `ContextVar`, so an override installed via
    `set_default_registry` / `use_default_registry` is visible to the current context and to async
    tasks/copies derived from it, and is automatically isolated from other contexts. New OS threads
    start with a fresh context and therefore fall back to the env-var / `DEFAULT_REGISTRY_ROOT`
    default — explicit overrides do not cross thread boundaries, but the configured default does.
    """
    registry = _default_registry.get()
    if registry is None:
        root = os.environ.get(DEFAULT_REGISTRY_ENV) or DEFAULT_REGISTRY_ROOT
        registry = FilesystemArtifactRegistry(root)
        _default_registry.set(registry)
    return registry


def set_default_registry(registry: ArtifactRegistry | None) -> None:
    """Install (or clear) the context-local default registry.

    Passing `None` clears it so the next `get_default_registry` call re-reads the environment. For
    scoped overrides that restore automatically, prefer `use_default_registry`.
    """
    _default_registry.set(registry)


@contextlib.contextmanager
def use_default_registry(registry: ArtifactRegistry) -> Iterator[ArtifactRegistry]:
    """Install `registry` as the context-local default for the duration of the `with` block.

    Restores the previous default on exit (even on exception). Primary consumers are tests and
    notebooks that need a non-default registry without threading `registry=` through every call.
    """
    token = _default_registry.set(registry)
    try:
        yield registry
    finally:
        _default_registry.reset(token)
