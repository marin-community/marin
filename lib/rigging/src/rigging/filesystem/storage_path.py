# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StoragePath` value type and its string helpers.

``StoragePath`` is a frozen, parsed storage location (scheme, authority, key
segments) with a bounded set of I/O verbs. The verbs resolve through the guarded
factory in :mod:`rigging.filesystem.factory`, which itself parses paths via
``StoragePath`` — so the verbs import the factory at call time (see
:func:`_url_to_fs`) to break that cycle, keeping this a leaf module.
"""

import dataclasses
import glob
import pathlib
from collections.abc import Generator
from datetime import datetime
from typing import Any

import braceexpand
import fsspec


def _url_to_fs(url: str) -> tuple[Any, str]:
    """Resolve ``(fs, path)`` through the guarded factory (deferred import; see module docstring)."""
    from rigging.filesystem.factory import url_to_fs  # noqa: PLC0415  (deferred: breaks storage_path<->factory cycle)

    return url_to_fs(url)


def _open_url(url: str, mode: str = "rb", **kwargs: Any) -> "fsspec.core.OpenFile":
    """Open ``url`` through the guarded factory (deferred import; see :func:`_url_to_fs`)."""
    from rigging.filesystem.factory import open_url  # noqa: PLC0415  (deferred: breaks storage_path<->factory cycle)

    return open_url(url, mode, **kwargs)


def _key_segments(key: str) -> tuple[str, ...]:
    return tuple(part for part in key.split("/") if part)


# Schemes whose ``scheme://`` prefix carries no authority: the entire remainder is the
# key, not ``authority/key``. The ``mirror://`` filesystem strips its protocol and
# treats what follows as a path relative to the local prefix, so ``mirror://a/b`` must
# parse to an empty authority with key ``a/b`` — matching ``parse("mirror://") / "a/b"``.
_EMPTY_AUTHORITY_SCHEMES: frozenset[str] = frozenset({"mirror"})


def _parse_parts(value: str) -> tuple[str | None, str, tuple[str, ...], bool]:
    """Split a storage URL / path into ``(scheme, netloc, segments, rooted)``."""
    if "://" in value:
        scheme, rest = value.split("://", 1)
        if scheme in _EMPTY_AUTHORITY_SCHEMES:
            # Empty-authority scheme: no netloc to split off; the whole remainder is the
            # key. ``rooted=False`` keeps the join convention (``mirror://`` + ``a/b`` ->
            # ``mirror://a/b``, no third slash).
            return scheme, "", _key_segments(rest), False
        netloc, sep, key = rest.partition("/")
        # With an authority the key is always /-separated, so rooted is pinned True
        # to keep value equality and relative_to independent of a trailing slash.
        return scheme, netloc, _key_segments(key), bool(netloc) or bool(sep)
    return None, "", _key_segments(value), value.startswith("/")


@dataclasses.dataclass(frozen=True, init=False)
class StoragePath:
    """A parsed storage location with a bounded set of I/O verbs.

    A frozen value type: URL scheme, authority, and key segments. Object-store keys are
    normalized on round-trip — joins are structural (a segment never contains a
    separator), so a doubled or trailing separator is unrepresentable and ``parse`` ->
    ``str`` collapses interior ``//`` and strips a trailing ``/``.

    Construct by parsing a string (``StoragePath("gs://b/k")``); :meth:`parse` is a thin
    alias. Paths at rest (configs, artifact records, CLI args) stay ``str``: parse at a
    boundary, manipulate, and ``str()`` back out. The I/O verbs are stateless — each
    resolves through the guarded :func:`url_to_fs`/:func:`open_url` factory (cross-region
    budget, ``mirror://``, finite S3 timeouts) rather than memoizing a filesystem handle
    on the frozen instance. The listing verbs (:meth:`ls`, :meth:`walk`, :meth:`glob`)
    return reopenable :class:`StoragePath` values.
    """

    scheme: str | None
    """URL scheme (``gs``, ``s3``, ``mirror``), or ``None`` for a local path."""
    netloc: str
    """Bucket/authority; empty for an empty-authority scheme like ``mirror://``."""
    segments: tuple[str, ...]
    """Key segments; never empty strings."""
    rooted: bool = True
    """Whether a ``/`` precedes the key: a local path's absoluteness (``/tmp/x`` vs
    ``rel/x``), or the empty-authority join convention (``file:///x`` vs ``mirror://x``).
    Irrelevant when ``netloc`` is non-empty."""

    def __init__(
        self,
        value: "str | StoragePath | None" = None,
        *,
        scheme: str | None = None,
        netloc: str = "",
        segments: tuple[str, ...] = (),
        rooted: bool = True,
    ):
        """Parse *value* (a URL/path string, or another :class:`StoragePath`); when
        *value* is omitted, build directly from the keyword parts.
        """
        # A positional value parses; the keyword parts build structurally. dataclasses
        # .replace passes only the fields (never value), so it takes the keyword branch —
        # keeping replace, __truediv__, and parent working under init=False.
        if value is not None:
            if isinstance(value, StoragePath):
                scheme, netloc, segments, rooted = value.scheme, value.netloc, value.segments, value.rooted
            else:
                scheme, netloc, segments, rooted = _parse_parts(value)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "netloc", netloc)
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "rooted", rooted)

    @classmethod
    def parse(cls, value: str) -> "StoragePath":
        """Alias for ``StoragePath(value)``, kept for existing call sites."""
        return cls(value)

    @staticmethod
    def normalize(value: str) -> str:
        """``value`` in canonical single-separator form (``str(StoragePath(value))``)."""
        return str(StoragePath(value))

    def __truediv__(self, relative: str) -> "StoragePath":
        if "://" in relative or relative.startswith("/"):
            raise ValueError(f"cannot join non-relative path {relative!r} onto {self}")
        return dataclasses.replace(self, segments=self.segments + _key_segments(relative))

    def relative_to(self, base: "StoragePath") -> str:
        """The ``/``-joined segments of this path under ``base``.

        Structural containment — compares parsed segments, not string prefixes, so a
        doubled separator on either side cannot fork the answer.
        """
        same_root = (self.scheme, self.netloc, self.rooted) == (base.scheme, base.netloc, base.rooted)
        if not same_root or self.segments[: len(base.segments)] != base.segments:
            raise ValueError(f"{self} is not under {base}")
        return "/".join(self.segments[len(base.segments) :])

    @property
    def bucket(self) -> str:
        """The object-store bucket (the authority); empty for a local or empty-authority path."""
        return self.netloc

    @property
    def key(self) -> str:
        """The ``/``-joined key segments beneath the authority (no leading or trailing ``/``)."""
        return "/".join(self.segments)

    @property
    def name(self) -> str:
        """The last key segment (basename), or ``""`` at the authority/filesystem root."""
        return self.segments[-1] if self.segments else ""

    @property
    def parent(self) -> "StoragePath":
        """This path with its last key segment removed; unchanged once at the root."""
        if not self.segments:
            return self
        return dataclasses.replace(self, segments=self.segments[:-1])

    def __str__(self) -> str:
        key = "/".join(self.segments)
        if self.scheme is None:
            return f"/{key}" if self.rooted else key
        root = f"{self.scheme}://{self.netloc}"
        if not key:
            return root
        if self.netloc or self.rooted:
            return f"{root}/{key}"
        return f"{root}{key}"

    # -- scheme predicates ---------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True for a local-disk path (no scheme, or the ``file`` scheme)."""
        return self.scheme in (None, "file")

    @property
    def is_remote(self) -> bool:
        """True for a remote object store (``gs``, ``s3``, ``mirror``, …)."""
        return not self.is_local

    # -- I/O verbs -----------------------------------------------------------
    #
    # Stateless: each resolves through the guarded url_to_fs/open_url factory so it
    # inherits the cross-region budget, mirror:// protocol, and S3 timeouts. The
    # factory is imported at call time (via _url_to_fs/_open_url) to keep this module
    # a leaf — see the module docstring.

    def exists(self) -> bool:
        fs, path = _url_to_fs(str(self))
        return fs.exists(path)

    def isdir(self) -> bool:
        fs, path = _url_to_fs(str(self))
        return fs.isdir(path)

    def size(self) -> int:
        fs, path = _url_to_fs(str(self))
        return fs.size(path)

    def mtime(self) -> datetime:
        fs, path = _url_to_fs(str(self))
        return fs.modified(path)

    def mkdirs(self, *, exist_ok: bool = True) -> None:
        fs, path = _url_to_fs(str(self))
        fs.makedirs(path, exist_ok=exist_ok)

    def glob(self) -> list["StoragePath"]:
        """Match this glob pattern against the filesystem.

        Brace-expands first (``{a,b}``), then globs each member and reattaches the
        filesystem's protocol so every result is a reopenable :class:`StoragePath`.
        Every member is treated as a pattern, so one that matches nothing — a magic
        pattern with no hits or a plain literal that is absent — contributes nothing
        and an all-missing input yields an empty list. Local matches stay scheme-less.
        Use :meth:`expand_glob` instead when a named-but-absent shard must be kept.
        """
        out: list[StoragePath] = []
        for pattern in braceexpand.braceexpand(str(self)):
            fs, path = _url_to_fs(pattern)
            out.extend(StoragePath(_reattach_protocol(fs, match)) for match in fs.glob(path))
        return out

    def expand_glob(self) -> list["StoragePath"]:
        """Resolve this shard specification into concrete paths, keeping named literals.

        Brace-expands first (``{a,b}``, ``{1..8}``); a member carrying glob magic
        (``*``, ``?``, ``[``) is matched against the filesystem, while a plain literal
        is kept as-is whether or not it exists. This is the difference from :meth:`glob`:
        an explicitly named but missing shard is preserved — so the caller can surface it
        — rather than silently dropped. A magic member that matches nothing still yields
        nothing.
        """
        out: list[StoragePath] = []
        for member in braceexpand.braceexpand(str(self)):
            fs, path = _url_to_fs(member)
            if glob.has_magic(path):
                out.extend(StoragePath(_reattach_protocol(fs, match)) for match in fs.glob(path))
            else:
                out.append(StoragePath(member))
        return out

    def open(self, mode: str = "rb", **kwargs: Any) -> "fsspec.core.OpenFile":
        """Open this path, returning a lazy ``fsspec`` ``OpenFile``.

        Delegates to :func:`open_url`, so it charges the cross-region budget for GCS
        reads and forwards ``compression=``/``encoding=``/``block_size=`` to ``fsspec``.
        """
        return _open_url(str(self), mode, **kwargs)

    def read_text(self, **kwargs: Any) -> str:
        with self.open("r", **kwargs) as f:
            return f.read()

    def read_bytes(self, **kwargs: Any) -> bytes:
        with self.open("rb", **kwargs) as f:
            return f.read()

    def write_text(self, data: str, **kwargs: Any) -> None:
        with self.open("w", **kwargs) as f:
            f.write(data)

    def write_bytes(self, data: bytes, **kwargs: Any) -> None:
        with self.open("wb", **kwargs) as f:
            f.write(data)

    def isfile(self) -> bool:
        fs, path = _url_to_fs(str(self))
        return fs.isfile(path)

    def ls(self) -> list["StoragePath"]:
        """List this directory's immediate children as reopenable paths.

        Non-recursive; each child carries its filesystem's protocol so it round-trips
        back through the verbs. For per-entry metadata (size/mtime) call
        :meth:`size`/:meth:`mtime` on a child, or drop to raw ``fs.ls(detail=True)``.
        """
        fs, path = _url_to_fs(str(self))
        return [StoragePath(_reattach_protocol(fs, child)) for child in fs.ls(path, detail=False)]

    def walk(self) -> "Generator[tuple[StoragePath, list[str], list[str]], None, None]":
        """Walk this tree top-down, yielding ``(dir, subdir_names, file_names)`` like ``os.walk``.

        ``dir`` is a reopenable :class:`StoragePath`; the name lists are plain strings, so
        a file is reached as ``dir / name``.
        """
        fs, path = _url_to_fs(str(self))
        for dirpath, dirnames, filenames in fs.walk(path):
            yield StoragePath(_reattach_protocol(fs, dirpath)), list(dirnames), list(filenames)

    def rm(self) -> None:
        fs, path = _url_to_fs(str(self))
        fs.rm(path)

    def rmtree(self) -> None:
        fs, path = _url_to_fs(str(self))
        fs.rm(path, recursive=True)

    def rename(self, target: "str | StoragePath") -> None:
        fs, path = _url_to_fs(str(self))
        fs.mv(path, str(target))

    def download_to(self, local_path: str, *, recursive: bool = False) -> None:
        """Copy this (remote) path down to ``local_path`` on the local disk."""
        fs, path = _url_to_fs(str(self))
        fs.get(path, local_path, recursive=recursive)

    def upload_from(self, local_path: str, *, recursive: bool = False) -> None:
        """Copy ``local_path`` from the local disk up to this (remote) path."""
        fs, path = _url_to_fs(str(self))
        fs.put(local_path, path, recursive=recursive)


def _reattach_protocol(fs: fsspec.AbstractFileSystem, path: str) -> str:
    """Re-attach ``fs``'s protocol to a bare ``path`` so it round-trips through ``url_to_fs``.

    ``fsspec`` glob/find results drop the protocol prefix (e.g. ``gs://``), which makes
    them ambiguous to reopen on a non-local filesystem. Local (``file``/protocol-less)
    paths and already-qualified paths are returned unchanged.
    """
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    if protocol in (None, "file"):
        return path
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


def prefix_join(prefix: str, relative: str) -> str:
    """Join a relative path onto a storage prefix with exactly one ``/`` separator.

    Object-store keys are not normalized: a naive ``f"{prefix}/{relative}"`` join of a
    trailing-slash prefix produces a doubled separator — a *different* key — silently
    splitting writers from slash-collapsing readers. ``str``-in/``str``-out
    convenience over :class:`StoragePath` for a single join; parse once and use ``/``
    for repeated manipulation.
    """
    return str(StoragePath.parse(prefix) / relative)


def split_gcs_path(gs_uri: str) -> tuple[str, pathlib.Path]:
    """Split a GCS URI into ``(bucket, Path(path/to/resource))``.

    Returns ``(bucket, Path("."))`` when the URI has no object path component.
    """
    parsed = StoragePath.parse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI `{gs_uri}`; expected URI of form `gs://BUCKET/path/to/resource`")

    key = parsed.key
    return parsed.bucket, pathlib.Path(key) if key else pathlib.Path(".")


def rebase_file_path(
    base_in_path: str,
    file_path: str,
    base_out_path: str,
    new_extension: str | None = None,
    old_extension: str | None = None,
) -> str:
    """Rebase ``file_path`` from under ``base_in_path`` to under ``base_out_path``.

    The path below ``base_in_path`` is preserved beneath ``base_out_path``, optionally
    swapping the file extension. Containment and joins are structural (via
    :class:`StoragePath`), so a trailing or doubled separator on any argument cannot
    double the output separator. ``file_path`` must lie under ``base_in_path``;
    otherwise a ``ValueError`` is raised.

    Args:
        base_in_path: The base directory of the input file.
        file_path: The path of the input file, under ``base_in_path``.
        base_out_path: The base directory of the output file.
        new_extension: New file extension including the dot (e.g. ``".parquet"``).
        old_extension: When given with ``new_extension``, the suffix of ``file_path`` to
            replace; a ``ValueError`` is raised if ``file_path`` does not end with it.
            When omitted (but ``new_extension`` is set), everything after the last dot is
            replaced; with no dot, ``new_extension`` is appended.
    """
    rel_path = StoragePath.parse(file_path).relative_to(StoragePath.parse(base_in_path))

    if old_extension and not new_extension:
        raise ValueError("old_extension requires new_extension to be set")

    if new_extension:
        if old_extension:
            # endswith (not rfind) so a mismatch fails loudly instead of silently
            # truncating: rfind returns -1 and rel_path[:-1] would drop a character.
            if not rel_path.endswith(old_extension):
                raise ValueError(
                    f"Cannot rebase {file_path!r}: relative path {rel_path!r} does not end with "
                    f"old_extension={old_extension!r}"
                )
            rel_path = rel_path[: -len(old_extension)] + new_extension
        else:
            dot_idx = rel_path.rfind(".")
            rel_path = (rel_path[:dot_idx] if dot_idx != -1 else rel_path) + new_extension
    return prefix_join(base_out_path, rel_path)
