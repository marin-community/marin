# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guarded fsspec entry points and atomic write-and-rename.

``url_to_fs``, ``open_url``, and ``filesystem`` are drop-in replacements for
``fsspec.core.url_to_fs``, ``fsspec.open``, and ``fsspec.filesystem`` that
automatically wrap GCS filesystems in a :class:`CrossRegionGuardedFS` and inject
finite botocore timeouts into S3/R2 filesystems (#6487). ``atomic_rename``
provides write-and-rename semantics via a sibling temp key, and
``fetch_file_atomic`` downloads a remote file to a local path the same way.
"""

import contextlib
import logging
import os
import uuid
from collections.abc import Generator
from typing import Any, cast

import fsspec
from fsspec.implementations.local import LocalFileSystem

from rigging.filesystem.cross_region import (
    CrossRegionGuardedFS,
    _fs_is_gcs,
    _is_gcs_protocol,
    _is_gcs_url,
)
from rigging.timing import ExponentialBackoff, retry_with_backoff

logger = logging.getLogger(__name__)

# Finite botocore timeouts/retries for every S3/R2 filesystem we build.
# s3fs/aiobotocore default to *no* read or connect timeout, so a silently dead
# R2 connection wedges ``upload_part`` forever (#6487): the blocked socket never
# raises, the shard never completes, and the sequential stage barrier stalls the
# whole job. With finite timeouts the wedge becomes a retryable error that fails
# the shard, which the coordinator then re-queues.
_S3_CONNECT_TIMEOUT = 30
_S3_READ_TIMEOUT = 120
_S3_RETRY_MAX_ATTEMPTS = 5


def _with_s3_timeout_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject finite botocore timeouts/retries into S3 filesystem kwargs.

    Caller-supplied ``config_kwargs`` values win; we only fill in keys the
    caller did not set. See :data:`_S3_READ_TIMEOUT` and #6487.

    We seed ``config_kwargs`` from the ``FSSPEC_S3`` config block first. fsspec
    builds the filesystem by shallow-merging ``{**conf, **kwargs}``, so a bare
    ``config_kwargs`` here would *replace* (not merge with) any ``config_kwargs``
    in ``FSSPEC_S3`` -- silently dropping settings like
    ``{"s3": {"addressing_style": "virtual"}}`` that S3-compatible endpoints
    (CoreWeave object storage) require, which then hangs/path-style-rejects.
    """
    conf_config_kwargs = (fsspec.config.conf.get("s3") or {}).get("config_kwargs") or {}
    config_kwargs = {**conf_config_kwargs, **dict(kwargs.get("config_kwargs") or {})}
    config_kwargs.setdefault("connect_timeout", _S3_CONNECT_TIMEOUT)
    config_kwargs.setdefault("read_timeout", _S3_READ_TIMEOUT)
    config_kwargs.setdefault("retries", {"max_attempts": _S3_RETRY_MAX_ATTEMPTS, "mode": "standard"})
    return {**kwargs, "config_kwargs": config_kwargs}


def url_to_fs(url: str, **kwargs: Any) -> tuple[Any, str]:
    """Like ``fsspec.core.url_to_fs`` but wraps GCS filesystems in a cross-region guard.

    Returns ``(fs, path)``.  For non-GCS URLs the filesystem is returned
    unwrapped.  ``mirror://`` URLs are handled by :class:`MirrorFileSystem`.
    S3/R2 URLs get finite timeouts injected (#6487).
    """
    if url.startswith("s3://"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if _fs_is_gcs(fs):
        fs = CrossRegionGuardedFS(fs)
    return fs, path


def is_remote_path(path: str) -> bool:
    """True if ``path`` resolves to a remote filesystem (e.g. ``gs://``, ``s3://``) rather than the
    local disk. A bare path or ``file://`` URL is local; anything with a remote scheme is not."""
    fs, _ = url_to_fs(path)
    return not isinstance(fs, LocalFileSystem)


def open_url(url: str, mode: str = "rb", **kwargs: Any) -> fsspec.core.OpenFile:
    """Like ``fsspec.open`` but checks the cross-region budget for GCS reads.

    For read modes on GCS URLs, eagerly stats the file and charges the
    transfer budget.  Then delegates to ``fsspec.open`` for the actual I/O.
    """
    if "r" in mode and _is_gcs_url(url):
        fs, path = fsspec.core.url_to_fs(url)
        guarded = CrossRegionGuardedFS(fs)
        guarded._guard_read(path)
    if url.startswith("s3://"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    return cast(fsspec.core.OpenFile, fsspec.open(url, mode, **kwargs))


def filesystem(protocol: str, **kwargs: Any) -> Any:
    """Like ``fsspec.filesystem`` but wraps GCS filesystems in a cross-region guard.

    S3/R2 filesystems get finite timeouts injected (#6487)."""
    if protocol in ("s3", "s3a"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    fs = fsspec.filesystem(protocol, **kwargs)
    if _is_gcs_protocol(protocol):
        fs = CrossRegionGuardedFS(fs)
    return fs


# ---------------------------------------------------------------------------
# Atomic write-and-rename
# ---------------------------------------------------------------------------


def unique_temp_path(output_path: str) -> str:
    """Return a unique temporary path derived from ``output_path``.

    Appends ``.tmp.<uuid>`` to avoid collisions when multiple writers target the
    same output path (e.g. during network-partition induced worker races).
    """
    return f"{output_path}.tmp.{uuid.uuid4().hex}"


# AWS error codes that are safe to retry on a server-side multipart copy
# (``s3fs.S3FileSystem.mv``). ``InvalidPart`` is the R2-specific symptom:
# every ``UploadPartCopy`` returns 200 but ``CompleteMultipartUpload`` then
# claims one or more parts are missing.
_TRANSIENT_S3_ERROR_CODES = frozenset(
    {
        "InvalidPart",
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "RequestTimeout",
        "RequestTimeTooSkewed",
    }
)

# Fragments matched against ``str(exc)`` for the case where s3fs has already
# translated the underlying ``botocore.ClientError`` into an ``OSError`` and
# the structured error code is no longer reachable.
_TRANSIENT_S3_MESSAGE_FRAGMENTS = (
    "specified parts could not be found",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
)


def _is_transient_s3_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_S3_ERROR_CODES:
            return True
    msg = str(exc)
    return any(frag in msg for frag in _TRANSIENT_S3_MESSAGE_FRAGMENTS)


def _mv_with_retry(fs: Any, src: str, dst: str) -> None:
    retry_with_backoff(
        lambda: fs.mv(src, dst, recursive=True),
        retryable=_is_transient_s3_error,
        max_attempts=4,
        backoff=ExponentialBackoff(initial=1.0, maximum=8.0, factor=2.0),
        operation=f"atomic_rename fs.mv {src} -> {dst}",
    )


@contextlib.contextmanager
def atomic_rename(output_path: str, fs: Any = None) -> Generator[str, None, None]:
    """Atomic write-and-rename via a sibling temp key.

    Yields ``<output_path>.tmp.<uuid>``; on clean exit, ``fs.mv`` renames the
    temp into the final path. On exception, the temp key is best-effort
    deleted and the original exception re-raised.

    Callers may pass a pre-constructed ``fs`` to reuse a configured
    filesystem (e.g. an ``S3FileSystem`` with ``fixed_upload_size=True``)
    instead of letting atomic_rename build a default one from ``output_path``.

    Example:
        with atomic_rename("output.jsonl.gz") as tmp_path:
            write_data(tmp_path)
        # File is now at output.jsonl.gz
    """
    temp_path = unique_temp_path(output_path)
    if fs is None:
        fs = url_to_fs(output_path)[0]
    try:
        yield temp_path
        _mv_with_retry(fs, temp_path, output_path)
    except Exception:
        # Best-effort cleanup: temp file may not exist (writer crashed before
        # creating it) so we tolerate any rm error and re-raise the original.
        with contextlib.suppress(Exception):
            fs.rm(temp_path)
        raise


def fetch_file_atomic(src_url: str, dest_path: str) -> bool:
    """Fetch ``src_url`` down to local ``dest_path`` atomically via a unique temp sibling.

    ``dest_path`` never holds a partial file — a concurrent reader sees the previous
    complete file or the new one, and a fetch killed midway leaves only an orphaned temp
    file instead of poisoning the destination (e.g. a shared cache). The unique temp name
    keeps concurrent fetches of the same destination from clobbering each other's
    in-flight files.

    Returns ``False`` if the source does not exist; re-raises all other errors.
    """
    tmp = unique_temp_path(dest_path)
    try:
        with open_url(src_url, "rb") as src:
            data = src.read()
        with open(tmp, "wb") as dst:
            dst.write(data)
        os.replace(tmp, dest_path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
