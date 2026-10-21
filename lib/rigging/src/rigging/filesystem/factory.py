# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guarded fsspec construction and open entry points.

``url_to_fs``, ``open_url``, and ``filesystem`` are drop-in replacements for
``fsspec.core.url_to_fs``, ``fsspec.open``, and ``fsspec.filesystem`` that
automatically wrap GCS filesystems in a :class:`CrossRegionGuardedFS` and inject
finite botocore timeouts into S3/R2 filesystems (#6487). Importing this module
also gives GCS and S3 directory listings a zero-second expiry unless fsspec was
explicitly configured otherwise.
"""

import logging
from typing import Any, cast

import fsspec
from fsspec.implementations.local import LocalFileSystem

from rigging.filesystem.cross_region import (
    CrossRegionGuardedFS,
    _fs_is_gcs,
    _is_gcs_protocol,
    _is_gcs_url,
)
from rigging.filesystem.listing_cache import configure_listing_cache_defaults
from rigging.filesystem.s3_compat import s3_python_config_kwargs

logger = logging.getLogger(__name__)

# Register by class path so importing the factory does not load the mirror's
# distributed-lock backends. fsspec imports the class on the first mirror:// use.
fsspec.register_implementation("mirror", "rigging.filesystem.mirror.MirrorFileSystem")

# fsspec has no constructor-default hook shared by raw and guarded entry points.
# Its environment/file config is loaded before this point, so explicit cache settings win.
configure_listing_cache_defaults()


def _with_s3_timeout_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject finite botocore timeouts/retries into S3 filesystem kwargs.

    Caller-supplied ``config_kwargs`` values win; we only fill in keys the
    caller did not set. See #6487.

    We seed ``config_kwargs`` from the ``FSSPEC_S3`` config block first. fsspec
    builds the filesystem by shallow-merging ``{**conf, **kwargs}``, so a bare
    ``config_kwargs`` here would *replace* (not merge with) any ``config_kwargs``
    in ``FSSPEC_S3`` -- silently dropping settings like
    ``{"s3": {"addressing_style": "virtual"}}`` that S3-compatible endpoints
    (CoreWeave object storage) require, which then hangs/path-style-rejects.
    """
    conf_config_kwargs = (fsspec.config.conf.get("s3") or {}).get("config_kwargs") or {}
    config_kwargs = {**conf_config_kwargs, **dict(kwargs.get("config_kwargs") or {})}
    for key, value in s3_python_config_kwargs().items():
        config_kwargs.setdefault(key, value)
    return {**kwargs, "config_kwargs": config_kwargs}


def url_to_fs(url: str, **kwargs: Any) -> tuple[Any, str]:
    """Like ``fsspec.core.url_to_fs`` but wraps GCS filesystems in a cross-region guard.

    Returns ``(fs, path)``.  For non-GCS URLs the filesystem is returned
    unwrapped.  ``mirror://`` URLs are handled by :class:`MirrorFileSystem`.
    GCS/S3 listings expire immediately by default, and S3/R2 URLs get finite
    timeouts injected (#6487).
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

    GCS/S3 listings expire immediately by default, and S3/R2 filesystems get
    finite timeouts injected (#6487)."""
    if protocol in ("s3", "s3a"):
        kwargs = _with_s3_timeout_defaults(kwargs)
    fs = fsspec.filesystem(protocol, **kwargs)
    if _is_gcs_protocol(protocol):
        fs = CrossRegionGuardedFS(fs)
    return fs
