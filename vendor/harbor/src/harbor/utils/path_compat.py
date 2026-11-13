"""Path-compatibility helpers for UPath / fsspec remote filesystems.

The job/trial config schemas use ``upath.UPath`` for ``jobs_dir`` /
``trials_dir`` so harbor can write artifacts to remote object stores
(gs://, s3://, …). Most of the codebase is already UPath-aware via
the pydantic field types, but stdlib ops (``shutil.rmtree``, ``open``,
``os.lstat``) only handle local files. This module provides
fsspec-routed equivalents that work for both local and remote UPaths.
"""

from __future__ import annotations

import shutil
from typing import Any


def _protocol(path: Any) -> str:
    """Return a UPath/PurePath's protocol, or '' for local paths."""
    proto = getattr(path, "protocol", "")
    if isinstance(proto, tuple):
        proto = proto[0] if proto else ""
    return proto


def is_remote(path: Any) -> bool:
    """True iff the path is a UPath with a remote scheme (gs, s3, az, …)."""
    return _protocol(path) not in ("", "file", "local")


def safe_rmtree(path: Any, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` that also handles remote UPath paths via fsspec.

    For local paths, delegates to ``shutil.rmtree`` (preserves existing
    semantics including ``ignore_errors``). For remote paths, uses the
    UPath's underlying fsspec filesystem to recursively remove the
    prefix.
    """
    if not is_remote(path):
        shutil.rmtree(str(path), ignore_errors=ignore_errors)
        return
    try:
        fs = path.fs
        fs.rm(str(path), recursive=True)
    except Exception:
        if not ignore_errors:
            raise
