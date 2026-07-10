# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-region read guard and cumulative transfer budget.

``TransferBudget`` tracks cumulative cross-region GCS bytes across all filesystem
instances in the process (default 10 GB). Both :class:`CrossRegionGuardedFS`
(direct reads) and the mirror filesystem (mirror copies) charge against the same
global budget. Prefer the guarded helpers in :mod:`rigging.filesystem.factory`
(``url_to_fs``, ``open_url``, ``filesystem``) over the raw fsspec equivalents;
they automatically wrap GCS filesystems in the guard. Set the
``MARIN_I_WILL_PAY_FOR_ALL_FEES`` env var to override the guard.
"""

import contextlib
import contextvars
import functools
import logging
import os
import threading
from collections.abc import Callable, Generator
from typing import Any

from rigging.filesystem.cluster_config import get_bucket_location, marin_region
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

MARIN_CROSS_REGION_OVERRIDE_ENV: str = "MARIN_I_WILL_PAY_FOR_ALL_FEES"
MARIN_MIRROR_BUDGET_ENV: str = "MARIN_MIRROR_BUDGET_GB"
_DEFAULT_TRANSFER_LIMIT_GB: int = 10


def _transfer_limit_bytes() -> int:
    raw = os.environ.get(MARIN_MIRROR_BUDGET_ENV, "")
    if raw:
        return int(float(raw) * 1024 * 1024 * 1024)
    return _DEFAULT_TRANSFER_LIMIT_GB * 1024 * 1024 * 1024


CROSS_REGION_TRANSFER_LIMIT_BYTES: int = _transfer_limit_bytes()

# GCS multi-region bucket locations are returned as "us", "eu", or "asia"
# rather than a specific region like "us-central1".  European regions use the
# prefix "europe-" (e.g. "europe-west4") so we map the multi-region label to
# the set of region prefixes it covers.
_MULTI_REGION_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "us": ("us-",),
    "eu": ("europe-", "eu-"),
    "asia": ("asia-",),
}


class TransferBudgetExceeded(Exception):
    """Raised when cumulative cross-region bytes exceed the budget."""

    def __init__(self, bytes_used: int, attempted: int, limit: int, path: str):
        self.bytes_used = bytes_used
        self.attempted = attempted
        self.limit = limit
        self.path = path
        # Pass the constructor arguments — not the rendered message — to
        # BaseException. The default exception reduce reconstructs via
        # ``TransferBudgetExceeded(*self.args)`` on unpickle, so ``args`` must
        # match this signature; storing the single message string instead made
        # the exception un-revivable (``TypeError: missing 3 required positional
        # arguments``) whenever it crossed a process boundary. The human-readable
        # message is rendered lazily by ``__str__``.
        super().__init__(bytes_used, attempted, limit, path)

    def __str__(self) -> str:
        return (
            f"Cross-region transfer budget exceeded: {self.path} "
            f"({self.attempted / (1024**2):.1f}MB) would bring total to "
            f"{(self.bytes_used + self.attempted) / (1024**3):.2f}GB, "
            f"exceeding the {self.limit / (1024**3):.0f}GB limit "
            f"(already transferred {self.bytes_used / (1024**3):.2f}GB). "
            f"Consider running in the source region instead."
        )


class TransferBudget:
    """Thread-safe cumulative byte budget for cross-region transfers.

    Shared by CrossRegionGuardedFS (direct reads) and MirrorFileSystem
    (mirror copies).  A single process-global instance tracks total
    cross-region bytes across all filesystem instances.
    """

    __slots__ = ("_bytes_used", "_limit_bytes", "_lock")

    def __init__(self, limit_bytes: int = CROSS_REGION_TRANSFER_LIMIT_BYTES):
        self._limit_bytes = limit_bytes
        self._bytes_used: int = 0
        self._lock = threading.Lock()

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    def record(self, size: int, path: str) -> None:
        """Atomically record *size* bytes.  Raise if budget exceeded.

        Does NOT increment on failure — the transfer hasn't happened yet.
        """
        with self._lock:
            new_total = self._bytes_used + size
            if new_total > self._limit_bytes:
                raise TransferBudgetExceeded(self._bytes_used, size, self._limit_bytes, path)
            self._bytes_used = new_total

    def reset(self, limit_bytes: int | None = None) -> None:
        """Reset counter to zero.  For testing only."""
        with self._lock:
            self._bytes_used = 0
            if limit_bytes is not None:
                self._limit_bytes = limit_bytes


_global_transfer_budget = TransferBudget()

_mirror_budget_ctx: contextvars.ContextVar[TransferBudget | None] = contextvars.ContextVar(
    "_mirror_budget_ctx", default=None
)


def set_mirror_budget(budget_gb: float) -> contextvars.Token:
    """Set the MirrorFileSystem transfer budget for the current context.

    Returns a token that can be used to reset the budget.
    """
    budget = TransferBudget(limit_bytes=int(budget_gb * 1024 * 1024 * 1024))
    return _mirror_budget_ctx.set(budget)


def reset_mirror_budget(token: contextvars.Token) -> None:
    """Reset the MirrorFileSystem transfer budget to its previous value."""
    _mirror_budget_ctx.reset(token)


@contextlib.contextmanager
def mirror_budget(budget_gb: float) -> Generator[None, None, None]:
    """Context manager to scope a MirrorFileSystem transfer budget."""
    token = set_mirror_budget(budget_gb)
    try:
        yield
    finally:
        reset_mirror_budget(token)


@functools.lru_cache(maxsize=1)
def cached_marin_region() -> str | None:
    """Return the current VM region, cached for the process lifetime (the VM region is stable)."""
    return marin_region()


@functools.lru_cache(maxsize=256)
def _cached_bucket_location(bucket_name: str) -> str | None:
    """Return the location of a GCS bucket, cached across calls."""
    try:
        return get_bucket_location(bucket_name)
    except Exception:
        logger.debug("Could not determine location for bucket %s", bucket_name, exc_info=True)
        return None


def _regions_match(vm_region: str, bucket_location: str) -> bool:
    """Return True if *vm_region* and *bucket_location* are the same region.

    Handles GCS multi-region buckets whose location is ``"us"``, ``"eu"``,
    or ``"asia"`` rather than a specific zone.
    """
    vm = vm_region.lower()
    bl = bucket_location.lower()
    if vm == bl:
        return True
    prefixes = _MULTI_REGION_TO_PREFIXES.get(bl)
    if prefixes is not None:
        return any(vm.startswith(p) for p in prefixes)
    return False


def _fs_is_gcs(fs: Any) -> bool:
    """Return True if *fs* is a GCS-backed fsspec filesystem."""
    proto = getattr(fs, "protocol", None)
    if isinstance(proto, tuple):
        return "gs" in proto or "gcs" in proto
    return proto in ("gs", "gcs")


def _is_gcs_url(url: str) -> bool:
    """Return True if *url* starts with a GCS scheme."""
    return url.startswith("gs://") or url.startswith("gcs://")


def _is_gcs_protocol(protocol: str) -> bool:
    """Return True if *protocol* names a GCS filesystem."""
    return protocol in ("gs", "gcs")


def _bucket_from_gcs_url(url: str) -> str | None:
    """Return the bucket name from a ``gs://``/``gcs://`` URL, or ``None``."""
    parsed = StoragePath.parse(url)
    if parsed.scheme in ("gs", "gcs"):
        return parsed.bucket
    return None


def _is_cross_region_url(url: str) -> bool:
    """Return True if *url* points to a GCS bucket in a different region than the VM."""
    if os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV):
        return False
    bucket = _bucket_from_gcs_url(url)
    if bucket is None:
        return False
    vm_region = cached_marin_region()
    if vm_region is None:
        return False
    bucket_location = _cached_bucket_location(bucket)
    if bucket_location is None:
        return False
    return not _regions_match(vm_region, bucket_location)


def is_cross_region_url(url: str) -> bool:
    """Return True if reading *url* would cross regions and be charged to the budget.

    Cheap: only cached region lookups, no listing or stat.  Callers can use this
    to skip an expensive size computation when a read would not be charged
    anyway (local paths, same-region buckets, unknown VM region, override set).
    """
    return _is_cross_region_url(url)


def record_transfer(size: int, url: str, *, budget: TransferBudget | None = None) -> None:
    """Charge *size* bytes against the cross-region transfer budget.

    Always safe to call: no-op for non-GCS URLs, same-region buckets, when the
    VM region is unknown, or when the override env var is set.  Raises
    :class:`TransferBudgetExceeded` if the recorded transfer would push the
    cumulative total past the budget.

    Used by callers (e.g. tensorstore-based code) that bypass fsspec but still
    want to charge against the shared cross-region transfer budget.

    Args:
        size: Number of bytes to charge.
        url: GCS URL (``gs://bucket/key``) being read or written.  Used both
            to decide whether the transfer is cross-region and as the path
            string in any raised :class:`TransferBudgetExceeded`.
        budget: Budget to charge against.  Defaults to the process-global
            singleton shared with :class:`CrossRegionGuardedFS` and
            :class:`MirrorFileSystem`.
    """
    if size <= 0:
        return
    if not _is_cross_region_url(url):
        return
    (budget if budget is not None else _global_transfer_budget).record(size, url)


class CrossRegionGuardedFS:
    """Wrapper around a GCS fsspec filesystem that enforces a cross-region transfer budget.

    Intercepts read operations (``open``, ``cat``, ``cat_file``, ``get_file``,
    ``get``) and records each cross-region read against a shared
    ``TransferBudget``.  Raises ``TransferBudgetExceeded`` when the cumulative
    cross-region bytes exceed the budget.

    Only constructed for GCS filesystems — the entry points (``url_to_fs``,
    ``open_url``, ``filesystem``) decide whether to wrap.

    Args:
        fs: The GCS fsspec filesystem to wrap.
        cross_region_checker: Optional callback ``(bucket_name) -> bool``
            used **only** for testing.  When provided, bypasses the default
            region-comparison logic.
        budget: Transfer budget to charge reads against.  Defaults to the
            process-global singleton.
    """

    __slots__ = ("_budget", "_cross_region_checker", "_current_region", "_fs")

    def __init__(
        self,
        fs: Any,
        *,
        cross_region_checker: Callable[[str], bool] | None = None,
        budget: TransferBudget | None = None,
    ):
        self._fs = fs
        self._cross_region_checker = cross_region_checker
        self._current_region = None if cross_region_checker else cached_marin_region()
        self._budget = budget if budget is not None else _global_transfer_budget

    # -- cross-region detection ----------------------------------------------

    def _is_cross_region(self, bucket_name: str) -> bool:
        if self._cross_region_checker is not None:
            return self._cross_region_checker(bucket_name)
        if self._current_region is None:
            return False
        bucket_location = _cached_bucket_location(bucket_name)
        if bucket_location is None:
            return False
        return not _regions_match(self._current_region, bucket_location)

    # -- read interception ---------------------------------------------------

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        if "r" in mode:
            self._guard_read(path)
        return self._fs.open(path, mode, **kwargs)

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        self._guard_read(path)
        return self._fs.cat_file(path, start=start, end=end, **kwargs)

    def cat(self, path: Any, recursive: bool = False, on_error: str = "raise", **kwargs: Any) -> Any:
        if isinstance(path, str):
            self._guard_read(path)
        elif isinstance(path, list):
            for p in path:
                self._guard_read(p)
        return self._fs.cat(path, recursive=recursive, on_error=on_error, **kwargs)

    def get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        self._guard_read(rpath)
        return self._fs.get_file(rpath, lpath, **kwargs)

    def get(self, rpath: Any, lpath: Any, recursive: bool = False, **kwargs: Any) -> None:
        """Guard each remote path before delegating the bulk download."""
        if isinstance(rpath, str):
            self._guard_read(rpath)
        elif isinstance(rpath, list):
            for p in rpath:
                self._guard_read(p)
        return self._fs.get(rpath, lpath, recursive=recursive, **kwargs)

    # -- guard logic ---------------------------------------------------------

    def _guard_read(self, path: str) -> None:
        if os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV):
            return

        # fsspec strips the protocol, so paths look like "bucket/key".
        bucket = path.split("/")[0] if "/" in path else path
        if not self._is_cross_region(bucket):
            return

        try:
            size = self._fs.size(path)
        except Exception:
            logger.warning("Failed to stat %s for cross-region guard check", path, exc_info=True)
            return

        if size is not None:
            self._budget.record(size, f"gs://{path}")

    # -- transparent delegation ----------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fs, name)
