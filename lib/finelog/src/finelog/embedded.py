# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process finelog server, backed by the native Rust extension.

``EmbeddedServer`` boots the same axum app the ``finelog-server`` binary serves,
on an owned tokio runtime bound to a local port. Callers then talk to it over
the normal RPC contract (:class:`finelog.client.LogClient` / proxies), so there
is exactly one server implementation. Iris's controller uses it as the local
log-server fallback when no external ``/system/log-server`` endpoint is set.

The extension ships inside the ``marin-finelog`` wheel. If it is missing (e.g. a
stale pure-Python install, or a source checkout that has not been built),
``EmbeddedServer`` is ``None`` and :func:`require_embedded_server` raises an
actionable error; :func:`is_available` lets callers probe without importing.
"""

try:
    from finelog._native import EmbeddedServer
except ImportError:
    EmbeddedServer = None  # type: ignore[assignment,misc]

_INSTALL_MSG = (
    "finelog native extension (finelog._native) is not available. It ships in "
    "the marin-finelog wheel; install the pre-built wheel, or build from source "
    "with 'python scripts/rust_mode.py dev && uv sync'."
)


def is_available() -> bool:
    """Whether the native in-process server extension is importable."""
    return EmbeddedServer is not None


def require_embedded_server() -> type:
    """Return :class:`EmbeddedServer`, raising a clear error if it is missing."""
    if EmbeddedServer is None:
        raise ImportError(_INSTALL_MSG)
    return EmbeddedServer
