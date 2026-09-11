# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Self

class EmbeddedServer:
    """An in-process finelog server backed by the native Rust implementation.

    Serves the same axum app as the ``finelog-server`` binary on a local port,
    over an owned tokio runtime. Talk to it over the normal RPC contract
    (``finelog.client.LogClient`` / proxies). Use as a context manager or call
    :meth:`stop`; the server also stops on garbage collection.
    """

    def __init__(
        self,
        log_dir: str | None = ...,
        remote_log_dir: str = ...,
        host: str = ...,
        port: int = ...,
        debug_admin: bool = ...,
        auth_policy: str | None = ...,
    ) -> None: ...
    @property
    def port(self) -> int:
        """The bound port (ephemeral when constructed with ``port=0``)."""

    @property
    def address(self) -> str:
        """Base URL, e.g. ``http://127.0.0.1:54321``."""

    def stop(self) -> None:
        """Trigger graceful shutdown and join the server. Idempotent."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...
