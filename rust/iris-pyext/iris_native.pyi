# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Self

PROXY_DECISION_PATH: str
DECISION_SECRET_HEADER: str
UPSTREAM_URL_HEADER: str
UPSTREAM_AUTHORIZATION_HEADER: str
PROXY_PREFIX_HEADER: str
PROXY_TIMEOUT_HEADER: str
DEFAULT_PROXY_TIMEOUT_SECONDS: int
PROXY_METHODS: list[str]

class NativeProxy:
    """Rust-owned Iris public listener with a private Python control plane."""

    def __init__(
        self,
        public_host: str,
        public_port: int,
        controller_url: str,
        decision_secret: str,
        auth_config_json: str,
        worker_threads: int = ...,
    ) -> None: ...

    @property
    def port(self) -> int: ...

    @property
    def address(self) -> str: ...

    @property
    def stats_json(self) -> str: ...

    def replace_registry(self, snapshot_json: str) -> None: ...

    def pause_registry(self) -> None: ...

    def update_mappings(self, delta_json: str) -> None: ...

    def stop(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...
