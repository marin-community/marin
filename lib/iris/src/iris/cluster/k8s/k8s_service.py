# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sService protocol for decoupling from the concrete Kubectl implementation."""

from __future__ import annotations

import subprocess
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from iris.cluster.k8s.k8s_types import KubectlLogResult


@runtime_checkable
class K8sService(Protocol):
    """Protocol matching the public API of Kubectl.

    Consumers that only need high-level Kubernetes operations should depend on
    this protocol rather than the concrete Kubectl class, enabling test doubles
    that don't shell out to kubectl.
    """

    @property
    def namespace(self) -> str: ...

    def apply_json(self, manifest: dict) -> None: ...

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None: ...

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]: ...

    def delete(
        self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True
    ) -> None: ...

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str: ...

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        byte_offset: int = 0,
    ) -> KubectlLogResult: ...

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None: ...

    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None: ...

    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None: ...

    def get_events(
        self,
        field_selector: str | None = None,
    ) -> list[dict]: ...

    def top_pod(self, pod_name: str) -> tuple[int, int] | None: ...

    def read_file(
        self,
        pod_name: str,
        path: str,
        *,
        container: str | None = None,
    ) -> bytes: ...

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None: ...

    def port_forward(
        self,
        service_name: str,
        remote_port: int,
        local_port: int | None = None,
        timeout: float = 90.0,
    ) -> AbstractContextManager[str]:
        """Port-forward to a K8s Service, yielding the local URL.

        The returned context manager keeps the tunnel alive for its duration.
        Implementations may reconnect transparently on transient failures.
        """
        ...
