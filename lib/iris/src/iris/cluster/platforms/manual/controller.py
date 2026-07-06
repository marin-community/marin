# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ManualControllerProvider for pre-existing hosts.

Implements ControllerProvider for manually managed hosts. Controller discovery
uses the static host from the ControllerVmConfig. Start/stop/restart use
vm_lifecycle.py which works uniformly across ManualWorkerProvider and
GcpWorkerProvider.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

from iris.cluster.config import ControllerVmConfig, IrisClusterConfig
from iris.cluster.platforms.manual.workers import ManualWorkerProvider
from iris.cluster.platforms.types import default_stop_all
from iris.cluster.platforms.vm_lifecycle import restart_controller as vm_restart_controller
from iris.cluster.platforms.vm_lifecycle import start_controller as vm_start_controller
from iris.cluster.platforms.vm_lifecycle import stop_controller as vm_stop_controller


@dataclass
class ManualControllerProvider:
    """Controller lifecycle for manually managed hosts, wrapping a ManualWorkerProvider.

    Implements ControllerProvider. Controller discovery uses the static host
    from the ControllerVmConfig. Start/stop/restart use vm_lifecycle.py which
    works uniformly across ManualWorkerProvider and GcpWorkerProvider.
    """

    worker_provider: ManualWorkerProvider

    def discover_controller(self, controller_config: ControllerVmConfig) -> str:
        manual = controller_config.manual
        port = manual.port or 10000
        return f"{manual.host}:{port}"

    def start_controller(self, config: IrisClusterConfig, *, fresh: bool = False) -> str:
        address, _vm = vm_start_controller(
            self.worker_provider,
            config,
            resolve_image=self.worker_provider.resolve_image,
            fresh=fresh,
        )
        return address

    def restart_controller(self, config: IrisClusterConfig) -> str:
        address, _vm = vm_restart_controller(
            self.worker_provider,
            config,
            resolve_image=self.worker_provider.resolve_image,
        )
        return address

    def stop_controller(self, config: IrisClusterConfig) -> None:
        vm_stop_controller(self.worker_provider, config)

    def stop_all(
        self,
        config: IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        # label_prefix is accepted for protocol compatibility but not yet wired to
        # list_all_slices filtering; the worker_provider always uses its own prefix.
        return default_stop_all(
            list_all_slices=self.worker_provider.list_all_slices,
            stop_controller=lambda: self.stop_controller(config),
            dry_run=dry_run,
        )

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(address)

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def debug_report(self) -> None:
        pass

    def shutdown(self) -> None:
        self.worker_provider.shutdown()
