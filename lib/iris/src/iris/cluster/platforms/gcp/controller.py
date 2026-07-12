# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GcpControllerProvider — controller lifecycle + connectivity for GCP.

Implements the ControllerProvider protocol: discover, start, restart, stop,
stop_all, tunnel, resolve_image, debug_report.
"""

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

from rigging.timing import ExponentialBackoff, retry_with_backoff

from iris.cluster.config import ControllerVmConfig, GcpControllerConfig, IrisClusterConfig, SshConfig
from iris.cluster.platforms.gcp.service import GcpService
from iris.cluster.platforms.gcp.ssh import ssh_impersonate_service_account
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.platforms.types import (
    Labels,
    default_stop_all,
    find_free_port,
    wait_for_port,
)
from iris.cluster.platforms.vm_lifecycle import restart_controller as vm_restart_controller
from iris.cluster.platforms.vm_lifecycle import start_controller as vm_start_controller
from iris.cluster.platforms.vm_lifecycle import stop_controller as vm_stop_controller
from iris.cluster.service_mode import ServiceMode

logger = logging.getLogger(__name__)


@dataclass
class GcpControllerProvider:
    """Controller lifecycle for GCP (and LOCAL mode), wrapping a GcpWorkerProvider."""

    worker_provider: GcpWorkerProvider
    controller_service_account: str | None = None

    def discover_controller(self, controller_config: ControllerVmConfig) -> str:
        """Discover controller by querying GCP for labeled controller VM.

        In LOCAL mode, returns the configured address directly without querying GCP.
        """
        gcp = controller_config.gcp or GcpControllerConfig()
        port = gcp.port or 10000

        if self.worker_provider.gcp_service.mode == ServiceMode.LOCAL:
            return f"localhost:{port}"

        vms = self.worker_provider.list_vms(
            zones=[gcp.zone],
            labels={self.worker_provider.iris_labels.iris_controller: "true"},
        )
        if not vms:
            raise RuntimeError(
                f"No controller VM found "
                f"(label={self.worker_provider.iris_labels.iris_controller}=true, "
                f"project={self.worker_provider.project_id})"
            )
        return f"{vms[0].internal_address}:{port}"

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
        if self.worker_provider.gcp_service.mode == ServiceMode.LOCAL:
            return nullcontext(address)
        return _gcp_tunnel(
            gcp_service=self.worker_provider.gcp_service,
            project=self.worker_provider.project_id,
            label_prefix=self.worker_provider.label_prefix,
            ssh_config=self.worker_provider.ssh_config,
            local_port=local_port,
        )

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return self.worker_provider.resolve_image(image, zone)

    def debug_report(self) -> None:
        pass

    def shutdown(self) -> None:
        self.worker_provider.shutdown()


# ============================================================================
# Tunnel
# ============================================================================


def _check_gcloud_ssh_key(key_file: str | None = None) -> None:
    """Verify that the gcloud compute SSH key exists.

    ``gcloud compute ssh`` expects an SSH key file. When the default key is
    key is missing, gcloud tries to generate one interactively — which hangs
    indefinitely in a non-interactive subprocess.
    """
    key_path = os.path.expanduser(key_file or "~/.ssh/google_compute_engine")
    if not os.path.exists(key_path):
        raise RuntimeError(
            f"SSH key not found at {key_path}. "
            "gcloud compute ssh requires this key to connect to VMs.\n"
            "To create it, run:\n"
            "  gcloud compute ssh --dry-run <any-vm> --zone=<zone>\n"
            "or:\n"
            "  ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -C \"$(gcloud config get account)\" -N ''\n"
            "Then re-run your command."
        )


def _build_tunnel_ssh_cmd(
    *,
    project: str,
    zone: str,
    vm_name: str,
    local_port: int,
    effective_service_account: str | None,
) -> list[str]:
    """Build a `gcloud compute ssh` tunnel command.

    No explicit user or key flag: gcloud auto-detects OS Login from the VM's
    enable-oslogin metadata and picks the right user/key itself.
    """
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        vm_name,
        f"--project={project}",
        f"--zone={zone}",
    ]
    if effective_service_account:
        cmd.append(f"--impersonate-service-account={effective_service_account}")
    cmd.extend(
        [
            "--",
            "-L",
            f"127.0.0.1:{local_port}:localhost:10000",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
        ]
    )
    return cmd


_TRANSIENT_SSH_ERROR_MARKERS = (
    "connection reset by peer",
    "connection refused",
    "connection timed out",
    "no route to host",
    "network is unreachable",
)


def _is_transient_ssh_error(error: str) -> bool:
    normalized = error.lower()
    return any(marker in normalized for marker in _TRANSIENT_SSH_ERROR_MARKERS)


def _establish_tunnel(
    *,
    project: str,
    zone: str,
    vm_name: str,
    local_port: int,
    effective_service_account: str | None,
    timeout: float,
) -> subprocess.Popen:
    """Open an SSH tunnel to the controller VM."""
    cmd = _build_tunnel_ssh_cmd(
        project=project,
        zone=zone,
        vm_name=vm_name,
        local_port=local_port,
        effective_service_account=effective_service_account,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    if not wait_for_port(local_port, host="127.0.0.1", timeout=timeout):
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.terminate()
        proc.wait()
        raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

    return proc


@contextmanager
def _gcp_tunnel(
    gcp_service: GcpService,
    project: str,
    label_prefix: str,
    ssh_config: SshConfig | None,
    local_port: int | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """SSH tunnel to the controller VM, yielding the local URL.

    Discovers the controller VM via the iris-controller label and shells out
    to `gcloud compute ssh`; gcloud auto-detects OS Login vs metadata SSH
    from the VM's own enable-oslogin metadata.
    """
    effective_service_account = ssh_impersonate_service_account(ssh_config)
    _check_gcloud_ssh_key()

    if local_port is None:
        local_port = find_free_port(start=10000)

    labels = Labels(label_prefix)
    vms = gcp_service.vm_list(zones=[], labels={labels.iris_controller: "true"})
    running = [vm for vm in vms if vm.status == "RUNNING"]
    if not running:
        raise RuntimeError(f"No controller VM found (label={labels.iris_controller}=true, project={project})")
    vm = running[0]

    logger.info("Establishing SSH tunnel to %s (zone=%s)...", vm.name, vm.zone)

    proc = retry_with_backoff(
        lambda: _establish_tunnel(
            project=project,
            zone=vm.zone,
            vm_name=vm.name,
            local_port=local_port,
            effective_service_account=effective_service_account,
            timeout=timeout,
        ),
        retryable=lambda e: isinstance(e, RuntimeError) and _is_transient_ssh_error(str(e)),
        max_attempts=3,
        backoff=ExponentialBackoff(initial=5.0, maximum=30.0, factor=2.0),
        operation=f"SSH tunnel to {vm.name}",
    )

    try:
        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:10000", local_port, vm.name)
        yield f"http://127.0.0.1:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
