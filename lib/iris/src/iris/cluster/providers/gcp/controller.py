# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GcpControllerProvider — controller lifecycle + connectivity for GCP.

Implements the ControllerProvider protocol: discover, start, restart, stop,
stop_all, tunnel, resolve_image, debug_report.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

from rigging.timing import ExponentialBackoff, retry_with_backoff

from iris.cluster.controller.vm_lifecycle import restart_controller as vm_restart_controller
from iris.cluster.controller.vm_lifecycle import start_controller as vm_start_controller
from iris.cluster.controller.vm_lifecycle import stop_controller as vm_stop_controller
from iris.cluster.providers.gcp.service import GcpService
from iris.cluster.providers.gcp.ssh import (
    auth_mode_from_label,
    auth_mode_to_label,
    ssh_impersonate_service_account,
    ssh_key_file,
)
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.remote_exec import resolve_current_os_login_user
from iris.cluster.providers.types import (
    InfraError,
    Labels,
    default_stop_all,
    find_free_port,
    wait_for_port,
)
from iris.cluster.service_mode import ServiceMode
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

_FALLBACK_SSH_ERROR_MARKERS = (
    "permission denied",
    "os login",
    "not in the sudoers",
    "requested access to the resource is denied",
    "could not fetch resource",
    "login profile",
)


@dataclass
class GcpControllerProvider:
    """Controller lifecycle for GCP (and LOCAL mode), wrapping a GcpWorkerProvider."""

    worker_provider: GcpWorkerProvider
    controller_service_account: str | None = None

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller by querying GCP for labeled controller VM.

        In LOCAL mode, returns the configured address directly without querying GCP.
        """
        gcp = controller_config.gcp
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

    def start_controller(self, config: config_pb2.IrisClusterConfig, *, fresh: bool = False) -> str:
        address, _vm = vm_start_controller(
            self.worker_provider,
            config,
            resolve_image=self.worker_provider.resolve_image,
            fresh=fresh,
        )
        return address

    def restart_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        address, _vm = vm_restart_controller(
            self.worker_provider,
            config,
            resolve_image=self.worker_provider.resolve_image,
        )
        return address

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        vm_stop_controller(self.worker_provider, config)

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
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


def _ssh_auth_mode(ssh_config: config_pb2.SshConfig | None) -> int:
    if ssh_config is None:
        return config_pb2.SshConfig.SSH_AUTH_MODE_METADATA
    return ssh_config.auth_mode or config_pb2.SshConfig.SSH_AUTH_MODE_METADATA


def resolve_controller_ssh_config(
    *,
    gcp_service: GcpService,
    platform_config: config_pb2.PlatformConfig,
    cluster_config: config_pb2.IrisClusterConfig | None,
    local_ssh_config: config_pb2.SshConfig | None,
) -> config_pb2.SshConfig | None:
    """Resolve the SSH auth mode the cluster's controller advertises.

    Reads ``Labels.iris_ssh_auth_mode`` off the controller VM via a cheap
    label-filtered ``vm_list`` call. If the controller advertises a different
    mode than the local YAML, returns a new SshConfig with the advertised
    auth_mode swapped in. Otherwise returns ``local_ssh_config`` unchanged.

    Returns silently in cases where there's no controller to read from:
    LOCAL service mode, missing controller.gcp.zone, no controller VM yet
    (e.g. ``cluster start``), or any GCP API error. The local config is the
    authority in those cases.
    """
    if gcp_service.mode == ServiceMode.LOCAL:
        return local_ssh_config
    if cluster_config is None or cluster_config.controller.WhichOneof("controller") != "gcp":
        return local_ssh_config
    zone = cluster_config.controller.gcp.zone
    if not zone:
        return local_ssh_config

    label_prefix = platform_config.label_prefix or "iris"
    labels = Labels(label_prefix)

    try:
        vms = gcp_service.vm_list(zones=[zone], labels={labels.iris_controller: "true"})
    except InfraError as e:
        logger.debug("ssh auth_mode probe: vm_list failed (%s); using local config", e)
        return local_ssh_config

    if not vms:
        return local_ssh_config

    advertised = vms[0].labels.get(labels.iris_ssh_auth_mode) if vms[0].labels else None
    if not advertised:
        return local_ssh_config
    advertised_mode = auth_mode_from_label(advertised)
    if advertised_mode is None:
        logger.warning("Controller VM advertises unknown ssh auth_mode=%r; using local config", advertised)
        return local_ssh_config

    local_mode = _ssh_auth_mode(local_ssh_config)
    if advertised_mode == local_mode:
        return local_ssh_config

    resolved = config_pb2.SshConfig()
    if local_ssh_config is not None:
        resolved.CopyFrom(local_ssh_config)
    resolved.auth_mode = advertised_mode
    logger.info(
        "using ssh auth_mode=%s (from controller label; local config: %s)",
        auth_mode_to_label(advertised_mode),
        auth_mode_to_label(local_mode),
    )
    return resolved


def _should_retry_metadata(stderr: str) -> bool:
    normalized = stderr.lower()
    return any(marker in normalized for marker in _FALLBACK_SSH_ERROR_MARKERS)


def _build_tunnel_ssh_cmd(
    *,
    project: str,
    zone: str,
    vm_name: str,
    local_port: int,
    ssh_config: config_pb2.SshConfig | None,
    effective_service_account: str | None,
    force_metadata: bool = False,
) -> list[str]:
    auth_mode = _ssh_auth_mode(ssh_config)
    use_os_login = auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN and not force_metadata
    key_file = ssh_key_file(ssh_config, effective_service_account)

    target = vm_name
    if use_os_login:
        os_login_user = ssh_config.os_login_user or resolve_current_os_login_user(
            impersonate_service_account=effective_service_account
        )
        target = f"{os_login_user}@{vm_name}"

    cmd = [
        "gcloud",
        "compute",
        "ssh",
        target,
        f"--project={project}",
        f"--zone={zone}",
    ]
    if effective_service_account:
        cmd.append(f"--impersonate-service-account={effective_service_account}")
    if key_file:
        cmd.append(f"--ssh-key-file={key_file}")
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
    ssh_config: config_pb2.SshConfig | None,
    effective_service_account: str | None,
    timeout: float,
) -> subprocess.Popen:
    """Start an SSH tunnel subprocess and wait for the port to be ready.

    Returns the running Popen; raises RuntimeError on failure.
    """
    auth_mode = _ssh_auth_mode(ssh_config)
    cmd = _build_tunnel_ssh_cmd(
        project=project,
        zone=zone,
        vm_name=vm_name,
        local_port=local_port,
        ssh_config=ssh_config,
        effective_service_account=effective_service_account,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    if auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN and not wait_for_port(
        local_port, host="127.0.0.1", timeout=timeout
    ):
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.terminate()
        proc.wait()
        if _should_retry_metadata(stderr):
            logger.warning("OS Login tunnel failed; retrying controller tunnel with metadata SSH fallback")
            cmd = _build_tunnel_ssh_cmd(
                project=project,
                zone=zone,
                vm_name=vm_name,
                local_port=local_port,
                ssh_config=ssh_config,
                effective_service_account=effective_service_account,
                force_metadata=True,
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        else:
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

    if not wait_for_port(local_port, host="127.0.0.1", timeout=timeout):
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.terminate()
        proc.wait()
        raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

    return proc


@contextmanager
def _gcp_tunnel(
    project: str,
    label_prefix: str,
    ssh_config: config_pb2.SshConfig | None,
    local_port: int | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """SSH tunnel to the controller VM, yielding the local URL.

    Binds explicitly to 127.0.0.1 to avoid conflicts with other processes
    that may be listening on the same port on a different address family (IPv6).
    Picks a free port automatically if none is specified.
    """
    effective_service_account = ssh_impersonate_service_account(ssh_config)
    key_file = ssh_key_file(ssh_config, effective_service_account)
    _check_gcloud_ssh_key(key_file)

    if local_port is None:
        local_port = find_free_port(start=10000)

    labels = Labels(label_prefix)
    label_filter = f"labels.{labels.iris_controller}=true AND status=RUNNING"
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter={label_filter}",
        "--format=value(name,zone)",
        "--limit=1",
    ]
    if effective_service_account:
        cmd.append(f"--impersonate-service-account={effective_service_account}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"No controller VM found (label={labels.iris_controller}=true, project={project})")

    parts = result.stdout.strip().split()
    vm_name = parts[0]
    zone = parts[1] if len(parts) > 1 else ""

    logger.info("Establishing SSH tunnel to %s (zone=%s)...", vm_name, zone)

    proc = retry_with_backoff(
        lambda: _establish_tunnel(
            project=project,
            zone=zone,
            vm_name=vm_name,
            local_port=local_port,
            ssh_config=ssh_config,
            effective_service_account=effective_service_account,
            timeout=timeout,
        ),
        retryable=lambda e: isinstance(e, RuntimeError) and _is_transient_ssh_error(str(e)),
        max_attempts=3,
        backoff=ExponentialBackoff(initial=5.0, maximum=30.0, factor=2.0),
        operation=f"SSH tunnel to {vm_name}",
    )

    try:
        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:10000", local_port, vm_name)
        yield f"http://127.0.0.1:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
