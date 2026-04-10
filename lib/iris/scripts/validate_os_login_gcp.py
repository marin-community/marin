#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate Iris OS Login SSH paths against disposable GCP resources.

This script exercises the same Iris-managed GCP code paths we use in production:

1. Create a disposable standalone GCE VM with OS Login enabled.
2. Create a disposable TPU slice with OS Login enabled.
3. Validate:
   - native ``gcloud compute ssh`` to the GCE VM
   - native ``gcloud compute tpus tpu-vm ssh`` to TPU worker 0
   - direct ``ssh`` to the TPU worker IP using an OS Login key

It generates a short-lived SSH key, adds it to the caller's OS Login profile,
and removes it during cleanup.

Usage:
    cd lib/iris
    uv run python scripts/validate_os_login_gcp.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from iris.cluster.config import get_ssh_config, load_config
from iris.cluster.providers.factory import create_provider_bundle
from iris.cluster.providers.gcp.handles import GcpSliceHandle, GcpVmSliceHandle
from iris.cluster.providers.remote_exec import DirectSshRemoteExec, resolve_current_os_login_user
from iris.cluster.providers.types import CloudSliceState, SliceHandle
from iris.rpc import config_pb2
from rigging.timing import Duration

logger = logging.getLogger("validate-os-login-gcp")
IRIS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "marin-dev.yaml"


@dataclass
class CommandCheck:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class ValidationReport:
    config: str
    vm_group: str
    tpu_group: str
    controller_service_account: str
    worker_service_account: str
    controller_os_login_user: str
    worker_os_login_user: str
    vm_name: str
    tpu_slice_id: str
    gce_native_ssh: CommandCheck
    tpu_native_ssh: CommandCheck
    tpu_direct_ssh: CommandCheck

    @property
    def ok(self) -> bool:
        return self.gce_native_ssh.ok and self.tpu_native_ssh.ok and self.tpu_direct_ssh.ok


def _run(cmd: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        logger.info("stdout:\n%s", result.stdout.strip())
    if result.stderr.strip():
        logger.info("stderr:\n%s", result.stderr.strip())
    return result


def _choose_scale_group(
    config: config_pb2.IrisClusterConfig,
    *,
    want_vm: bool,
    explicit_name: str | None,
) -> str:
    if explicit_name:
        if explicit_name not in config.scale_groups:
            raise ValueError(f"Scale group {explicit_name!r} not found in config")
        return explicit_name

    candidates: list[tuple[int, str]] = []
    for name, group in config.scale_groups.items():
        if not group.HasField("slice_template") or not group.slice_template.HasField("gcp"):
            continue
        is_vm_group = group.slice_template.gcp.mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
        if is_vm_group != want_vm:
            continue
        priority = group.priority if group.HasField("priority") else 100
        candidates.append((priority, name))

    if not candidates:
        kind = "VM-backed" if want_vm else "TPU-backed"
        raise ValueError(f"No {kind} GCP scale group found in config")

    candidates.sort()
    return candidates[0][1]


def _make_temp_ssh_key(ttl: str) -> tuple[Path, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="iris-oslogin-"))
    private_key = temp_dir / "google_compute_engine"
    comment = f"iris-oslogin-{int(time.time())}"

    _run(["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", str(private_key), "-N", "", "-q", "-C", comment])
    return private_key, comment


def _os_login_key_command(
    action: str,
    *,
    private_key: Path,
    service_account: str,
    ttl: str | None = None,
) -> list[str]:
    pub_path = private_key.with_suffix(private_key.suffix + ".pub")
    cmd = [
        "gcloud",
        "compute",
        "os-login",
        "ssh-keys",
        action,
        f"--key-file={pub_path}",
        f"--impersonate-service-account={service_account}",
    ]
    if ttl:
        cmd.append(f"--ttl={ttl}")
    return cmd


def _add_temp_ssh_key(private_key: Path, *, service_account: str, ttl: str) -> None:
    _run(
        _os_login_key_command(
            "add",
            private_key=private_key,
            service_account=service_account,
            ttl=ttl,
        )
    ).check_returncode()


def _remove_temp_ssh_key(private_key: Path, *, service_account: str) -> None:
    pub_path = private_key.with_suffix(private_key.suffix + ".pub")
    try:
        _run(_os_login_key_command("remove", private_key=private_key, service_account=service_account), timeout=30)
    except Exception as exc:
        logger.warning("Failed to remove OS Login key %s: %s", pub_path, exc)


def _wait_for_vm_connection(vm_handle, timeout: float = 600.0) -> None:
    if not vm_handle.wait_for_connection(timeout=Duration.from_seconds(timeout)):
        raise RuntimeError(f"VM {vm_handle.vm_id} did not become reachable within {timeout}s")


def _wait_for_tpu_ready(handle: SliceHandle, timeout: float = 900.0) -> tuple[GcpSliceHandle | GcpVmSliceHandle, str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = handle.describe()
        if status.state == CloudSliceState.READY and status.workers:
            worker = status.workers[0]
            preferred_address = worker.external_address or worker.internal_address
            if preferred_address:
                return handle, preferred_address
        time.sleep(10)
    raise RuntimeError(f"TPU slice {handle.slice_id} did not become READY within {timeout}s")


def _command_check(name: str, cmd: list[str], *, timeout: float = 120.0) -> CommandCheck:
    result = _run(cmd, timeout=timeout)
    return CommandCheck(
        name=name,
        command=cmd,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Iris cluster config to use")
    parser.add_argument("--vm-group", default=None, help="Explicit VM-backed scale group to use")
    parser.add_argument("--tpu-group", default=None, help="Explicit TPU-backed scale group to use")
    parser.add_argument("--ttl", default="2h", help="TTL for the temporary OS Login SSH key")
    parser.add_argument("--keep-resources", action="store_true", help="Skip deleting the disposable VM and TPU slice")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    cluster_config = load_config(args.config)
    vm_group = _choose_scale_group(cluster_config, want_vm=True, explicit_name=args.vm_group)
    tpu_group = _choose_scale_group(cluster_config, want_vm=False, explicit_name=args.tpu_group)
    tpu_capacity = cluster_config.scale_groups[tpu_group].resources.capacity_type
    if tpu_capacity != config_pb2.CAPACITY_TYPE_PREEMPTIBLE:
        raise ValueError(f"Scale group {tpu_group!r} must use capacity_type=preemptible for this validator")

    private_key, _comment = _make_temp_ssh_key(args.ttl)
    controller_service_account = cluster_config.controller.gcp.service_account
    worker_service_account = cluster_config.scale_groups[tpu_group].slice_template.gcp.service_account
    ssh_service_account = cluster_config.defaults.ssh.impersonate_service_account or controller_service_account
    if not controller_service_account:
        raise ValueError("controller.gcp.service_account is required for OS Login validation")
    if not worker_service_account:
        raise ValueError(f"Scale group {tpu_group!r} is missing slice_template.gcp.service_account")
    if not ssh_service_account:
        raise ValueError("defaults.ssh.impersonate_service_account or controller.gcp.service_account is required")

    vm_handle = None
    tpu_handle = None
    try:
        _add_temp_ssh_key(private_key, service_account=ssh_service_account, ttl=args.ttl)

        controller_os_login_user = resolve_current_os_login_user(impersonate_service_account=ssh_service_account)
        worker_os_login_user = resolve_current_os_login_user(impersonate_service_account=ssh_service_account)

        config = copy.deepcopy(cluster_config)
        config.defaults.ssh.auth_mode = config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN
        config.defaults.ssh.key_file = str(private_key)

        ssh_config = get_ssh_config(config)
        bundle = create_provider_bundle(
            platform_config=config.platform,
            cluster_config=config,
            ssh_config=ssh_config,
        )
        worker_provider = bundle.workers
        if worker_provider is None:
            raise RuntimeError("Expected a worker provider for GCP config")

        vm_group_config = config.scale_groups[vm_group]
        vm_zone = vm_group_config.slice_template.gcp.zone
        vm_name = f"iris-oslogin-check-{int(time.time())}"

        vm_config = config_pb2.VmConfig(name=vm_name)
        vm_config.gcp.zone = vm_zone
        vm_config.gcp.machine_type = vm_group_config.slice_template.gcp.machine_type
        vm_config.gcp.service_account = controller_service_account
        vm_handle = worker_provider.create_vm(vm_config)
        _wait_for_vm_connection(vm_handle)

        tpu_group_config = copy.deepcopy(config.scale_groups[tpu_group].slice_template)
        tpu_handle = worker_provider.create_slice(tpu_group_config)
        tpu_handle, tpu_worker_ip = _wait_for_tpu_ready(tpu_handle)

        gce_native = _command_check(
            "gce_native_ssh",
            [
                "gcloud",
                "compute",
                "ssh",
                vm_name,
                f"--project={config.platform.gcp.project_id}",
                f"--zone={vm_zone}",
                f"--ssh-key-file={private_key}",
                f"--impersonate-service-account={ssh_service_account}",
                "--quiet",
                "--command",
                "echo ok-gce-oslogin",
            ],
        )

        tpu_native = _command_check(
            "tpu_native_ssh",
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                tpu_handle.slice_id,
                f"--project={config.platform.gcp.project_id}",
                f"--zone={tpu_handle.zone}",
                f"--ssh-key-file={private_key}",
                f"--impersonate-service-account={ssh_service_account}",
                "--worker=0",
                "--quiet",
                "--command",
                "echo ok-tpu-gcloud-oslogin",
            ],
            timeout=180.0,
        )

        direct_exec = DirectSshRemoteExec(
            host=tpu_worker_ip,
            user=worker_os_login_user,
            key_file=str(private_key),
            connect_timeout=Duration.from_seconds(30),
        )
        direct_result = direct_exec.run("echo ok-tpu-direct-oslogin", timeout=Duration.from_seconds(60))
        tpu_direct = CommandCheck(
            name="tpu_direct_ssh",
            command=direct_exec._build_cmd("echo ok-tpu-direct-oslogin"),
            returncode=direct_result.returncode,
            stdout=direct_result.stdout,
            stderr=direct_result.stderr,
        )

        report = ValidationReport(
            config=str(Path(args.config).resolve()),
            vm_group=vm_group,
            tpu_group=tpu_group,
            controller_service_account=controller_service_account,
            worker_service_account=worker_service_account,
            controller_os_login_user=controller_os_login_user,
            worker_os_login_user=worker_os_login_user,
            vm_name=vm_name,
            tpu_slice_id=tpu_handle.slice_id,
            gce_native_ssh=gce_native,
            tpu_native_ssh=tpu_native,
            tpu_direct_ssh=tpu_direct,
        )
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.ok else 1
    finally:
        if not args.keep_resources:
            if vm_handle is not None:
                try:
                    vm_handle.terminate()
                except Exception as exc:
                    logger.warning("Failed to delete VM %s: %s", vm_handle.vm_id, exc)
            if tpu_handle is not None:
                try:
                    tpu_handle.terminate()
                except Exception as exc:
                    logger.warning("Failed to delete TPU slice %s: %s", tpu_handle.slice_id, exc)
        _remove_temp_ssh_key(private_key, service_account=ssh_service_account)


if __name__ == "__main__":
    raise SystemExit(main())
