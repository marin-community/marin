# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import getpass
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import requests  # type: ignore

from levanter.infra.docker import make_docker_run_command

logger = logging.getLogger(__name__)

_SENSITIVE_ENV_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
    }
)
_EMBEDDED_ENV_ASSIGNMENT_RE = re.compile(r"(?P<prefix>-e\s+)(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")


@dataclass(frozen=True)
class TpuWorker:
    """Concrete location for a TPU worker within a pod."""

    ordinal: int
    slice_index: int
    slice_name: str
    worker_index: int
    ip_address: str | None


def setup_vm_docker(tpu_name, zone, node_count):
    """Change docker permissions on `tpu_name`, remove any old runs, and setup the cache volume."""
    tpu_ssh(
        tpu_name,
        zone,
        node_count,
        "sudo",
        "usermod",
        "-aG",
        "docker",
        getpass.getuser(),
        "&&",
        "sudo",
        "docker",
        "volume",
        "create",
        "--driver=local",
        "levanter",
        "&&",
        "sudo",
        "docker",
        "rm",
        "-f",
        "levanter",
    )


def describe_tpu_queued_resource(tpu_name, zone):
    try:
        return json.loads(
            subprocess.check_output(
                [
                    "gcloud",
                    "alpha",
                    "compute",
                    "tpus",
                    "queued-resources",
                    "describe",
                    tpu_name,
                    f"--zone={zone}",
                    "--format=json(name.basename(), state)",
                    "--quiet",
                ],
                stderr=subprocess.DEVNULL,
            )
        )
    except subprocess.CalledProcessError:
        return None


def _wait_for_queued_resource_active(tpu_name: str, zone: str) -> None:
    print("Checking TPU creation status every minute...")
    waited = 0
    while True:
        time.sleep(60)
        waited += 1

        tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
        assert tpu_stat is not None, f"{tpu_name} creation failed."
        state = tpu_stat["state"]["state"]

        match state:
            case "ACTIVE":
                return
            case "FAILED":
                raise RuntimeError(
                    f"{tpu_name} creation failed: {tpu_stat['state']['failedData']['error']['message']}"
                )
            case _:
                print(f"Status is {state}. Waited {waited} minutes...")


def _wait_for_queued_resource_absent(tpu_name: str, zone: str) -> None:
    print("Waiting for queued resource deletion to finish...")
    waited = 0
    while True:
        time.sleep(30)
        waited += 1
        tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
        if tpu_stat is None:
            return
        state = tpu_stat["state"]["state"]
        print(f"Delete still in progress with state {state}. Waited {waited * 30} seconds...")


def start_tpu_vm_queued_resources(tpu_name, *, tpu_type, capacity_type, version, zone, node_count):
    # ensure alpha is enabled (best-effort; may fail on managed installs like homebrew)
    try:
        run_command("gcloud", "components", "install", "alpha", "--quiet")
    except RuntimeError:
        pass
    if version is None:
        version = "tpu-ubuntu2204-base"
    tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
    if tpu_stat is not None:
        state = tpu_stat["state"]["state"]
        if state in ["FAILED", "SUSPENDED", "SUSPENDING", "DELETING"]:
            print(f"TPU is in reclaim/deletion state {state}, deleting before recreate...", file=sys.stderr)
            if state != "DELETING":
                run_command(
                    "gcloud",
                    "alpha",
                    "compute",
                    "tpus",
                    "queued-resources",
                    "delete",
                    tpu_name,
                    "--quiet",
                    f"--zone={zone}",
                    "--force",
                )
            _wait_for_queued_resource_absent(tpu_name, zone)
        elif state == "ACTIVE":
            print(f"TPU {tpu_name} already exists and is in state {state}.", file=sys.stderr)
            return
        else:
            print(f"TPU {tpu_name} already exists and is in state {state}. Waiting for ACTIVE.", file=sys.stderr)
            _wait_for_queued_resource_active(tpu_name, zone)
            return

    print(f"Creating new TPU {tpu_name} in {zone} of type {tpu_type}...", file=sys.stderr)
    command = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        tpu_name,
        f"--accelerator-type={tpu_type}",
        f"--zone={zone}",
        "--quiet",
    ]

    if version is not None:
        command.append(f"--runtime-version={version}")

    normalized_capacity_type = capacity_type.replace("_", "-") if capacity_type is not None else None

    if normalized_capacity_type in ["preemptible", "spot"]:
        command.append("--spot")
    elif normalized_capacity_type == "best-effort":
        command.append("--best-effort")
    elif normalized_capacity_type == "reserved":
        command.append("--reserved")
    elif normalized_capacity_type == "on-demand" or normalized_capacity_type is None:
        pass
    else:
        raise ValueError(f"Unknown capacity type: {capacity_type}")

    if node_count == 1:
        command.append(f"--node-id={tpu_name}")
    else:
        command.append(f"--node-count={node_count}")

    run_command(*command)

    _wait_for_queued_resource_active(tpu_name, zone)


def launch_job(
    command: list[str],
    tpu_name: str,
    tpu_type: str,
    capacity_type: str,
    zone: str,
    node_count: int,
    full_image_id: str,
    env: dict[str, str],
    foreground: bool,
    version: Optional[str] = None,
):
    start_tpu_vm_queued_resources(
        tpu_name=tpu_name,
        tpu_type=tpu_type,
        capacity_type=capacity_type,
        version=version,
        zone=zone,
        node_count=node_count,
    )

    # We don't technically need to setup on every run, but if we are working on a
    # stale VM or a VM from e.g. spin-up-vm.sh, this ensures things always work.
    setup_vm_docker(
        tpu_name=tpu_name,
        zone=zone,
        node_count=node_count,
    )

    docker_command = make_docker_run_command(full_image_id, command, env=env, foreground=foreground)

    print(f"Running on tpu_name... {tpu_name}")
    tpu_ssh(tpu_name, zone, node_count, *docker_command)


def _redact_env_assignment(arg: str) -> str:
    if "=" not in arg:
        return arg
    key, value = arg.split("=", 1)
    if key not in _SENSITIVE_ENV_VARS:
        return arg
    return f"{key}=<redacted>"


def _redact_embedded_env_assignments(arg: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group("key")
        if key not in _SENSITIVE_ENV_VARS:
            return match.group(0)
        return f"{match.group('prefix')}{key}=<redacted>"

    return _EMBEDDED_ENV_ASSIGNMENT_RE.sub(_replace, arg)


def _redacted_command(args: Sequence[object]) -> tuple[str, ...]:
    redacted: list[str] = []
    expect_env_assignment = False

    for raw_arg in args:
        arg = str(raw_arg)
        if expect_env_assignment:
            redacted.append(_redact_env_assignment(arg))
            expect_env_assignment = False
            continue
        if arg == "-e":
            redacted.append(arg)
            expect_env_assignment = True
            continue
        redacted.append(_redact_embedded_env_assignments(arg))

    return tuple(redacted)


def run_command(*args, **kwargs):
    redacted_args = _redacted_command(args)
    print("Running:", " ".join(redacted_args))
    result = subprocess.run(args, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(redacted_args)}")
    return result.returncode


def add_ssh_key(ssh_key_filename):
    # format 3072 SHA256:... key-name (RSA)
    try:
        key_hash = (
            subprocess.check_output(["ssh-keygen", "-lf", ssh_key_filename], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .split()[1]
        )
        existing_keys = (
            subprocess.check_output(["ssh-add", "-l"], stderr=subprocess.STDOUT).decode("utf-8").split("\n")
        )
        for key in existing_keys:
            if key_hash in key:
                print("Existing key found in keychain, skipping ssh-add")
                return

        print("SSH key not in key-chain, adding.")
        subprocess.check_call(["ssh-add", ssh_key_filename])
    except subprocess.CalledProcessError:
        raise


def describe_tpu_workers(tpu_name: str, zone: str, node_count: int) -> list[TpuWorker]:
    """Describe every worker in a TPU pod.

    Args:
        tpu_name: Base TPU name. Multi-slice pods are expected to use the
            `name-<slice_index>` naming pattern.
        zone: TPU zone.
        node_count: Number of slices in the pod.

    Returns:
        A flattened list of workers in global ordinal order.
    """

    workers: list[TpuWorker] = []
    ordinal = 0

    for slice_index, slice_name in enumerate(_slice_names(tpu_name, node_count)):
        try:
            description = json.loads(
                subprocess.check_output(
                    [
                        "gcloud",
                        "alpha",
                        "compute",
                        "tpus",
                        "tpu-vm",
                        "describe",
                        slice_name,
                        f"--zone={zone}",
                        "--format=json(networkEndpoints)",
                        "--quiet",
                    ],
                    stderr=subprocess.DEVNULL,
                )
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to describe TPU slice {slice_name} in zone {zone}") from exc

        network_endpoints = description.get("networkEndpoints", [])
        if not network_endpoints:
            raise RuntimeError(f"TPU slice {slice_name} has no network endpoints")

        for worker_index, endpoint in enumerate(network_endpoints):
            workers.append(
                TpuWorker(
                    ordinal=ordinal,
                    slice_index=slice_index,
                    slice_name=slice_name,
                    worker_index=worker_index,
                    ip_address=endpoint.get("ipAddress"),
                )
            )
            ordinal += 1

    return workers


def ssh_tpu_worker(
    tpu_name: str,
    zone: str,
    node_count: int,
    worker_ordinal: int,
    *args,
    ignore_failure: bool = False,
):
    """Run a command on one TPU worker by global worker ordinal."""

    try:
        add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    except subprocess.CalledProcessError as e:
        print("Failed to add ssh key. This may lead to problems.", e)

    worker = _resolve_tpu_worker(tpu_name, zone, node_count, worker_ordinal)

    try:
        return run_command(*_build_tpu_ssh_command(worker.slice_name, zone, worker.worker_index, args))
    except subprocess.CalledProcessError as e:
        if ignore_failure:
            print("Ignoring failure:", e)
            return None
        raise


def run_container_on_worker(
    *,
    tpu_name: str,
    zone: str,
    node_count: int,
    worker_ordinal: int,
    full_image_id: str,
    command: Sequence[str],
    env: dict[str, str],
    foreground: bool,
    name: str,
):
    """Run a named Docker container on one TPU worker."""

    docker_command = make_docker_run_command(full_image_id, command, env=env, foreground=foreground, name=name)
    return ssh_tpu_worker(tpu_name, zone, node_count, worker_ordinal, *docker_command)


def run_container_all_workers(
    *,
    tpu_name: str,
    zone: str,
    node_count: int,
    full_image_id: str,
    command: Sequence[str],
    env: dict[str, str],
    foreground: bool,
    name: str,
):
    """Run the same named Docker container on all workers in a TPU pod."""

    docker_command = make_docker_run_command(full_image_id, command, env=env, foreground=foreground, name=name)
    return tpu_ssh(tpu_name, zone, node_count, *docker_command)


def stop_container_on_worker(
    tpu_name: str, zone: str, node_count: int, worker_ordinal: int, *, name: str, ignore_missing: bool = True
):
    """Remove a named Docker container from one worker."""

    stop_command = f"sudo docker rm -f {shlex.quote(name)}"
    if ignore_missing:
        stop_command = f"{stop_command} || true"

    return ssh_tpu_worker(tpu_name, zone, node_count, worker_ordinal, *_bash_command(stop_command))


def stop_container_all_workers(tpu_name: str, zone: str, node_count: int, *, name: str, ignore_missing: bool = True):
    """Remove a named Docker container from every worker."""

    stop_command = f"sudo docker rm -f {shlex.quote(name)}"
    if ignore_missing:
        stop_command = f"{stop_command} || true"

    return tpu_ssh(tpu_name, zone, node_count, *_bash_command(stop_command))


def container_exists_on_worker(tpu_name: str, zone: str, node_count: int, worker_ordinal: int, *, name: str) -> bool:
    """Return True if a named Docker container exists on a worker."""

    result = _ssh_tpu_worker_output(
        tpu_name,
        zone,
        node_count,
        worker_ordinal,
        *_bash_command(
            f"if sudo docker container inspect {shlex.quote(name)} >/dev/null 2>&1; "
            "then echo present; else echo absent; fi"
        ),
    )
    return result.decode("utf-8").strip() == "present"


def container_exists_on_all_workers(tpu_name: str, zone: str, node_count: int, *, name: str) -> dict[int, bool]:
    """Return container presence for every worker in global ordinal order."""

    return {
        worker.ordinal: container_exists_on_worker(tpu_name, zone, node_count, worker.ordinal, name=name)
        for worker in describe_tpu_workers(tpu_name, zone, node_count)
    }


def probe_tpu_worker_health(tpu_name: str, zone: str, node_count: int, worker_ordinal: int):
    """Verify SSH and Docker responsiveness on one TPU worker."""

    return ssh_tpu_worker(
        tpu_name,
        zone,
        node_count,
        worker_ordinal,
        *_bash_command("hostname >/dev/null && sudo docker info >/dev/null"),
    )


def probe_tpu_all_workers_health(tpu_name: str, zone: str, node_count: int):
    """Verify SSH and Docker responsiveness across the full TPU pod."""

    return tpu_ssh(tpu_name, zone, node_count, *_bash_command("hostname >/dev/null && sudo docker info >/dev/null"))


def run_python_on_worker(
    tpu_name: str,
    zone: str,
    node_count: int,
    worker_ordinal: int,
    *,
    python_executable: str,
    module: str | None = None,
    script_path: str | None = None,
    args: Sequence[str] = (),
):
    """Run a Python entrypoint directly on one worker.

    Exactly one of `module` or `script_path` must be specified.
    """

    if (module is None) == (script_path is None):
        raise ValueError("Exactly one of module or script_path must be specified")

    python_command = [python_executable]
    if module is not None:
        python_command.extend(["-m", module])
    else:
        python_command.append(script_path)
    python_command.extend(args)

    return ssh_tpu_worker(tpu_name, zone, node_count, worker_ordinal, *python_command)


def tpu_ssh(tpu_name, zone, node_count, *args, ignore_failure=False):
    try:
        add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    except subprocess.CalledProcessError as e:
        print("Failed to add ssh key. This may lead to problems.", e)
        pass

    try:
        if node_count > 1:
            return _tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=ignore_failure)

        return run_command(*_build_tpu_ssh_command(tpu_name, zone, "all", args))
    except subprocess.CalledProcessError as e:
        if ignore_failure:
            print("Ignoring failure:", e)
        else:
            raise


def _tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=False):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_command,
                *_build_tpu_ssh_command(f"{tpu_name}-{i}", zone, "all", args),
            )
            for i in range(node_count)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                if ignore_failure:
                    print("Ignoring failure:", e)
                else:
                    raise


def _slice_names(tpu_name: str, node_count: int) -> list[str]:
    if node_count < 1:
        raise ValueError(f"node_count must be >= 1, got {node_count}")

    if node_count == 1:
        return [tpu_name]

    return [f"{tpu_name}-{slice_index}" for slice_index in range(node_count)]


def _resolve_tpu_worker(tpu_name: str, zone: str, node_count: int, worker_ordinal: int) -> TpuWorker:
    workers = describe_tpu_workers(tpu_name, zone, node_count)
    if worker_ordinal < 0 or worker_ordinal >= len(workers):
        raise ValueError(f"Worker ordinal {worker_ordinal} is out of range for TPU {tpu_name}: {len(workers)} workers")

    return workers[worker_ordinal]


def _build_tpu_ssh_command(tpu_name: str, zone: str, worker_selector: str | int, args: Sequence[str]) -> list[str]:
    return [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        "--quiet",
        f"--worker={worker_selector}",
        f"--zone={zone}",
        "--command=%s" % " ".join(args),
    ]


def _ssh_tpu_worker_output(tpu_name: str, zone: str, node_count: int, worker_ordinal: int, *args) -> bytes:
    try:
        add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    except subprocess.CalledProcessError as e:
        print("Failed to add ssh key. This may lead to problems.", e)

    worker = _resolve_tpu_worker(tpu_name, zone, node_count, worker_ordinal)
    return subprocess.check_output(_build_tpu_ssh_command(worker.slice_name, zone, worker.worker_index, args))


def _bash_command(command: str) -> tuple[str, str, str]:
    return ("bash", "-lc", shlex.quote(command))


GCE_TPU_ACCELERATOR_ENDPOINT = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}


def get_current_tpu_is_preempted() -> bool:
    """curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/preempted"""
    try:
        preempted_request = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            headers=GCE_TPU_HEADERS,
        )
        if preempted_request.status_code == 200:
            return preempted_request.text == "TRUE"
        else:
            logging.warning(
                "Unable to poll TPU preempted status. Got "
                f"status code: {preempted_request.status_code} and "
                f"content: {preempted_request.text}"
            )
            return False
    except requests.RequestException as e:
        logging.debug("Unable to poll TPU preempted status: %s", e)
        raise e
