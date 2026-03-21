# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import getpass
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

import requests  # type: ignore

from levanter.infra.docker import make_docker_run_command

logger = logging.getLogger(__name__)


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


def start_tpu_vm_queued_resources(tpu_name, *, tpu_type, capacity_type, version, zone, node_count):
    # ensure alpha is enabled
    run_command("gcloud", "components", "install", "alpha", "--quiet")
    if version is None:
        version = "tpu-ubuntu2204-base"
    tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
    if tpu_stat is not None:
        if tpu_stat["state"]["state"] in ["FAILED", "SUSPENDED"]:
            print("TPU suspended,  deleting...", file=sys.stderr)

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
        else:
            print(f"TPU {tpu_name} already exists and is in state {tpu_stat['state']['state']}.", file=sys.stderr)
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

    # wait for queued resource to complete
    print("Checking TPU creation status every minute...")
    waited = 0
    while True:
        time.sleep(60)
        waited += 1

        tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
        assert tpu_stat is not None, f"{tpu_name} creation failed."

        match tpu_stat["state"]["state"]:
            case "ACTIVE":
                break
            case "FAILED":
                raise RuntimeError(
                    f"{tpu_name} creation failed: {tpu_stat['state']['failedData']['error']['message']}"
                )
            case _:
                print(f"Status is {tpu_stat['state']['state']}. Waited {waited} minutes...")


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


def run_command(*args, **kwargs):
    print("Running:", " ".join(list(args)))
    return subprocess.check_call(args, **kwargs)


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


def tpu_ssh(tpu_name, zone, node_count, *args, ignore_failure=False):
    try:
        add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    except subprocess.CalledProcessError as e:
        print("Failed to add ssh key. This may lead to problems.", e)
        pass

    try:
        if node_count > 1:
            return _tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=ignore_failure)

        return run_command(
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            tpu_name,
            "--quiet",
            "--worker=all",
            f"--zone={zone}",
            "--command=%s" % " ".join(args),
        )
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
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                f"{tpu_name}-{i}",
                "--worker=all",
                "--quiet",
                f"--zone={zone}",
                "--command=%s" % " ".join(args),
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


GCE_TPU_ACCELERATOR_ENDPOINT = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}


# ---------------------------------------------------------------------------
# Helpers for alternating RL: per-worker and existing-pod operations
# ---------------------------------------------------------------------------


def ssh_tpu_worker(tpu_name: str, zone: str, worker_ordinal: int, *args, ignore_failure: bool = False):
    """SSH into a specific worker of a TPU VM."""
    try:
        add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    except subprocess.CalledProcessError as e:
        print("Failed to add ssh key. This may lead to problems.", e)

    try:
        return run_command(
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            tpu_name,
            "--quiet",
            f"--worker={worker_ordinal}",
            f"--zone={zone}",
            "--command=%s" % " ".join(args),
        )
    except subprocess.CalledProcessError as e:
        if ignore_failure:
            print("Ignoring failure:", e)
        else:
            raise


def run_container_on_worker(
    tpu_name: str,
    zone: str,
    worker_ordinal: int,
    image_id: str,
    command: list[str],
    env: dict[str, str],
    name: str = "levanter",
    foreground: bool = True,
):
    """Run a Docker container on a specific TPU worker."""
    docker_command = make_docker_run_command(image_id, command, env=env, foreground=foreground, name=name)
    ssh_tpu_worker(tpu_name, zone, worker_ordinal, *docker_command)


def run_container_all_workers(
    tpu_name: str,
    zone: str,
    num_workers: int,
    image_id: str,
    command: list[str],
    env: dict[str, str],
    name: str = "levanter",
    foreground: bool = True,
):
    """Run a Docker container on all TPU workers simultaneously."""
    docker_command = make_docker_run_command(image_id, command, env=env, foreground=foreground, name=name)
    tpu_ssh(tpu_name, zone, num_workers, *docker_command)


def stop_container_on_worker(
    tpu_name: str,
    zone: str,
    worker_ordinal: int,
    name: str = "levanter",
    ignore_failure: bool = True,
):
    """Stop and remove a Docker container on a specific TPU worker."""
    ssh_tpu_worker(
        tpu_name,
        zone,
        worker_ordinal,
        "sudo",
        "docker",
        "rm",
        "-f",
        name,
        ignore_failure=ignore_failure,
    )


def stop_container_all_workers(
    tpu_name: str,
    zone: str,
    num_workers: int,
    name: str = "levanter",
    ignore_failure: bool = True,
):
    """Stop and remove a Docker container on all TPU workers."""
    tpu_ssh(
        tpu_name,
        zone,
        num_workers,
        "sudo",
        "docker",
        "rm",
        "-f",
        name,
        ignore_failure=ignore_failure,
    )


def describe_tpu_workers(tpu_name: str, zone: str) -> dict | None:
    """Get TPU VM details including worker network endpoints."""
    try:
        return json.loads(
            subprocess.check_output(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "describe",
                    tpu_name,
                    f"--zone={zone}",
                    "--format=json",
                    "--quiet",
                ],
                stderr=subprocess.DEVNULL,
            )
        )
    except subprocess.CalledProcessError:
        return None


def worker_health_probe(
    tpu_name: str,
    zone: str,
    worker_ordinal: int,
) -> bool:
    """Run a simple health check on a TPU worker. Returns True if healthy."""
    try:
        ssh_tpu_worker(tpu_name, zone, worker_ordinal, "echo", "ok")
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_image_digest(image_tag: str) -> str:
    """Resolve a Docker image tag to its full digest."""
    result = subprocess.check_output(
        ["gcloud", "container", "images", "describe", image_tag, "--format=value(image_summary.digest)"],
        text=True,
    ).strip()
    # Return full image reference with digest
    repo = image_tag.rsplit(":", 1)[0]
    return f"{repo}@{result}"


def ensure_tpu_exists(
    tpu_name: str,
    tpu_type: str,
    zone: str,
    capacity_type: str = "on-demand",
    version: str | None = None,
) -> None:
    """Create a TPU if it doesn't already exist."""
    tpu_stat = describe_tpu_queued_resource(tpu_name, zone)
    if tpu_stat is not None and tpu_stat["state"]["state"] == "ACTIVE":
        logger.info("TPU %s already exists and is active", tpu_name)
        return

    # Determine node count from TPU type
    parts = tpu_type.split("-")
    total_chips = int(parts[1])
    chips_per_host = 4
    node_count = max(1, total_chips // chips_per_host)

    start_tpu_vm_queued_resources(
        tpu_name=tpu_name,
        tpu_type=tpu_type,
        capacity_type=capacity_type,
        version=version,
        zone=zone,
        node_count=node_count,
    )
    setup_vm_docker(tpu_name=tpu_name, zone=zone, node_count=node_count)


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
