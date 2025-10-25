# Copyright 2025 The Levanter Authors
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


def list_tpus(zone):
    return json.loads(
        subprocess.check_output(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "queued-resources",
                "list",
                f"--zone={zone}",
                "--format=json(name.basename(), state)",
                "--quiet",
            ]
        )
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


def describe_tpu_vm(tpu_name, zone):
    try:
        return json.loads(
            subprocess.check_output(
                [
                    "gcloud",
                    "alpha",
                    "compute",
                    "tpus",
                    "tpu-vm",
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
    if capacity_type in ["best-effort", "preemptible"]:
        command.append("--best-effort")
        command.extend(["--provisioning-model", "spot"])
    elif capacity_type == "reserved":
        command.append("--reserved")
    elif capacity_type == "spot":
        command.append("--spot")
        command.append(["--provisioning-model", "spot"])
    elif capacity_type == "on-demand" or capacity_type is None:
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


def get_current_tpu_metadata(key: str) -> Optional[str]:
    # cribbed from Ray.
    """Poll and get TPU metadata. This only works on a **TPU VM**."""
    try:
        accelerator_type_request = requests.get(
            os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, key),
            headers=GCE_TPU_HEADERS,
        )
        if accelerator_type_request.status_code == 200 and accelerator_type_request.text:
            return accelerator_type_request.text
        else:
            logging.debug(
                "Unable to poll TPU GCE Metadata. Got "
                f"status code: {accelerator_type_request.status_code} and "
                f"content: {accelerator_type_request.text}"
            )
    except requests.RequestException as e:
        logging.debug("Unable to poll the TPU GCE Metadata: %s", e)
    return None


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
