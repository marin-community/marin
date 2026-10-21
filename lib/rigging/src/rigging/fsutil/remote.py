# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an fsutil transfer on the CoreWeave US-EAST-02A Iris cluster."""

import os
import subprocess

import click
from huggingface_hub import get_token

from rigging.filesystem.storage_path import StoragePath

REMOTE_CLUSTER = "cw-us-east-02a"


def run_remote_transfer(
    operation: str,
    urls: tuple[str, ...],
    *,
    recursive: bool,
    no_clobber: bool = False,
) -> None:
    """Submit one CPU job and stream its logs through Iris's waiting CLI."""
    for url in urls:
        if StoragePath(url).is_local:
            raise click.UsageError(f"--remote requires storage URLs; {url!r} is local to the submitting machine")

    remote_command = ["fsutil", operation]
    if recursive:
        remote_command.append("--recursive")
    if no_clobber:
        remote_command.append("--no-clobber")
    remote_command.extend(urls)

    # Iris forwards HF_TOKEN from the submitter's environment. A token saved by
    # `hf auth login` is otherwise only present on this machine, not the worker.
    environment = os.environ.copy()
    if any(StoragePath(url).scheme == "hf" for url in urls):
        token = get_token()
        if token:
            environment["HF_TOKEN"] = token

    command = [
        "iris",
        f"--cluster={REMOTE_CLUSTER}",
        "job",
        "run",
        "--cpu",
        "1",
        "--memory",
        "2GB",
        "--disk",
        "5GB",
        "--sync-package",
        "marin-rigging",
        "--",
        *remote_command,
    ]
    try:
        result = subprocess.run(command, env=environment, check=False)
    except FileNotFoundError as error:
        raise click.ClickException("Iris is not installed; run fsutil from the Marin workspace") from error
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)
