# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy Shellbox artifacts into local or object-storage trial paths."""

import asyncio
import tempfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Machine


async def download_artifact(
    machine: Machine, source: str, target: str, *, directory: bool
) -> None:
    destination = StoragePath(target)
    if destination.is_local:
        local = (
            Path("/" + destination.key)
            if destination.scheme == "file"
            else Path(str(destination))
        )
        await machine.download(source, local)
        return
    with tempfile.TemporaryDirectory() as staging:
        local = Path(staging) / "download"
        await machine.download(source, local)
        await asyncio.to_thread(
            destination.upload_from,
            str(local) + "/" if directory else str(local),
            recursive=directory,
        )
