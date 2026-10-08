# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write file transfers with bounded memory and atomic target replacement."""

import shlex
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from shellbox.machine import Command, DownloadLimitExceeded, Machine, UnsupportedMachineSpec

DOWNLOAD_CHUNK_BYTES = 64 * 1024


async def file_chunks(
    machine: Machine, source: str, max_bytes: int, *, busybox: str = ""
) -> AsyncGenerator[bytes, None]:
    quoted = shlex.quote(source)
    tools = f"{shlex.quote(busybox)} " if busybox else ""
    regular = await machine.run(Command(("test", "-f", source)))
    if regular.exit_code != 0:
        raise UnsupportedMachineSpec("A download byte limit requires a regular file")
    offset = 0
    while offset <= max_bytes:
        length = min(DOWNLOAD_CHUNK_BYTES, max_bytes + 1 - offset)
        script = f"{tools}dd if={quoted} bs={DOWNLOAD_CHUNK_BYTES} skip={offset // DOWNLOAD_CHUNK_BYTES} count=1"
        result = await machine.run(Command(("sh", "-c", script), output_limit_bytes=length))
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot download file: {result.stderr.decode(errors='replace')}")
        if result.stdout_truncated:
            raise DownloadLimitExceeded(f"Candidate file exceeds the {max_bytes}-byte download limit")
        data = result.stdout
        if not data:
            return
        yield data
        offset += len(data)
        if len(data) < length:
            return


async def write_download(chunks: AsyncIterator[bytes], target: Path, max_bytes: int) -> None:
    if max_bytes < 0:
        raise ValueError("Download limit must be nonnegative")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".shellbox-download-", dir=target.parent) as directory:
        staged = Path(directory) / "file"
        received = 0
        with staged.open("wb") as file:
            async for chunk in chunks:
                received += len(chunk)
                if received > max_bytes:
                    raise DownloadLimitExceeded(f"Candidate file exceeds the {max_bytes}-byte download limit")
                file.write(chunk)
        staged.replace(target)
