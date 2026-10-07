# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check concurrent guest isolation and pristine base bytes using a real bundle."""

import argparse
import asyncio
import hashlib
import tempfile
from pathlib import Path

from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory
from shellbox.machine import Command, MachineSpec, QemuBundle


def disk_digest(bundle: Path) -> str:
    with (bundle / "rootfs.ext4").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def main(bundle: Path) -> None:
    bundle = bundle.resolve()
    original = disk_digest(bundle)
    # Commas in host paths must survive QEMU's drive-option parsing.
    with tempfile.TemporaryDirectory(prefix="shellbox,isolation-") as directory:
        mounted = Path(directory)
        for child in bundle.iterdir():
            (mounted / child.name).symlink_to(child)
        factory = QemuMachineFactory(Acceleration.TCG)
        spec = MachineSpec(QemuBundle(mounted))
        first = await factory.create(spec)
        try:
            written = await first.run(Command(("/bin/sh", "-c", "echo first > /isolation-marker; sync")))
            assert written.exit_code == 0, written
            second = await factory.create(spec)
            try:
                absent = await second.run(Command(("test", "!", "-e", "/isolation-marker")))
                assert absent.exit_code == 0, absent
                written = await second.run(Command(("/bin/sh", "-c", "echo second > /isolation-marker; sync")))
                assert written.exit_code == 0, written
                for machine, expected in ((first, b"first\n"), (second, b"second\n")):
                    result = await machine.run(Command(("cat", "/isolation-marker")))
                    assert result.exit_code == 0 and result.stdout == expected, result
            finally:
                await second.close()
        finally:
            await first.close()
        fresh = await factory.create(spec)
        try:
            absent = await fresh.run(Command(("test", "!", "-e", "/isolation-marker")))
            assert absent.exit_code == 0, absent
        finally:
            await fresh.close()
    assert disk_digest(bundle) == original, "A guest modified the base disk"
    print("Concurrent writes are isolated, fresh guests reset, and the base disk is unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.bundle))
