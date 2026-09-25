# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the shared command and file contract with prepared Docker and QEMU images."""

import argparse
import asyncio
import tempfile
from pathlib import Path

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.qemu.machine import Acceleration, QemuMachineFactory
from shellbox.machine import Command, DockerImage, ExitReason, MachineFactory, MachineSpec, QemuBundle


async def check(factory: MachineFactory, spec: MachineSpec) -> None:
    machine = await factory.create(spec)
    try:
        first = await machine.run(Command(("/bin/sh", "-c", "printf 'persistent' > state.txt")))
        assert first.reason is ExitReason.EXITED and first.exit_code == 0, first
        second = await machine.run(Command(("/bin/sh", "-c", "cat state.txt; printf 'warning' >&2; exit 7")))
        assert second.exit_code == 7 and second.stdout == b"persistent" and second.stderr == b"warning", second
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            target = Path(directory) / "target.txt"
            source.write_bytes(b"binary\x00content")
            await machine.upload(source, "/tmp/machine-file.txt")
            await machine.download("/tmp/machine-file.txt", target)
            assert target.read_bytes() == source.read_bytes()
    finally:
        await machine.close()


async def main(bundle: Path, image: str) -> None:
    await check(DockerMachineFactory(), MachineSpec(DockerImage(image), workdir="/tmp"))
    print("Docker machine passed")
    await check(QemuMachineFactory(Acceleration.TCG), MachineSpec(QemuBundle(bundle), workdir="/tmp"))
    print("QEMU machine passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("image")
    args = parser.parse_args()
    asyncio.run(main(args.bundle, args.image))
