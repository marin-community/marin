# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one registry image through Docker and QEMU."""

import argparse
import asyncio
from pathlib import Path

from harbor_qemu.docker_machine import DockerMachineFactory
from harbor_qemu.image import QemuAssets, RegistryImage
from harbor_qemu.machine import Command, MachineFactory, MachineSpec
from harbor_qemu.qemu_machine import Acceleration, QemuMachineFactory


async def check(factory: MachineFactory, spec: MachineSpec) -> bytes:
    machine = await factory.create(spec)
    try:
        result = await machine.run(Command(("/bin/sh", "-c", "cat /etc/os-release")))
        assert result.exit_code == 0, result
        return result.stdout
    finally:
        await machine.close()


async def main(reference: str, cache: Path, tools: Path, base_bundle: Path, busybox: Path) -> None:
    skopeo = tools / "usr/bin/skopeo"
    spec = MachineSpec(source=RegistryImage(reference), workdir="/tmp")
    docker_output = await check(DockerMachineFactory(skopeo=skopeo, image_cache=cache / "oci"), spec)
    assets = QemuAssets(
        qemu=base_bundle / "qemu-system-x86_64",
        kernel=base_bundle / "vmlinuz",
        busybox=busybox,
        firmware=base_bundle / "firmware",
        libraries=base_bundle / "lib",
        umoci=tools / "usr/bin/umoci",
        disk_size_mb=256,
        runtime_id="local-qemu-kernel-busybox-v1",
    )
    qemu_output = await check(
        QemuMachineFactory(
            Acceleration.TCG,
            assets=assets,
            bundle_cache=cache / "bundles",
            image_cache=cache / "oci",
            skopeo=skopeo,
        ),
        spec,
    )
    assert docker_output == qemu_output, (docker_output, qemu_output)
    print(f"Docker and QEMU matched for {reference}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("cache", type=Path)
    parser.add_argument("tools", type=Path)
    parser.add_argument("base_bundle", type=Path)
    parser.add_argument("busybox", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.reference, args.cache, args.tools, args.base_bundle, args.busybox))
