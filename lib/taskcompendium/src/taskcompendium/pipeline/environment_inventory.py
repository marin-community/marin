# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded file inventories through Shellbox for static task review."""

from shellbox.machine import Command, MachineFactory, MachineSpec

from taskcompendium.pipeline.models import EnvironmentInventory


async def inspect_environment(
    factory: MachineFactory,
    spec: MachineSpec,
    *,
    environment_id: str,
    roots: tuple[str, ...],
    timeout: float,
    output_limit_bytes: int,
) -> EnvironmentInventory:
    """Inspect files in a fresh machine, preserving scope and truncation evidence.

    The caller supplies the pinned image or guest identity and selected backend.
    ShellSim inventories describe its virtual filesystem, not a Harbor OCI image.
    """
    if not roots or any(not root.startswith("/") for root in roots):
        raise ValueError("Inventory roots must be explicit absolute paths")
    machine = await factory.create(spec)
    try:
        result = await machine.run(
            Command(
                ("find", *roots, "-type", "f", "-print0"),
                cwd="/",
                timeout=timeout,
                output_limit_bytes=output_limit_bytes,
            )
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Environment inventory failed: {result.stderr.decode(errors='replace')}")
        # Discard a partial final path if the machine truncated the byte stream.
        paths = result.stdout.split(b"\0")[:-1]
        return EnvironmentInventory(
            environment_id=environment_id,
            origin=f"Shellbox {type(factory).__module__}.{type(factory).__name__}: find -type f -print0",
            roots=roots,
            paths=tuple(sorted(path.decode("utf-8", errors="replace") for path in paths)),
            complete=not result.stdout_truncated,
        )
    finally:
        await machine.close()
