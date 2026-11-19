# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live backend probes that run against real machines, unlike the offline survey.

The first probe answers one question per remote backend: can this backend pull an
internal ``ghcr.io/marin-community`` image? Iris task pods pull through the
namespace service account's image pull secrets and Daytona sandboxes pull through
the org's registry configuration; a backend without a working credential cannot run
any source whose task or grader image is an internal GHCR package, so the survey
records the cell UNSUPPORTED before any gate run depends on it.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

logger = logging.getLogger(__name__)

# A digest-pinned internal image every credentialed backend already pulls for real
# work; the probe exercises the same pull path without transferring more than the
# platform manifest and layers of one task image.
CANARY_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:" "c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)

DAYTONA_KEY_ENV = "DAYTONA_API_KEY"

ProbeState = Literal["supported", "unsupported"]


@dataclass(frozen=True)
class PullProbeRecord:
    backend: str
    state: ProbeState
    detail: str
    credential_env: str | None = None
    """The environment variable whose value selects the pull credential, never its value."""


def _factory(backend: str, *, storage_mb: int):
    if backend == "daytona":
        from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415

        return DaytonaMachineFactory()
    if backend == "iris":
        from shellbox.backends.iris.machine import IrisMachineFactory  # noqa: PLC0415

        return IrisMachineFactory(cluster="cw-rno2a")
    raise ValueError(f"The pull probe supports remote backends only: {backend}")


def _canary_spec(backend: str, *, cpus: int, memory_mb: int, storage_mb: int) -> MachineSpec:
    if backend == "iris":
        # Enforcing an Iris storage limit needs a direct Kubernetes controller
        # (the survey keeps that cell VERIFY); the pull probe must not gate on it.
        return MachineSpec(RegistryImage(CANARY_IMAGE), cpus=cpus, memory_mb=memory_mb)
    return MachineSpec(RegistryImage(CANARY_IMAGE), cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb)


async def probe_internal_image_pull(
    backend: str, *, cpus: int = 1, memory_mb: int = 2048, storage_mb: int = 8192
) -> PullProbeRecord:
    """Create one machine from the internal canary image and report whether the pull works."""
    factory = _factory(backend, storage_mb=storage_mb)
    credential_env = DAYTONA_KEY_ENV if backend == "daytona" else None
    spec = _canary_spec(backend, cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb)
    machine = None
    try:
        machine = await factory.create(spec)
        return PullProbeRecord(
            backend=backend,
            state="supported",
            detail="the backend pulled the internal canary image and started the machine",
            credential_env=credential_env,
        )
    except UnsupportedMachineSpec as error:
        return PullProbeRecord(
            backend=backend,
            state="unsupported",
            detail=f"the backend refused the internal canary image: {error}",
            credential_env=credential_env,
        )
    except Exception as error:
        logger.warning("pull probe failed for %s: %s", backend, error)
        return PullProbeRecord(
            backend=backend,
            state="unsupported",
            detail=f"the backend could not pull the internal canary image: {error}",
            credential_env=credential_env,
        )
    finally:
        if machine is not None:
            try:
                await machine.close()
            except Exception as error:
                logger.warning("pull-probe machine close failed for %s: %s", backend, error)


def run_pull_probe(backend: str, **kwargs) -> PullProbeRecord:
    """Synchronous entry point for the survey's live-qualification runner."""
    return asyncio.run(probe_internal_image_pull(backend, **kwargs))
