# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared local ports runtime for Python-executing simulator tests."""

import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Backend, Machine, MachineFactory, MachineSpec, PackageSpec


@pytest.fixture(scope="session")
def shellsim_factory():
    # ShellSim is optional for suites that only use native backends.
    import shellsim  # noqa: PLC0415
    from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: PLC0415

    catalog = os.environ.get("SHELLSIM_TEST_CATALOG")
    if catalog is None:
        raise RuntimeError(
            "Run uv run python infra/ci/prepare_shellsim_ports.py before pytest, "
            "then set SHELLSIM_TEST_CATALOG to the printed file:// URI"
        )
    uri = urlparse(catalog)
    if uri.scheme != "file" or uri.netloc or not Path(uri.path).is_file():
        raise ValueError("SHELLSIM_TEST_CATALOG must identify a prepared local file:// catalog")
    return ShellSimMachineFactory(context=shellsim.Context(pypi_catalog=catalog))


class ShellSimImageFactory(DockerMachineFactory):
    """Stand in for a pinned image with ShellSim and optional prepared Python ports."""

    backend = Backend.DOCKER

    def __init__(self, simulator: MachineFactory | None = None):
        self.packages = PackageSpec(pypi_packages=("pytest",)) if simulator is not None else PackageSpec()
        if simulator is None:
            # Import the optional backend only when a simulated image is requested.
            from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: PLC0415

            simulator = ShellSimMachineFactory()
        self.simulator = simulator

    async def create(self, spec: MachineSpec) -> Machine:
        return await self.simulator.create(replace(spec, source=self.packages, workdir=spec.workdir or "/workspace"))
