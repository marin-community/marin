# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare dashboard ports before serving; workers consume the local cache."""

import shellsim

TOOL_PACKAGES = ("numpy",)


def prepare_tool_runtime(catalog: str) -> None:
    shellsim.Context(pypi_catalog=catalog).Environment(pypi_packages=TOOL_PACKAGES)


def tool_environment(catalog: str, *, cpu: int, output: int) -> shellsim.Environment:
    return shellsim.Context(pypi_catalog=catalog, offline=True).Environment(
        pypi_packages=TOOL_PACKAGES,
        cpu=cpu,
        memory=1024 * 1024 * 1024,
        disk=512 * 1024 * 1024,
        output=output,
    )
