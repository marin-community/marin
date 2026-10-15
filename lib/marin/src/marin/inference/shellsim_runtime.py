# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare dashboard ports before serving; workers consume the local cache.

Preparation and offline workers must share the same precompiled trust policy,
which selects the cache namespace.
"""

import shellsim

TOOL_PACKAGES = ("numpy",)


def prepare_tool_runtime(catalog: str) -> None:
    shellsim.Context(
        pypi_catalog=catalog,
        trusted_precompiled=catalog == shellsim.DEFAULT_PYPI_CATALOG,
    ).Environment(pypi_packages=TOOL_PACKAGES)


def tool_environment(catalog: str, *, cpu: int, output: int) -> shellsim.Environment:
    return shellsim.Context(
        pypi_catalog=catalog,
        offline=True,
        trusted_precompiled=catalog == shellsim.DEFAULT_PYPI_CATALOG,
    ).Environment(
        pypi_packages=TOOL_PACKAGES,
        cpu=cpu,
        memory=1024 * 1024 * 1024,
        disk=512 * 1024 * 1024,
        output=output,
    )
