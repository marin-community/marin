# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EnvironmentSpec resolves the user setup scripts onto the wire."""

import pytest
from iris.cluster.setup_scripts import default_setup_script, iris_runtime_setup_script
from iris.cluster.types import EnvironmentSpec


@pytest.mark.parametrize(
    "setup_scripts, expected",
    [
        # Default: iris builds one project-setup script. The iris runtime-deps
        # script is appended later, in build_runtime_entrypoint — not here.
        (None, None),
        # Custom scripts pass through verbatim, in order.
        (["echo a", "echo b"], ["echo a", "echo b"]),
        # Whitespace-only entries are dropped.
        (["echo a", "   "], ["echo a"]),
        # No setup at all.
        ([], []),
    ],
)
def test_to_proto_resolves_user_setup_scripts(setup_scripts, expected):
    resolved = list(EnvironmentSpec(setup_scripts=setup_scripts).to_proto().setup_scripts)

    if expected is None:
        assert len(resolved) == 1  # the generated default
    else:
        assert resolved == expected


def test_default_setup_scripts_do_not_suppress_uv_output():
    """uv runs at default verbosity so its progress streams into task logs.

    ``--quiet`` hides the only signal a live setup gives (which package is
    resolving/downloading, and whether it is making progress), so the generated
    scripts must never pass it.
    """
    scripts = [
        default_setup_script(extras=["gpu"], pip_packages=["foo"], python_version="3.12"),
        iris_runtime_setup_script(),
        next(iter(EnvironmentSpec(setup_scripts=None).to_proto().setup_scripts)),
    ]
    for script in scripts:
        assert "uv " in script  # sanity: these actually invoke uv
        assert "--quiet" not in script
