# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EnvironmentSpec resolves the user setup scripts onto the wire."""

import pytest
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
