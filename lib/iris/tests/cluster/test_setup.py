# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EnvironmentSpec composes setup scripts onto the wire."""

import pytest
from iris.cluster.setup import iris_runtime_setup_script
from iris.cluster.types import EnvironmentSpec


@pytest.mark.parametrize(
    "setup_scripts, expected_user_scripts",
    [
        # Default: iris builds one project-setup script for the user.
        (None, None),
        # Custom scripts pass through verbatim, in order.
        (["echo a", "echo b"], ["echo a", "echo b"]),
        # Whitespace-only entries are dropped.
        (["echo a", "   "], ["echo a"]),
        # No setup at all: nothing runs, not even the iris script.
        ([], []),
    ],
)
def test_to_proto_composes_user_scripts_then_iris(setup_scripts, expected_user_scripts):
    resolved = list(EnvironmentSpec(setup_scripts=setup_scripts).to_proto().setup_scripts)

    if expected_user_scripts == []:
        assert resolved == []
        return

    # iris always appends its own runtime-deps script as the final step so its
    # features keep working regardless of what the user setup does.
    assert resolved[-1] == iris_runtime_setup_script()
    user_scripts = resolved[:-1]
    if expected_user_scripts is None:
        assert len(user_scripts) == 1  # the generated default
    else:
        assert user_scripts == expected_user_scripts
