# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource exposure and integrity across the task export boundary."""

import hashlib
import stat

import pytest

from taskcompendium.resources import (
    ResourceReference,
    ResourceVisibility,
    TaskResource,
    materialize_resources,
    validate_resources,
)


def test_materialize_resources_exposes_only_selected_files(tmp_path):
    resources = (
        TaskResource(path="inputs/run.sh", visibility=ResourceVisibility.AGENT, executable=True, content="echo ready\n"),
        TaskResource(path="expected.json", visibility=ResourceVisibility.VERIFIER, content='{"ok": true}\n'),
    )
    destination = tmp_path / "agent"

    materialize_resources(resources, destination, visibility=ResourceVisibility.AGENT)

    script = destination / "inputs" / "run.sh"
    assert script.read_text() == "echo ready\n"
    assert script.stat().st_mode & stat.S_IXUSR
    assert not (destination / "expected.json").exists()


def test_materialize_resources_validates_private_reference_before_writing_agent_files(tmp_path):
    resources = (
        TaskResource(path="agent.txt", visibility=ResourceVisibility.AGENT, content="visible"),
        TaskResource(
            path="private.txt",
            visibility=ResourceVisibility.VERIFIER,
            reference=ResourceReference(locator="dataset://row-0", sha256=hashlib.sha256(b"expected").hexdigest()),
        ),
    )
    destination = tmp_path / "agent"

    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_resources(
            resources,
            destination,
            visibility=ResourceVisibility.AGENT,
            trusted_resolver=lambda reference: b"wrong",
        )

    assert not destination.exists()


def test_materialize_resources_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "agent"
    destination.mkdir()
    (destination / "inputs").symlink_to(outside, target_is_directory=True)
    resource = TaskResource(path="inputs/secret.txt", visibility=ResourceVisibility.AGENT, content="secret")

    with pytest.raises(ValueError, match="Unsafe resource parent"):
        materialize_resources((resource,), destination, visibility=ResourceVisibility.AGENT)

    assert not (outside / "secret.txt").exists()


@pytest.mark.parametrize("paths", [("a", "a/b"), ("A/x", "a/X"), ("a/../b",)])
def test_validate_resources_rejects_path_collisions_and_traversal(paths):
    with pytest.raises(ValueError):
        validate_resources(TaskResource(path=path, visibility=ResourceVisibility.AGENT, content="x") for path in paths)
