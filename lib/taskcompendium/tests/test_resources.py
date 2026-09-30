# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource exposure and integrity across the task export boundary."""

import base64
import hashlib
import stat

import pytest

from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.resources import (
    ResourceReference,
    ResourceVisibility,
    TaskResource,
    materialize_resources,
    validate_resources,
)
from taskcompendium.submission import PlainText


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


def test_materialize_inline_binary_resource_preserves_exact_bytes(tmp_path):
    payload = b"\x00\xff\x80\n"
    resource = TaskResource(
        path="inputs/blob.bin",
        visibility=ResourceVisibility.AGENT,
        content_base64=base64.b64encode(payload).decode("ascii"),
    )

    materialize_resources((resource,), tmp_path / "agent", visibility=ResourceVisibility.AGENT)

    assert (tmp_path / "agent/inputs/blob.bin").read_bytes() == payload


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


def test_materialize_pinned_reference_into_private_directory(tmp_path):
    payload = b'{"expected":true}\n'
    reference = ResourceReference(locator="dataset://row-0", sha256=hashlib.sha256(payload).hexdigest())
    resource = TaskResource(path="expected/state.json", visibility=ResourceVisibility.VERIFIER, reference=reference)

    materialize_resources(
        (resource,),
        tmp_path / "private",
        visibility=ResourceVisibility.VERIFIER,
        trusted_resolver=lambda selected: payload if selected == reference else b"",
    )

    assert (tmp_path / "private/expected/state.json").read_bytes() == payload


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


@pytest.mark.parametrize(
    "visibility", [ResourceVisibility.AGENT, ResourceVisibility.VERIFIER, ResourceVisibility.ORACLE]
)
def test_host_chat_rejects_resources_before_export(tmp_path, visibility):
    specification = TaskSpec(
        id="workspace-resource",
        context=ConversationInput(events=(TextMessage(role="user", content="Read the file."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("ready"),
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
        resources=(TaskResource(path="input.txt", visibility=visibility, content="ready"),),
    )
    convention = PlainText(id="plain")
    config = HarborEnvironmentConfig()
    destination = tmp_path / "task"

    assert compatible_lowerings(specification, (convention,), (config,)) == ()
    with pytest.raises(ValueError, match="Host chat cannot expose task resources"):
        lower_to_harbor(specification, convention, config, destination)
    assert not destination.exists()
