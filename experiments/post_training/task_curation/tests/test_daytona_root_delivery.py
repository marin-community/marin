# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The census reads a root-delivery repair's archived source configuration."""

import pytest
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.daytona_identity import native_task_toml


def _root_delivery_task(toml: bytes, *, with_archive: bool) -> dict:
    verifier = [inline_resource("source_archive/tests/test.sh", b"test")]
    if with_archive:
        verifier.append(inline_resource("source_archive/task.toml", toml))
    task = TaskSpec(
        id="repaired-native",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the repository"),)),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        source=Source(dataset="fixture", revision="fixture", row="repaired-native", importer_revision="1"),
        environment_requirements=EnvironmentRequirements(
            command_semantics="linux_process", docker_image="actor@sha256:" + "a" * 64
        ),
        grader=ScriptGrader(
            argv=("bash", "tests/test.sh"),
            environment=EnvironmentRequirements(
                command_semantics="linux_process", docker_image="actor@sha256:" + "a" * 64
            ),
        ),
        resources=ResourceGroups(verifier=tuple(verifier)),
        integrity_contract=RootDeliveryContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            verifier_files=(DeliveredVerifierFile(name="test.sh", mode=493, sha256="b" * 64),),
        ),
        tags=("harbor", "root-delivery", "prepared-images"),
    )
    return task.model_dump(mode="json")


def test_native_task_toml_reads_the_root_delivery_archived_package():
    """A root-delivery repair archives the original package beside the delivered grader."""
    toml = b"[agent]\ntimeout_sec = 1800\n[verifier]\ntimeout_sec = 1800\n"
    task = _root_delivery_task(toml, with_archive=True)
    assert native_task_toml(task) == toml


def test_native_task_toml_refuses_a_harbor_package_without_its_configuration():
    task = _root_delivery_task(b"", with_archive=False)
    with pytest.raises(ValueError, match="neither the delivered nor the archived"):
        native_task_toml(task)
