# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published cohort fixtures shared by importer and oracle receipt tests."""

import json
from dataclasses import replace

from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


def prebuilt_source(info: SourceInfo) -> RlDataSource[PrebuiltCohort]:
    count = info.count if info.count is not None else 1
    return RlDataSource(
        info=replace(
            info,
            within_source_dedup=info.within_source_dedup
            or deduplication_evidence("open-athena/fixture-tasks", "a" * 40, "fixture-1", count),
            lexical_contamination=info.lexical_contamination
            or decontamination_evidence("open-athena/fixture-tasks", "a" * 40, count),
        ),
        config=PrebuiltCohort(
            name=info.id.rsplit(":", 1)[-1],
            version="fixture-1",
            repo="open-athena/fixture-tasks",
            revision="a" * 40,
            expected_rows=count,
        ),
        pipeline=import_taskcompendium_prebuilt,
    )


def prebuilt_task(task_id: str, *, image: str | None = IMAGE, capabilities: tuple[str, ...] = ()) -> dict:
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=image,
        working_directory="/app",
        setup_commands=("true",),
        capabilities=capabilities,
    )
    grader = ScriptGrader(
        argv=("true",),
        environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=requirements,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="d", revision="r", row=task_id, importer_revision="1"),
        resources=ResourceGroups(),
    )
    return {
        "path": task_id,
        "lowered_task_spec": json.dumps({"task": spec.model_dump(mode="json"), "runtime": {"task_machine": {}}}),
    }
