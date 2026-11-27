# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The heredoc repair normalizes only the whitespace form; the lineage records each file."""

import pytest
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.shell_heredoc_repair import (
    REPAIR_VERSION,
    lineage_of,
    repair_dockerfile,
    repair_task,
)


def task_spec(dockerfile: bytes, extra_file: bytes = b"echo ok") -> TaskSpec:
    build = DockerBuildContext(
        files=(
            inline_resource("Dockerfile", dockerfile),
            inline_resource("task_file/setup.sh", extra_file),
        )
    )
    return TaskSpec(
        id="task-under-repair",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="do the task"),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=build,
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("cp", "/source-reward.txt", "/logs/verifier/reward.txt"),
            cwd="/",
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_build=build,
            ),
            answer_path=None,
        ),
        resources=ResourceGroups(
            oracle=(inline_resource("solution/solve.sh", b"#!/bin/bash\necho solved\n"),),
        ),
        tags=("harbor",),
    )


BROKEN_DOCKERFILE = b"""FROM ubuntu:22.04
RUN apt-get update && apt-get install -y git gnupg
RUN cat << 'EOF' > /app/task_file/gpg_wrapper.sh
#!/bin/bash
gpg --homedir /app/task_file/custom_gnupg "$@"
EOF
COPY << 'PHPEOF' /app/index.php
<?php echo 1;
PHPEOF
"""

CLEAN_DOCKERFILE = b"""FROM ubuntu:22.04
RUN cat <<'EOF' > /app/ok.sh
echo tight
EOF
RUN cat <<EOF > /app/unquoted.sh
echo unquoted
EOF
"""


def test_repair_dockerfile_normalizes_only_the_whitespace_form():
    fixed, count = repair_dockerfile(BROKEN_DOCKERFILE.decode())
    assert count == 2
    assert "<<'EOF' > /app/task_file/gpg_wrapper.sh" in fixed
    assert "<<'PHPEOF' /app/index.php" in fixed
    # idempotent
    again, again_count = repair_dockerfile(fixed)
    assert again == fixed and again_count == 0


@pytest.mark.parametrize(
    "text",
    [
        "RUN cat <<EOF > f\nbody\nEOF\n",
        "RUN cat <<- 'EOF' > f\nbody\nEOF\n",
        "RUN echo '<< quoted >>'\n",
    ],
)
def test_repair_dockerfile_leaves_valid_forms_alone(text):
    fixed, count = repair_dockerfile(text)
    assert fixed == text
    assert count == 0


def test_repair_task_fixes_both_environments_and_records_lineage():
    task = task_spec(BROKEN_DOCKERFILE)
    parent_sha = task_spec_sha256(task)
    repaired = repair_task(task)
    assert repaired is not task
    assert task_spec_sha256(repaired) != parent_sha
    for build in (repaired.environment_requirements.docker_build, repaired.grader.environment.docker_build):
        files = {file.path: resource_bytes(file) for file in build.files}
        assert b"<<'EOF'" in files["Dockerfile"]
        assert b"<< 'EOF'" not in files["Dockerfile"]
        assert b"<<'PHPEOF'" in files["Dockerfile"]
        assert files["task_file/setup.sh"] == b"echo ok"  # non-Dockerfile context files pass through
    lineage = lineage_of(repaired)
    assert lineage is not None and lineage.version == REPAIR_VERSION
    assert lineage.input_task_sha256 == parent_sha
    assert {change.file for change in lineage.files} == {"actor:Dockerfile", "grader:Dockerfile"}
    assert repaired.labels["heredoc_repair_version"] == REPAIR_VERSION
    assert lineage_of(task) is None


def test_repair_task_passes_through_clean_tasks():
    task = task_spec(CLEAN_DOCKERFILE)
    repaired = repair_task(task)
    assert repaired is task
    assert lineage_of(repaired) is None
    assert "heredoc_repair_version" not in repaired.labels
