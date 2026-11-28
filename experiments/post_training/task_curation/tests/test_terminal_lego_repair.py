# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Lego-X repair grants the shell capability and patches the verifier bytes in place."""

import base64
import gzip
import io
import tarfile

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
    VerifierCommand,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.terminal_lego_repair import (
    PATCH_MARKER,
    REPAIR_VERSION,
    lineage_of,
    repair_terminal_lego_task,
    repaired_test_script,
)

CANARY = (
    "# Terminal-Bench Canary String, DO NOT REMOVE:\n"
    "# terminal-bench-canary GUID 4cdea708-0000-0000-0000-55576302\n"
    "#!/bin/bash\n"
    "\n"
)

SOURCE_TEST_SCRIPT = CANARY + (
    "apt-get update\n"
    "apt-get install -y curl\n"
    "\n"
    "# Install uv\n"
    "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n"
    "\n"
    "source $HOME/.local/bin/env\n"
    "\n"
    'if [ "$PWD" = "/" ]; then\n'
    '    echo "Error: No working directory set."\n'
    "    exit 1\n"
    "fi\n"
    "\n"
    "uvx \\\n"
    "  -p 3.13 \\\n"
    "  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA\n"
    "\n"
    "if [ $? -eq 0 ]; then\n"
    "  echo 1 > /logs/verifier/reward.txt\n"
    "else\n"
    "  echo 0 > /logs/verifier/reward.txt\n"
    "fi\n"
)


def lego_task(capabilities: tuple[str, ...] = ()) -> TaskSpec:
    build = DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM ubuntu:22.04\nWORKDIR /app\n"),))
    capture = io.BytesIO()
    with tarfile.open(fileobj=capture, mode="w:gz") as tar:
        for name, content in (
            ("test.sh", SOURCE_TEST_SCRIPT.encode()),
            ("test_outputs.py", b"def test_ok():\n    assert True\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(content))
    embedded = (
        "mkdir -p /tests && printf '%s' "
        + base64.b64encode(capture.getvalue()).decode()
        + " | base64 -d > /tests/.capture.tar.gz"
    )
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=build,
        capabilities=capabilities,
    )
    return TaskSpec(
        id="lego-x-terminal-lego-15k-fixture",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="configure git signing"),)),
        environment_requirements=environment,
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
            collect=(
                VerifierCommand(argv=("bash", "-c", "quiesce"), cwd="/"),
                VerifierCommand(argv=("bash", "-c", embedded), cwd="/"),
                VerifierCommand(
                    argv=(
                        "bash",
                        "-c",
                        "bash /tests/test.sh > /logs/verifier/test-output.txt 2>&1 || s=$?; "
                        "printf '%s\\n' \"$s\" > /logs/verifier/exit-code.txt; exit 0",
                    )
                ),
            ),
        ),
        resources=ResourceGroups(
            verifier=(
                inline_resource("source_archive/tests/test.sh", SOURCE_TEST_SCRIPT.encode()),
                inline_resource("source_archive/tests/test_outputs.py", b"def test_ok():\n    assert True\n"),
            ),
            oracle=(inline_resource("solution/solve.sh", b"#!/bin/bash\necho solved\n"),),
        ),
        tags=("harbor",),
    )


def capture_test_script(task: TaskSpec) -> str:
    embedded = next(
        command.argv[2]
        for command in task.grader.collect
        if command.argv[2].endswith(" | base64 -d > /tests/.capture.tar.gz")
    )
    payload = base64.b64decode(
        embedded[len("mkdir -p /tests && printf '%s' ") : -len(" | base64 -d > /tests/.capture.tar.gz")]
    )
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload))) as tar:
        return tar.extractfile("test.sh").read().decode()


def test_repair_grants_shell_and_patches_both_verifier_surfaces():
    task = lego_task()
    parent_sha = task_spec_sha256(task)
    repaired = repair_terminal_lego_task(task)
    assert repaired.environment_requirements.capabilities == ("shell",)
    live = capture_test_script(repaired)
    archived = resource_bytes(
        next(r for r in repaired.resources.verifier if r.path == "source_archive/tests/test.sh")
    ).decode()
    for script in (live, archived):
        assert PATCH_MARKER in script
        assert 'APT::Sandbox::User "root"' in script
        assert "_harbor_write_missing_reward" in script
        assert 'if [ "$PWD" = "/" ]; then cd /app 2>/dev/null || true; fi' in script
        # the pinned uv release is pre-installed sandbox-safe before the source chain
        assert "astral-sh/uv/releases/download/0.9.5/uv-" in script
        assert "--no-same-owner -C /usr/local/bin" in script
        # the canary header and the source's own body stay verbatim
        assert script.startswith(CANARY)
        assert "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh" in script
        assert "uvx \\" in script
        assert "pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA" in script
    assert live == archived
    lineage = lineage_of(repaired)
    assert lineage is not None and lineage.version == REPAIR_VERSION
    assert lineage.capabilities_before == [] and lineage.capabilities_after == ["shell"]
    assert lineage.input_task_sha256 == parent_sha
    assert lineage.capture_sha256_before != lineage.capture_sha256_after
    assert repaired.labels["terminal_lego_repair_version"] == REPAIR_VERSION


def test_repair_is_idempotent():
    task = repair_terminal_lego_task(lego_task())
    again = repair_terminal_lego_task(task)
    assert again.model_dump_json() == task.model_dump_json()


def test_repair_rejects_missing_capture():
    task = lego_task()
    task = task.model_copy(update={"grader": task.grader.model_copy(update={"collect": ()})})
    with pytest.raises(ValueError, match="no capture command"):
        repair_terminal_lego_task(task)


def test_repaired_test_script_keeps_unrelated_tasks_intact():
    # a verifier without apt-get still gets the trap; the body is untouched otherwise
    script = CANARY + "uvx -p 3.13 pytest /tests/test_outputs.py -rA\n"
    patched = repaired_test_script(script)
    assert PATCH_MARKER in patched and "apt" not in patched.split(PATCH_MARKER)[1].split("\n\n", 1)[-1]


def test_repaired_test_script_skips_uv_preinstall_without_the_source_installer():
    # no astral installer line: no uv pre-install is templated into the patch
    script = CANARY + "apt-get update\nuvx -p 3.13 pytest /tests/test_outputs.py -rA\n"
    patched = repaired_test_script(script)
    assert PATCH_MARKER in patched
    assert "astral-sh/uv/releases/download" not in patched
    assert "_harbor_uv_preinstall" not in patched


def test_repaired_test_script_reuses_the_pinned_uv_version():
    # a script pinned to a different uv release pre-installs exactly that release
    script = CANARY + "curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh\nuvx pytest\n"
    patched = repaired_test_script(script)
    assert "astral-sh/uv/releases/download/0.9.7/uv-" in patched
    assert "download/0.9.5/" not in patched
