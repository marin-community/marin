# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cohort conversion keeps the package harness and refuses what it cannot express."""

import hashlib
import tomllib

import pytest
from finestore.rl.curation import TaskFormat
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.task_format import FormatRow, detect_task_format
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.harbor_cohort_conversion import (
    CONVERSION_PROTOCOL,
    COPIED_TEST_STDOUT,
    COPIED_VERDICT,
    CohortConversionRejected,
    _task_environment,
    _task_verifier,
    convert_cohort_task,
)

CONTROL_IMAGE = "registry/control-task-image@sha256:" + "a" * 64


def _entry(name: str, content: bytes, mode: str = "644") -> object:
    return inline_resource(name, content).model_copy(update={"mode": mode})


TASK_TOML_TEMPLATE = """\
[environment]
docker_image = "{image}"

[verifier]
timeout_sec = 900.0
env = {{ VERIFIER_MODE = "strict" }}
"""


def _package(
    *,
    task_toml: str | None = None,
    with_verifier_config: bool = True,
    harbor_tag: bool = True,
    grader: NoGrader | None = None,
    test_script_mode: str = "755",
    extra_files: tuple = (),
) -> TaskSpec:
    toml = task_toml if task_toml is not None else TASK_TOML_TEMPLATE.format(image=CONTROL_IMAGE)
    verifier_files = [_entry("tests/test.sh", b"#!/bin/bash\nexit 0\n", test_script_mode)]
    if with_verifier_config:
        verifier_files.append(_entry("tests/verifier.toml", b'mode = "pytest"\npaths = ["tests/test_sample.py"]\n'))
    verifier_files.extend(
        [
            _entry("tests/trusted_test_paths.txt", b"tests/test_sample.py\n"),
            _entry("instruction.md", b"Fix the repository.\n"),
            _entry("task.toml", toml.encode()),
            _entry("environment/Dockerfile", b"FROM ubuntu@sha256:" + b"b" * 60 + b"\n"),
            _entry("environment/setup.sh", b"#!/bin/bash\ntrue\n"),
        ]
    )
    verifier_files.extend(extra_files)
    return TaskSpec(
        id="Marin:cohort-fixture/task-1",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the repository."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader if grader is not None else NoGrader(reason="Harbor grades the completed trial", contract={}),
        resources=ResourceGroups(verifier=tuple(verifier_files)),
        tags=("harbor",) if harbor_tag else (),
    )


def _convert(task: TaskSpec):
    return convert_cohort_task(task, input_task_sha256=task_spec_sha256(task))


def test_conversion_preserves_the_payload_and_rewrites_only_the_grading_fields():
    task = _package()
    out = _convert(task)

    assert out.task.context == task.context
    assert out.task.resources == task.resources
    assert out.task.answer_type == task.answer_type
    assert out.task.grader.kind == "script"
    assert out.task.tags == ()
    assert out.task.labels["harbor_cohort_conversion"] == CONVERSION_PROTOCOL
    assert out.task.environment_requirements.docker_image == CONTROL_IMAGE
    assert out.changed_fields == ("environment_requirements", "execution_requirements", "grader", "tags", "labels")
    assert out.input_task_sha256 == task_spec_sha256(task)
    assert out.output_task_sha256 == hashlib.sha256(out.task.model_dump_json().encode()).hexdigest()


def test_converted_row_detects_as_a_taskspec_row():
    out = _convert(_package())
    assert (
        detect_task_format(FormatRow(raw={"task_spec": out.task.model_dump_json()}, task=out.task))
        is TaskFormat.TASKSPEC
    )


def test_grader_replays_the_harbor_verifier_steps_in_order():
    task = _package(
        task_toml=(
            TASK_TOML_TEMPLATE.format(image=CONTROL_IMAGE)
            + '\n[[verifier.collect]]\ncommand = "tar -czf /tmp/keep.tgz /testbed"\n'
        )
    )
    out = _convert(task)
    grader = out.task.grader
    steps = list(grader.collect)
    extract = next(command for command in steps if "tar -C /tests -xzf" in command.argv[-1])
    assert extract.argv[-1].startswith("set -euo pipefail")
    assert steps[-2].argv == ("bash", "-c", "tar -czf /tmp/keep.tgz /testbed")
    assert steps[-1].argv == (
        "bash",
        "-c",
        "rm -rf /logs/verifier && mkdir -p /logs/verifier && "
        "( ( /tests/test.sh ) > /logs/verifier/test-stdout.txt 2>&1 || status=$?; "
        'printf "%s\\n" "${status-0}" > /logs/verifier/exit-code.txt ); exit 0',
    )
    assert steps[-1].env == {"VERIFIER_MODE": "strict"}
    assert all(command.argv[:1] != ("chmod",) for command in steps)
    assert grader.answer_path is None
    assert grader.timeout == 900.0
    assert [artifact.target for artifact in grader.artifacts] == [
        COPIED_TEST_STDOUT,
        COPIED_VERDICT,
        "/source-exit-code.txt",
    ]
    assert grader.argv == ("cp", COPIED_VERDICT, "/converted-verdict.json")
    assert grader.reward.files[0].path == "/converted-verdict.json"
    assert grader.reward.files[0].format.value == "verdict_json"
    assert grader.environment.docker_image == CONTROL_IMAGE


def test_the_restored_tree_keeps_the_exec_bit_without_a_chmod_and_adds_one_without_it():
    executable = _convert(_package(test_script_mode="755"))
    assert all(command.argv[:1] != ("chmod",) for command in executable.task.grader.collect)
    unpackaged = _convert(_package(test_script_mode="644"))
    chmods = [command for command in unpackaged.task.grader.collect if command.argv[:2] == ("chmod", "+x")]
    assert [command.argv for command in chmods] == [("chmod", "+x", "/tests/test.sh")]


def test_verifier_config_absent_falls_back_to_the_reward_files():
    out = _convert(_package(with_verifier_config=False))
    grader = out.task.grader
    targets = [artifact.target for artifact in grader.artifacts]
    assert targets == [COPIED_TEST_STDOUT, "/source-reward.json", "/source-reward.txt", "/source-exit-code.txt"]
    assert [file.format.value for file in grader.reward.files] == ["json", "number"]
    assert grader.reward.files[0].path == "/converted-reward.json"
    assert grader.reward.files[1].path == "/converted-reward.txt"
    assert grader.argv == (
        "bash",
        "-c",
        "cp /source-reward.json /converted-reward.json 2>/dev/null || cp /source-reward.txt /converted-reward.txt",
    )


def test_declared_execution_limits_lift_only_when_the_declaration_is_complete():
    """Incomplete declarations stay unlifted: the harness fills no defaults."""
    complete = _package(
        task_toml=(
            '[environment]\ndocker_image = "' + CONTROL_IMAGE + '"\n'
            "cpus = 4\nmemory_mb = 4096\nstorage_mb = 10240\n\n"
            "[agent]\ntimeout_sec = 1800.0\n\n[verifier]\ntimeout_sec = 900.0\n"
        )
    )
    out = _convert(complete)
    requirements = out.task.execution_requirements
    assert requirements is not None
    assert requirements.cpus == 4 and requirements.memory_mb == 4096 and requirements.storage_mb == 10240
    assert requirements.agent_timeout_sec == 1800.0 and requirements.verifier_timeout_sec == 900.0
    partial = _package(
        task_toml='[environment]\ndocker_image = "'
        + CONTROL_IMAGE
        + '"\ncpus = 4\nmemory_mb = 4096\nstorage_mb = 10240\n'
    )
    assert _convert(partial).task.execution_requirements is None


def test_build_from_the_environment_dockerfile_when_no_image_is_declared():
    task = _package(
        task_toml='[environment]\nworkdir = "/repo"\nenv = { REPO = "/repo" }\n\n[verifier]\ntimeout_sec = 300.0\n'
    )
    out = _convert(task)
    build = out.task.environment_requirements.docker_build
    assert build is not None and sorted(file.path for file in build.files) == ["Dockerfile", "setup.sh"]
    assert out.task.environment_requirements.docker_image is None
    assert out.task.environment_requirements.working_directory == "/repo"
    assert out.task.environment_requirements.environment_variables == {"REPO": "/repo"}
    assert out.task.grader.environment.docker_build is build
    assert out.task.grader.timeout == 300.0


def test_not_a_native_package_is_rejected():
    """The package contract itself refuses the row before any conversion rewrite."""
    with pytest.raises(CohortConversionRejected, match="not a native Harbor package"):
        _convert(_package(harbor_tag=False))
    stripped = _package()
    stripped = stripped.model_copy(
        update={
            "resources": ResourceGroups(
                verifier=tuple(resource for resource in stripped.resources.verifier if resource.path != "task.toml")
            )
        }
    )
    with pytest.raises(CohortConversionRejected, match="not a native Harbor package"):
        _convert(stripped)


def test_missing_tests_tree_is_refused_by_the_package_contract():
    """Stripping the tests tree breaks the native package contract itself."""
    task = _package()
    stripped = task.model_copy(
        update={
            "resources": ResourceGroups(
                verifier=tuple(
                    resource for resource in task.resources.verifier if not resource.path.startswith("tests/")
                )
            )
        }
    )
    with pytest.raises(CohortConversionRejected, match="not a native Harbor package"):
        _convert(stripped)


def test_non_root_verifier_user_is_rejected():
    task = _package(task_toml=f'[environment]\ndocker_image = "{CONTROL_IMAGE}"\n\n[verifier]\nuser = "agent"\n')
    with pytest.raises(CohortConversionRejected, match="declares user 'agent'"):
        _convert(task)


def test_separate_verifier_container_is_rejected():
    task = _package(
        task_toml=(
            f'[environment]\ndocker_image = "{CONTROL_IMAGE}"\n\n'
            f'[verifier.environment]\ndocker_image = "registry/other@sha256:' + "b" * 64 + '"\n'
        )
    )
    with pytest.raises(CohortConversionRejected, match="separate verifier container"):
        _convert(task)


def test_multi_step_and_windows_are_rejected():
    steps = f'[environment]\ndocker_image = "{CONTROL_IMAGE}"\n\n[[steps]]\nname = "first"\n'
    with pytest.raises(CohortConversionRejected, match="multi-step"):
        _convert(_package(task_toml=steps))
    windows = '[environment]\ndocker_image = "registry/x@sha256:' + "a" * 64 + '"\nos = "windows"\n'
    with pytest.raises(CohortConversionRejected, match="Windows"):
        _convert(_package(task_toml=windows))


def test_the_read_matches_the_vendored_harbor_parser_where_harbor_is_installed():
    """Parity with the vendored parser; Harbor is absent from the base environment."""
    task_config = pytest.importorskip("harbor.models.task.config")

    toml_text = (
        f'[environment]\ndocker_image = "{CONTROL_IMAGE}"\ncpus = 4\n\n'
        '[verifier]\ntimeout_sec = 777.0\nenv = { VERIFIER_MODE = "strict" }\n'
        '[[verifier.collect]]\ncommand = "tar -czf /tmp/keep.tgz /testbed"\n'
    )
    config = task_config.TaskConfig.model_validate_toml(toml_text)
    data = tomllib.loads(toml_text)
    environment, verifier = _task_environment(data), _task_verifier(data)
    assert environment["docker_image"] == config.environment.docker_image
    assert environment["os"] == config.environment.os.value
    assert environment["cpus"] == config.environment.cpus
    assert verifier["timeout_sec"] == config.verifier.timeout_sec
    assert verifier["env"] == config.verifier.env
    assert verifier["user"] == config.verifier.user
    assert verifier["collect"][0]["command"] == config.verifier.collect[0].command
