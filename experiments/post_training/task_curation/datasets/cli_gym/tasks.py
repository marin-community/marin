# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI-Gym's original test command with separately parsed, pinned source verdicts."""

import hashlib
import json
from pathlib import Path

import tomlkit
import yaml
from taskcompendium.harbor.root_delivery import DeliveredVerifier, delivery_commands, shell_quiesce_command
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import inline_resource

HERE = Path(__file__).parent
VERSION = "2026.11.20.cli-gym-native.1"
UPSTREAM_REVISION = "48bb920b728a25a55a5b442303e901919654599e"
PARSER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
LOG = "/logs/verifier/cli-gym-test-output.txt"
EXIT = "/logs/verifier/cli-gym-exit-code.txt"
REWARD = "/logs/verifier/reward.txt"


def convert_cli(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    try:
        config = yaml.safe_load(data["task_yaml"])
    except yaml.YAMLError as error:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="malformed_task_yaml",
            detail=json.dumps(
                {
                    "task_id": data["task_id"],
                    "yaml_sha256": hashlib.sha256(data["task_yaml"].encode()).hexdigest(),
                    "error": str(error),
                }
            ),
        )
    compose = yaml.safe_load(data["docker_compose"])
    services = compose["services"]
    if len(services) != 1:
        raise ValueError("CLI-Gym single-service contract changed")
    service = next(iter(services.values()))
    expected = {"build", "container_name", "command", "volumes"}
    optional = {"environment", "cpus"}
    if (
        not expected <= service.keys()
        or service.keys() - expected - optional
        or service["build"] != {"context": ".", "dockerfile": "Dockerfile"}
    ):
        raise ValueError("Unmeasured CLI-Gym Compose fields require a faithful implementation")
    if service["command"] != ["sh", "-c", "sleep infinity"] or service.get("environment", []) not in (
        [],
        ["TEST_DIR=${T_BENCH_TEST_DIR}"],
    ):
        raise ValueError("CLI-Gym service startup or environment changed")
    if set(service["volumes"]) != {
        "${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}",
        "${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}",
    }:
        raise ValueError("CLI-Gym has non-harness volume semantics")
    if config["parser_name"] not in {"pytest", "swebench"} or config["run_tests_in_same_shell"]:
        raise ValueError("Unmeasured CLI-Gym parser or shell-session contract")
    agent_deadline = config["max_agent_timeout_sec"]
    verifier_deadline = config["max_test_timeout_sec"]
    native_config = tomlkit.dumps(
        {
            "environment": {"cpus": service["cpus"]} if "cpus" in service else {},
            "agent": {"timeout_sec": agent_deadline},
            "verifier": {"timeout_sec": verifier_deadline},
        }
    ).encode()
    tests = data["run_tests"].encode()
    env = EnvironmentRequirements(
        capabilities=("shell",),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", data["dockerfile"].encode()),)),
        environment_variables={"TEST_DIR": "/tests"} if service.get("environment") else {},
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=config["instruction"]),)),
        environment_requirements=env,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=(
                "python",
                "-I",
                "/tests/score.py",
                "--parser",
                config["parser_name"],
                "--output",
                "/source-test-output.txt",
                "--reward",
                REWARD,
            ),
            cwd="/",
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=PARSER_IMAGE
            ),
            answer_path=None,
            timeout=verifier_deadline,
            collect=(
                shell_quiesce_command(),
                *delivery_commands(
                    DeliveredVerifier({"run-tests.sh": tests}, frozenset({"run-tests.sh"})),
                    entrypoint="run-tests.sh",
                    output_path=LOG,
                    exit_code_path=EXIT,
                ),
            ),
            artifacts=(
                VerifierArtifact(source=LOG, target="/source-test-output.txt", kind=ArtifactKind.FILE),
                VerifierArtifact(source=EXIT, target="/source-test-exit-code.txt", kind=ArtifactKind.FILE),
            ),
            reward=FileReward(files=(RewardFile(path=REWARD, format=RewardFileFormat.NUMBER),)),
        ),
        resources=ResourceGroups(
            verifier=(
                inline_resource("score.py", (HERE / "score.py").read_bytes()),
                inline_resource("upstream.json", (HERE / "upstream.json").read_bytes()),
                inline_resource("LICENSE", (HERE / "LICENSE").read_bytes()),
                inline_resource("task.toml", native_config),
                *(
                    inline_resource("source_archive/" + name, data[key].encode())
                    for name, key in (
                        ("task.yaml", "task_yaml"),
                        ("Dockerfile", "dockerfile"),
                        ("docker-compose.yaml", "docker_compose"),
                        ("run-tests.sh", "run_tests"),
                    )
                ),
            )
        ),
        integrity_contract=RootDeliveryContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            verifier_files=(
                DeliveredVerifierFile(name="run-tests.sh", mode=0o755, sha256=hashlib.sha256(tests).hexdigest()),
            ),
        ),
        tags=("cli-gym", "root-delivery"),
        labels={
            "native_import_version": VERSION,
            "upstream_parser_revision": UPSTREAM_REVISION,
            "source_deadlines": json.dumps({"agent": agent_deadline, "verifier": verifier_deadline}),
            "original_oracle": "absent",
        },
    )


def decode_cli(row: dict, _context: ConversionContext) -> dict:
    """Retain the source's task-folder identifier as pipeline provenance."""
    return {**row, "path": row["task_id"]}
