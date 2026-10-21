# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert offline, single-workspace Skill2Env tasks without rewriting native tests."""

import hashlib
import json
import re
import shlex
import tarfile
import tomllib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import unsupported
from taskcompendium.models import (
    AnswerType,
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
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, OracleCommand, RawRow
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.skill2env import grade

MEMORY_MB = 4096
WORKSPACE = "/workspace"
ADAPTER_PATH = "curation-adapter.py"
CONTRACT_PATH = "curation-contract.json"
RELEASE = "skill2env/skill2env@sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c"


def content_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, blob in sorted(files.items()):
        digest.update(f"{name}\0{hashlib.sha256(blob).hexdigest()}\n".encode())
    return digest.hexdigest()


def read_task(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Read one native archive, retaining file contents and permission modes."""
    manifest = json.loads((path.parent.parent / "release.json").read_text())
    if manifest["release"] != RELEASE:
        raise ValueError("Input belongs to another Skill2Env release")
    selected = set(manifest["selected"])
    if not selected.issubset(manifest["task_refs"]):
        raise ValueError("Selected tasks are absent from the release manifest")
    selected_names = {hashlib.sha256(ref.encode()).hexdigest() + ".tar.gz" for ref in selected}
    if path.name not in selected_names:
        return
    files = {}
    modes = {}
    with path.open("rb") as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir():
                continue
            if not member.isfile():
                yield {"release": RELEASE, "unsupported": "unsupported_package_links"}
                return
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or name in files:
                raise ValueError(f"Invalid or duplicate archive path: {name}")
            content = archive.extractfile(member)
            assert content is not None
            files[name] = content.read()
            modes[name] = format(member.mode & 0o7777, "o")
    config = tomllib.loads(files["task.toml"].decode())
    task_ref = f"{config['task']['name']}@sha256:{content_digest(files)}"
    if path.name != hashlib.sha256(task_ref.encode()).hexdigest() + ".tar.gz":
        raise ValueError("Archive contents differ from the pinned task filename")
    yield {"release": RELEASE, "task_ref": task_ref, "files": files, "modes": modes}


def convert_task(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    data = row.data
    if data["release"] != RELEASE:
        raise ValueError("Input belongs to another Skill2Env release")
    if "unsupported" in data:
        return unsupported(data["unsupported"], "Native export could not preserve this task")
    files = data["files"]
    task_name, task_digest = data["task_ref"].rsplit("@", 1)
    if task_digest != f"sha256:{content_digest(files)}":
        raise ValueError("Exported files differ from the pinned task")
    config = tomllib.loads(files["task.toml"].decode())
    if config["task"]["name"] != task_name:
        raise ValueError("Task identity differs from its native configuration")
    if any(path.startswith("steps/") for path in files):
        return unsupported("native_steps", "Multi-step tasks require a session execution contract")
    dockerfile = files["environment/Dockerfile"].decode()
    final_stage = re.split(r"(?im)^[ \t]*FROM[ \t]+", dockerfile)[-1]
    workdirs = re.findall(r"(?im)^[ \t]*WORKDIR[ \t]+([^\r\n]*)", final_stage)
    if not workdirs or re.fullmatch(r"\S+", workdirs[-1].strip()) is None:
        return unsupported("native_workspace", "Requires a final literal absolute WORKDIR within /workspace")
    workdir = PurePosixPath(workdirs[-1].strip())
    if (
        not workdir.is_relative_to(WORKSPACE)
        or ".." in workdir.parts
        or any(character in str(workdir) for character in ("$", "\\", "'", '"'))
    ):
        return unsupported("native_workspace", "Requires a final literal absolute WORKDIR within /workspace")
    working_directory = str(workdir)
    if not all(path.startswith(WORKSPACE + "/") for path in config.get("artifacts", [])):
        return unsupported("native_outputs", "Outputs outside /workspace need another capture contract")
    for section in ("environment", "agent", "verifier"):
        if config[section].get("network_mode") != "no-network" or config[section].get("allowed_hosts"):
            return unsupported("native_network", "Only network-disabled task contracts are supported")
    if any(
        config["environment"].get(key) != value
        for key, value in (("cpus", 2), ("memory_mb", MEMORY_MB), ("storage_mb", 4096))
    ):
        return unsupported("native_resources", "This recipe supports 2 CPU / 4096 MiB RAM / 4096 MiB storage tasks")
    if config["environment"].get("mcp_servers") or config["verifier"].get("collect"):
        return unsupported("native_services", "MCP services and verifier collect hooks need explicit adapters")
    if "tests/test.sh" not in files or "solution/solve.sh" not in files:
        return unsupported("native_commands", "Expected native verifier and reference solution scripts")
    if any(path in files for path in (f"tests/{ADAPTER_PATH}", f"tests/{CONTRACT_PATH}")):
        return unsupported("adapter_collision", "Native tests occupy the adapter's private paths")
    context = tuple(
        native_resource(files, data["modes"], path, path.removeprefix("environment/"))
        for path in sorted(files)
        if path.startswith("environment/")
    )
    actor_build = DockerBuildContext(files=context)
    # A pristine workspace would resurrect files deleted by the submitted solution.
    verifier_dockerfile = (
        dockerfile.rstrip()
        + f"\nUSER root\nRUN rm -rf /workspace && mkdir -p {shlex.quote(working_directory)}\n"
        + f"WORKDIR {working_directory}\n"
    )
    grader_build = DockerBuildContext(
        files=tuple(
            inline_resource("Dockerfile", verifier_dockerfile.encode()) if resource.path == "Dockerfile" else resource
            for resource in context
        )
    )
    environment_variables = dict(config["environment"].get("env", {}))
    actor = EnvironmentRequirements(
        capabilities=("shell", "filesystem"),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=actor_build,
        working_directory=working_directory,
        environment_variables={**environment_variables, **config["agent"].get("env", {})},
    )
    verifier_environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=grader_build,
        working_directory=working_directory,
        environment_variables=environment_variables,
    )
    verifier = tuple(
        native_resource(files, data["modes"], path, path.removeprefix("tests/"))
        for path in files
        if path.startswith("tests/")
    )
    provenance = {
        "release": data["release"],
        "task_ref": data["task_ref"],
        "native_config": config,
        "capture": WORKSPACE,
        "additional_llm_rubric": "preserved but not executed",
    }
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=files["instruction.md"].decode()),)),
        environment_requirements=actor,
        resources=ResourceGroups(
            verifier=(
                *verifier,
                inline_resource(ADAPTER_PATH, Path(grade.__file__).read_bytes()),
                inline_resource(CONTRACT_PATH, json.dumps(provenance).encode()),
            ),
            oracle=tuple(
                native_resource(files, data["modes"], path, path) for path in files if path.startswith("solution/")
            ),
        ),
        output_paths=(WORKSPACE,),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python3", f"/tests/{ADAPTER_PATH}"),
            cwd=working_directory,
            answer_path=None,
            environment=verifier_environment,
            env=config["verifier"].get("env", {}),
            timeout=config["verifier"]["timeout_sec"],
            reward=FileReward(
                files=(RewardFile(path=str(grade.ADAPTED_REWARD), format=RewardFileFormat.JSON),),
            ),
        ),
        tags=("agentic", "multi-turn", "native-harbor", "workspace-files"),
    )


def native_resource(files: dict[str, bytes], modes: dict[str, str], path: str, target: str) -> TaskResource:
    return inline_resource(target, files[path]).model_copy(update={"mode": modes[path]})


def golden(task: TaskSpec) -> OracleCommand:
    contract = next(resource for resource in task.resources.verifier if resource.path == CONTRACT_PATH)
    env = json.loads(resource_bytes(contract))["native_config"].get("solution", {}).get("env", {})
    return OracleCommand(
        shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), "bash", "/solution/solve.sh"])
    )
