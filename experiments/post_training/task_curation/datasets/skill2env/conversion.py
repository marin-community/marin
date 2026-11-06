# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert offline Skill2Env tasks without rewriting native tests."""

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

# These native verifiers read complete state directories, not just the declared artifacts.
# Scope exceptions to the audited package contents; later releases need another audit.
CAPTURE_ROOTS = {
    # task_c-memory_qi67nu17: journal plus protected memory and session files.
    "sha256:a9b30e4581483338bc51c87e08ae83a4b5d4f42a4fbb2d3394e4fc4c86d9ee85": (
        WORKSPACE,
        "/root/.claude/memory",
    ),
    # task_gws-sheets-append_njhq0ebv: workbook state and append audit log.
    "sha256:a3a76ed72a278586f3a76af9543d89bd514f137f751d42f78d1af1409f4b52de": (
        "/work",
        "/var/lib/gws-local",
    ),
    # task_agent-platform-model-registry_822ely73.
    "sha256:10574c3520c8e12a5c3dd18ef6882e9cfe47ce2efeb7c0cff2cfd3c5b558dd13": (
        "/workspace",
        "/var/lib/parcelpulse",
        "/opt/parcelpulse/seed",
    ),
    # task_agent-manager_l5mfu3fn.
    "sha256:bb79b1b83cd419659efee2f947080faac6e3056b05cf28513adb95e127553f2b": (
        "/workspace",
        "/var/lib/fake-cron",
    ),
    # task_apple-container_x5m7xq5j.
    "sha256:a0e38b933973c61f91a108c79973c7f32b0b16084f9e64a7760090be2437afc1": (
        "/workspace",
        "/home/cedar",
        "/opt/cedar-container",
    ),
    # task_bigquery-basics_fbebs224.
    "sha256:c288a78e4cba2ec26b9236efa193b701c987c787d6e0e7d7c27330fd04f18724": (
        "/workspace",
        "/var/lib/harbor-bq",
        "/opt/harbor-bq",
    ),
    # task_brief_uhjgxwy4.
    "sha256:e7675e79e5de8a9fb3ece130a9b77f507122839aca1b61e23c1600c36305c239": (
        "/workspace",
        "/home/agent/.claude",
    ),
    # task_c-memory_sp3smf6l.
    "sha256:6cbc1ed415104b1fc6c4907145da9dee6b3e8977b2da3299658b9dea94e4e7d0": (
        "/workspace",
        "/home/harbor/.claude/memory",
    ),
    # task_c-speakers_sst0k2od.
    "sha256:7c2a3bce70ed42ccd165fc0ce9e1ae9290ae0653a9ee9a7e72ab79e2d54c74a7": (
        "/workspace",
        "/var/lib/station-speakers",
        "/var/log/station-speakers",
        "/opt/station-speakers/initial-state.json",
    ),
    # task_c-tasks_za86wnfq.
    "sha256:28b55a1540fd712898dabcfaa9d297bee03aba71a0ea4d661465d52054930ec1": (
        "/workspace",
        "/var/lib/things-cli",
        "/var/log/things-cli",
    ),
    # task_claimable-postgres_ot8kysxf.
    "sha256:7b72396bbf228f1466a34edae25b3e44b08b61c679daea884ac8453184c8fa62": (
        "/workspace",
        "/var/lib/neon-new",
    ),
    # task_coreos-cve-validator_8w8gflo5.
    "sha256:a785aff14c957bcec9eebfeae0c084d7ea3ac1fed8f250e349513627cd0c4b2c": (
        "/workspace",
        "/tmp/coreos-cache",
        "/opt/coreos-validator/fixtures",
    ),
    # task_cs-onboard_y0zu81um.
    "sha256:32a9fd519d177302ac2cd3b733507cdf25a775097b1bae1cf58d9d6d28e5c086": (
        "/workspace",
        "/home/agent/.claude",
        "/var/lib/relayforge/approval.json",
        "/var/log/relayforge",
        "/opt/relayforge/baseline-context.md",
    ),
    # task_cs-onboard_5if8gco8.
    "sha256:12af6b1ab47911924ec8eed08b12e4fd6752116f9f151f7475ba73ecd529db6b": (
        "/workspace",
        "/root",
    ),
    # task_design-system_8nqaiu0q.
    "sha256:723e9639fc67b91998e09072007e54a5abd7d5c8d7fcdfcca1ee500e2d876227": (
        "/workspace",
        "/home/agent",
    ),
    # task_firebase-firestore_phatmneo.
    "sha256:b15a8e436a359e35259421978d1fe22daec3bc1379f110529dff2f17d8ea77b9": (
        "/workspace",
        "/var/lib/firebase-mock/control-plane.json",
        "/var/lib/firebase-mock/deployments.json",
        "/var/lib/firebase-mock/invocations.jsonl",
    ),
    # task_gke-app-onboarding_8br2b29h.
    "sha256:d638f9cda48e53ca53d238ec7e1a5c0bb75659ea25a5384e04c1de8550508df9": (
        "/workspace",
        "/var/lib/gke-sim",
        "/opt/gke-sim/initial_state.json",
    ),
    # task_gke-scaling_kypx2wz3.
    "sha256:58e95df516dafa9f8915fbde8ec58d2e503c9904ac4ff5e03429207b1192d5c5": (
        "/workspace",
        "/var/lib/kube-sim",
        "/var/log/kube-sim",
    ),
    # task_gws-chat-send_867rfdcr.
    "sha256:fe6a38bc7b5197cc38a9759c4ff862b741115cf15e04294f466d3d07c9ede802": (
        "/workspace",
        "/var/lib/gws-chat",
        "/opt/gws-chat/seed.json",
    ),
    # task_gws-script-push_fvlax5x4.
    "sha256:78e6f1bd9d289a7a7613b30ca1dcf19e5b60af1afbc18c5c802e07927fdbb74a": (
        "/workspace",
        "/var/lib/gws-emulator",
        "/var/log/gws-emulator",
    ),
    # task_mcore-run-on-slurm_44f7slmf.
    "sha256:75b4b8224baa7bcc447b6616ccf79048631e7d07d58fc967ba4e9af93adb9f65": (
        "/workspace",
        "/shared/orion/atlas",
    ),
    # task_neon_au39bwyi.
    "sha256:1c0ef2f2d0a1034c0e8bd68a824ad88946cfe25cddfbb5d5e9cb397f8cf82777": (
        "/workspace",
        "/opt/neon-emulator",
    ),
    # task_recipe-create-task-list_7s3otz05.
    "sha256:6f193281c632a3ca0a3ea2535f74d7bb95f7973ef9d12537e325c96d953ccab8": (
        "/workspace",
        "/var/lib/gws",
        "/var/log/gws",
        "/opt/gws",
    ),
    # task_recipe-create-events-from-sheet_f3un7a2r.
    "sha256:21d1cde6c798fb20a118548687d922684b63722a11d305cd09b6b221b38b38f8": (
        "/workspace",
        "/var/lib/gws",
        "/opt/gws-fixtures",
    ),
    # task_recipe-organize-drive-folder_o28wlb1u.
    "sha256:8e9991465014b7bc01f8e74fd1850c51e60446e0f346091e12397f44e013f0c8": (
        "/workspace",
        "/var/lib/gws",
        "/opt/gws",
    ),
    # task_research-junshi_z2s3k43l.
    "sha256:44ab48d33e37c86e1900d09390df17fbd67fe081cb4d2eab98255b1cb24918ba": (
        "/workspace",
        "/home/researcher",
    ),
    # task_vertical-logistics_4uqzjad5.
    "sha256:c6c4e2d1590ded5b67c3bb63a83beba7989560726a17ac8e6b5c4da117c8b9e8": ("/app",),
    # task_cs-onboard_bntq7mho.
    "sha256:a99f7a8aac0840533069f3ad1bea9f2cd9e7cedefaf564a2b345f93fefb3667b": (
        "/workspace",
        "/root/.claude",
    ),
    # task_recipe-reschedule-meeting_1a12lgde.
    "sha256:171ad1db7303ae520e563b4f2f9db0ed8757bd8114708c43c388bcd1f4baaffd": (
        "/workspace",
        "/var/lib/local-calendar",
        "/opt/local-calendar/seed_state.json",
    ),
}

# This native Node image has no Python interpreter for the grading adapter.
PYTHON_GRADER_TASK = "sha256:7b72396bbf228f1466a34edae25b3e44b08b61c679daea884ac8453184c8fa62"

# Invoke the released entrypoint through the existing actor setup interface.
STARTUP_COMMANDS = {
    "sha256:78e6f1bd9d289a7a7613b30ca1dcf19e5b60af1afbc18c5c802e07927fdbb74a": (
        "/usr/local/bin/gws-emulator-entrypoint true",
    ),
}


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
    output_paths = CAPTURE_ROOTS.get(task_digest, (WORKSPACE,))
    dockerfile = files["environment/Dockerfile"].decode()
    final_stage = re.split(r"(?im)^[ \t]*FROM[ \t]+", dockerfile)[-1]
    workdirs = re.findall(r"(?im)^[ \t]*WORKDIR[ \t]+([^\r\n]*)", final_stage)
    if not workdirs or (task_digest not in CAPTURE_ROOTS and re.fullmatch(r"\S+", workdirs[-1].strip()) is None):
        return unsupported("native_workspace", "Requires a final literal absolute WORKDIR within the capture roots")
    workdir = PurePosixPath(workdirs[-1].strip())
    if (
        not any(workdir.is_relative_to(root) for root in output_paths)
        or ".." in workdir.parts
        or any(character in str(workdir) for character in ("$", "\\", "'", '"'))
    ):
        return unsupported("native_workspace", "Requires a final literal absolute WORKDIR within the capture roots")
    working_directory = str(workdir)
    if not all(
        any(PurePosixPath(path).is_relative_to(root) for root in output_paths) for path in config.get("artifacts", [])
    ):
        return unsupported("native_outputs", "Outputs outside the audited capture roots need another capture contract")
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
    # Clear every captured directory so the image cannot resurrect submitted deletions.
    clear_paths = shlex.join(output_paths)
    verifier_dockerfile = (
        dockerfile.rstrip()
        + f"\nUSER root\nRUN rm -rf {clear_paths} && mkdir -p {shlex.quote(working_directory)}\n"
        + f"WORKDIR {working_directory}\n"
    )
    if task_digest == PYTHON_GRADER_TASK:
        verifier_dockerfile += (
            "RUN apt-get update && apt-get install -y --no-install-recommends python3"
            " && rm -rf /var/lib/apt/lists/*\n"
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
        setup_commands=STARTUP_COMMANDS.get(task_digest, ()),
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
        "capture": output_paths,
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
        output_paths=output_paths,
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
