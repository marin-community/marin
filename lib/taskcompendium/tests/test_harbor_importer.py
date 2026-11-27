# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, NetworkPolicy, PackageSpec, UnsupportedMachineSpec

from taskcompendium.evaluation.spec import AgentEpisode, EpisodeInputs, EpisodeLimits, TerminusAgent
from taskcompendium.importers.harbor.config import UnsupportedHarborFeature
from taskcompendium.importers.harbor.package import (
    AnswerFile,
    HarborTask,
    JobSettings,
    PackageTests,
    image_references,
    lower_package,
)
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    CommandSemantics,
    EndpointRequirement,
    EnvironmentRequirements,
    GraderMachine,
    ReadinessRequirement,
    ScriptGrader,
    Source,
    TaskSpec,
    VerifyitGrader,
)
from taskcompendium.runtime.environment import validate_machine_spec

# Each package maps a path relative to the package root to that file's contents.

PREBUILT_SEPARATE = {
    "task.toml": (
        """\
version = "1.0"

[metadata]
author_name = "Fixture Author"
difficulty = "medium"
category = "file-operations"
tags = ["bash", "text"]

[verifier]
timeout_sec = 300.0
environment_mode = "separate"
env = { JUDGE_MODE = "strict" }

[[verifier.collect]]
command = "cp /app/out.txt /logs/artifacts/out.txt"
timeout_sec = 30.0

[agent]
timeout_sec = 900.0

[environment]
docker_image = "alexgshaw/count-lines:20251031"
workdir = "/app"
cpus = 2
memory = "2G"
storage_mb = 10240
allow_internet = false
env = { DATA_DIR = "/setup_files" }

[environment.healthcheck]
command = "test -d /app"
interval_sec = 2.0
timeout_sec = 5.0
retries = 4
"""
    ),
    "instruction.md": "Count the lines in /setup_files/data.txt and write the number to /app/out.txt.\n",
    "environment/README.md": "Notes copied into the agent's working directory.\n",
    "setup_files/data.txt": "alpha\nbeta\ngamma\n",
    "solution/solve.sh": "#!/bin/bash\nwc -l < /setup_files/data.txt > /app/out.txt\n",
    "tests/test.sh": (
        """\
#!/bin/bash
if [ "$(cat /logs/artifacts/out.txt)" = "3" ]; then
  echo '{"reward": 1.0, "lines_checked": 3}' > /logs/verifier/reward.json
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
    ),
}

DOCKERFILE_MULTISTAGE = {
    "task.toml": (
        """\
version = "1.1"

[metadata]
difficulty = "hard"

[verifier]
timeout_sec = 600.0
environment_mode = "separate"

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 900.0
"""
    ),
    "instruction.md": "Build the tool and make `tool --version` print 1.0.\n",
    "environment/Dockerfile": (
        """\
# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim
FROM golang:1.22 AS builder
WORKDIR /src
COPY main.go .
RUN go build -o /out/tool main.go

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS runtime
COPY --from=builder /out/tool /usr/local/bin/tool
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv
RUN <<EOT
echo "FROM ubuntu:24.04" > /tmp/not-an-image
EOT
WORKDIR /workspace
WORKDIR project
"""
    ),
    "environment/main.go": "package main\n\nfunc main() {}\n",
    "tests/Dockerfile": "FROM python:3.12-slim\nRUN pip install pytest==8.3.3\n",
    "tests/check_tool.py": (
        """\
import subprocess


def test_version():
    assert subprocess.run(["tool", "--version"], capture_output=True, text=True).stdout.strip() == "1.0"
"""
    ),
    "tests/test.sh": (
        "#!/bin/bash\n"
        "if pytest /tests/check_tool.py; then echo 1 > /logs/verifier/reward.txt;"
        " else echo 0 > /logs/verifier/reward.txt; fi\n"
    ),
}

COMPOSE_SIDECAR = {
    "task.toml": (
        """\
version = "1.0"

[verifier]
timeout_sec = 120.0

[verifier.environment]
docker_image = "python:3.12-slim"
allow_internet = false

[agent]
timeout_sec = 600.0

[environment]
allow_internet = false

[[environment.mcp_servers]]
name = "filings"
transport = "http"
url = "http://tools:8000/mcp"
"""
    ),
    "instruction.md": "Use the filings tool to find the 2023 revenue and write it to /app/answer.txt.\n",
    "environment/Dockerfile": "FROM ubuntu:24.04\nWORKDIR /app\n",
    "environment/docker-compose.yaml": (
        """\
version: "3.8"
services:
  main:
    build: .
    environment:
      FILINGS_URL: http://tools:8000/mcp
      SEARCH_API_KEY: ${SEARCH_API_KEY}
    depends_on:
      tools:
        condition: service_healthy
    volumes:
      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
  tools:
    build:
      context: tools
    expose:
      - 8000
    command: python /srv/server.py --port 8000
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 2s
      timeout: 5s
      retries: 10
      start_period: 1s
  cache:
    image: redis:7.2
    expose:
      - "6379"
"""
    ),
    "environment/tools/Dockerfile": "FROM python:3.12-slim\nCOPY server.py /srv/server.py\n",
    "environment/tools/server.py": 'print("serving filings")\n',
    "tests/test.sh": (
        "#!/bin/bash\n"
        "if grep -q 383285 /app/answer.txt; then echo 1 > /logs/verifier/reward.txt;"
        " else echo 0 > /logs/verifier/reward.txt; fi\n"
    ),
}

MAIN_OVERRIDE_COMPOSE = "environment/docker-compose.yaml"

COMPOSE_MAIN_OVERRIDE = {
    "task.toml": (
        """\
version = "1.0"

[verifier]
timeout_sec = 600

[agent]
timeout_sec = 3600

[environment]
cpus = 1
memory_mb = 2048
storage_mb = 10240
"""
    ),
    "instruction.md": "Use /app/tools/edgar_search.py to find the filing and write the answer to /app/answer.txt.\n",
    "environment/Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY edgar_search.py /app/tools/edgar_search.py\n",
    "environment/edgar_search.py": 'import os\n\nprint(bool(os.environ.get("SEC_EDGAR_API_KEY")))\n',
    MAIN_OVERRIDE_COMPOSE: (
        """\
services:
  main:
    build:
      context: ${CONTEXT_DIR}
    image: ${MAIN_IMAGE_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    network_mode: ${NETWORK_MODE:-bridge}
    environment:
      - TEST_DIR=${TEST_DIR}
      - SEC_EDGAR_API_KEY=${SEC_EDGAR_API_KEY}
      - MODEL_FOR_TOOLS=${MODEL_FOR_TOOLS}
    volumes:
      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
    deploy:
      resources:
        limits:
          cpus: ${CPUS}
          memory: ${MEMORY}
"""
    ),
    "tests/test.sh": (
        "#!/bin/bash\n"
        "if [ -s /app/answer.txt ]; then echo 1 > /logs/verifier/reward.txt;"
        " else echo 0 > /logs/verifier/reward.txt; fi\n"
    ),
}

SHARED_DEFAULT = {
    "task.toml": (
        """\
version = "1.0"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
docker_image = "alexgshaw/fix-git:20251031"
"""
    ),
    "instruction.md": "Recover the lost commit.\n",
    "tests/test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
}


def write_package(root: Path, name: str, files: Mapping[str, str]) -> Path:
    """Write ``files`` under ``root / name``; shell scripts are executable, as in Harbor packages.

    The importer labels a task with its directory name, so ``name`` is part of the lowered task.
    """
    directory = root / name
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if path.suffix == ".sh":
            path.chmod(0o755)
    return directory


def replaced(files: Mapping[str, str], path: str, old: str, new: str) -> dict[str, str]:
    """``files`` with one substring of the file at ``path`` replaced."""
    assert old in files[path]
    return {**files, path: files[path].replace(old, new)}


def digest(reference: str) -> str:
    return f"sha256:{hashlib.sha256(reference.encode()).hexdigest()}"


def pinned(reference: str) -> str:
    return f"{reference}@{digest(reference)}"


def lower(directory: Path, rule=PackageTests(), job=JobSettings()) -> HarborTask:
    return lower_package(
        directory,
        task_id=f"harbor-{directory.name}",
        source=Source(dataset="harbor-fixtures", revision="1", row=directory.name, importer_revision="1"),
        rule=rule,
        image_digests={reference: digest(reference) for reference in image_references(directory)},
        job=job,
    )


def build_file(environment: EnvironmentRequirements, path: str) -> str:
    assert environment.docker_build is not None
    resource = next(file for file in environment.docker_build.files if file.path == path)
    return base64.b64decode(resource.source.content_base64).decode()


def test_prebuilt_image_with_separate_verifier(tmp_path):
    directory = write_package(tmp_path, "prebuilt-separate", PREBUILT_SEPARATE)

    lowered = lower(directory)

    task = lowered.task
    assert task.context.events[0].content == PREBUILT_SEPARATE["instruction.md"]
    assert task.answer_type == AnswerType.WORKSPACE_STATE
    agent = task.environment_requirements
    assert agent.docker_image == pinned("alexgshaw/count-lines:20251031")
    assert (agent.working_directory, agent.environment_variables) == ("/app", {"DATA_DIR": "/setup_files"})
    assert "network" not in agent.capabilities
    assert agent.readiness is not None
    assert agent.readiness.argv[-1] == "test -d /app"
    assert (agent.readiness.attempts, agent.readiness.command_timeout) == (4, 5.0)
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert (grader.argv, grader.cwd, grader.answer_path, grader.timeout) == (
        ("bash", "/tests/test.sh"),
        "/app",
        None,
        300,
    )
    placement = grader.placement
    assert isinstance(placement, GraderMachine)
    assert placement.environment.docker_image == agent.docker_image
    assert placement.environment.environment_variables == {"DATA_DIR": "/setup_files", "JUDGE_MODE": "strict"}
    assert [(hook.argv[-1], hook.timeout) for hook in placement.collect] == [
        ("cp /app/out.txt /logs/artifacts/out.txt", 30.0)
    ]
    assert [artifact.source for artifact in placement.artifacts] == ["/logs/artifacts"]
    test_script = task.resources.verifier[0]
    assert (test_script.path, test_script.mode, test_script.mtime_ns) == ("test.sh", "0755", None)
    assert [file.path for file in task.resources.oracle] == ["solution/solve.sh"]
    assert [file.path for file in task.resources.worker] == ["setup_files/data.txt", "app/README.md"]
    assert task.labels == {
        "category": "file-operations",
        "difficulty": "medium",
        "tags": "bash,text",
        "harbor_path": "prebuilt-separate",
    }
    assert lowered.episode == EpisodeInputs(
        agent_timeout=900,
        verifier_timeout=300,
        cpus=2,
        memory_mb=2048,
        storage_mb=10240,
        network=NetworkPolicy.DENY,
    )
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


def test_dockerfile_build_pins_every_external_base(tmp_path):
    directory = write_package(tmp_path, "dockerfile-multistage", DOCKERFILE_MULTISTAGE)

    references = image_references(directory)
    lowered = lower(directory).task

    assert references == ("golang:1.22", "python:3.12-slim", "ghcr.io/astral-sh/uv:0.5.0")
    agent = lowered.environment_requirements
    assert build_file(agent, "Dockerfile") == (
        "# syntax=docker/dockerfile:1\n"
        "ARG PYTHON_IMAGE=python:3.12-slim\n"
        f"FROM {pinned('golang:1.22')} AS builder\n"
        "WORKDIR /src\n"
        "COPY main.go .\n"
        "RUN go build -o /out/tool main.go\n"
        "\n"
        f"FROM --platform=linux/amd64 {pinned('python:3.12-slim')} AS runtime\n"
        "COPY --from=builder /out/tool /usr/local/bin/tool\n"
        f"COPY --from={pinned('ghcr.io/astral-sh/uv:0.5.0')} /uv /usr/local/bin/uv\n"
        "RUN <<EOT\n"
        'echo "FROM ubuntu:24.04" > /tmp/not-an-image\n'
        "EOT\n"
        "WORKDIR /workspace\n"
        "WORKDIR project\n"
    )
    assert build_file(agent, "main.go") == DOCKERFILE_MULTISTAGE["environment/main.go"]
    assert agent.working_directory == "/workspace/project"
    assert "network" in agent.capabilities
    # Harbor builds a separate verifier from tests/, not environment/.
    grader = lowered.grader
    assert isinstance(grader, ScriptGrader)
    assert isinstance(grader.placement, GraderMachine)
    assert build_file(grader.placement.environment, "Dockerfile").startswith(f"FROM {pinned('python:3.12-slim')}\n")
    # The verifier's Dockerfile sets no WORKDIR, so Harbor grades in the image's default directory.
    assert grader.cwd is None
    with pytest.raises(ValueError, match=re.escape("ghcr.io/astral-sh/uv:0.5.0")):
        lower_package(
            directory,
            task_id="unpinned",
            source=lowered.source,
            rule=PackageTests(),
            image_digests={reference: digest(reference) for reference in references[:2]},
            job=JobSettings(),
        )


def test_compose_sidecars_become_services_with_endpoints_and_readiness(tmp_path):
    directory = write_package(tmp_path, "compose-sidecar", COMPOSE_SIDECAR)

    assert image_references(directory) == ("ubuntu:24.04", "python:3.12-slim", "redis:7.2")
    agent = lower(directory).task.environment_requirements

    assert build_file(agent, "Dockerfile") == f"FROM {pinned('ubuntu:24.04')}\nWORKDIR /app\n"
    assert agent.environment_variables == {
        "FILINGS_URL": "http://tools:8000/mcp",
        "SEARCH_API_KEY": "${SEARCH_API_KEY}",
    }
    assert agent.tool_providers["filings"].initial_state == {
        "transport": "streamable-http",
        "url": "http://tools:8000/mcp",
    }
    tools, cache = agent.services
    assert tools.name == "tools"
    assert build_file(tools, "Dockerfile").startswith(f"FROM {pinned('python:3.12-slim')}\n")
    assert tools.command == ("python", "/srv/server.py", "--port", "8000")
    assert tools.endpoints == (EndpointRequirement(name="tcp-8000", port=8000),)
    assert tools.readiness == ReadinessRequirement(
        argv=("python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"),
        command_timeout=5,
        timeout=1 + 10 * (5 + 2),
        interval=2,
        start_period=1,
        attempts=10,
    )
    assert (cache.name, cache.docker_image, cache.endpoints) == (
        "cache",
        pinned("redis:7.2"),
        (EndpointRequirement(name="tcp-6379", port=6379),),
    )
    # Runtimes without a multi-container task factory refuse the group instead of running main alone.
    with pytest.raises(UnsupportedMachineSpec, match="Service groups"):
        validate_machine_spec(agent, ShellSimMachineFactory(), MachineSpec(PackageSpec()))


def test_harbor_main_override_keeps_only_the_tasks_own_variables(tmp_path):
    """Harbor's Docker-backend override of ``main`` restates Harbor's settings, except the task's variables."""
    lowered = lower(write_package(tmp_path, "compose-main-override", COMPOSE_MAIN_OVERRIDE))

    agent = lowered.task.environment_requirements
    assert build_file(agent, "Dockerfile").startswith(f"FROM {pinned('python:3.11-slim')}\n")
    assert agent.services == ()
    assert agent.environment_variables == {
        "SEC_EDGAR_API_KEY": "${SEC_EDGAR_API_KEY}",
        "MODEL_FOR_TOOLS": "${MODEL_FOR_TOOLS}",
    }
    assert "network" in agent.capabilities
    # Harbor fills the ${CPUS} and ${MEMORY} limits from task.toml.
    assert (lowered.episode.cpus, lowered.episode.memory_mb, lowered.episode.network) == (1, 2048, NetworkPolicy.ALLOW)


def test_harbor_main_override_without_a_network_denies_the_agent_network(tmp_path):
    files = replaced(COMPOSE_MAIN_OVERRIDE, MAIN_OVERRIDE_COMPOSE, "${NETWORK_MODE:-bridge}", "none")

    lowered = lower(write_package(tmp_path, "package", files))

    assert "network" not in lowered.task.environment_requirements.capabilities
    assert lowered.episode.network == NetworkPolicy.DENY


@pytest.mark.parametrize(
    "replacement,reason",
    [
        (("cpus: ${CPUS}", "cpus: '2'"), "Compose resource limits other than Harbor's ${CPUS} and ${MEMORY}"),
        (("limits:", "reservations:"), "unsupported Compose configuration"),
        (("${NETWORK_MODE:-bridge}", "host"), "Compose network_mode 'host'"),
        (("sleep infinity", "python /app/server.py"), "cannot override its process"),
    ],
)
def test_harbor_main_override_rejects_settings_harbor_does_not_own(tmp_path, replacement, reason):
    directory = write_package(tmp_path, "package", replaced(COMPOSE_MAIN_OVERRIDE, MAIN_OVERRIDE_COMPOSE, *replacement))

    with pytest.raises(UnsupportedHarborFeature, match=re.escape(reason)):
        lower(directory)


def test_environments_without_services_keep_their_serialized_form():
    environment = EnvironmentRequirements(
        docker_image=pinned("python:3.12-slim"), command_semantics=CommandSemantics.LINUX_PROCESS
    )

    serialized = environment.model_dump(mode="json")

    assert "services" not in serialized and "readiness" not in serialized
    assert EnvironmentRequirements.model_validate(serialized) == environment


@pytest.mark.parametrize(
    "files,reason",
    [
        ({"task.toml": 'version = "2.0"\n'}, "schema 2.0"),
        ({"task.toml": '[[steps]]\nname = "first"\n'}, "multi-step"),
        ({"task.toml": "[environment]\ngpus = 1\n"}, "GPUs"),
        ({"task.toml": '[environment.tpu]\ntype = "v6e"\ntopology = "2x2"\n'}, "TPU"),
        ({"task.toml": '[environment]\nos = "Windows"\n'}, "only Linux"),
        ({"task.toml": '[agent]\nuser = "agent"\n'}, "only root"),
        (
            {"task.toml": '[[verifier.collect]]\ncommand = "true"\nuser = "grader"\n'},
            "collect hooks with their own users",
        ),
        (
            {"task.toml": '[[environment.mcp_servers]]\nname = "search"\nurl = "http://localhost:9000/sse"\n'},
            "MCP servers outside a Compose service group",
        ),
        (
            {"environment/Dockerfile": "ARG BASE\nFROM ${BASE}\n"},
            "build argument BASE without a default",
        ),
        (
            {"environment/docker-compose.yaml": "services:\n  db:\n    image: postgres:16\n    ports: ['5432:5432']\n"},
            "unsupported Compose configuration",
        ),
    ],
)
def test_unsupported_features_are_rejected_with_reasons(tmp_path, files, reason):
    directory = write_package(tmp_path, "package", {**DOCKERFILE_MULTISTAGE, **files})

    with pytest.raises(UnsupportedHarborFeature, match=re.escape(reason)) as rejected:
        lower(directory)

    assert reason in rejected.value.reason


def test_shared_verifier_grades_in_the_agent_machine(tmp_path):
    """Harbor's default runs tests/test.sh in the agent's container, in its working directory.

    The package names no directory, so the grader runs in the image's own ``WORKDIR``.
    """
    task = lower(write_package(tmp_path, "shared-default", SHARED_DEFAULT)).task

    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert (grader.placement, grader.argv, grader.cwd, grader.env) == (
        AgentMachine(),
        ("bash", "/tests/test.sh"),
        None,
        {},
    )
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


@pytest.mark.parametrize(
    "verifier,reason",
    [
        ('[[verifier.collect]]\ncommand = "true"\n', "collect hooks for a verifier in the agent's container"),
        ('[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n', "verifier variables OPENAI_API_KEY that read host"),
    ],
)
def test_shared_verifier_rejects_what_the_agent_machine_cannot_run(tmp_path, verifier, reason):
    files = {**SHARED_DEFAULT, "task.toml": SHARED_DEFAULT["task.toml"] + verifier}
    directory = write_package(tmp_path, "package", files)

    with pytest.raises(UnsupportedHarborFeature, match=re.escape(reason)):
        lower(directory)


def test_answer_file_is_captured_and_graded_on_the_host(tmp_path):
    grader = VerifyitGrader(mode="exact", parameters={"expected": ["204"]})
    rule = AnswerFile(path="/app/answer.txt", grader=grader)

    task = lower(write_package(tmp_path, "shared-default", SHARED_DEFAULT), rule).task

    assert (task.output_paths, task.answer_type, task.grader) == (("/app/answer.txt",), AnswerType.FILE, grader)


def test_job_settings_replace_the_package_timeouts_and_storage(tmp_path):
    files = replaced(PREBUILT_SEPARATE, "task.toml", "timeout_sec = 900.0", "timeout_sec = 3000.0")
    directory = write_package(tmp_path, "package", files)
    episode = AgentEpisode(
        harness=TerminusAgent(),
        limits=EpisodeLimits(
            max_turns=None,
            max_total_tokens=None,
            model_turn_timeout=600,
            total_turn_timeout=None,
            verifier_timeout=None,
            attempt_timeout=4 * 60 * 60,
        ),
    )

    lowered = lower(directory, job=JobSettings(agent_timeout=1800, verifier_timeout=600, storage_mb=4096))

    limits = episode.limits_for(lowered.episode)
    assert (limits.total_turn_timeout, limits.verifier_timeout) == (1800, 600)
    # The package's own 300-second verifier limit no longer caps its test script.
    grader = lowered.task.grader
    assert isinstance(grader, ScriptGrader) and grader.timeout == 600
    assert (lowered.episode.storage_mb, lowered.episode.memory_mb) == (4096, 2048)


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(PREBUILT_SEPARATE, id="prebuilt-separate"),
        pytest.param(DOCKERFILE_MULTISTAGE, id="dockerfile-multistage"),
    ],
)
def test_agent_install_builds_after_the_package_image(tmp_path, files):
    directory = write_package(tmp_path, "package", files)
    package = lower(directory).task

    task = lower(directory, job=JobSettings(agent_install="RUN install-agent\n")).task

    agent, original = task.environment_requirements, package.environment_requirements
    if original.docker_build is None:
        base, context = f"FROM {original.docker_image}\n", {"Dockerfile"}
    else:
        base, context = build_file(original, "Dockerfile"), {file.path for file in original.docker_build.files}
    assert build_file(agent, "Dockerfile") == f"{base}RUN install-agent\n"
    assert agent.docker_build is not None and {file.path for file in agent.docker_build.files} == context
    assert agent.working_directory == original.working_directory
    # Harbor installs the agent for the agent alone, so a separate verifier's machine is unchanged.
    assert task.grader == package.grader
