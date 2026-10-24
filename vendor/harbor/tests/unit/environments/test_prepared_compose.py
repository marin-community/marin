# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import tau3's service contract without giving Compose host authority."""

import json

import pytest
import yaml

from harbor.environments.prepared_compose import import_prepared_task
from harbor.models.task.config import EnvironmentConfig, MCPServerConfig
from harbor.utils.env import task_environment
from shellbox.machine import NetworkPolicy

MAIN_IMAGE = "example.test/main@sha256:" + "a" * 64
RUNTIME_IMAGE = "example.test/runtime@sha256:" + "b" * 64
MODEL_ENVIRONMENT = {
    "OPENAI_API_KEY": "EMPTY",
    "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
}


@pytest.fixture
def tau3_environment(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (tmp_path / "runtime-server").mkdir()
    (tmp_path / "runtime-server" / "Dockerfile").write_text('FROM python:3.12-slim\nCMD ["python3", "server.py"]\n')
    (tmp_path / "shellbox-images.json").write_text(
        json.dumps(
            {
                "main": {"image": MAIN_IMAGE, "workdir": "/app"},
                "tau3-runtime": {
                    "image": RUNTIME_IMAGE,
                    "workdir": "/app",
                    "agent_logs": "publish",
                },
            }
        )
    )
    compose = {
        "services": {
            "main": {"depends_on": {"tau3-runtime": {"condition": "service_healthy"}}},
            "tau3-runtime": {
                "build": {"context": "./runtime-server"},
                "environment": [
                    "OPENAI_API_KEY=${OPENAI_API_KEY:-}",
                    "OPENAI_BASE_URL=${OPENAI_BASE_URL:-}",
                    "TAU2_USER_MODEL=${TAU2_USER_MODEL:-gpt-5.2}",
                ],
                "volumes": ["${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}"],
                "expose": ["8000"],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python3",
                        "-c",
                        "import socket; socket.create_connection(('localhost',8000),timeout=2).close()",
                    ],
                    "interval": "2s",
                    "timeout": "5s",
                    "retries": 15,
                    "start_period": "5s",
                },
            },
        }
    }
    (tmp_path / "docker-compose.yaml").write_text(yaml.safe_dump(compose))
    return tmp_path


def test_tau3_import_preserves_prepared_services_and_private_model_capability(tau3_environment, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-provider-key")
    task = EnvironmentConfig(
        allow_internet=False,
        env={
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
        },
        mcp_servers=[
            MCPServerConfig(
                name="tau3-runtime",
                transport="streamable-http",
                url="http://tau3-runtime:8000/mcp",
            )
        ],
    )
    with task_environment(MODEL_ENVIRONMENT):
        plan = import_prepared_task(task, tau3_environment)
    (runtime,) = plan.group.services

    assert plan.group.task.source.reference == MAIN_IMAGE
    assert runtime.machine.source.reference == RUNTIME_IMAGE
    assert plan.group.task.network is runtime.machine.network is NetworkPolicy.DENY
    assert plan.group.task.env["OPENAI_API_KEY"] == runtime.machine.env["OPENAI_API_KEY"] == "EMPTY"
    assert runtime.machine.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert runtime.machine.env["TAU2_USER_MODEL"] == "gpt-5.2"
    assert runtime.arguments is None and runtime.entrypoint is None
    assert runtime.endpoints[0].port == 8000
    assert runtime.readiness.command.argv[:2] == ("python3", "-c")
    assert runtime.readiness.attempts == 15
    assert plan.service_log_paths == (("tau3-runtime", "/logs/agent"),)


@pytest.mark.parametrize(
    "escape",
    [
        {"privileged": True},
        {"network_mode": "host"},
        {"volumes": ["/run/secrets:/secrets"]},
        {"env_file": ".env"},
        {"ports": ["8000:8000"]},
    ],
)
def test_compose_rejects_host_authority_before_submitting_any_service(tau3_environment, escape):
    path = tau3_environment / "docker-compose.yaml"
    document = yaml.safe_load(path.read_text())
    document["services"]["tau3-runtime"].update(escape)
    path.write_text(yaml.safe_dump(document))
    with task_environment(MODEL_ENVIRONMENT), pytest.raises(ValueError):
        import_prepared_task(EnvironmentConfig(allow_internet=False), tau3_environment)


def test_compose_does_not_resolve_storage_secret_through_an_alias(tau3_environment, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "storage-private")
    path = tau3_environment / "docker-compose.yaml"
    document = yaml.safe_load(path.read_text())
    document["services"]["tau3-runtime"]["environment"] = ["UNRELATED_NAME=${AWS_SECRET_ACCESS_KEY}"]
    path.write_text(yaml.safe_dump(document))
    with task_environment(MODEL_ENVIRONMENT), pytest.raises(ValueError):
        import_prepared_task(EnvironmentConfig(allow_internet=False), tau3_environment)


def test_compose_requires_every_build_to_have_a_prepared_image(tau3_environment):
    (tau3_environment / "shellbox-images.json").write_text(
        json.dumps({"main": {"image": MAIN_IMAGE, "workdir": "/app"}})
    )
    path = tau3_environment / "docker-compose.yaml"
    document = yaml.safe_load(path.read_text())
    document["services"]["tau3-runtime"].pop("volumes")
    path.write_text(yaml.safe_dump(document))
    with (
        task_environment(MODEL_ENVIRONMENT),
        pytest.raises(ValueError, match="prepared image"),
    ):
        import_prepared_task(EnvironmentConfig(allow_internet=False), tau3_environment)
