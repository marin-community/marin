# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight rejects unsafe task inputs before any runtime allocation."""

import json
import shutil

import pytest
from marin.evaluation.harbor.driver_config import HarborErrorTaxonomy
from marin.evaluation.harbor.runner import _read_trial
from rigging.filesystem.storage_path import StoragePath

from tests.evaluation.harbor_runtime import preflight


@pytest.fixture
def prepared_policy(tmp_path):
    task = tmp_path / "dataset" / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "instruction.md").write_text("Write an answer in /workspace.\n")
    (task / "task.toml").write_text(
        '[task]\nname = "test/offline"\n[environment]\n'
        f'docker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "environment": {"type": "local"},
                "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    return policy


def test_runtime_failures_use_published_infrastructure_taxonomy_in_marin(tmp_path, prepared_policy):
    report = json.loads(preflight(tmp_path, [(prepared_policy, {})]).stdout)[0]
    taxonomy = HarborErrorTaxonomy(**{role: frozenset(names) for role, names in report["error_taxonomy"].items()})
    for failure in ("RuntimeServiceError", "RuntimeLogCollectionError", "EnvironmentStopError"):
        result_path = tmp_path / f"{failure}.json"
        result_path.write_text(
            json.dumps(
                {
                    "task_name": "test/offline",
                    "verifier_result": {"rewards": {"reward": 1}},
                    "exception_info": {"exception_type": failure, "exception_message": "failed"},
                }
            )
        )
        trial = _read_trial(StoragePath(str(result_path)), taxonomy)
        assert not trial.scored
        assert trial.error is not None
        assert trial.error["type"] == failure


@pytest.mark.parametrize(
    ("section", "replacement", "diagnostic"),
    [
        ("agents", [{"name": "mini-swe-agent"}], "requires terminus-2, oracle, Pi, or pi-acp"),
        (
            "agents",
            [{"name": "pi", "kwargs": {"thinking_format": "chat-template"}, "env": {"STOLEN": "${HF_TOKEN}"}}],
            "environment forwarding",
        ),
        (
            "agents",
            [
                {
                    "name": "acp",
                    "kwargs": {
                        "registry_entry": {
                            "id": "another-agent",
                            "name": "Another agent",
                            "version": "1.0.0",
                            "description": "An alternate ACP agent",
                            "distribution": {"npx": {"package": "another-agent@1.0.0"}},
                        }
                    },
                }
            ],
            "requires terminus-2, oracle, Pi, or pi-acp",
        ),
        ("agents", [{"name": "terminus-2", "env": {"STOLEN": "${AWS_SECRET_ACCESS_KEY}"}}], "environment forwarding"),
        (
            "agents",
            [{"name": "terminus-2", "kwargs": {"extra_env": {"STOLEN": "${HF_TOKEN}"}}}],
            "alternate agent environment",
        ),
        ("agents", [{"name": "terminus-2", "mcp_servers": [{"name": "host", "url": "http://host"}]}], "MCP servers"),
        ("verifier", {"env": {"OPENAI_API_KEY": "${AWS_SECRET_ACCESS_KEY}"}}, "AWS_SECRET_ACCESS_KEY"),
        ("verifier", {"import_path": "host.verifier:Verifier"}, "without a custom import"),
        (
            "user_agent",
            {"import_path": "harbor.agents.terminus_2:Terminus2", "bridge": {"kind": "acp"}},
            "sandboxed Tau3 ACP simulated user",
        ),
    ],
)
def test_preflight_rejects_host_escape_routes_before_submission(
    tmp_path, prepared_policy, section, replacement, diagnostic
):
    document = json.loads(prepared_policy.read_text())
    document[section] = replacement
    prepared_policy.write_text(json.dumps(document))
    completed = preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert diagnostic in completed.stderr


@pytest.mark.parametrize(
    ("environment", "location"),
    [
        ({"type": "docker"}, ["environment", "type"]),
        ({"import_path": "harbor.environments.docker.docker:DockerEnvironment"}, ["environment", "import_path"]),
    ],
)
def test_preflight_reports_rejected_legacy_environment_field(tmp_path, prepared_policy, environment, location):
    document = json.loads(prepared_policy.read_text())
    document["environment"] = environment
    prepared_policy.write_text(json.dumps(document))
    completed = preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert completed.returncode == 2
    assert any(error["loc"] == location for error in json.loads(completed.stderr.strip().splitlines()[-1]))


@pytest.mark.parametrize(
    ("unsafe_configuration", "diagnostic"),
    [
        ('docker_image = "example.test/task:latest"\nallow_internet = false\n', "pinned with @sha256"),
        (f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = true\n', "allow_internet=false"),
        (
            f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = false\n'
            '[environment.env]\nSTOLEN = "${AWS_SECRET_ACCESS_KEY}"\n',
            "AWS_SECRET_ACCESS_KEY",
        ),
        (
            f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = false\n'
            '[verifier.env]\nSTOLEN = "${HF_TOKEN}"\n',
            "HF_TOKEN",
        ),
        (
            f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = false\n'
            '[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\nOPENAI_BASE_URL = "${OPENAI_BASE_URL}"\n',
            "OPENAI_API_KEY",
        ),
    ],
)
def test_preflight_checks_each_resolved_task_before_submission(
    tmp_path, prepared_policy, unsafe_configuration, diagnostic
):
    invalid_task = tmp_path / "dataset" / "z-invalid-task"
    shutil.copytree(tmp_path / "dataset/task", invalid_task)
    (invalid_task / "task.toml").write_text('[environment]\nworkdir = "/workspace"\n' + unsafe_configuration)
    completed = preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert diagnostic in completed.stderr


@pytest.mark.parametrize("other_role", ["REFUSAL_JUDGE_MODEL", "EXPANSION_MODEL"])
def test_preflight_rejects_distinct_judge_roles_before_allocation(tmp_path, prepared_policy, other_role, monkeypatch):
    monkeypatch.setenv("DAYTONA_TARGET", "test-target")
    document = json.loads(prepared_policy.read_text())
    document["environment"] = {"type": "daytona"}
    document["verifier"] = {
        "env": {
            "OPENAI_BASE_URL": "https://api.together.xyz/v1",
            "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
            "JUDGE_MODEL": "together_ai/openai/gpt-oss-120b",
            other_role: "together_ai/another-model",
        }
    }
    prepared_policy.write_text(json.dumps(document))
    completed = preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Distinct verifier model roles" in completed.stderr
