# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for Marin's Harbor task execution boundary."""

import json
import shutil
import textwrap

import pytest

from tests.evaluation.harbor_runtime import external_python as _external_python
from tests.evaluation.harbor_runtime import preflight as _preflight
from tests.evaluation.harbor_runtime import write_runtime_overlay


@pytest.fixture
def prepared_policy(tmp_path):
    task = tmp_path / "dataset" / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "instruction.md").write_text("Write an answer in /workspace.\n")
    (task / "task.toml").write_text(
        '[task]\nname = "test/offline"\n[environment]\n'
        f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = false\nworkdir = "/workspace"\n'
    )
    policy = tmp_path / "policy-source.json"
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


def test_prepared_preflight_resolves_offline_tasks_without_allocating_a_sandbox(tmp_path, prepared_policy):
    completed = _preflight(tmp_path, [(prepared_policy, {})])
    result = json.loads(completed.stdout)[0]

    assert result["environment"] == json.loads(prepared_policy.read_text())["environment"]["type"]
    assert result["benchmark_metadata"]["n_benchmark"] == 1
    assert result["verifier_env_keys"] == []


def test_prepared_compose_preflight_accepts_pinned_peers_without_building(tmp_path, prepared_policy):
    environment = tmp_path / "dataset" / "task" / "environment"
    (environment / "compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {"depends_on": {"runtime": {"condition": "service_healthy"}}},
                    "runtime": {
                        "build": {"context": "runtime"},
                        "expose": [8000],
                        "healthcheck": {"test": ["CMD", "true"]},
                    },
                }
            }
        )
    )
    (environment / "shellbox-images.json").write_text(
        json.dumps({"runtime": {"image": f"example.test/runtime@sha256:{'b' * 64}", "workdir": "/runtime"}})
    )
    completed = _preflight(tmp_path, [(prepared_policy, {})])
    assert json.loads(completed.stdout)[0]["benchmark_metadata"]["n_benchmark"] == 1


def test_local_preflight_applies_resource_overrides_before_admission(tmp_path, prepared_policy):
    document = json.loads(prepared_policy.read_text())
    document["environment"] = {"type": "local", "override_cpus": 2}
    prepared_policy.write_text(json.dumps(document))
    rejected = _preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert rejected.returncode == 2
    assert rejected.stdout == ""

    document["environment"]["cpu_enforcement_policy"] = "ignore"
    prepared_policy.write_text(json.dumps(document))
    accepted = _preflight(tmp_path, [(prepared_policy, {})])
    assert json.loads(accepted.stdout)[0]["benchmark_metadata"]["n_benchmark"] == 1

    document["environment"]["override_storage_mb"] = 128
    prepared_policy.write_text(json.dumps(document))
    rejected = _preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert rejected.returncode == 2
    assert rejected.stdout == ""


@pytest.mark.parametrize(
    "environment",
    [
        {"type": "docker"},
        {"import_path": "harbor.environments.docker.docker:DockerEnvironment"},
    ],
)
def test_preflight_rejects_docker_execution_before_task_launch(tmp_path, prepared_policy, environment):
    document = json.loads(prepared_policy.read_text())
    document["environment"] = environment
    prepared_policy.write_text(json.dumps(document))

    completed = _preflight(tmp_path, [(prepared_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("agents", [{"name": "mini-swe-agent"}]),
        ("agents", [{"name": "terminus-2", "env": {"STOLEN": "${AWS_SECRET_ACCESS_KEY}"}}]),
        ("agents", [{"name": "terminus-2", "kwargs": {"extra_env": {"STOLEN": "${HF_TOKEN}"}}}]),
        ("agents", [{"name": "terminus-2", "mcp_servers": [{"name": "host", "url": "http://host"}]}]),
        ("verifier", {"env": {"OPENAI_API_KEY": "${AWS_SECRET_ACCESS_KEY}"}}),
        ("verifier", {"import_path": "host.verifier:Verifier"}),
    ],
)
def test_prepared_preflight_rejects_agent_and_verifier_escape_routes(tmp_path, prepared_policy, section, replacement):
    document = json.loads(prepared_policy.read_text())
    document[section] = replacement
    prepared_policy.write_text(json.dumps(document))

    completed = _preflight(tmp_path, [(prepared_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "task_configuration",
    [
        '[environment]\ndocker_image = "example.test/task:latest"\nallow_internet = false\nworkdir = "/workspace"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = true\nworkdir = "/workspace"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
        '[environment.env]\nSTOLEN = "${AWS_SECRET_ACCESS_KEY}"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
        '[verifier.env]\nSTOLEN = "${HF_TOKEN}"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
        '[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\nOPENAI_BASE_URL = "${OPENAI_BASE_URL}"\n',
    ],
)
def test_prepared_preflight_checks_each_resolved_task_before_submission(tmp_path, prepared_policy, task_configuration):
    invalid_task = tmp_path / "dataset" / "z-invalid-task"
    shutil.copytree(tmp_path / "dataset" / "task", invalid_task)
    (invalid_task / "task.toml").write_text(task_configuration)

    completed = _preflight(tmp_path, [(prepared_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


def test_prepared_runtime_keeps_hosted_judge_credentials_out_of_task_environment(tmp_path, prepared_policy):
    preflight = json.loads(_preflight(tmp_path, [(prepared_policy, {})]).stdout)[0]
    policy = tmp_path / "stable-policy.json"
    stable_policy = json.loads(preflight["stable_policy_json"])
    stable_policy["environment"]["kwargs"] = {"image_cache": str(tmp_path / "image-cache")}
    policy.write_text(json.dumps(stable_policy))
    overlay = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay,
        tmp_path,
        job_name="test",
        dataset_path=str(tmp_path / "dataset"),
        endpoint_url="https://model.test/v1",
        served_model="test",
        task_limit=None,
        verifier_env={
            "OPENAI_API_KEY": "hosted-judge-secret",
            "OPENAI_BASE_URL": "https://judge.test/v1",
            "MODEL_NAME": "judge",
        },
        archive_dataset="test",
    )
    completed = _external_python(
        "-c",
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            from marin.evaluation.harbor.trial_driver import effective_job_config
            policy, overlay = Path({str(policy)!r}), Path({str(overlay)!r})
            config = effective_job_config(policy, overlay)
            print(json.dumps(config.verifier.env))
            """
        ),
    )

    assert json.loads(completed.stdout) == {
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
        "MODEL_NAME": "judge",
    }
    assert "hosted-judge-secret" not in completed.stdout


def test_prepared_tau2_models_resolve_openai_provider_and_hosted_judge_in_task_containers(tmp_path, prepared_policy):
    environment = tmp_path / "dataset" / "task" / "environment"
    model_templates = {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
        "TAU2_USER_MODEL": "${TAU2_USER_MODEL:-gpt-5.2}",
    }
    task_config = environment.parent / "task.toml"
    task_config.write_text(
        task_config.read_text() + '\n[verifier.env]\nTAU2_NL_ASSERTIONS_MODEL = "${TAU2_NL_ASSERTIONS_MODEL:-gpt-5.2}"\n'
    )
    (environment / "compose.yaml").write_text(
        json.dumps({"services": {"main": {}, "runtime": {"environment": model_templates}}})
    )
    (environment / "shellbox-images.json").write_text(
        json.dumps({"runtime": {"image": f"example.test/runtime@sha256:{'b' * 64}", "workdir": "/runtime"}})
    )
    overlay = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay,
        tmp_path,
        dataset_path=str(tmp_path / "dataset"),
        verifier_env={
            "OPENAI_API_KEY": "hosted-judge-secret",
            "OPENAI_BASE_URL": "https://judge.test/v1",
            "MODEL_NAME": "Qwen/Qwen3-8B",
        },
    )
    completed = _external_python(
        "-c",
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            from harbor.environments.prepared_compose import import_prepared_task
            from harbor.models.task.task import Task
            from harbor.utils.env import resolve_env_vars, task_environment
            from marin.evaluation.harbor.trial_driver import effective_job_config, _model_environment

            config = effective_job_config(Path({str(prepared_policy)!r}), Path({str(overlay)!r}))
            task = Task(Path({str(environment.parent)!r}))
            with task_environment(_model_environment(config)):
                imported = import_prepared_task(task.config.environment, task.paths.environment_dir)
                verifier = resolve_env_vars(task.config.verifier.env)
            runtime = next(container.machine for container in imported.task.containers if container.name == 'runtime')
            print(json.dumps({{'runtime': runtime.env, 'verifier': verifier}}))
            """
        ),
    )

    guest = json.loads(completed.stdout)
    # Tau2 uses LiteLLM: the first segment selects its transport, even for namespaced model IDs.
    assert guest["runtime"]["TAU2_USER_MODEL"].split("/", 1) == ["openai", "Qwen/Qwen3-8B"]
    assert guest["verifier"]["TAU2_NL_ASSERTIONS_MODEL"].split("/", 1) == ["openai", "Qwen/Qwen3-8B"]
    assert guest["runtime"]["OPENAI_API_KEY"] == "EMPTY"
    assert guest["runtime"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert "hosted-judge-secret" not in completed.stdout


@pytest.mark.parametrize("other_role", ["REFUSAL_JUDGE_MODEL", "EXPANSION_MODEL"])
def test_distinct_judge_roles_require_separate_capabilities_before_allocation(tmp_path, prepared_policy, other_role):
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
    completed = _preflight(tmp_path, [(prepared_policy, {})], check=False)
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize("roles", [("JUDGE_MODEL", "EXPANSION_MODEL"), ("EXPANSION_MODEL",)])
def test_verifier_model_roles_call_hosted_judge_through_guest_loopback(tmp_path, prepared_policy, roles):
    document = json.loads(prepared_policy.read_text())
    document["environment"] = {"type": "daytona"}
    document["verifier"] = {
        "env": {
            "OPENAI_BASE_URL": "https://api.together.xyz/v1",
            "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
            **dict.fromkeys(roles, "together_ai/openai/gpt-oss-120b"),
        }
    }
    prepared_policy.write_text(json.dumps(document))
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        verifier_env={
            "OPENAI_BASE_URL": "https://hosted-qwen.test/v1",
            "OPENAI_API_KEY": "host-judge-private",
            "MODEL_NAME": "Qwen/Qwen3-8B",
        },
    )
    completed = _external_python(
        "-c",
        textwrap.dedent(
            f"""
            import asyncio, json
            from pathlib import Path
            import httpx
            from openai import AsyncOpenAI
            from harbor_config import JobConfig
            from harbor.environments import model_relay
            from harbor.environments.model_relay import ModelRelay, host_model_authentication
            from shellbox.http_service import HttpRequest
            from marin.evaluation.harbor.trial_driver import (
                effective_job_config, RuntimeOverlay, _judge_authentication,
            )

            policy_path, overlay_path = Path({str(prepared_policy)!r}), Path({str(overlay_path)!r})
            policy = JobConfig.model_validate_json(policy_path.read_text())
            overlay = RuntimeOverlay.model_validate_json(overlay_path.read_text())
            config = effective_job_config(policy_path, overlay_path)
            provider_models = []
            original_client = httpx.AsyncClient
            def provider(request):
                assert str(request.url) == 'https://hosted-qwen.test/v1/chat/completions'
                assert request.headers['authorization'] == 'Bearer host-judge-private'
                payload = json.loads(request.content)
                assert payload['model'] == 'Qwen/Qwen3-8B'
                provider_models.append(payload['model'])
                return httpx.Response(200, json={{
                    'id':'judge', 'object':'chat.completion', 'created':0,
                    'model':payload['model'], 'choices':[{{'index':0, 'finish_reason':'stop',
                        'message':{{'role':'assistant', 'content':payload['messages'][0]['content']}}}}],
                }})
            class WireClient(original_client):
                def __init__(self, **kwargs):
                    kwargs.setdefault('transport', httpx.MockTransport(provider))
                    super().__init__(**kwargs)
            model_relay.httpx.AsyncClient = WireClient
            with host_model_authentication(_judge_authentication(policy, overlay)):
                relay = ModelRelay(config.environment.kwargs['model_endpoint'], config.environment.kwargs['model_name'])
            async def loopback(request):
                assert str(request.url) == 'http://127.0.0.1:8787/v1/chat/completions'
                assert request.headers['authorization'] == 'Bearer EMPTY'
                assert json.loads(request.content)['model'] == 'Qwen/Qwen3-8B'
                response = await relay.handle(HttpRequest('POST', request.url.path, request.content))
                return httpx.Response(
                    response.status, content=response.body, headers={{'content-type':'application/json'}})
            async def guest():
                replies = {{}}
                for role in {roles!r}:
                    transport, model = config.verifier.env[role].split('/', 1)
                    assert transport == 'openai', 'guest must use the OpenAI-compatible transport'
                    async with AsyncOpenAI(
                        api_key=config.verifier.env['OPENAI_API_KEY'],
                        base_url=config.verifier.env['OPENAI_API_BASE'],
                        http_client=WireClient(transport=httpx.MockTransport(loopback)),
                    ) as client:
                        response = await client.chat.completions.create(
                            model=model, messages=[{{'role':'user', 'content':role}}])
                        replies[role] = response.choices[0].message.content
                return replies
            print(json.dumps({{'replies':asyncio.run(guest()), 'provider_models':provider_models}}))
            """
        ),
    )
    result = json.loads(completed.stdout)
    assert result["replies"] == {role: role for role in roles}
    assert result["provider_models"] == ["Qwen/Qwen3-8B"] * len(roles)
    assert "host-judge-private" not in completed.stdout


@pytest.mark.parametrize("agent_name", ["terminus-2", "sotopia"])
def test_daytona_declared_judge_auth_stays_host_only_and_preserves_provider_routing(
    tmp_path, agent_name, prepared_policy
):
    agent = {"name": agent_name}
    verifier = {
        "env": {
            "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
            "TOGETHER_API_KEY": "${TOGETHER_API_KEY}",
            "OPENAI_BASE_URL": "https://api.together.xyz/v1",
            "JUDGE_MODEL": "together_ai/openai/gpt-oss-120b",
        }
    }
    if agent_name == "sotopia":
        agent["import_path"] = "harbor.agents.installed.sotopia:SotopiaAgent"
        agent.update(
            {
                "env": {"CUSTOM_API_KEY": "${TOGETHER_API_KEY}", "OPENAI_API_KEY": "${TOGETHER_API_KEY}"},
                "kwargs": {
                    "partner_model": "openai/openai/gpt-oss-120b",
                    "evaluator_model": "openai/openai/gpt-oss-120b",
                    "partner_api_base": "https://api.together.xyz/v1",
                    "evaluator_api_base": "https://api.together.xyz/v1",
                },
            }
        )
        verifier = {}
    policy_path = tmp_path / "policy.json"
    document = {
        "environment": {"type": "daytona"},
        "agents": [agent],
        "datasets": [{"path": "dataset"}],
        "verifier": verifier,
    }
    if agent_name == "sotopia":
        document["sotopia_cohort"] = {"version": "test-v1", "models": []}
    policy_path.write_text(json.dumps(document))
    preflight_result = _preflight(tmp_path, [(policy_path, {})], check=False)
    assert preflight_result.returncode == 0, preflight_result.stderr
    preflight = json.loads(preflight_result.stdout)[0]
    assert preflight["verifier_env_keys"] == ["TOGETHER_API_KEY"]
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        job_name="test",
        endpoint_url="http://candidate.test/v1",
        served_model="candidate",
        task_limit=None,
        archive_dataset="test",
    )
    completed = _external_python(
        "-c",
        textwrap.dedent(
            f"""
        import asyncio, json, os
        from pathlib import Path
        import httpx
        from harbor_config import JobConfig
        from harbor.environments import model_relay
        from harbor.environments.model_relay import ModelRelay, host_model_authentication
        from shellbox.http_service import HttpRequest
        from marin.evaluation.harbor.trial_driver import effective_job_config, RuntimeOverlay, _judge_authentication
        os.environ['TOGETHER_API_KEY'] = 'host-provider-private'
        policy = JobConfig.model_validate_json(Path({str(policy_path)!r}).read_text())
        overlay = RuntimeOverlay.model_validate_json(Path({str(overlay_path)!r}).read_text())
        config = effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r}))
        original_client = httpx.AsyncClient
        def provider(request):
            assert request.url == 'https://api.together.xyz/v1/chat/completions'
            assert request.headers['authorization'] == 'Bearer host-provider-private'
            assert json.loads(request.content)['model'] == 'openai/gpt-oss-120b'
            return httpx.Response(200, json={{'ok':True}})
        model_relay.httpx.AsyncClient = lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs)
        with host_model_authentication(_judge_authentication(policy, overlay)):
            relay = ModelRelay(config.environment.kwargs['model_endpoint'], config.environment.kwargs['model_name'])
            request = HttpRequest('POST','/v1/chat/completions',b'{{"model":"untrusted-override"}}')
            response = asyncio.run(relay.handle(request))
            assert response.status == 200
        print(json.dumps({{'verifier':config.verifier.env,
            'agent':config.agents[0].env, 'kwargs':config.agents[0].kwargs}}))
    """
        ),
    )
    result = json.loads(completed.stdout)
    assert result["verifier"]["OPENAI_API_KEY"] == "EMPTY"
    assert result["verifier"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert "host-provider-private" not in completed.stdout
    if agent_name == "sotopia":
        assert result["agent"] == {
            "CUSTOM_API_KEY": "EMPTY",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": "http://127.0.0.1:8788/v1",
            "OPENAI_API_BASE": "http://127.0.0.1:8788/v1",
        }
        assert result["kwargs"]["partner_api_base"] == "http://127.0.0.1:8787/v1"
        assert result["kwargs"]["evaluator_api_base"] == "http://127.0.0.1:8787/v1"
    else:
        assert result["verifier"]["JUDGE_MODEL"] == "openai/openai/gpt-oss-120b"
