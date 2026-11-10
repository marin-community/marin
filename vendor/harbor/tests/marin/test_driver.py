# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin's Harbor adapter, run directly in Harbor's locked runtime environment."""

import json
from pathlib import Path

import httpx
import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.environments import model_relay
from harbor.environments.docker_translate import import_prepared_task
from harbor.environments.model_relay import ModelRelay
from harbor.models.task.task import Task
from harbor.utils.env import resolve_env_vars
from harbor_config import JobConfig
from harbor_config.models.trial.config import TaskConfig, TrialConfig
from marin.evaluation.harbor.trial_driver import (
    RuntimeOverlay,
    _model_authentication,
    _model_environment,
    _preflight_one,
    effective_job_config,
)
from marin.evaluation.harbor.agent_context import CANDIDATE_API_KEY_ENV
from openai import AsyncOpenAI
from shellbox.http_service import HttpRequest

_ROOT = Path(__file__).parents[4]
_CANDIDATE_ENDPOINT = "https://candidate.test/proxy/t/private-capability/v1"
_JUDGE_ENDPOINT = "https://judge.test/v1"


@pytest.fixture
def policy():
    return JobConfig.model_validate(
        {
            "environment": {"type": "daytona"},
            "agents": [{"name": "terminus-2"}],
            "datasets": [{"path": "dataset"}],
        }
    )


@pytest.fixture
def overlay(tmp_path):
    return RuntimeOverlay(
        job_name="test",
        jobs_dir=str(tmp_path / "jobs"),
        dataset_path=None,
        endpoint_url=_CANDIDATE_ENDPOINT,
        served_model="candidate-alias",
        model_identity="org/candidate@revision",
        task_limit=None,
        model_agent_kwargs={"thinking_format": "qwen-chat-template"},
        verifier_env={
            "OPENAI_BASE_URL": _JUDGE_ENDPOINT,
            "OPENAI_API_KEY": "judge-private",
            "MODEL_NAME": "judge-alias",
        },
        judge_model_identity="org/judge@revision",
        archive_root=str(tmp_path / "archive"),
        archive_dataset="test",
    )


@pytest.fixture
def prepared_task(tmp_path):
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
    return task


def _trial(config):
    return TrialConfig(
        task=TaskConfig(name="test/offline", ref="sha256:" + "a" * 64),
        agent=config.agents[0],
        environment=config.environment,
        verifier=config.verifier,
    )


def test_model_transport_relocation_preserves_saved_trial_and_changed_scoring_inputs_do_not(
    policy, overlay
):
    original = effective_job_config(policy, overlay)
    saved = TrialConfig.model_validate_json(_trial(original).model_dump_json())
    relocated = overlay.model_copy(
        update={
            "endpoint_url": "https://candidate-replacement.test/v1",
            "served_model": "candidate-replacement",
            "verifier_env": {
                "OPENAI_BASE_URL": "https://judge-replacement.test/v1",
                "OPENAI_API_KEY": "judge-replacement-private",
                "MODEL_NAME": "judge-replacement",
            },
        }
    )
    assert saved.matches_identity(_trial(effective_job_config(policy, relocated)))
    for changes in (
        {"model_identity": "org/other-candidate@revision"},
        {"judge_model_identity": "org/other-judge@revision"},
        {"model_agent_kwargs": {"temperature": 0.8}},
    ):
        assert not saved.matches_identity(
            _trial(effective_job_config(policy, relocated.model_copy(update=changes)))
        )


@pytest.mark.parametrize(
    "identity",
    ["org/model", "gs://models/export/step-10", "s3://models/export/step-10"],
)
def test_factory_reports_original_model_identity_when_serving_alias_changes(
    tmp_path, policy, overlay, identity
):
    original = effective_job_config(
        policy, overlay.model_copy(update={"model_identity": identity})
    )
    resumed = effective_job_config(
        policy,
        overlay.model_copy(
            update={"model_identity": identity, "served_model": "resumed-alias"}
        ),
    )
    agent = AgentFactory.create_agent_from_config(
        resumed.agents[0], logs_dir=tmp_path / "logs"
    )
    assert agent.to_agent_info().model_info.model_dump() == {
        "name": identity,
        "provider": "hosted_vllm",
    }
    assert _trial(original).matches_identity(_trial(resumed))
    assert resumed.agents[0].model_alias == "hosted_vllm/resumed-alias"


def test_host_agent_receives_live_capability_while_saved_config_redacts_it(
    policy, overlay
):
    config = effective_job_config(policy, overlay)
    assert config.agents[0].kwargs["api_base"] == _CANDIDATE_ENDPOINT
    assert "private-capability" not in config.model_dump_json()


@pytest.mark.parametrize(
    "agent_name", ["pi", "opencode", "mini-swe-agent", "acp:pi-acp", "sotopia"]
)
def test_installed_candidate_cannot_receive_host_endpoint_or_credentials(
    policy, overlay, agent_name
):
    configured = policy.agents[0].model_copy(
        update={
            "name": agent_name,
            "env": {
                "OPENAI_API_KEY": "candidate-private",
                "OPENAI_API_BASE": _CANDIDATE_ENDPOINT,
            },
        }
    )
    if agent_name == "sotopia":
        configured = configured.model_copy(
            update={
                "name": None,
                "import_path": "harbor.agents.installed.sotopia:SotopiaAgent",
            }
        )
    if agent_name == "opencode":
        configured = configured.model_copy(
            update={
                "kwargs": {
                    "opencode_config": {
                        "provider": {
                            "hosted_vllm": {
                                "options": {
                                    "baseURL": _CANDIDATE_ENDPOINT,
                                    "apiKey": "private",
                                }
                            }
                        }
                    }
                }
            }
        )
    config = effective_job_config(
        policy.model_copy(update={"agents": [configured]}), overlay
    )
    agent = config.agents[0]
    serialized = json.dumps(agent.model_dump(mode="python"))
    assert "private" not in serialized
    assert agent.kwargs["api_base"] == "http://127.0.0.1:8788/v1"
    assert config.environment.kwargs["candidate_model_endpoint"] == _CANDIDATE_ENDPOINT


@pytest.mark.parametrize(
    ("name", "import_path", "selected_class", "expected_endpoint"),
    [
        (
            "terminus-2",
            "harbor.agents.installed.pi:Pi",
            "Terminus2",
            _CANDIDATE_ENDPOINT,
        ),
        ("pi", "harbor.agents.terminus_2:Terminus2", "Pi", "http://127.0.0.1:8788/v1"),
        (
            "acp:pi-acp",
            "harbor.agents.terminus_2:Terminus2",
            "AcpAgent",
            "http://127.0.0.1:8788/v1",
        ),
    ],
)
def test_candidate_routing_agrees_with_factory_precedence(
    tmp_path, policy, overlay, name, import_path, selected_class, expected_endpoint
):
    candidate = policy.agents[0].model_copy(
        update={"name": name, "import_path": import_path}
    )
    config = effective_job_config(
        policy.model_copy(update={"agents": [candidate]}), overlay
    )
    agent = AgentFactory.create_agent_from_config(
        config.agents[0], logs_dir=tmp_path / "logs"
    )
    assert type(agent).__name__ == selected_class
    assert config.agents[0].kwargs["api_base"] == expected_endpoint


def test_host_authentication_uses_owned_snapshot_and_rejects_endpoint_credential_conflict(
    policy, overlay, monkeypatch
):
    source = {CANDIDATE_API_KEY_ENV: "candidate-private"}
    monkeypatch.setenv(CANDIDATE_API_KEY_ENV, "later-ambient-private")
    assert _model_authentication(policy, overlay, host_environment=source) == {
        _CANDIDATE_ENDPOINT: "candidate-private",
        _JUDGE_ENDPOINT: "judge-private",
    }
    with pytest.raises(ValueError) as error:
        _model_authentication(
            policy,
            overlay.model_copy(update={"endpoint_url": _JUDGE_ENDPOINT}),
            host_environment=source,
        )
    assert "private" not in str(error.value)


async def test_candidate_transport_keeps_the_policy_openai_key_for_its_judge(
    policy, overlay, monkeypatch
):
    candidate = policy.agents[0].model_copy(
        update={"env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}}
    )
    verifier = policy.verifier.model_copy(
        update={
            "env": {
                "OPENAI_BASE_URL": _JUDGE_ENDPOINT,
                "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                "MODEL_NAME": "judge-alias",
            }
        }
    )
    policy = policy.model_copy(update={"agents": [candidate], "verifier": verifier})
    authentication = _model_authentication(
        policy,
        overlay.model_copy(update={"verifier_env": {}}),
        host_environment={
            CANDIDATE_API_KEY_ENV: "candidate-private",
            "OPENAI_API_KEY": "judge-private",
        },
    )
    observed = {}

    def provider(request):
        observed[str(request.url)] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": []})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(provider), **kwargs),
    )
    for endpoint in (_CANDIDATE_ENDPOINT, _JUDGE_ENDPOINT):
        relay = ModelRelay.from_endpoint(
            endpoint, "model", authentication=authentication
        )
        response = await relay.handle(
            HttpRequest("POST", "/v1/chat/completions", b'{"messages": []}')
        )
        assert response.status == 200
    assert observed == {
        _CANDIDATE_ENDPOINT + "/chat/completions": "Bearer candidate-private",
        _JUDGE_ENDPOINT + "/chat/completions": "Bearer judge-private",
    }


def test_tau3_candidate_and_simulated_user_have_separate_model_capabilities(
    prepared_task, overlay
):
    campaign = (
        _ROOT
        / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs/tau3-pi.yaml"
    )
    document = yaml.safe_load(campaign.read_text())
    document.update(
        environment={"type": "local"}, datasets=[{"path": str(prepared_task.parent)}]
    )
    policy = JobConfig.model_validate(document)
    config = effective_job_config(policy, overlay)
    assert config.agents[0].kwargs["api_base"] == "http://127.0.0.1:8788/v1"
    assert config.user_agent.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert config.user_agent.env["OPENAI_API_KEY"] == "EMPTY"
    assert config.environment.kwargs["judge_access"] == "runtime"
    assert "private" not in config.agents[0].model_dump_json()
    untrusted = policy.user_agent.model_copy(
        update={"env": {"OPENAI_BASE_URL": "http://outside.test/v1"}}
    )
    with pytest.raises(ValueError):
        effective_job_config(
            policy.model_copy(update={"user_agent": untrusted}), overlay
        )


def test_tau2_compose_and_verifier_models_use_hosted_judge_without_parent_secrets(
    prepared_task, policy, overlay
):
    environment = prepared_task / "environment"
    (environment / "compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {},
                    "runtime": {
                        "environment": {
                            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
                            "TAU2_USER_MODEL": "${TAU2_USER_MODEL:-gpt-5.2}",
                        }
                    },
                }
            }
        )
    )
    (environment / "shellbox-images.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "image": "example.test/runtime@sha256:" + "b" * 64,
                    "workdir": "/runtime",
                }
            }
        )
    )
    config = effective_job_config(policy, overlay)
    guest_environment = _model_environment(config)
    task = Task(prepared_task)
    imported = import_prepared_task(
        task.config.environment, environment, guest_environment=guest_environment
    )
    runtime = next(
        container.machine
        for container in imported.task.containers
        if container.name == "runtime"
    )
    verifier = resolve_env_vars(
        {"TAU2_NL_ASSERTIONS_MODEL": "${TAU2_NL_ASSERTIONS_MODEL}"},
        source=guest_environment,
    )
    assert (
        runtime.env["TAU2_USER_MODEL"]
        == verifier["TAU2_NL_ASSERTIONS_MODEL"]
        == "openai/org/judge@revision"
    )
    assert runtime.env["OPENAI_API_KEY"] == "EMPTY"
    assert runtime.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert "judge-private" not in json.dumps(runtime.env)


@pytest.mark.parametrize(
    "roles", [("JUDGE_MODEL", "EXPANSION_MODEL"), ("EXPANSION_MODEL",), ("MODEL_NAME",)]
)
@pytest.mark.parametrize("hosted_judge_key", ["EMPTY", "judge-private"])
async def test_verifier_roles_reach_fixed_provider_with_host_authentication(
    policy, overlay, roles, hosted_judge_key, monkeypatch
):
    verifier = policy.verifier.model_copy(
        update={
            "env": {
                "OPENAI_API_BASE": (
                    "https://api.together.xyz/proxy/t/policy-capability/v1"
                ),
                "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
                "TOGETHER_API_KEY": "${TOGETHER_API_KEY}",
                **dict.fromkeys(roles, "together_ai/openai/gpt-oss-120b"),
            }
        }
    )
    policy = policy.model_copy(update={"verifier": verifier})
    overlay = overlay.model_copy(
        update={
            "verifier_env": {**overlay.verifier_env, "OPENAI_API_KEY": hosted_judge_key}
        }
    )
    config = effective_job_config(policy, overlay)
    authentication = _model_authentication(
        policy,
        overlay,
        host_environment={"TOGETHER_API_KEY": "policy-together-private"},
    )
    provider_models = []

    def provider(request):
        assert str(request.url) == _JUDGE_ENDPOINT + "/chat/completions"
        assert request.headers["authorization"] == "Bearer " + hosted_judge_key
        payload = json.loads(request.content)
        provider_models.append(payload["model"])
        return httpx.Response(
            200,
            json={
                "id": "judge",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": payload["messages"][0],
                    }
                ],
            },
        )

    client_type = httpx.AsyncClient

    class WireClient(client_type):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", httpx.MockTransport(provider))
            super().__init__(**kwargs)

    monkeypatch.setattr(model_relay.httpx, "AsyncClient", WireClient)
    relay = ModelRelay.from_endpoint(
        config.environment.kwargs["model_endpoint"],
        config.environment.kwargs["model_name"],
        authentication=authentication,
    )

    async def loopback(request):
        assert request.headers["authorization"] == "Bearer EMPTY"
        response = await relay.handle(
            HttpRequest("POST", request.url.path, request.content)
        )
        return httpx.Response(
            response.status,
            content=response.body,
            headers={"content-type": "application/json"},
        )

    async with AsyncOpenAI(
        api_key=config.verifier.env["OPENAI_API_KEY"],
        base_url=config.verifier.env["OPENAI_API_BASE"],
        http_client=WireClient(transport=httpx.MockTransport(loopback)),
    ) as client:
        for role in roles:
            model = config.verifier.env[role]
            if role != "MODEL_NAME":
                transport, model = model.split("/", 1)
                assert transport == "openai"
            response = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": role}]
            )
            assert response.choices[0].message.content == role
    assert provider_models == ["judge-alias"] * len(roles)


def test_local_acp_shorthand_survives_preflight_normalization(tmp_path, prepared_task):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "environment": {"type": "local"},
                "agents": [{"name": "acp:pi-acp@0.0.33"}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    report = _preflight_one(path, {}, None, host_environment={})
    assert report["benchmark_metadata"]["n_benchmark"] == 1


@pytest.mark.parametrize(
    ("backend", "location"),
    [("local", "primary"), ("daytona", "primary"), ("local", "peer")],
)
@pytest.mark.parametrize("endpoint_name", ["OPENAI_API_BASE", "OPENAI_BASE_URL"])
@pytest.mark.parametrize("authority", ["missing-judge", "verifier-only", "runtime"])
def test_preflight_requires_runtime_judge_for_primary_and_peer_model_access(
    tmp_path, prepared_task, backend, location, endpoint_name, authority
):
    values = {endpoint_name: "http://127.0.0.1:8787/v1"}
    if location == "primary":
        task_config = prepared_task / "task.toml"
        task_config.write_text(
            task_config.read_text()
            + f'\n[environment.env]\n{endpoint_name} = "{values[endpoint_name]}"\n'
        )
    else:
        (prepared_task / "environment/compose.yaml").write_text(
            json.dumps(
                {
                    "services": {
                        "main": {},
                        "runtime": {
                            "image": "example.test/runtime@sha256:" + "b" * 64,
                            "working_dir": "/runtime",
                            "environment": values,
                        },
                    }
                }
            )
        )
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "environment": {
                    "type": backend,
                    "kwargs": (
                        {"judge_access": "runtime"} if authority == "runtime" else {}
                    ),
                },
                "agents": [{"name": "terminus-2"}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    judge = None if authority == "missing-judge" else "judge-alias"
    if authority == "runtime":
        report = _preflight_one(
            path, {}, judge, host_environment={"DAYTONA_TARGET": "test-target"}
        )
        assert report["benchmark_metadata"]["n_benchmark"] == 1
    else:
        diagnostic = (
            "host judge" if authority == "missing-judge" else "judge_access='runtime'"
        )
        with pytest.raises(ValueError, match=diagnostic):
            _preflight_one(
                path, {}, judge, host_environment={"DAYTONA_TARGET": "test-target"}
            )


def test_preflight_validates_later_peer_aliases_after_primary_model_endpoint(
    tmp_path, prepared_task
):
    task_config = prepared_task / "task.toml"
    task_config.write_text(
        task_config.read_text()
        + '\n[environment.env]\nOPENAI_BASE_URL="http://127.0.0.1:8787/v1"\n'
    )
    (prepared_task / "environment/compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {},
                    "runtime": {
                        "image": "example.test/runtime@sha256:" + "b" * 64,
                        "working_dir": "/runtime",
                        "environment": {
                            "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
                            "OPENAI_API_BASE": "http://different/v1",
                        },
                    },
                },
            }
        )
    )
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "environment": {"type": "local", "kwargs": {"judge_access": "runtime"}},
                "agents": [{"name": "terminus-2"}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    with pytest.raises(ValueError, match="Conflicting OpenAI model endpoint aliases"):
        _preflight_one(path, {}, "judge-alias", host_environment={})


def test_preflight_daytona_requires_target_before_resolving_tasks(
    tmp_path, prepared_task
):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "terminus-2"}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    with pytest.raises(ValueError, match="DAYTONA_TARGET"):
        _preflight_one(path, {}, None, host_environment={})
    report = _preflight_one(
        path, {}, None, host_environment={"DAYTONA_TARGET": "test-target"}
    )
    assert report["benchmark_metadata"]["n_benchmark"] == 1


@pytest.mark.parametrize("backend", ["local", "daytona"])
def test_preflight_standalone_pi_requires_matching_prepared_main_image(
    tmp_path, prepared_task, backend
):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "environment": {"type": backend},
                "agents": [{"name": "pi"}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    kwargs = {"thinking_format": "chat-template"}
    host = {"DAYTONA_TARGET": "test-target"}
    with pytest.raises(
        ValueError, match="Standalone Pi requires a prepared main image"
    ):
        _preflight_one(path, kwargs, None, host_environment=host)
    manifest = prepared_task / "environment/shellbox-images.json"
    manifest.write_text(
        json.dumps(
            {
                "main": {
                    "image": "example.test/task@sha256:" + "b" * 64,
                    "workdir": "/workspace",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="(must match|conflicts with).*prepared"):
        _preflight_one(path, kwargs, None, host_environment=host)
    manifest.write_text(
        json.dumps(
            {
                "main": {
                    "image": "example.test/task@sha256:" + "a" * 64,
                    "workdir": "/workspace",
                }
            }
        )
    )
    report = _preflight_one(path, kwargs, None, host_environment=host)
    assert report["benchmark_metadata"]["n_benchmark"] == 1
