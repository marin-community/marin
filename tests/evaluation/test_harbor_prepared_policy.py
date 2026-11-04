# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for Marin's Harbor task execution boundary."""

import json
import shutil
import textwrap
from pathlib import Path

import pytest
import yaml
from marin.evaluation.harbor.driver_config import HarborErrorTaxonomy
from marin.evaluation.harbor.runner import _read_trial
from rigging.filesystem.storage_path import StoragePath

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


def test_prepared_compose_preflight_accepts_pinned_peers(tmp_path, prepared_policy):
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


def test_runtime_failures_use_published_infrastructure_taxonomy_in_marin(tmp_path, prepared_policy):
    report = json.loads(_preflight(tmp_path, [(prepared_policy, {})]).stdout)[0]
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
        assert trial.scored is False
        assert trial.error is not None
        assert trial.error["type"] == failure


def test_prepared_tau3_campaign_policy_uses_separate_host_candidate_and_judge_capabilities(tmp_path, prepared_policy):
    campaign = (
        Path(__file__).parents[2]
        / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/harbor-configs/tau3-pi.yaml"
    )
    document = yaml.safe_load(campaign.read_text())
    document["environment"] = {"type": "local"}
    document["datasets"] = [{"path": "dataset"}]
    prepared_policy.write_text(json.dumps(document))
    preflight = json.loads(_preflight(tmp_path, [(prepared_policy, {})], judge_model="judge").stdout)[0]
    policy = tmp_path / "stable-policy.json"
    policy.write_text(preflight["stable_policy_json"])
    overlay = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay,
        tmp_path,
        endpoint_url="https://candidate.test/proxy/t/private-capability/v1",
        judge_model_identity="org/judge@revision",
        verifier_env={
            "OPENAI_API_KEY": "judge-private",
            "OPENAI_BASE_URL": "https://judge.test/v1",
            "MODEL_NAME": "judge",
        },
    )
    completed = _external_python(
        "-c",
        textwrap.dedent(
            f"""
            import json, os
            from pathlib import Path
            from harbor_config import JobConfig
            from harbor_config.models.trial.config import TaskConfig, TrialConfig
            from marin.evaluation.harbor.trial_driver import (
                effective_job_config, RuntimeOverlay, _model_authentication,
            )
            policy, overlay_path = Path({str(policy)!r}), Path({str(overlay)!r})
            original = JobConfig.model_validate_json(policy.read_text())
            overlay = RuntimeOverlay.model_validate_json(overlay_path.read_text())
            config = effective_job_config(original, overlay)
            untrusted_user = original.user_agent.model_copy(update={{'env': {{'OPENAI_BASE_URL': 'http://outside.test/v1'}}}})
            try:
                effective_job_config(original.model_copy(update={{'user_agent': untrusted_user}}), overlay)
            except ValueError:
                pass
            else:
                raise AssertionError('source policy forwarded simulated-user environment')
            host_environment = {{'OPENAI_API_KEY': 'candidate-private'}}
            os.environ['OPENAI_API_KEY'] = 'later-ambient-key'
            authentication = _model_authentication(original, overlay, host_environment=host_environment)
            assert authentication == {{
                'https://candidate.test/proxy/t/private-capability/v1': 'candidate-private',
                'https://judge.test/v1': 'judge-private',
            }}
            conflicting = overlay.model_copy(update={{'endpoint_url': 'https://judge.test/v1'}})
            try:
                _model_authentication(original, conflicting, host_environment=host_environment)
            except ValueError as error:
                assert 'candidate-private' not in str(error) and 'judge-private' not in str(error)
            else:
                raise AssertionError('conflicting candidate and judge credentials accepted')
            def trial(job):
                return TrialConfig(
                    task=TaskConfig(name='test/offline', ref='sha256:' + 'a' * 64),
                    agent=job.agents[0], environment=job.environment, verifier=job.verifier,
                )
            saved_trial = TrialConfig.model_validate_json(trial(config).model_dump_json())
            relocated = overlay.model_copy(update={{
                'endpoint_url': 'https://candidate-replacement.test/v1',
                'served_model': 'candidate-replacement-alias',
                'verifier_env': {{
                    'OPENAI_API_KEY': 'replacement-judge-key',
                    'OPENAI_BASE_URL': 'https://judge-replacement.test/v1',
                    'MODEL_NAME': 'judge-replacement-alias',
                }},
            }})
            changed_judge = relocated.model_copy(update={{'judge_model_identity': 'org/different-judge@revision'}})
            changed_generation = relocated.model_copy(update={{'model_agent_kwargs': {{'temperature': 0.8}}}})
            print(json.dumps({{
                'agent': config.agents[0].model_dump(mode='python'),
                'user_environment': config.user_agent.env,
                'verifier': config.verifier.env,
                'capabilities': config.environment.kwargs,
                'resume_relocated': saved_trial.matches_identity(trial(effective_job_config(original, relocated))),
                'resume_changed_judge': saved_trial.matches_identity(
                    trial(effective_job_config(original, changed_judge))),
                'resume_changed_generation': saved_trial.matches_identity(
                    trial(effective_job_config(original, changed_generation))),
            }}))
            """
        ),
    )
    result = json.loads(completed.stdout)
    assert result["agent"]["kwargs"]["api_base"] == "http://127.0.0.1:8788/v1"
    assert result["agent"]["kwargs"]["api_key"] == "EMPTY"
    assert result["verifier"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert result["user_environment"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert result["user_environment"]["OPENAI_API_KEY"] == "EMPTY"
    assert result["capabilities"]["judge_access"] == "runtime"
    assert result["capabilities"]["candidate_model_endpoint"].endswith("private-capability/v1")
    assert result["resume_relocated"] is True
    assert result["resume_changed_judge"] is False
    assert result["resume_changed_generation"] is False
    assert "private-capability" not in json.dumps(result["agent"])
    assert "candidate-private" not in completed.stdout
    assert "judge-private" not in completed.stdout


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
        ("agents", [{"name": "pi", "env": {"STOLEN": "${HF_TOKEN}"}}]),
        ("agents", [{"name": "acp", "kwargs": {"registry_entry": {"id": "another-agent"}}}]),
        ("agents", [{"name": "terminus-2", "env": {"STOLEN": "${AWS_SECRET_ACCESS_KEY}"}}]),
        ("agents", [{"name": "terminus-2", "kwargs": {"extra_env": {"STOLEN": "${HF_TOKEN}"}}}]),
        ("agents", [{"name": "terminus-2", "mcp_servers": [{"name": "host", "url": "http://host"}]}]),
        ("verifier", {"env": {"OPENAI_API_KEY": "${AWS_SECRET_ACCESS_KEY}"}}),
        ("verifier", {"import_path": "host.verifier:Verifier"}),
        (
            "user_agent",
            {"import_path": "harbor.agents.terminus_2:Terminus2", "bridge": {"kind": "acp"}},
        ),
        (
            "user_agent",
            {
                "import_path": "harbor.agents.tau3_acp_relay.harbor_agent:Tau3HarborACPRelay",
                "bridge": {"kind": "acp"},
                "mcp_servers": [{"name": "tau3-runtime", "transport": "streamable-http", "url": "http://host/mcp"}],
            },
        ),
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
        judge_model_identity="org/judge@revision",
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
            from harbor_config import JobConfig
            from marin.evaluation.harbor.trial_driver import effective_job_config, RuntimeOverlay
            policy, overlay = Path({str(policy)!r}), Path({str(overlay)!r})
            config = effective_job_config(
                JobConfig.model_validate_json(policy.read_text()),
                RuntimeOverlay.model_validate_json(overlay.read_text()),
            )
            print(json.dumps({{
                'verifier': config.verifier.env,
                'candidate': config.agents[0].env,
                'judge_access': config.environment.kwargs['judge_access'],
            }}))
            """
        ),
    )

    result = json.loads(completed.stdout)
    assert result["candidate"] == {}
    assert result["judge_access"] == "verifier"
    assert result["verifier"] == {
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
        "MODEL_NAME": "org/judge@revision",
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
        judge_model_identity="org/judge@revision",
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
            from harbor.environments.docker_translate import import_prepared_task
            from harbor.models.task.task import Task
            from harbor.utils.env import resolve_env_vars
            from harbor_config import JobConfig
            from marin.evaluation.harbor.trial_driver import effective_job_config, RuntimeOverlay, _model_environment

            config = effective_job_config(
                JobConfig.model_validate_json(Path({str(prepared_policy)!r}).read_text()),
                RuntimeOverlay.model_validate_json(Path({str(overlay)!r}).read_text()),
            )
            task = Task(Path({str(environment.parent)!r}))
            guest_environment = _model_environment(config)
            imported = import_prepared_task(
                task.config.environment, task.paths.environment_dir, guest_environment=guest_environment
            )
            verifier = resolve_env_vars(task.config.verifier.env, source=guest_environment)
            runtime = next(container.machine for container in imported.task.containers if container.name == 'runtime')
            print(json.dumps({{'runtime': runtime.env, 'verifier': verifier}}))
            """
        ),
    )

    guest = json.loads(completed.stdout)
    # Tau2 uses LiteLLM: the first segment selects its transport, even for namespaced model IDs.
    assert guest["runtime"]["TAU2_USER_MODEL"].split("/", 1) == ["openai", "org/judge@revision"]
    assert guest["verifier"]["TAU2_NL_ASSERTIONS_MODEL"].split("/", 1) == ["openai", "org/judge@revision"]
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
        judge_model_identity="org/judge@revision",
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
            from harbor_config.models.trial.config import TaskConfig, TrialConfig
            from harbor.environments import model_relay
            from harbor.environments.model_relay import ModelRelay
            from shellbox.http_service import HttpRequest
            from marin.evaluation.harbor.trial_driver import (
                effective_job_config, RuntimeOverlay, _judge_authentication,
            )

            policy_path, overlay_path = Path({str(prepared_policy)!r}), Path({str(overlay_path)!r})
            policy = JobConfig.model_validate_json(policy_path.read_text())
            overlay = RuntimeOverlay.model_validate_json(overlay_path.read_text())
            config = effective_job_config(policy, overlay)
            saved_trial = TrialConfig(
                task=TaskConfig(name='test/offline', ref='sha256:' + 'a' * 64),
                agent=config.agents[0], environment=config.environment, verifier=config.verifier,
            )
            provider_models = []
            expected_provider_url = 'https://hosted-qwen.test/v1/chat/completions'
            expected_provider_model = 'Qwen/Qwen3-8B'
            original_client = httpx.AsyncClient
            def provider(request):
                assert str(request.url) == expected_provider_url
                assert request.headers['authorization'] == 'Bearer host-judge-private'
                payload = json.loads(request.content)
                assert payload['model'] == expected_provider_model
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
            authentication = _judge_authentication(policy, overlay, source={{}})
            relay = ModelRelay(
                config.environment.kwargs['model_endpoint'], config.environment.kwargs['model_name'],
                api_key=authentication[config.environment.kwargs['model_endpoint']],
            )
            async def loopback(request):
                assert str(request.url) == 'http://127.0.0.1:8787/v1/chat/completions'
                assert request.headers['authorization'] == 'Bearer EMPTY'
                assert json.loads(request.content)['model'] == 'org/judge@revision'
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
            replies = asyncio.run(guest())
            relocated_overlay = overlay.model_copy(update={{'verifier_env': {{
                'OPENAI_BASE_URL': 'https://replacement-judge.test/v1',
                'OPENAI_API_KEY': 'host-judge-private', 'MODEL_NAME': 'replacement-alias',
            }}}})
            config = effective_job_config(policy, relocated_overlay)
            relocated_trial = saved_trial.model_copy(update={{
                'agent':config.agents[0], 'environment':config.environment, 'verifier':config.verifier,
            }})
            assert saved_trial.matches_identity(relocated_trial)
            expected_provider_url = 'https://replacement-judge.test/v1/chat/completions'
            expected_provider_model = 'replacement-alias'
            relay = ModelRelay(config.environment.kwargs['model_endpoint'],
                config.environment.kwargs['model_name'], api_key='host-judge-private')
            print(json.dumps({{'replies':replies, 'relocated_replies':asyncio.run(guest()),
                'provider_models':provider_models}}))
            """
        ),
    )
    result = json.loads(completed.stdout)
    assert result["replies"] == {role: role for role in roles}
    assert result["relocated_replies"] == result["replies"]
    assert result["provider_models"] == ["Qwen/Qwen3-8B"] * len(roles) + ["replacement-alias"] * len(roles)
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
        from harbor_config.models.trial.config import TaskConfig, TrialConfig
        from harbor.environments import model_relay
        from harbor.environments.model_relay import ModelRelay
        from shellbox.http_service import HttpRequest
        from marin.evaluation.harbor.trial_driver import effective_job_config, RuntimeOverlay, _judge_authentication
        host_environment = {{'TOGETHER_API_KEY': 'host-provider-private'}}
        os.environ['TOGETHER_API_KEY'] = 'later-ambient-key'
        policy = JobConfig.model_validate_json(Path({str(policy_path)!r}).read_text())
        overlay = RuntimeOverlay.model_validate_json(Path({str(overlay_path)!r}).read_text())
        config = effective_job_config(policy, overlay)
        original_client = httpx.AsyncClient
        def provider(request):
            assert request.url == 'https://api.together.xyz/v1/chat/completions'
            assert request.headers['authorization'] == 'Bearer host-provider-private'
            assert json.loads(request.content)['model'] == 'openai/gpt-oss-120b'
            return httpx.Response(200, json={{'ok':True}})
        model_relay.httpx.AsyncClient = lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs)
        authentication = _judge_authentication(policy, overlay, source=host_environment)
        relay = ModelRelay(
            config.environment.kwargs['model_endpoint'], config.environment.kwargs['model_name'],
            api_key=authentication[config.environment.kwargs['model_endpoint']],
        )
        request = HttpRequest('POST','/v1/chat/completions',b'{{"model":"untrusted-override"}}')
        response = asyncio.run(relay.handle(request))
        assert response.status == 200
        hosted_overlay = overlay.model_copy(update={{
            'judge_model_identity': 'org/judge@revision',
            'verifier_env': {{'OPENAI_BASE_URL': 'https://judge-a.test/v1',
                'OPENAI_API_KEY': 'host-provider-private', 'MODEL_NAME': 'alias-a'}},
        }})
        hosted = effective_job_config(policy, hosted_overlay)
        relocated = effective_job_config(policy, hosted_overlay.model_copy(update={{
            'verifier_env': {{'OPENAI_BASE_URL': 'https://judge-b.test/v1',
                'OPENAI_API_KEY': 'host-provider-private', 'MODEL_NAME': 'alias-b'}},
        }}))
        def trial(job):
            return TrialConfig(task=TaskConfig(name='test/offline', ref='sha256:' + 'a' * 64),
                agent=job.agents[0], environment=job.environment, verifier=job.verifier)
        saved = TrialConfig.model_validate_json(trial(hosted).model_dump_json())
        print(json.dumps({{'verifier':config.verifier.env,
            'agent':config.agents[0].env, 'kwargs':config.agents[0].kwargs,
            'role_resume_same':saved.matches_identity(trial(relocated))}}))
    """
        ),
    )
    result = json.loads(completed.stdout)
    assert result["verifier"]["OPENAI_API_KEY"] == "EMPTY"
    assert result["verifier"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert result["role_resume_same"] is True
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
