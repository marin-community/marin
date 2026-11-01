# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the vendored Harbor policy boundary."""

import json
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml
from marin.evaluation.harbor.dataset import local_harbor_dataset_path
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from tests.evaluation.harbor_runtime import external_python, write_runtime_overlay
from tests.evaluation.harbor_runtime import preflight as _preflight

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_POLICIES = _ROOT / "experiments/evaluation/configs/harbor"
_INVALID_SOURCE_DOCUMENTS = {
    "hf-uri-in-path": {
        "environment": {"type": "daytona"},
        "agents": [{"name": "terminus-2"}],
        "datasets": [{"path": "hf://org/repository"}],
    },
    "nested-hf-repository": {
        "environment": {"type": "daytona"},
        "agents": [{"name": "terminus-2"}],
        "datasets": [{"name": "hf://org/repository/nested"}],
    },
    "unknown-job-field": {
        "unknown_job_field": True,
        "environment": {"type": "daytona"},
        "agents": [{"name": "terminus-2"}],
        "datasets": [{"name": "aime"}],
    },
    "multiple-agents": {
        "environment": {"type": "daytona"},
        "agents": [{"name": "terminus-2"}, {"name": "opencode"}],
        "datasets": [{"name": "aime"}],
    },
}


def test_selected_runtime_imports_vendored_harbor_and_local_marin_packages():
    script = textwrap.dedent(
        """
        import inspect
        import json
        import finestore
        import harbor_config
        import rigging

        print(json.dumps({
            "harbor_config": inspect.getfile(harbor_config.JobConfig),
            "rigging": rigging.__file__,
            "finestore": finestore.__file__,
        }))
        """
    )

    imported = json.loads(external_python("-c", script).stdout)

    assert Path(imported["harbor_config"]).is_relative_to(_ROOT / "vendor/harbor/src")
    assert Path(imported["rigging"]).is_relative_to(_ROOT / "lib/rigging/src")
    assert Path(imported["finestore"]).is_relative_to(_ROOT / "lib/finestore/src")


def test_external_driver_uses_locked_archive_dependencies():
    external_python(
        "-c",
        textwrap.dedent(
            f"""
            from pathlib import Path

            import finestore
            import rigging
            from harbor import archive
            from marin.evaluation.harbor import trial_driver

            assert archive.EvaluationStore is not None
            assert trial_driver.RuntimeOverlay is not None
            assert Path(finestore.__file__).resolve().is_relative_to(Path({str(_ROOT / "lib/finestore/src")!r}))
            assert Path(rigging.__file__).resolve().is_relative_to(Path({str(_ROOT / "lib/rigging/src")!r}))
            """
        ),
    )


def test_external_driver_retries_spurious_path_style_response():
    result = external_python(
        "-c",
        textwrap.dedent(
            """
            from io import BytesIO
            from unittest.mock import MagicMock, patch

            from botocore.exceptions import ClientError
            from rigging.filesystem.conditional_object import S3ConditionalObject

            client = MagicMock()
            client.get_object.side_effect = [
                ClientError(
                    {"Error": {"Code": "PathStyleRequestNotAllowed", "Message": "use virtual-host addressing"}},
                    "GetObject",
                ),
                {"Body": BytesIO(b"one"), "ETag": '"v1"'},
            ]
            S3ConditionalObject._client = staticmethod(lambda _endpoint: client)
            with patch("rigging.timing.time.sleep", lambda _delay: None):
                result = S3ConditionalObject("s3://bucket/HEAD").read()
            assert result is not None
            assert result.data == b"one"
            """
        ),
    )

    assert result.returncode == 0


@pytest.fixture(scope="module")
def checked_policies(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("harbor-policies")
    paths = sorted(_POLICIES.glob("*.yaml"))
    completed = _preflight(tmp_path, [(path, {}) for path in paths])
    return dict(zip((path.name for path in paths), json.loads(completed.stdout), strict=True))


def test_preflight_digest_is_stable_across_hash_seeds(tmp_path, checked_policies):
    path = _POLICIES / "ot-tblite.yaml"

    seeded = [json.loads(_preflight(tmp_path, [(path, {})], hash_seed=seed).stdout)[0] for seed in ("1", "8675309")]

    expected = checked_policies[path.name]
    assert all(result["stable_policy_json"] == expected["stable_policy_json"] for result in seeded)
    assert all(result["digest"] == expected["digest"] for result in seeded)
    assert expected["trials_per_task"] == 3


def test_preflight_resolves_hugging_face_datasets_in_harbor(tmp_path):
    (result,) = json.loads(_preflight(tmp_path, [(_POLICIES / "swebench-recovery.yaml", {})]).stdout)
    assert result["dataset_kind"] == "hugging_face"
    assert result["dataset_selector"] == "DCAgent2/swebench-verified-random-100-folders"
    assert result["benchmark_metadata"]["task"] == "hf://DCAgent2/swebench-verified-random-100-folders"
    assert result["benchmark_metadata"]["n_benchmark"] == 100


def test_preflight_accepts_pinned_harbor_acp_registry_agent(tmp_path):
    policy_path = tmp_path / "acp.yaml"
    policy_path.write_text(
        """
environment:
  type: daytona
agents:
  - name: acp:pi-acp@0.0.33
datasets:
  - name: aime
"""
    )

    (result,) = json.loads(_preflight(tmp_path, [(policy_path, {})]).stdout)
    assert json.loads(result["stable_policy_json"])["agents"][0]["name"] == "acp:pi-acp@0.0.33"


def test_preflight_reports_only_verifier_host_environment_dependencies(tmp_path):
    policy_path = tmp_path / "external-judge.yaml"
    policy_path.write_text(
        """
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: simpleqa
    version: "1.0"
verifier:
  env:
    OPENAI_API_KEY: "${TOGETHER_API_KEY}"
    OPENAI_BASE_URL: "https://api.together.xyz/v1"
    MODEL_NAME: "openai/gpt-oss-120b"
"""
    )

    payload = json.loads(_preflight(tmp_path, [(policy_path, {})]).stdout)[0]
    stable_policy = json.loads(payload["stable_policy_json"])

    assert payload["verifier_env_keys"] == ["TOGETHER_API_KEY"]
    assert stable_policy["verifier"]["env"] == {
        "MODEL_NAME": "openai/gpt-oss-120b",
        "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
        "OPENAI_BASE_URL": "https://api.together.xyz/v1",
    }
    assert stable_policy["agents"][0]["env"] == {}


def test_preflight_exports_harbor_error_taxonomy(checked_policies):
    taxonomies = [payload["error_taxonomy"] for payload in checked_policies.values()]

    assert all(taxonomy == taxonomies[0] for taxonomy in taxonomies)
    assert "LLMRequestTimeoutError" in taxonomies[0]["infrastructure"]
    assert {"AgentTimeoutError", "ContextLengthExceededError"} <= set(taxonomies[0]["agent"])
    assert "OutputLengthExceededError" in taxonomies[0]["passthrough"]
    assert set(taxonomies[0]["undecided"]) == {
        "TrialNotScoredError",
        "VerificationNotCompletedError",
        "VerifierTimeoutError",
    }


def test_preflight_reports_agent_context_resolved_from_the_served_model(tmp_path):
    served = {"model_info": {"max_input_tokens": 1048576, "max_output_tokens": 393216}}

    (result,) = json.loads(_preflight(tmp_path, [(_POLICIES / "tb2.yaml", served)]).stdout)

    assert result["max_input_tokens"] == 1048576
    assert result["max_output_tokens"] == 393216


def test_preflight_keeps_a_policy_agent_context_below_the_served_window(tmp_path):
    served = {"model_info": {"max_input_tokens": 65536}}

    (result,) = json.loads(_preflight(tmp_path, [(_POLICIES / "ot-tblite.yaml", served)]).stdout)

    assert result["max_input_tokens"] == 64512


def test_preflight_rejects_a_policy_agent_context_above_the_served_window(tmp_path):
    served = {"model_info": {"max_input_tokens": 32768}}

    completed = _preflight(tmp_path, [(_POLICIES / "ot-tblite.yaml", served)], check=False)

    assert completed.returncode == 2
    assert "64512" in completed.stderr
    assert "32768" in completed.stderr


@pytest.mark.parametrize(
    ("setup_parameters", "run_parameters", "callback", "keywords"),
    [
        ("_environment", "instruction, environment, context", "setup", "environment"),
        ("environment", "instruction, environment, _context", "run", "instruction, environment, context"),
        (
            "environment",
            "instruction, environment, context, scratch_dir",
            "run",
            "instruction, environment, context",
        ),
    ],
)
def test_preflight_rejects_agent_callback_keyword_mismatch(
    tmp_path,
    setup_parameters,
    run_parameters,
    callback,
    keywords,
):
    module_path = tmp_path / "keyword_mismatched_agent.py"
    module_path.write_text(
        textwrap.dedent(
            f"""
            class KeywordMismatchedAgent:
                async def setup(self, {setup_parameters}):
                    pass

                async def run(self, {run_parameters}):
                    pass
            """
        )
    )
    document = yaml.safe_load((_POLICIES / "aime-smoke.yaml").read_text())
    document["agents"][0]["import_path"] = "keyword_mismatched_agent:KeywordMismatchedAgent"
    policy_path = tmp_path / "keyword-mismatched-agent.yaml"
    policy_path.write_text(yaml.safe_dump(document))

    completed = _preflight(
        tmp_path,
        [(policy_path, {})],
        check=False,
        extra_python_path=tmp_path,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert f"keyword_mismatched_agent:KeywordMismatchedAgent.{callback}()" in completed.stderr
    assert all(keyword in completed.stderr for keyword in keywords.split(", "))


def test_terminus_policies_retry_transient_endpoint_errors(tmp_path, checked_policies):
    policies = {
        name: payload["stable_policy_json"]
        for name, payload in checked_policies.items()
        if json.loads(payload["stable_policy_json"])["agents"][0]["name"] == "terminus-2"
    }
    assert policies
    policies_path = tmp_path / "policies.json"
    policies_path.write_text(json.dumps(policies))
    script = textwrap.dedent(
        """
        import asyncio
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import harbor.trial.queue as queue_module
        from harbor.models.job.config import JobConfig
        from harbor.trial.queue import TrialQueue
        from harbor.trial.trial import Trial

        async def main():
            policies = json.loads(Path(sys.argv[1]).read_text())
            outcomes = {}
            for name, serialized_policy in policies.items():
                retry = JobConfig.model_validate_json(serialized_policy).retry
                attempts = 0
                waits = []
                failed_result = SimpleNamespace(
                    exception_info=SimpleNamespace(exception_type="InternalServerError")
                )

                class FailedTrial:
                    paths = SimpleNamespace(trial_dir=Path("/tmp/unused-harbor-trial"))

                    async def run(self):
                        nonlocal attempts
                        attempts += 1
                        return failed_result

                    def add_hook(self, _event, _hook):
                        pass

                async def create_trial(_config):
                    return FailedTrial()

                async def record_wait(delay):
                    waits.append(delay)

                Trial.create = staticmethod(create_trial)
                queue_module.asyncio.sleep = record_wait
                queue_module.safe_rmtree = lambda *_args, **_kwargs: None
                result = await TrialQueue(n_concurrent=1, retry_config=retry)._run_trial(
                    SimpleNamespace(trial_name="endpoint-failure")
                )
                assert result is failed_result
                outcomes[name] = {"attempts": attempts, "wait_seconds": sum(waits)}

            print(json.dumps(outcomes, sort_keys=True))

        asyncio.run(main())
        """
    )

    outcomes = json.loads(external_python("-c", script, str(policies_path)).stdout)

    assert outcomes == {name: {"attempts": 11, "wait_seconds": 303.0} for name in policies}


def test_local_source_is_rebased_onto_worker_workspace(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory(prefix=".harbor-local-", dir=_ROOT) as launch_dir_string:
        launch_dir = Path(launch_dir_string)
        policy_path = launch_dir / "policy.yaml"
        policy_path.write_text(
            """
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - path: tasks
"""
        )
        task_dir = launch_dir / "tasks" / "task-one"
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text('version = "1.0"\n[task]\nname = "marin/task-one"\n[environment]\n')
        (task_dir / "instruction.md").write_text("Solve the task.")
        (task_dir / "environment").mkdir()
        (task_dir / "tests").mkdir()
        (task_dir / "tests/test.sh").write_text("exit 0\n")

        (config,) = preflight_harbor_configs([(policy_path, {})], judge_model=None)
        assert config.benchmark.n_benchmark == 1

        worker_workspace = tmp_path / "worker"
        worker_dataset = worker_workspace / launch_dir.relative_to(_ROOT) / "tasks"
        worker_dataset.mkdir(parents=True)
        monkeypatch.setattr(
            "marin.evaluation.harbor.dataset.find_project_root",
            lambda: worker_workspace,
        )

        assert local_harbor_dataset_path(config) == worker_dataset


def test_effective_job_applies_runtime_precedence_and_validates_nested_updates(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(yaml.safe_load((_POLICIES / "ot-tblite.yaml").read_text())))
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        served_model="served-grug",
        task_limit=3,
        model_agent_kwargs={
            "extra_body": '{"chat_template_kwargs":{"enable_thinking":true}}',
            "model_info": {"max_input_tokens": 64512, "max_output_tokens": 16384},
            "trajectory_config": {"raw_content": True},
        },
        verifier_env={
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": "https://judge.example/capability/v1",
            "MODEL_NAME": "judge-model",
        },
        archive_dataset="hf://DCAgent/dev_set_v2",
    )
    script = (
        "import json; "
        "from pathlib import Path; "
        "from pydantic import SecretStr; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"config=effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r})); "
        "payload=config.model_dump(mode='json'); "
        "payload['verifier']['env']={key: (value.get_secret_value() if isinstance(value, SecretStr) else value) "
        "for key, value in config.verifier.env.items()}; "
        "print(json.dumps(payload))"
    )

    effective = json.loads(external_python("-c", script).stdout)

    assert effective["job_name"] == "runtime-job"
    assert effective["jobs_dir"] == str(tmp_path / "jobs")
    assert effective["datasets"][0]["name"] == "hf://DCAgent/dev_set_v2"
    assert effective["datasets"][0]["n_tasks"] == 3
    agent = effective["agents"][0]
    assert agent["model_name"] == "hosted_vllm/org/original-model"
    assert agent["model_alias"] == "hosted_vllm/served-grug"
    assert agent["kwargs"]["api_base"] == "http://127.0.0.1:8788/v1"
    assert agent["kwargs"]["extra_body"] == '{"chat_template_kwargs":{"enable_thinking":true}}'
    assert agent["kwargs"]["trajectory_config"] == {"raw_content": False, "linear_history": True}
    assert agent["kwargs"]["model_info"] == {
        "max_input_tokens": 64512,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert agent["kwargs"]["opencode_config"]["provider"]["hosted_vllm"]["options"] == {
        "baseURL": "http://127.0.0.1:8788/v1",
        "apiKey": "EMPTY",
    }
    assert effective["verifier"]["env"] == {
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
        "MODEL_NAME": "judge-model",
    }


def test_effective_acp_job_routes_model_requests_to_served_endpoint(tmp_path):
    task_dir = tmp_path / "tasks" / "task-one"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('version = "1.0"\n[task]\nname = "test/task-one"\n[environment]\n')
    (task_dir / "instruction.md").write_text("Solve the task.")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [
                    {
                        "name": "acp:pi-acp@0.0.33",
                        "env": {
                            "OPENAI_API_KEY": "test-key",
                            "OPENAI_BASE_URL": "https://stale.example/v1",
                        },
                    }
                ],
                "datasets": [{"path": "tasks"}],
            }
        )
    )

    preflight = json.loads(_preflight(tmp_path, [(policy_path, {})]).stdout)
    assert preflight[0]["agent"] == "acp:pi-acp@0.0.33"

    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        served_model="served-pi",
        archive_dataset="tau3-pi",
    )
    script = (
        "from pathlib import Path; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"config=effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r})); "
        "print(config.model_dump_json())"
    )
    effective = json.loads(external_python("-c", script).stdout)

    assert effective["agents"][0]["name"] == "acp:pi-acp@0.0.33"
    assert effective["agents"][0]["env"] == {
        "OPENAI_API_KEY": "****",
        "OPENAI_BASE_URL": "http://127.0.0.1:8788/v1",
        "OPENAI_API_BASE": "http://127.0.0.1:8788/v1",
    }


def test_packaged_sotopia_agent_resolves_in_vendored_runtime():
    script = (
        "from harbor.agents.installed.sotopia import SotopiaAgent; "
        "from marin.evaluation.harbor.trial_driver import _validate_agent_callbacks; "
        '_validate_agent_callbacks(SotopiaAgent, "harbor.agents.installed.sotopia:SotopiaAgent"); '
        "print(SotopiaAgent.name())"
    )

    completed = external_python("-c", script)

    assert completed.stdout.strip() == "sotopia"


@pytest.mark.parametrize(
    ("agent_name", "policy_kwargs", "expected_policy_version", "expected_runtime_version", "user_agent", "user_version"),
    [
        ("pi", {}, "0.87.0", "0.87.0", None, None),
        ("pi", {"version": "1.3.0"}, "0.87.0", "0.87.0", None, None),
        ("pi", {"version": "0.87.0"}, "0.87.0", "0.87.0", None, None),
        ("terminus-2", {"version": "1.3.0"}, "1.3.0", "1.3.0", None, None),
        ("terminus-2", {}, None, "9.9.9", None, None),
        ("acp:pi-acp@0.0.33", {}, None, "9.9.9", None, None),
        ("pi", {}, "0.87.0", "0.87.0", {"name": "pi", "kwargs": {}}, "0.87.0"),
        ("pi", {"version": "1.3.0"}, "0.87.0", "0.87.0", {"name": "pi", "kwargs": {"version": "1.3.0"}}, "0.87.0"),
        (
            "pi",
            {"version": "1.3.0"},
            "0.87.0",
            "0.87.0",
            {"name": "gemini-cli", "kwargs": {"version": "1.3.0"}},
            "1.3.0",
        ),
        ("terminus-2", {"version": "1.3.0"}, "1.3.0", "1.3.0", {"name": "pi", "kwargs": {"version": "1.3.0"}}, "0.87.0"),
    ],
)
def test_agent_version_survives_preflight_and_runtime_overlay(
    tmp_path, agent_name, policy_kwargs, expected_policy_version, expected_runtime_version, user_agent, user_version
):
    task = tmp_path / "dataset" / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nname = "test/version-pin"\n')
    (task / "instruction.md").write_text("Complete the task.\n")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    if user_agent is not None:
        user_agent = {
            **user_agent,
            "model_name": "openai/user-model",
            "kwargs": {**user_agent["kwargs"], "temperature": 0.4},
            "bridge": {"kind": "acp"},
        }
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": agent_name, "kwargs": policy_kwargs}],
                "datasets": [{"path": "dataset"}],
                "user_agent": user_agent,
            }
        )
    )
    model_kwargs = {
        "version": "9.9.9",
        "thinking_format": "chat-template",
        "model_info": {"max_input_tokens": 65536, "max_output_tokens": 16384},
    }
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"path": str(source), "model_agent_kwargs": model_kwargs, "judge_model": None}]))
    policy = tmp_path / "policy.json"
    overlay = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay,
        tmp_path,
        job_name="version-pin",
        dataset_path=str(task.parent),
        served_model="test-model",
        model_agent_kwargs=model_kwargs,
        archive_dataset="dataset",
    )
    script = textwrap.dedent(
        """
        import io
        import json
        import sys
        from contextlib import redirect_stdout
        from pathlib import Path

        from harbor.trial.simulated_user import validate_user_agent_version_pin
        from marin.evaluation.harbor.trial_driver import effective_job_config, main

        driver, requests, source, policy, overlay = sys.argv[1:]
        sys.argv = [driver, "preflight", requests]
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        (validated,) = json.loads(output.getvalue())
        Path(policy).write_text(validated["stable_policy_json"])
        effective = effective_job_config(Path(policy), Path(overlay))
        raw_effective = effective_job_config(Path(source), Path(overlay))
        # Exercise the same shared-container version check as trial startup.
        for config in (effective, raw_effective):
            if config.user_agent is not None:
                validate_user_agent_version_pin(
                    config.agents[0].name, config.agents[0].kwargs.get("version"),
                    config.user_agent.name, config.user_agent.kwargs.get("version"),
                )
        print(json.dumps({
            "stable": json.loads(validated["stable_policy_json"]),
            "effective": effective.model_dump(mode="json"),
            "raw_effective": raw_effective.model_dump(mode="json"),
        }))
        """
    )
    rendered = external_python("-c", script, str(_DRIVER), str(requests), str(source), str(policy), str(overlay))
    result = json.loads(rendered.stdout)
    stable = result["stable"]
    assert stable["agents"][0]["name"] == agent_name
    assert stable["agents"][0]["kwargs"].get("version") == expected_policy_version
    for config in (result["effective"], result["raw_effective"]):
        agent = config["agents"][0]
        assert agent["name"] == agent_name
        assert agent["kwargs"]["version"] == expected_runtime_version
        assert agent["kwargs"]["thinking_format"] == "chat-template"
        assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384
    for config in (stable, result["effective"], result["raw_effective"]):
        if user_agent is None:
            assert config["user_agent"] is None
        else:
            user = config["user_agent"]
            assert user["name"] == user_agent["name"]
            assert user["kwargs"] == {**user_agent["kwargs"], "version": user_version}
            assert user["model_name"] == "openai/user-model"
            assert user["bridge"]["kind"] == "acp"


@pytest.mark.parametrize(
    ("model_kwargs", "policy_kwargs", "expected_format"),
    [
        ({"thinking_format": "chat-template"}, {}, "chat-template"),
        ({"thinking_format": "qwen-chat-template"}, {}, "qwen-chat-template"),
        ({"thinking_format": "qwen-chat-template"}, {"thinking_format": "chat-template"}, "chat-template"),
    ],
)
def test_hosted_pi_accepts_effective_thinking_format(tmp_path, model_kwargs, policy_kwargs, expected_format):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "pi", "kwargs": policy_kwargs}],
                "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            }
        )
    )
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        served_model="served-qwen",
        model_agent_kwargs={
            **model_kwargs,
            "model_info": {"max_input_tokens": 65536, "max_output_tokens": 32768},
        },
    )
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from harbor.agents.factory import AgentFactory
        from harbor_config.models.agent.name import AgentName
        from marin.evaluation.harbor.trial_driver import effective_job_config

        config = effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r}))
        agent = config.agents[0]
        AgentFactory.create_agent_from_name(
            AgentName.PI,
            logs_dir=Path({str(tmp_path / "logs")!r}),
            model_name=agent.model_name,
            model_alias=agent.model_alias,
            **agent.kwargs,
        )
        print(config.model_dump_json())
        """
    )

    effective = json.loads(external_python("-c", script).stdout)

    assert effective["agents"][0]["kwargs"]["thinking_format"] == expected_format


@pytest.mark.parametrize("thinking_format", [None, "unsupported"])
def test_pi_preflight_rejects_missing_or_invalid_thinking_format_before_dataset_access(tmp_path, thinking_format):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "pi"}],
                "datasets": [{"name": "hf://unavailable/dataset"}],
            }
        )
    )
    kwargs = {} if thinking_format is None else {"thinking_format": thinking_format}

    result = _preflight(tmp_path, [(policy_path, kwargs)], check=False)

    assert result.returncode != 0
    assert "thinking_format" in result.stderr


@pytest.mark.parametrize(("policy_max_tokens", "expected_max_tokens"), [(None, 32768), (16384, 16384)])
def test_effective_terminus_job_applies_output_limit_to_llm_requests(tmp_path, policy_max_tokens, expected_max_tokens):
    agent: dict[str, object] = {"name": "terminus-2"}
    if policy_max_tokens is not None:
        agent["kwargs"] = {"llm_call_kwargs": {"max_tokens": policy_max_tokens}}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [agent],
                "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            }
        )
    )
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        served_model="served-glm",
        model_agent_kwargs={
            "model_info": {"max_input_tokens": 65536, "max_output_tokens": 32768},
        },
    )
    script = (
        "from pathlib import Path; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"config=effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r})); "
        "print(config.model_dump_json())"
    )

    effective = json.loads(external_python("-c", script).stdout)

    assert effective["agents"][0]["kwargs"]["llm_call_kwargs"]["max_tokens"] == expected_max_tokens


def test_effective_aime_job_preserves_capability_url_in_live_config_and_redacts_dump(tmp_path):
    capability_token = "dummy-capability-token"
    capability_url = f"https://iris.example/proxy/t/{capability_token}/serve.inference-test/v1"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(yaml.safe_load((_POLICIES / "aime-smoke.yaml").read_text())))
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path,
        tmp_path,
        dataset_path=str(tmp_path / "tasks"),
        endpoint_url=capability_url,
        served_model="served-qwen",
        task_limit=3,
        archive_dataset="aime",
    )
    script = (
        "import json; "
        "from pathlib import Path; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"config=effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r})); "
        'print(json.dumps({"api_base": config.agents[0].kwargs["api_base"], '
        '"job_dir": str(config.jobs_dir / config.job_name), '
        '"serialized": config.model_dump(mode="json")}))'
    )

    result = json.loads(external_python("-c", script).stdout)

    assert result["api_base"] == capability_url
    assert result["job_dir"] == str(tmp_path / "jobs" / "runtime-job")
    serialized = result["serialized"]
    assert serialized["agents"][0]["kwargs"]["api_base"] == (
        "https://iris.example/proxy/t/<redacted>/serve.inference-test/v1"
    )
    serialized_json = json.dumps(serialized)
    assert capability_token not in serialized_json
    assert "<redacted>" in serialized_json


@pytest.mark.parametrize("agent_name", ["pi", "opencode", "mini-swe-agent", "acp:pi-acp", "sotopia", "terminus-2"])
def test_effective_candidate_endpoint_stays_on_host_for_installed_agents(tmp_path, agent_name):
    endpoint = "https://iris.example/proxy/t/private-capability/serve.candidate/v1"
    policy_path = tmp_path / "policy.json"
    policy_agent = (
        {"import_path": "harbor.agents.installed.sotopia:SotopiaAgent"}
        if agent_name == "sotopia"
        else {"name": agent_name}
    )
    policy_agent["env"] = {"OPENAI_API_KEY": "${OPENAI_API_KEY}", "OPENAI_API_BASE": endpoint}
    if agent_name == "opencode":
        policy_agent["kwargs"] = {
            "opencode_config": {
                "provider": {
                    "hosted_vllm": {
                        "options": {
                            "baseURL": endpoint,
                            "apiKey": "private-provider-key",
                        }
                    }
                }
            }
        }
    policy_path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [policy_agent],
                "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            }
        )
    )
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(
        overlay_path, tmp_path, endpoint_url=endpoint, model_agent_kwargs={"thinking_format": "qwen-chat-template"}
    )
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from marin.evaluation.harbor.trial_driver import effective_job_config
        config = effective_job_config(Path(sys.argv[1]), Path(sys.argv[2]))
        agent = config.agents[0].model_dump(mode='python')
        agent['env'] = dict(config.agents[0].env)
        print(json.dumps({
            'agent': agent,
            'live_endpoint': config.agents[0].kwargs['api_base'],
            'candidate_endpoint': config.environment.kwargs.get('candidate_model_endpoint'),
            'candidate_model': config.environment.kwargs.get('candidate_model_name'),
        }))
    """
    )
    result = json.loads(external_python("-c", script, str(policy_path), str(overlay_path)).stdout)
    agent = result["agent"]
    if agent_name == "terminus-2":
        assert result["live_endpoint"] == endpoint
        assert result["candidate_endpoint"] is None
        return
    assert agent["kwargs"]["api_base"] == "http://127.0.0.1:8788/v1"
    assert agent["kwargs"]["api_key"] == "EMPTY"
    assert agent["env"]["OPENAI_API_KEY"] == "EMPTY"
    assert agent["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8788/v1"
    assert "private-capability" not in json.dumps(agent)
    assert "private-provider-key" not in json.dumps(agent)
    assert result["candidate_endpoint"] == endpoint
    assert result["candidate_model"] == "served-model"
    if agent_name == "opencode":
        options = agent["kwargs"]["opencode_config"]["provider"]["hosted_vllm"]["options"]
        assert options == {"baseURL": "http://127.0.0.1:8788/v1", "apiKey": "EMPTY"}


@pytest.mark.parametrize(
    ("name", "import_path", "selected_class", "expected_endpoint"),
    [
        ("terminus-2", "harbor.agents.installed.pi:Pi", "Terminus2", "https://iris.example/capability/v1"),
        ("pi", "harbor.agents.terminus_2:Terminus2", "Pi", "http://127.0.0.1:8788/v1"),
        ("acp:pi-acp", "harbor.agents.terminus_2:Terminus2", "AcpAgent", "http://127.0.0.1:8788/v1"),
    ],
)
def test_candidate_routing_matches_factory_when_name_and_import_path_are_both_set(
    tmp_path, name, import_path, selected_class, expected_endpoint
):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": name, "import_path": import_path}],
                "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            }
        )
    )
    overlay_path = tmp_path / "overlay.json"
    write_runtime_overlay(overlay_path, tmp_path, model_agent_kwargs={"thinking_format": "qwen-chat-template"})
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from harbor.agents.factory import AgentFactory
        from marin.evaluation.harbor.trial_driver import effective_job_config
        config = effective_job_config(Path(sys.argv[1]), Path(sys.argv[2]))
        agent = AgentFactory.create_agent_from_config(config.agents[0], logs_dir=Path(sys.argv[3]))
        print(json.dumps({'selected_class': type(agent).__name__, 'endpoint': config.agents[0].kwargs['api_base']}))
    """
    )
    result = json.loads(
        external_python("-c", script, str(policy_path), str(overlay_path), str(tmp_path / "logs")).stdout
    )
    assert result == {"selected_class": selected_class, "endpoint": expected_endpoint}


@pytest.mark.parametrize(
    "document",
    _INVALID_SOURCE_DOCUMENTS.values(),
    ids=_INVALID_SOURCE_DOCUMENTS,
)
def test_preflight_rejects_invalid_source_policies(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document))

    completed = _preflight(tmp_path, [(path, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


def test_preflight_rejects_malformed_effective_provider_kwargs(tmp_path):
    path = _POLICIES / "aime-harbor.yaml"

    completed = _preflight(tmp_path, [(path, {"model_info": []})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""
