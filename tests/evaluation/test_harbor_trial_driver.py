# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the pinned Harbor policy boundary."""

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_EXTERNAL_PROJECT = _ROOT / "config/external/harbor"
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


def _external_python(*args: str, hash_seed: str = "0", check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONPATH"] = str(_ROOT / "lib/marin/src")
    return subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(_EXTERNAL_PROJECT),
            "--frozen",
            "python",
            *args,
        ],
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def _preflight(
    tmp_path: Path,
    requests: list[tuple[Path, dict[str, object]]],
    *,
    hash_seed: str = "0",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    request_path = tmp_path / f"requests-{hash_seed}.json"
    request_path.write_text(
        json.dumps(
            [{"path": str(path), "model_agent_kwargs": kwargs} for path, kwargs in requests],
            separators=(",", ":"),
        )
    )
    return _external_python(str(_DRIVER), "preflight", str(request_path), hash_seed=hash_seed, check=check)


def _run_single_turn_aime_agent(
    tmp_path: Path,
    *,
    content: str,
    finish_reason: str | None = None,
    transient_failures: int = 0,
) -> dict[str, object]:
    """Exercise the agent in Harbor's frozen environment with fake HTTP and sandbox I/O boundaries."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    answer_path = tmp_path / "answer.txt"
    choice: dict[str, object] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    response_bytes = json.dumps({"choices": [choice]}).encode()
    script = textwrap.dedent(
        f"""
        import asyncio
        import io
        import json
        import subprocess
        import urllib.error
        from types import SimpleNamespace

        from marin.evaluation.harbor import single_turn_aime_agent as agent_module


        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.close()


        class UrlOpen:
            def __init__(self):
                self.attempts = 0

            def __call__(self, request, timeout):
                self.attempts += 1
                if self.attempts <= {transient_failures}:
                    raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", None, None)
                return Response({response_bytes!r})


        class Environment:
            async def exec(self, command, **kwargs):
                completed = subprocess.run(
                    ["/bin/sh", "-c", command],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return SimpleNamespace(
                    return_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )


        urlopen = UrlOpen()
        agent_module.urllib.request.urlopen = urlopen
        agent = agent_module.SingleTurnAimeAgent(
            logs_dir=agent_module.Path({str(logs_dir)!r}),
            model_name="hosted_vllm/iceball-micro",
            api_base="https://inference.example/v1",
            answer_path={str(answer_path)!r},
            max_tokens=256,
            request_retry_initial=0.001,
        )
        asyncio.run(agent.run("Solve this AIME problem", Environment(), object()))
        print(json.dumps({{
            "answer": agent_module.Path({str(answer_path)!r}).read_text(),
            "response": agent_module.Path({str(logs_dir / "response.txt")!r}).read_text(),
            "attempts": urlopen.attempts,
        }}))
        """
    )
    return json.loads(_external_python("-c", script).stdout)


@pytest.fixture(scope="module")
def checked_policies(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("harbor-policies")
    paths = sorted(_POLICIES.glob("*.yaml"))
    completed = _preflight(tmp_path, [(path, {}) for path in paths])
    return dict(zip((path.name for path in paths), json.loads(completed.stdout), strict=True))


def test_preflight_digest_is_stable_across_hash_seeds(tmp_path, checked_policies):
    path = _POLICIES / "grug-opencode-id.yaml"

    seeded = [json.loads(_preflight(tmp_path, [(path, {})], hash_seed=seed).stdout)[0] for seed in ("1", "8675309")]

    expected = checked_policies[path.name]
    assert all(result["stable_policy_json"] == expected["stable_policy_json"] for result in seeded)
    assert all(result["digest"] == expected["digest"] for result in seeded)


def test_single_turn_aime_agent_grades_length_finished_response(tmp_path):
    content = "A long incomplete derivation mentions 12. The final result is \\boxed{137}."
    result = _run_single_turn_aime_agent(tmp_path, content=content, finish_reason="length")

    assert result == {"answer": "137\n", "response": content, "attempts": 1}


def test_single_turn_aime_agent_retries_transient_proxy_failure(tmp_path):
    result = _run_single_turn_aime_agent(
        tmp_path,
        content="Final answer: 42",
        transient_failures=1,
    )

    assert result == {"answer": "42\n", "response": "Final answer: 42", "attempts": 2}


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

    outcomes = json.loads(_external_python("-c", script, str(policies_path)).stdout)

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
        (launch_dir / "tasks").mkdir()

        (config,) = preflight_harbor_configs([(policy_path, {})])

        worker_workspace = tmp_path / "worker"
        worker_dataset = worker_workspace / launch_dir.relative_to(_ROOT) / "tasks"
        worker_dataset.mkdir(parents=True)
        monkeypatch.setattr(
            "marin.evaluation.harbor.dataset.find_project_root",
            lambda: worker_workspace,
        )

        assert materialize_harbor_dataset(config, tmp_path / "workdir", hf_token=None) == worker_dataset


def test_effective_job_applies_runtime_precedence_and_validates_nested_updates(tmp_path, checked_policies):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(checked_policies["grug-opencode-id.yaml"]["stable_policy_json"])
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "job_name": "runtime-job",
                "jobs_dir": str(tmp_path / "jobs"),
                "dataset_path": str(tmp_path / "tasks"),
                "endpoint_url": "https://iris.example/capability/v1",
                "served_model": "served-grug",
                "task_limit": 3,
                "model_agent_kwargs": {
                    "extra_body": '{"chat_template_kwargs":{"enable_thinking":true}}',
                    "model_info": {"max_input_tokens": 123},
                    "trajectory_config": {"raw_content": True},
                },
            }
        )
    )
    script = (
        "from pathlib import Path; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"config=effective_job_config(Path({str(policy_path)!r}), Path({str(overlay_path)!r})); "
        "print(config.model_dump_json())"
    )

    effective = json.loads(_external_python("-c", script).stdout)

    assert effective["job_name"] == "runtime-job"
    assert effective["jobs_dir"] == str(tmp_path / "jobs")
    assert effective["datasets"][0]["path"] == str(tmp_path / "tasks")
    assert effective["datasets"][0]["n_tasks"] == 3
    agent = effective["agents"][0]
    assert agent["model_name"] == "hosted_vllm/served-grug"
    assert agent["kwargs"]["api_base"] == "https://iris.example/capability/v1"
    assert agent["kwargs"]["extra_body"] == '{"chat_template_kwargs":{"enable_thinking":true}}'
    assert agent["kwargs"]["trajectory_config"] == {"raw_content": False, "linear_history": True}
    assert agent["kwargs"]["model_info"] == {
        "max_input_tokens": 64512,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    assert agent["kwargs"]["opencode_config"]["provider"]["hosted_vllm"]["options"] == {
        "baseURL": "https://iris.example/capability/v1"
    }


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
