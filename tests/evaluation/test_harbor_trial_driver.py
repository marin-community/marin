# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the pinned Harbor policy boundary."""

import json
import os
import subprocess
import tempfile
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
