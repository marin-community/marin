# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The isolated Harbor driver's JSON protocol and dataset staging boundary."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from marin.evaluation.harbor.dataset import local_harbor_dataset_path
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from tests.evaluation.harbor_runtime import preflight

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

_ROOT = Path(__file__).parents[2]
_POLICIES = _ROOT / "experiments/evaluation/configs/harbor"


@pytest.fixture(autouse=True)
def daytona_target(monkeypatch):
    monkeypatch.setenv("DAYTONA_TARGET", "test-target")


@pytest.fixture(scope="module")
def checked_policies(tmp_path_factory):
    paths = sorted(_POLICIES.glob("*.yaml"))
    completed = preflight(tmp_path_factory.mktemp("harbor-policies"), [(path, {}) for path in paths])
    return dict(zip((path.name for path in paths), json.loads(completed.stdout), strict=True))


def test_preflight_digest_is_stable_across_hash_seeds(tmp_path, checked_policies):
    path = _POLICIES / "ot-tblite.yaml"
    seeded = [json.loads(preflight(tmp_path, [(path, {})], hash_seed=seed).stdout)[0] for seed in ("1", "8675309")]
    expected = checked_policies[path.name]
    assert all(result["stable_policy_json"] == expected["stable_policy_json"] for result in seeded)
    assert all(result["digest"] == expected["digest"] for result in seeded)


def test_preflight_resolves_hugging_face_datasets_in_harbor(tmp_path):
    (result,) = json.loads(preflight(tmp_path, [(_POLICIES / "swebench-recovery.yaml", {})]).stdout)
    assert result["dataset_kind"] == "hugging_face"
    assert result["dataset_selector"] == "DCAgent2/swebench-verified-random-100-folders"
    assert result["benchmark_metadata"]["task"] == "hf://DCAgent2/swebench-verified-random-100-folders"
    assert result["benchmark_metadata"]["n_benchmark"] == 100


def test_preflight_reports_declared_host_environment_dependencies(tmp_path):
    policy = tmp_path / "external-judge.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "terminus-2"}],
                "datasets": [{"name": "simpleqa", "version": "1.0"}],
                "verifier": {
                    "env": {
                        "OPENAI_API_KEY": "${TOGETHER_API_KEY}",
                        "OPENAI_BASE_URL": "https://api.together.xyz/v1",
                        "MODEL_NAME": "openai/gpt-oss-120b",
                    }
                },
            }
        )
    )
    (result,) = json.loads(preflight(tmp_path, [(policy, {})]).stdout)
    assert result["verifier_env_keys"] == ["TOGETHER_API_KEY"]
    assert json.loads(result["stable_policy_json"])["verifier"]["env"]["OPENAI_API_KEY"] == "${TOGETHER_API_KEY}"


def test_preflight_reports_agent_context_resolved_from_served_model(tmp_path):
    served = {"model_info": {"max_input_tokens": 1048576, "max_output_tokens": 393216}}
    (result,) = json.loads(preflight(tmp_path, [(_POLICIES / "tb2.yaml", served)]).stdout)
    assert (result["max_input_tokens"], result["max_output_tokens"]) == (1048576, 393216)


def test_preflight_rejects_policy_context_above_served_window(tmp_path):
    completed = preflight(
        tmp_path, [(_POLICIES / "ot-tblite.yaml", {"model_info": {"max_input_tokens": 32768}})], check=False
    )
    assert completed.returncode == 2
    assert "64512" in completed.stderr
    assert "32768" in completed.stderr


def test_local_source_is_rebased_onto_worker_workspace(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory(prefix=".harbor-local-", dir=_ROOT) as launch_dir_string:
        launch_dir = Path(launch_dir_string)
        policy_path = launch_dir / "policy.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "environment": {"type": "daytona"},
                    "agents": [{"name": "terminus-2"}],
                    "datasets": [{"path": "tasks"}],
                }
            )
        )
        task_dir = launch_dir / "tasks/task-one"
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "task.toml").write_text('[task]\nname = "marin/task-one"\n[environment]\n')
        (task_dir / "instruction.md").write_text("Solve the task.")
        (task_dir / "tests/test.sh").write_text("exit 0\n")
        (config,) = preflight_harbor_configs([(policy_path, {})], judge_model=None)
        worker_workspace = tmp_path / "worker"
        worker_dataset = worker_workspace / launch_dir.relative_to(_ROOT) / "tasks"
        worker_dataset.mkdir(parents=True)
        monkeypatch.setattr("marin.evaluation.harbor.dataset.find_project_root", lambda: worker_workspace)
        assert local_harbor_dataset_path(config) == worker_dataset


@pytest.mark.parametrize("thinking_format", [None, "unsupported"])
def test_pi_preflight_rejects_invalid_format_before_dataset_access(tmp_path, thinking_format):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "environment": {"type": "daytona"},
                "agents": [{"name": "pi"}],
                "datasets": [{"name": "hf://unavailable/dataset"}],
            }
        )
    )
    kwargs = {} if thinking_format is None else {"thinking_format": thinking_format}
    result = preflight(tmp_path, [(policy, kwargs)], check=False)
    assert result.returncode != 0
    assert "thinking_format" in result.stderr
