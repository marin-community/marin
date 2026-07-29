# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the pinned Harbor policy boundary."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborEnvironmentConfig,
    HarborRetryConfig,
    HarborRunConfig,
    HarborVerifierConfig,
    legacy_harbor_policy_document,
)

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_EXTERNAL_PROJECT = _ROOT / "config/external/harbor"
_POLICIES = _ROOT / "experiments/evaluation/configs/harbor"
_GRUG_EXCLUDED_EXCEPTIONS = (
    "AgentTimeoutError",
    "AgentEnvironmentTimeoutError",
    "VerifierTimeoutError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
    "SandboxBuildFailedError",
    "VerifierRuntimeError",
    "SummarizationTimeoutError",
    "ContextLengthExceededError",
)


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
    paths = [
        _POLICIES / "aime.yaml",
        _POLICIES / "aime-smoke.yaml",
        _POLICIES / "grug-opencode-id.yaml",
    ]
    completed = _preflight(tmp_path, [(path, {}) for path in paths])
    return dict(zip((path.name for path in paths), json.loads(completed.stdout), strict=True))


def test_preflight_digest_is_stable_across_hash_seeds(tmp_path, checked_policies):
    path = _POLICIES / "grug-opencode-id.yaml"

    seeded = [
        json.loads(_preflight(tmp_path, [(path, {})], hash_seed=seed).stdout)[0]
        for seed in ("1", "8675309")
    ]

    expected = checked_policies[path.name]
    assert all(result["stable_policy_json"] == expected["stable_policy_json"] for result in seeded)
    assert all(result["digest"] == expected["digest"] for result in seeded)


def test_checked_yaml_matches_the_staged_python_policies(tmp_path, checked_policies):
    aime = HarborRunConfig(
        dataset="aime",
        revision="1.0",
        agent=HarborAgentConfig(name="terminus-2"),
        environment=HarborEnvironmentConfig(environment_type="daytona"),
    )
    grug = HarborRunConfig(
        dataset="hf://DCAgent/dev_set_v2",
        revision="377118ff3031c934f5a647ae2c425eb74eef3b21",
        agent=HarborAgentConfig(
            name="opencode",
            max_output_tokens=16384,
            max_timeout=7200,
            setup_timeout=600,
            kwargs={
                "opencode_config": {"compaction": {"auto": False}},
                "model_info": {
                    "max_input_tokens": 64512,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "trajectory_config": {"raw_content": False, "linear_history": True},
            },
        ),
        environment=HarborEnvironmentConfig(
            environment_type="daytona",
            force_build=True,
            delete=True,
            cpus=2,
            memory_mb=8192,
            storage_mb=8192,
            kwargs={"auto_snapshot": True},
        ),
        n_concurrent=256,
        attempts=3,
        timeout_multiplier=2.0,
        retry=HarborRetryConfig(
            max_retries=6,
            exclude_exceptions=_GRUG_EXCLUDED_EXCEPTIONS,
            wait_multiplier=2.0,
            min_wait=1.0,
            max_wait=90.0,
        ),
        verifier=HarborVerifierConfig(max_timeout=14400),
    )
    legacy_paths = []
    for name, policy in (("aime", aime), ("grug-opencode-id", grug)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(legacy_harbor_policy_document(policy)))
        legacy_paths.append(path)

    completed = _preflight(tmp_path, [(path, {}) for path in legacy_paths])
    legacy = dict(zip((path.stem for path in legacy_paths), json.loads(completed.stdout), strict=True))

    assert legacy["aime"]["stable_policy_json"] == checked_policies["aime.yaml"]["stable_policy_json"]
    assert legacy["aime"]["digest"] == checked_policies["aime.yaml"]["digest"]
    assert (
        legacy["grug-opencode-id"]["stable_policy_json"]
        == checked_policies["grug-opencode-id.yaml"]["stable_policy_json"]
    )
    assert legacy["grug-opencode-id"]["digest"] == checked_policies["grug-opencode-id.yaml"]["digest"]


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
    "policy",
    [
        """
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - path: hf://org/repository
""",
        """
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: hf://org/repository/nested
""",
        """
unknown_job_field: true
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: aime
""",
        """
environment:
  type: daytona
agents:
  - name: terminus-2
  - name: opencode
datasets:
  - name: aime
""",
    ],
)
def test_preflight_rejects_invalid_source_policies_without_echoing_inputs(tmp_path, policy):
    path = tmp_path / "invalid.yaml"
    path.write_text(policy)

    completed = _preflight(tmp_path, [(path, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "https://errors.pydantic.dev" not in completed.stderr
    assert "input_value" not in completed.stderr


def test_preflight_rejects_malformed_effective_provider_kwargs(tmp_path):
    path = _POLICIES / "aime.yaml"

    completed = _preflight(tmp_path, [(path, {"model_info": []})], check=False)

    assert completed.returncode == 2
    assert "model_info must be a mapping" in completed.stderr
