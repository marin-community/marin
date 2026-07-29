# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the pinned Harbor policy boundary."""

import json
import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_EXTERNAL_PROJECT = _ROOT / "config/external/harbor"
_POLICIES = _ROOT / "experiments/evaluation/configs/harbor"


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


def test_catalog_policy_digests_preserve_reviewed_job_identity(checked_policies):
    assert {name: policy["digest"] for name, policy in checked_policies.items()} == {
        "aider.yaml": "sha256:5cb46786e2f1eacf43fb3b5eac849b5ad6a1cf29a956b47177de682fce1c2ab0",
        "aime-harbor.yaml": "sha256:d6061d09ff3994e8baf5bed293ff826caad47702eafef26835d9ba786460302c",
        "aime-smoke.yaml": "sha256:2554c05ee716c4a4be7c26bd2ae445f237d60bbea8f6fd2c084a6ab67ff53f9f",
        "bfcl.yaml": "sha256:47ce3c5ec0afd851ecced3b99643aabbe84f98a6bf4337ebf75b487a576c4712",
        "financeagent.yaml": "sha256:347afaa2ca78088f020db6ceacbf13e32ba16f00e647d24a6084269bad754e03",
        "gaia.yaml": "sha256:77e4522db26d408ac488e55d0ba579f822e4d794c135fd545dc92f3de656364a",
        "grug-opencode-id.yaml": "sha256:e2bcc00724d70dad05eca5c78ccb612bae18530d34c3816660dea33f9eab3873",
        "medagentbench.yaml": "sha256:08166bdab2e3ccd26012b202db7a7b8223eb09d6c97c217c363910ad52f3d45b",
        "swebench-full.yaml": "sha256:e78b976fd6fa8ee925d6a89b0d1d357483c7a8f99b1f47688ba5b4f597ecac11",
        "swebench-lite.yaml": "sha256:6638a989caf7c101f70b16f542de94375b778496d0478a9e4f51ad21c6c54d1d",
        "swebench.yaml": "sha256:834a966205d8a95339f4ee02c8173f12fe1071985165ccaa211d3be034acf425",
        "tb2-lite.yaml": "sha256:995430d2abbabe39c35c7cce7a3ddf27cd2a32de1a02b19e9c0f6e9b883ece65",
        "tb2.yaml": "sha256:4ae314abd378080983e87ab391473620df031539918cbb1f274d92b84f4640a9",
    }


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
    path = _POLICIES / "aime-harbor.yaml"

    completed = _preflight(tmp_path, [(path, {"model_info": []})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "https://errors.pydantic.dev" not in completed.stderr
    assert "input_value" not in completed.stderr
