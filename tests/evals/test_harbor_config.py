# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.harbor_config import HarborRuntime, resolve_harbor_config


def _runtime() -> HarborRuntime:
    return HarborRuntime(
        job_name="harbor_aime_123",
        jobs_dir="/tmp/harbor-results",
        dataset="aime",
        version="1.0",
        task_limit=2,
        served_model_name="grug",
        api_base="https://capability.example/secret/v1",
    )


def test_harbor_preset_retains_policy_patch_and_reapplies_runtime_owned_fields():
    resolved = resolve_harbor_config(
        preset="standard",
        n_concurrent=4,
        agent="terminus-2",
        environment="daytona",
        default_max_output_tokens=7168,
        agent_kwargs={"compaction": {"enabled": True}},
        runtime=_runtime(),
        patch={
            "n_attempts": 3,
            "retry": {"max_retries": 6},
            "environment": {"override_cpus": 8, "override_memory_mb": 16384},
            "agents": [
                {
                    "name": "opencode",
                    "model_name": "incorrect-model",
                    "kwargs": {"model_info": {"context_window": 65536, "max_output_tokens": 4096}},
                    "env": {"DAYTONA_EVAL_API_KEY": "${DAYTONA_EVAL_API_KEY}"},
                }
            ],
            "datasets": [{"name": "incorrect-dataset", "version": "0.0", "registry_url": "https://registry.example"}],
        },
    )

    document = resolved.document
    assert document["n_attempts"] == 3
    assert document["retry"]["max_retries"] == 6
    assert document["environment"]["override_cpus"] == 8
    assert document["environment"]["override_memory_mb"] == 16384
    assert document["agents"][0]["name"] == "opencode"
    assert document["agents"][0]["model_name"] == "hosted_vllm/grug"
    assert document["agents"][0]["kwargs"]["api_base"] == _runtime().api_base
    assert document["agents"][0]["kwargs"]["model_info"] == {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "context_window": 65536,
    }
    assert document["datasets"][0]["name"] == "aime"
    assert document["datasets"][0]["version"] == "1.0"
    assert document["datasets"][0]["n_tasks"] == 2
    assert document["datasets"][0]["registry_url"] == "https://registry.example"
    metadata = resolved.persisted_metadata()
    assert metadata["config_sha256"] == resolved.sha256
    assert metadata["job_config"]["agents"][0]["kwargs"]["api_base"] == "<redacted>"


def test_harbor_config_rejects_multiple_runtime_agents_or_datasets():
    with pytest.raises(ValueError, match="exactly one agent"):
        resolve_harbor_config(
            preset="standard",
            n_concurrent=4,
            agent="terminus-2",
            environment="daytona",
            default_max_output_tokens=7168,
            agent_kwargs={},
            runtime=_runtime(),
            document={"agents": [{"name": "one"}, {"name": "two"}], "datasets": [{"name": "aime"}]},
        )
    with pytest.raises(ValueError, match="exactly one dataset"):
        resolve_harbor_config(
            preset="standard",
            n_concurrent=4,
            agent="terminus-2",
            environment="daytona",
            default_max_output_tokens=7168,
            agent_kwargs={},
            runtime=_runtime(),
            document={"agents": [{"name": "one"}], "datasets": [{"name": "aime"}, {"name": "math"}]},
        )
