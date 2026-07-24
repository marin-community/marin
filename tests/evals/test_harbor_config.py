# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.harbor_config import (
    HarborPolicyDefaults,
    HarborRuntimeBinding,
    resolve_harbor_policy,
)


def _defaults() -> HarborPolicyDefaults:
    return HarborPolicyDefaults(
        dataset="aime",
        version="1.0",
        agent="terminus-2",
        environment="daytona",
        n_concurrent_trials=4,
        max_output_tokens=7168,
        agent_kwargs={"compaction": {"enabled": True}},
        task_limit=2,
    )


def test_harbor_policy_keeps_artifact_fields_until_the_narrow_runtime_binding():
    policy = resolve_harbor_policy(
        _defaults(),
        patch={
            "n_attempts": 3,
            "retry": {"max_retries": 6},
            "environment": {"override_cpus": 8, "override_memory_mb": 16384},
            "agents": [
                {
                    "name": "opencode",
                    "kwargs": {"model_info": {"context_window": 65536, "max_output_tokens": 4096}},
                    "env": {"DAYTONA_EVAL_API_KEY": "${DAYTONA_EVAL_API_KEY}"},
                }
            ],
            "datasets": [{"name": "incorrect-dataset", "version": "0.0", "registry_url": "https://registry.example"}],
        },
    )

    assert policy.document["n_attempts"] == 3
    assert policy.document["retry"]["max_retries"] == 6
    assert policy.document["environment"]["override_cpus"] == 8
    assert policy.document["agents"][0]["name"] == "opencode"
    assert policy.document["agents"][0]["kwargs"]["model_info"]["max_output_tokens"] == 4096
    assert policy.document["datasets"][0]["name"] == "aime"
    assert policy.document["datasets"][0]["version"] == "1.0"
    assert policy.document["datasets"][0]["n_tasks"] == 2
    assert policy.document["datasets"][0]["registry_url"] == "https://registry.example"

    bound = policy.bind(
        HarborRuntimeBinding(
            job_name="harbor_aime_123",
            jobs_dir="/tmp/harbor-results",
            served_model_name="grug",
            api_base="https://capability.example/secret/v1",
        )
    )
    assert bound.document["agents"][0]["name"] == "opencode"
    assert bound.document["agents"][0]["model_name"] == "hosted_vllm/grug"
    assert bound.document["agents"][0]["kwargs"]["api_base"] == "https://capability.example/secret/v1"
    assert bound.document["agents"][0]["kwargs"]["model_info"]["max_output_tokens"] == 4096
    assert bound.persisted_metadata()["job_config"]["agents"][0]["kwargs"]["api_base"] == "<redacted>"


def test_harbor_policy_rejects_multiple_agents_before_runtime_binding():
    with pytest.raises(ValueError, match="exactly one agent"):
        resolve_harbor_policy(
            _defaults(),
            document={"agents": [{"name": "one"}, {"name": "two"}], "datasets": [{"name": "aime"}]},
        ).bind(
            HarborRuntimeBinding(
                job_name="job",
                jobs_dir="/tmp/job",
                served_model_name="model",
                api_base="https://endpoint/v1",
            )
        )


def test_grug_opencode_preset_retains_harbor_policy_and_binds_runtime_endpoint():
    policy = resolve_harbor_policy(
        HarborPolicyDefaults(
            dataset="DCAgent/dev_set_v2",
            version="1.0",
            agent="opencode",
            environment="daytona",
            n_concurrent_trials=256,
            max_output_tokens=16384,
            agent_kwargs={},
            preset="grug-opencode-id",
        )
    )

    assert policy.document["n_attempts"] == 3
    assert policy.document["n_concurrent_trials"] == 256
    assert policy.document["retry"]["max_retries"] == 6
    assert policy.document["verifier"]["max_timeout_sec"] == 14400
    assert policy.document["agents"][0]["kwargs"]["model_info"] == {
        "max_input_tokens": 64512,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }

    bound = policy.bind(
        HarborRuntimeBinding(
            job_name="grug-id-123",
            jobs_dir="/tmp/grug-id-123",
            served_model_name="grug",
            api_base="https://capability.example/secret/v1",
        )
    )
    assert bound.document["agents"][0]["model_name"] == "hosted_vllm/grug"
    assert bound.document["agents"][0]["kwargs"]["api_base"] == "https://capability.example/secret/v1"
