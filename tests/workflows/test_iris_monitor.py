# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from rigging.redaction import REDACTED_VALUE

from scripts.workflows import iris_monitor


def test_redact_pod_doc_redacts_env_values_and_preserves_context():
    pod = {
        "metadata": {"name": "worker-0"},
        "spec": {
            "containers": [
                {
                    "name": "runner",
                    "image": "registry.example/iris-runner:sha",
                    "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    "env": [
                        {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA_TEST_ACCESS"},
                        # Low-entropy secret only caught via name-based lift.
                        {"name": "WANDB_API_KEY", "value": "wandb-test-secret"},
                        {
                            "name": "IRIS_JOB_ENV",
                            "value": json.dumps(
                                {
                                    "AWS_SECRET_ACCESS_KEY": "nested-secret-key",
                                    "HF_TOKEN": "nested-hf-token",
                                    "LOG_LEVEL": "debug",
                                }
                            ),
                        },
                        {"name": "NORMAL_ENV", "value": "normal-env-value"},
                        {
                            "name": "HF_TOKEN",
                            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
                        },
                    ],
                }
            ]
        },
    }

    redacted = iris_monitor._redact_pod_doc(pod)
    env_by_name = {entry["name"]: entry for entry in redacted["spec"]["containers"][0]["env"]}

    assert env_by_name["AWS_ACCESS_KEY_ID"]["value"] == REDACTED_VALUE
    assert env_by_name["WANDB_API_KEY"]["value"] == REDACTED_VALUE
    assert env_by_name["NORMAL_ENV"]["value"] == "normal-env-value"

    nested = json.loads(env_by_name["IRIS_JOB_ENV"]["value"])
    assert nested == {
        "AWS_SECRET_ACCESS_KEY": REDACTED_VALUE,
        "HF_TOKEN": REDACTED_VALUE,
        "LOG_LEVEL": "debug",
    }

    # valueFrom entries pass through untouched and never gain a phantom `value`.
    assert "value" not in env_by_name["HF_TOKEN"]
    assert env_by_name["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "hf-token"

    # Non-env pod context stays intact.
    assert redacted["spec"]["containers"][0]["image"] == "registry.example/iris-runner:sha"
    assert redacted["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "8"
