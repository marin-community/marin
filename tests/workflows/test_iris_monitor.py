# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from rigging.redaction import REDACTED_VALUE

from scripts.workflows import iris_monitor


def _pod(name: str, *, phase: str = "Running", ready: bool = True, deleting: bool = False) -> dict:
    metadata = {"name": name}
    if deleting:
        metadata["deletionTimestamp"] = "2026-05-06T12:00:00Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def _statuses(*pods: dict) -> list[iris_monitor.K8sPodStatus]:
    return iris_monitor._controller_pods_from_json(json.dumps({"items": list(pods)}))


def test_settled_coreweave_controller_requires_exactly_one_ready_pod() -> None:
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new"))) == "iris-controller-new"

    assert iris_monitor._settled_controller_pod_name(_statuses()) is None
    assert (
        iris_monitor._settled_controller_pod_name(
            _statuses(
                _pod("iris-controller-old", deleting=True),
                _pod("iris-controller-new"),
            )
        )
        is None
    )
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", ready=False))) is None
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", phase="Pending"))) is None


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
