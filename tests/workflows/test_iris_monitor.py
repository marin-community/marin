# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

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


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


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


def test_collect_coreweave_redacts_sensitive_env_values(monkeypatch, tmp_path):
    slack_token = "xox" + "b-1234567890-abcdefghijklmnopqrstuvwxyz"
    pod_json = {
        "items": [
            {
                "metadata": {"name": "worker-0"},
                "spec": {
                    "containers": [
                        {
                            "name": "runner",
                            "image": "registry.example/iris-runner:sha",
                            "resources": {"limits": {"nvidia.com/gpu": "8"}},
                            "env": [
                                {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA_TEST_ACCESS"},
                                {"name": "WANDB_API_KEY", "value": "wandb-test-secret"},
                                {
                                    "name": "CACHE_BUSTER",
                                    "value": "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
                                },
                                {
                                    "name": "IRIS_JOB_ENV",
                                    "value": json.dumps(
                                        {
                                            "AWS_SECRET_ACCESS_KEY": "nested-secret-key",
                                            "HF_TOKEN": "nested-hf-token",
                                            "WANDB_API_KEY": "nested-wandb-key",
                                            "LOG_LEVEL": "debug",
                                        }
                                    ),
                                },
                                {"name": "NORMAL_ENV", "value": "normal-env-value"},
                                {
                                    "name": "HF_TOKEN",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "hf-token",
                                            "key": "HF_TOKEN",
                                        }
                                    },
                                },
                            ],
                        }
                    ]
                },
                "status": {"phase": "Pending"},
            }
        ]
    }
    controller_json = {
        "items": [
            {
                "metadata": {"name": "iris-controller"},
                "spec": {
                    "containers": [
                        {
                            "name": "controller",
                            "image": "registry.example/iris-controller:sha",
                            "env": [
                                {"name": "GITHUB_TOKEN", "value": "controller-secret-token"},
                                {"name": "NORMAL_ENV", "value": "controller-context"},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    worker_json = {
        "metadata": {"name": "worker-0"},
        "spec": {
            "containers": [
                {
                    "name": "runner",
                    "image": "registry.example/iris-runner:sha",
                    "resources": {"requests": {"nvidia.com/gpu": "8"}},
                    "env": [
                        {"name": "AWS_SECRET_ACCESS_KEY", "value": "worker-secret-key"},
                        {"name": "AWS_SESSION_TOKEN", "value": "worker-session-token"},
                        {"name": "HF_TOKEN", "value": "hf-literal-token"},
                        {"name": "CACHE_BUSTER", "value": slack_token},
                        {
                            "name": "AWS_ACCESS_KEY_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "r2-creds",
                                    "key": "AWS_ACCESS_KEY_ID",
                                }
                            },
                        },
                    ],
                }
            ]
        },
    }

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess:
        if cmd == [
            "kubectl",
            "-n",
            "iris",
            "get",
            "pods",
            "-l=iris.job_id=canary-job",
            "-o",
            "json",
        ]:
            return _completed(json.dumps(pod_json))
        if cmd == [
            "kubectl",
            "-n",
            "iris",
            "logs",
            "-l",
            "app=iris-controller",
            "--tail=-1",
            "--all-containers",
        ]:
            return _completed("controller log\n")
        if cmd == [
            "kubectl",
            "-n",
            "iris",
            "logs",
            "-l",
            "app=iris-controller",
            "--tail=-1",
            "--all-containers",
            "--previous",
        ]:
            return _completed("previous controller log\n")
        if cmd == ["kubectl", "-n", "iris", "get", "pods", "-l", "app=iris-controller", "-o", "json"]:
            return _completed(json.dumps(controller_json))
        if cmd == ["kubectl", "-n", "iris", "get", "pods", "-l", "managed=true", "-o", "name"]:
            return _completed("pod/worker-0\n")
        if cmd == ["kubectl", "-n", "iris", "logs", "pod/worker-0", "--tail=-1", "--all-containers"]:
            return _completed("worker log\n")
        if cmd == ["kubectl", "-n", "iris", "get", "pod/worker-0", "-o", "json"]:
            return _completed(json.dumps(worker_json))
        if cmd == ["kubectl", "-n", "iris", "get", "events", "--sort-by=.lastTimestamp"]:
            return _completed(
                "LAST SEEN  TYPE     REASON            MESSAGE\n1m         Warning  FailedScheduling  pending\n"
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(iris_monitor, "_run", fake_run)

    files, errors = iris_monitor._collect_coreweave(
        tmp_path,
        "canary-job",
        "iris",
        None,
        managed_label="managed",
        include_cluster_context=False,
        iris_cmd=["iris"],
    )

    assert errors == []
    assert "kubernetes-pods.json" in files
    assert "controller-pods.json" in files
    assert "pod-worker-0.json" in files

    artifact_text = "\n".join(path.read_text() for path in tmp_path.iterdir())
    for secret in [
        "AKIA_TEST_ACCESS",
        "wandb-test-secret",
        "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "nested-secret-key",
        "nested-hf-token",
        "nested-wandb-key",
        "controller-secret-token",
        "worker-secret-key",
        "worker-session-token",
        "hf-literal-token",
        slack_token,
    ]:
        assert secret not in artifact_text

    assert "normal-env-value" in artifact_text
    assert "controller-context" in artifact_text
    assert "debug" in artifact_text
    assert "registry.example/iris-runner:sha" in artifact_text
    assert "nvidia.com/gpu" in artifact_text
    assert "FailedScheduling" in artifact_text
    assert "secretKeyRef" in artifact_text

    pods = json.loads((tmp_path / "kubernetes-pods.json").read_text())
    env = pods["items"][0]["spec"]["containers"][0]["env"]
    env_by_name = {entry["name"]: entry for entry in env}
    assert env_by_name["AWS_ACCESS_KEY_ID"]["value"] == REDACTED_VALUE
    assert env_by_name["WANDB_API_KEY"]["value"] == REDACTED_VALUE
    assert env_by_name["CACHE_BUSTER"]["value"] == REDACTED_VALUE
    assert env_by_name["NORMAL_ENV"]["value"] == "normal-env-value"
    iris_job_env = json.loads(env_by_name["IRIS_JOB_ENV"]["value"])
    assert iris_job_env["AWS_SECRET_ACCESS_KEY"] == REDACTED_VALUE
    assert iris_job_env["HF_TOKEN"] == REDACTED_VALUE
    assert iris_job_env["WANDB_API_KEY"] == REDACTED_VALUE
    assert iris_job_env["LOG_LEVEL"] == "debug"
    assert env_by_name["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "hf-token"
    assert "value" not in env_by_name["HF_TOKEN"]

    controller_pods = json.loads((tmp_path / "controller-pods.json").read_text())
    controller_env = controller_pods["items"][0]["spec"]["containers"][0]["env"]
    controller_env_by_name = {entry["name"]: entry for entry in controller_env}
    assert controller_env_by_name["GITHUB_TOKEN"]["value"] == REDACTED_VALUE
    assert controller_env_by_name["NORMAL_ENV"]["value"] == "controller-context"

    worker_pod = json.loads((tmp_path / "pod-worker-0.json").read_text())
    worker_env = {entry["name"]: entry for entry in worker_pod["spec"]["containers"][0]["env"]}
    assert worker_env["AWS_SECRET_ACCESS_KEY"]["value"] == REDACTED_VALUE
    assert worker_env["AWS_SESSION_TOKEN"]["value"] == REDACTED_VALUE
    assert worker_env["HF_TOKEN"]["value"] == REDACTED_VALUE
    assert worker_env["CACHE_BUSTER"]["value"] == REDACTED_VALUE
    assert worker_env["AWS_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"]["name"] == "r2-creds"
    assert "value" not in worker_env["AWS_ACCESS_KEY_ID"]
