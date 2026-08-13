# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Kubernetes secrets, verification, and rollback."""

import base64
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import click
import pytest
from finelog.deploy import _k8s
from finelog.deploy._k8s import (
    K8sRevision,
    K8sRevisionHistory,
    _build_env_secret_manifest,
    k8s_pulumi_rollout,
    k8s_rollback,
    k8s_verify_ingest_ready,
    select_rollback_revision,
)
from finelog.deploy.bootstrap import HEALTH_OK
from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    K8sDeployment,
)
from rigging.secrets import SecretResolutionError


def _s3_config(**k8s_overrides) -> FinelogConfig:
    k8s = {
        "namespace": "iris",
        "object_storage_endpoint": "https://acct.r2.cloudflarestorage.com",
    }
    k8s.update(k8s_overrides)
    return FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="img",
        remote_log_dir="s3://bucket/finelog/cw",
        deployment=Deployment(k8s=K8sDeployment(**k8s)),
    )


def _secret_data(config: FinelogConfig) -> dict[str, str]:
    manifest = _build_env_secret_manifest(config)
    assert manifest is not None
    encoded = json.loads(manifest)["data"]
    return {key: base64.b64decode(value).decode() for key, value in encoded.items()}


def test_env_secret_minted_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")

    assert _secret_data(_s3_config()) == {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SEKRIT",
        "AWS_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
        "AWS_REGION": "auto",
        "AWS_DEFAULT_REGION": "auto",
    }


def test_no_secret_for_non_s3_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")
    config = FinelogConfig(
        name="finelog",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
    )

    assert _build_env_secret_manifest(config) is None


def test_env_secret_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "SEKRIT")

    with pytest.raises(click.ClickException, match="object_storage_endpoint"):
        _build_env_secret_manifest(_s3_config(object_storage_endpoint=None))


def test_env_secret_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_KEY_ID", raising=False)
    monkeypatch.delenv("R2_KEY_SECRET", raising=False)

    with pytest.raises(click.ClickException, match="R2_KEY_ID"):
        _build_env_secret_manifest(_s3_config())


FORWARDING = ForwardingConfig(
    target="https://finelog.oa.dev",
    cluster="cw-rno2a",
    signing_key=("env:TEST_FINELOG_SIGNING_KEY",),
)


def _forwarding_config() -> FinelogConfig:
    return FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="img",
        remote_log_dir="gs://bucket/logs",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
        forwarding=FORWARDING,
    )


def test_forwarding_signing_key_is_written_to_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    key_pem = "-----BEGIN PRIVATE KEY-----\nSEKRIT\n-----END PRIVATE KEY-----"
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", key_pem)

    assert _secret_data(_forwarding_config()) == {"FINELOG_SIGNING_KEY": key_pem}


def test_env_secret_carries_s3_credentials_and_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_KEY_ID", "AKID")
    monkeypatch.setenv("R2_KEY_SECRET", "R2SEKRIT")
    monkeypatch.setenv("TEST_FINELOG_SIGNING_KEY", "PRIVKEY")
    config = replace(_s3_config(), forwarding=FORWARDING)

    data = _secret_data(config)

    assert data["AWS_ACCESS_KEY_ID"] == "AKID"
    assert data["FINELOG_SIGNING_KEY"] == "PRIVKEY"


def test_env_secret_fails_when_signing_key_source_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_FINELOG_SIGNING_KEY", raising=False)

    with pytest.raises(SecretResolutionError):
        _build_env_secret_manifest(_forwarding_config())


def _health_bodies(monkeypatch: pytest.MonkeyPatch, bodies: list[str]) -> list[str]:
    monkeypatch.setattr(_k8s.time, "sleep", lambda _: None)
    remaining = list(bodies)
    observed = []

    def run(argv, **_kwargs):
        body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        observed.append(body)
        return subprocess.CompletedProcess(argv, 0, stdout=body, stderr="")

    monkeypatch.setattr(
        subprocess,
        "run",
        run,
    )
    return observed


def test_ingest_verification_rejects_registration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _health_bodies(monkeypatch, ["degraded: telemetry_v1: registration failed: column type mismatch"])

    with pytest.raises(click.ClickException, match="serving but not ingesting"):
        k8s_verify_ingest_ready(_forwarding_config())


def test_ingest_verification_waits_for_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = ["degraded: telemetry_v1: registration pending", HEALTH_OK]
    observed = _health_bodies(monkeypatch, responses)

    k8s_verify_ingest_ready(_forwarding_config())

    assert observed == responses


def test_ingest_verification_rejects_registration_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _health_bodies(monkeypatch, ["degraded: telemetry_v1: registration pending"])

    with pytest.raises(click.ClickException, match="registration pending"):
        k8s_verify_ingest_ready(_forwarding_config(), max_attempts=3)


def _revision(name: str, revision: int, age: int) -> K8sRevision:
    return K8sRevision(
        replica_set=name,
        revision=revision,
        created_at=datetime(2026, 8, 12, tzinfo=UTC) - timedelta(days=age),
        image=f"image@sha256:{name}",
        source_revision=name,
    )


def test_rollback_selection_walks_back_from_active_replica_set() -> None:
    failed_newer = _revision("failed-newer", 5, 0)
    current = _revision("current", 6, 2)
    previous = _revision("previous", 2, 3)
    history = K8sRevisionHistory(
        deployment_uid="deployment-uid",
        current=current,
        revisions=(failed_newer, current, previous),
    )

    assert select_rollback_revision(history) == previous


class FakeK8sDeploy:
    def __init__(self, *, current: K8sRevision, revisions: tuple[K8sRevision, ...]) -> None:
        self.current = current
        self.revisions = {revision.replica_set: revision for revision in revisions}
        self.pulumi_failure_target: str | None = None
        self.unhealthy: set[str] = set()
        self.refreshed = False
        self.change_on_second_deployment_read: str | None = None
        self.deployment_reads = 0

    def _deployment(self) -> dict:
        return {
            "metadata": {
                "uid": "deployment-uid",
                "annotations": {"deployment.kubernetes.io/revision": str(self.current.revision)},
            }
        }

    def _replica_sets(self) -> dict:
        return {
            "items": [
                {
                    "metadata": {
                        "name": revision.replica_set,
                        "creationTimestamp": revision.created_at.isoformat().replace("+00:00", "Z"),
                        "annotations": {"deployment.kubernetes.io/revision": str(revision.revision)},
                        "ownerReferences": [{"uid": "deployment-uid", "kind": "Deployment", "controller": True}],
                    },
                    "spec": {
                        "template": {
                            "metadata": {"annotations": {"finelog.marin/source-revision": revision.source_revision}},
                            "spec": {"containers": [{"name": "finelog", "image": revision.image}]},
                        }
                    },
                }
                for revision in self.revisions.values()
            ]
        }

    def _activate(self, revision_number: int) -> None:
        target = next(revision for revision in self.revisions.values() if revision.revision == revision_number)
        next_revision = max(revision.revision for revision in self.revisions.values()) + 1
        activated = replace(target, revision=next_revision)
        self.revisions[target.replica_set] = activated
        self.current = activated

    def run(self, argv, **_kwargs):
        if argv[0] == "pulumi":
            if argv[1] == "refresh":
                self.refreshed = True
                return subprocess.CompletedProcess(argv, 0)
            if self.pulumi_failure_target is not None:
                target = self.revisions[self.pulumi_failure_target]
                self._activate(target.revision)
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0)

        command = argv[1:]
        if command[0:2] == ["get", "deployment/finelog-cw"]:
            self.deployment_reads += 1
            if self.deployment_reads == 2 and self.change_on_second_deployment_read is not None:
                target = self.revisions[self.change_on_second_deployment_read]
                self._activate(target.revision)
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self._deployment()), stderr="")
        if command[0:2] == ["get", "replicasets"]:
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self._replica_sets()), stderr="")
        if command[0:3] == ["rollout", "undo", "deployment/finelog-cw"]:
            revision_number = int(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--to-revision=")))
            self._activate(revision_number)
            return subprocess.CompletedProcess(argv, 0)
        if command[0:3] == ["rollout", "status", "deployment/finelog-cw"]:
            expected = int(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--revision=")))
            assert expected == self.current.revision
            return subprocess.CompletedProcess(argv, 0)
        if command[0] == "exec":
            body = "degraded: registration failed" if self.current.replica_set in self.unhealthy else HEALTH_OK
            return subprocess.CompletedProcess(argv, 0, stdout=body, stderr="")
        raise AssertionError(argv)


def test_failed_pulumi_update_restores_captured_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = _revision("finelog-good", 4, 1)
    attempted = _revision("finelog-bad", 5, 0)
    deploy = FakeK8sDeploy(current=previous, revisions=(previous, attempted))
    deploy.pulumi_failure_target = attempted.replica_set
    monkeypatch.setattr(subprocess, "run", deploy.run)

    with pytest.raises(click.ClickException, match="restored Kubernetes revision 4"):
        k8s_pulumi_rollout(_s3_config(), stack="cw", yes=True)

    assert deploy.current.replica_set == previous.replica_set
    assert deploy.refreshed


def test_failed_manual_rollback_restores_source_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _revision("finelog-current", 5, 0)
    target = _revision("finelog-unhealthy", 4, 1)
    deploy = FakeK8sDeploy(current=current, revisions=(current, target))
    deploy.unhealthy.add(target.replica_set)
    monkeypatch.setattr(subprocess, "run", deploy.run)

    with pytest.raises(click.ClickException, match="restored finelog-current"):
        k8s_rollback(_s3_config(), stack="cw", to_revision=target.revision, yes=True)

    assert deploy.current.replica_set == current.replica_set
    assert deploy.refreshed


def test_manual_rollback_does_not_override_concurrent_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _revision("finelog-current", 5, 0)
    target = _revision("finelog-target", 4, 1)
    concurrent = _revision("finelog-concurrent", 6, 0)
    deploy = FakeK8sDeploy(current=current, revisions=(current, target, concurrent))
    deploy.change_on_second_deployment_read = concurrent.replica_set
    monkeypatch.setattr(subprocess, "run", deploy.run)

    with pytest.raises(click.ClickException, match="plan the rollback again"):
        k8s_rollback(_s3_config(), stack="cw", to_revision=target.revision, yes=True)

    assert deploy.current.replica_set == concurrent.replica_set
    assert not deploy.refreshed
