# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove sustained Loom→Iris native S3 enforcement, then bound revocation observation.

Use a policy binding granting only GetObject and PutObject on the supplied unique
prefix. The operator must create the nonempty denied-key fixture beforehand and
remove probe-owned sentinels afterwards. No provider credential is returned.
"""

import argparse
import json
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack, closing
from dataclasses import asdict
from pathlib import Path

import httpx
import jwt
import yaml
from botocore.exceptions import ClientError
from botocore.session import Session
from iris.cluster.platforms.k8s.types import K8sResource
from rigging.auth import RefreshingTokenProvider
from rigging.filesystem.s3_compat import CoreWeaveFederation
from rigging.grant_auth import ACCESS_TOKEN_TYPE, JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.shared_grants import SharedGrantConfig
from rigging.timing import Duration, ExponentialBackoff

from infra.auth.client import StorageCredentials
from infra.auth.proof_fga import free_port
from infra.auth.proof_http import serve
from infra.auth.proof_identity import controller_identity_file, google_identity, refreshing_identity_file
from infra.auth.prove_execution import PROOF_PYTHON, ExecutionConfig, execution_agent_command, execution_controller
from infra.auth.prove_loom import isolated_loom, prepare_loom_profile
from infra.auth.service import LoomConfig, ServiceConfig, SqliteDatabaseConfig, create_service
from infra.auth.storage_probe import (
    OBSERVER_TOKEN_FILE,
    PROGRESS_MARKER,
    StorageProbeConfig,
    storage_client,
    validate_progress,
)
from infra.auth.verify_storage import CredentialExpirations, native_session, read_fixture


def latest_progress(log: str) -> dict | None:
    lines = [line.split(PROGRESS_MARKER, 1)[1] for line in log.splitlines() if PROGRESS_MARKER in line]
    return json.loads(lines[-1]) if lines else None


def observe_residual(s3, *, bucket: str, key: str, expires_at: float, limit: float = 120) -> dict:
    """Observe one frozen native credential without any post-revocation renewal."""
    started = time.time()
    last_success = None
    while time.time() - started < limit:
        try:
            read_fixture(s3, bucket, key)
            last_success = time.time()
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("AccessDenied", "ExpiredToken", "InvalidAccessKeyId"):
                raise
            return {
                "native_denied_at": time.time(),
                "native_error_code": error.response["Error"]["Code"],
                "native_http_status": error.response["ResponseMetadata"]["HTTPStatusCode"],
                "last_native_success_at": last_success,
                "provider_expires_at": expires_at,
                "residual_access_observed": last_success is not None,
            }
        threading.Event().wait(min(5, max(0, limit - (time.time() - started))))
    return {
        "native_denied_at": None,
        "last_native_success_at": last_success,
        "provider_expires_at": expires_at,
        "residual_access_observed": last_success is not None,
        "observation_limit": limit,
        "limit": "Already-issued provider credential remained usable; expiry is an upper bound, not observed denial.",
    }


def run_storage_execution(runtime, loom, *, client, config, storage, loom_token, iris_token, identity_file, result_file):
    def refresh_iris(previous: str) -> str:
        response = client.post(
            "/oauth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": JWT_TOKEN_TYPE,
                "subject_token": identity_file.read_text().strip(),
                "grant_id": jwt.decode(previous, options={"verify_signature": False})["grant_id"],
                "audience": "iris",
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]

    iris_credentials = RefreshingTokenProvider(iris_token, refresh_iris, refresh_margin=10)
    response = client.post(
        "/v1/loom/launch",
        headers={"Authorization": f"Bearer {loom_token}"},
        json={
            "template": "storage-task-proof",
            "title": "Synthetic sustained storage proof",
            "goal": "say:task submitted",
        },
    )
    response.raise_for_status()
    run = response.json()
    job_id = None
    try:
        ExponentialBackoff().wait_until_or_raise(
            result_file.exists, timeout=Duration.from_seconds(30), error_message="Loom did not submit the storage task"
        )
        submission = json.loads(result_file.read_text())
        if submission["session_id"] != run["session_id"]:
            raise RuntimeError("The storage task belongs to another Loom session")
        job_id = submission["job_id"]
        deadline = time.monotonic() + storage.duration + 90
        pod_name = None
        progress = None
        while time.monotonic() < deadline:
            iris_credentials.get_token()
            pods = runtime.kubernetes.list_json(K8sResource.PODS)
            matching = [
                p for p in pods if p.get("metadata", {}).get("annotations", {}).get("iris.task_id") == job_id + "/0"
            ]
            if matching:
                pod = matching[0]
                pod_name = pod["metadata"]["name"]
                progress = latest_progress(runtime.kubernetes.logs(pod_name, container="task", tail=20))
                if (
                    progress is not None
                    and progress["elapsed"] >= 900
                    and progress["native_rotated"]
                    and progress["observed_at"] > progress["initial_provider_expires_at"]
                    and progress["grant_rotations"] >= 2
                ):
                    validate_progress(progress, storage)
                    break
                if pod.get("status", {}).get("phase") in ("Failed", "Succeeded"):
                    raise RuntimeError("The storage task terminated before sustaining 15 minutes of operations")
            threading.Event().wait(5)
        else:
            raise TimeoutError("The storage task did not complete its sustained proof")
        assert pod_name is not None and progress is not None
        # This copy remains in observer memory and a task-owned temporary file;
        # the log and evidence contain only its public grant identifier.
        captured = runtime.kubernetes.exec(pod_name, ["cat", OBSERVER_TOKEN_FILE], container="task", timeout=5)
        if captured.returncode:
            raise RuntimeError("Could not obtain the task's refreshed credential for the controlled observer")
        token = captured.stdout.strip()
        grant_id = jwt.decode(token, options={"verify_signature": False})["grant_id"]
        with tempfile.TemporaryDirectory(prefix="auth-storage-observer-") as directory:
            token_file = Path(directory) / "native.jwt"
            credentials = StorageCredentials(
                auth_url=config.trust.authority_url,
                grant_token=token,
                binding=storage.binding,
                token_file=token_file,
                ca_bundle=config.authority_ca_file,
            )
            metadata = CredentialExpirations(config.authority_ca_file)
            session = native_session(
                CoreWeaveFederation(
                    credentials.credentials_url,
                    str(token_file),
                ),
                metadata,
            )
            frozen = session.get_credentials().get_frozen_credentials()
            fixed = Session()
            fixed.set_credentials(frozen.access_key, frozen.secret_key, frozen.token)
            with closing(storage_client(fixed, storage)) as native:
                read_fixture(native, storage.bucket, progress["key"])
                assert metadata.latest is not None
                client.post(
                    f"/v1/grants/{grant_id}/revoke",
                    headers={"Authorization": "Bearer " + identity_file.read_text().strip()},
                ).raise_for_status()
                revoked_at = time.time()
                denied = client.post(
                    "/oauth/token",
                    data={
                        "grant_type": TOKEN_EXCHANGE_GRANT,
                        "subject_token_type": ACCESS_TOKEN_TYPE,
                        "subject_token": token,
                        "grant_id": grant_id,
                        "audience": "cw_s3",
                    },
                )
                if denied.status_code != 403:
                    raise RuntimeError("The revoked attempt retained broker refresh authority")
                storage_denied = client.get(
                    f"/v1/storage/{storage.binding}/credentials", headers={"Authorization": f"Bearer {token}"}
                )
                if storage_denied.status_code != 403:
                    raise RuntimeError("The revoked attempt retained native credential issuance authority")
                residual = observe_residual(
                    native, bucket=storage.bucket, key=progress["key"], expires_at=metadata.latest.timestamp()
                )
        return {
            "job_id": job_id,
            "session_id": run["session_id"],
            "attempt_grant_id": grant_id,
            "revoked_at": revoked_at,
            "broker_refresh_denied": True,
            "native_issuance_denied": True,
            "task_progress": progress,
            "observer_provenance": (
                "Frozen native credentials minted from the running attempt immediately before revocation; "
                "no refresh afterwards."
            ),
            "residual": residual,
        }
    finally:
        try:
            if job_id is not None:
                with httpx.Client(
                    base_url=runtime.controller_url,
                    headers={"Authorization": f"Bearer {iris_credentials.get_token()}"},
                ) as iris:
                    iris.post("/iris.cluster.ControllerService/TerminateJob", json={"jobId": job_id}).raise_for_status()
        finally:
            loom.post("sessions/archive", {"session": run["session_id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "service-config",
        "identity-token-file",
        "controller-identity-token-file",
        "authority-ca-file",
        "loom-executable",
        "loom-source",
        "output-dir",
        "storage-config",
        "controller-openbao-token-file",
        "google-login-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--public-key-file", type=Path, action="append", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--bao-address", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    service = ServiceConfig.model_validate(yaml.safe_load(args.service_config.read_text()))
    storage = StorageProbeConfig(**json.loads(args.storage_config.read_text()))
    config = ExecutionConfig(
        trust=SharedGrantConfig(
            issuer=service.issuer,
            public_keys=tuple(p.read_text() for p in args.public_key_file),
            authority_url=service.issuer,
            ca_bundle=str(args.authority_ca_file),
        ),
        controller_identity_file=args.controller_identity_token_file,
        task_image=args.task_image,
        authority_ca_file=args.authority_ca_file,
        expected_canary_sha256="unused",
    )
    with ExitStack() as stack:
        stack.enter_context(
            controller_identity_file(
                args.controller_identity_token_file,
                address=args.bao_address,
                token_file=args.controller_openbao_token_file,
                ca_file=str(args.authority_ca_file),
            )
        )
        stack.enter_context(
            refreshing_identity_file(
                args.identity_token_file,
                lambda: google_identity(
                    args.google_login_directory,
                    address=args.bao_address,
                    issuer=service.bao_issuer,
                    ca_file=str(args.authority_ca_file),
                ),
            )
        )
        runtime = stack.enter_context(execution_controller(args.output_dir / "controller", config))
        command, result_file = execution_agent_command(
            args.output_dir,
            fixture=args.loom_source / "crates/loom/tests/fixtures/fake-acp-agent.mjs",
            runtime=runtime,
            config=config,
            job_name="/proof/storage-" + uuid.uuid4().hex[:12],
            task_argv=[PROOF_PYTHON, "-m", "infra.auth.storage_probe", "--config-json", json.dumps(asdict(storage))],
            task_timeout=int((storage.duration + 180) * 1000),
        )
        with isolated_loom(
            args.output_dir / "loom", executable=args.loom_executable, source=args.loom_source, agent_command=command
        ) as loom:
            local = service.model_copy(
                update={
                    "database": SqliteDatabaseConfig(backend="sqlite", path=args.output_dir / "loom.sqlite"),
                    "loom": LoomConfig(
                        base_url=loom.base_url,
                        observer_token_file=loom.observer_token_file,
                        callback_token_file=loom.callback_token_file,
                    ),
                    "iris": None,
                }
            )
            app, authority = create_service(local)
            with serve(app, free_port()) as broker_url:
                prepare_loom_profile(
                    loom,
                    trust=config.trust,
                    template_name="storage-task-proof",
                    credentials_url=broker_url + "/v1/loom/credentials",
                )
                identity = args.identity_token_file.read_text().strip()
                principal = authority.identity_verifier.verify(identity)
                with httpx.Client(base_url=broker_url, timeout=15) as client:
                    grant = client.post(
                        "/v1/grants",
                        headers={"Authorization": f"Bearer {identity}"},
                        json={
                            "actor": principal,
                            "permissions": [
                                {"service": "loom", "actions": ["session.create"], "resource": "/storage-task-proof/"},
                                {
                                    "service": "iris",
                                    "actions": ["job.submit", "job.read", "job.cancel"],
                                    "resource": f"/clusters/{config.cluster_name}/jobs/proof/",
                                },
                                {
                                    "service": "cw_s3",
                                    "actions": ["s3:GetObject", "s3:PutObject"],
                                    "resource": f"s3://{storage.bucket}/{storage.prefix}",
                                },
                            ],
                            "expires_at": time.time() + 1800,
                        },
                    )
                    grant.raise_for_status()
                    grant_id = grant.json()["id"]
                    try:
                        tokens = {}
                        for audience in ("loom", "iris"):
                            result = client.post(
                                "/oauth/token",
                                data={
                                    "grant_type": TOKEN_EXCHANGE_GRANT,
                                    "subject_token_type": JWT_TOKEN_TYPE,
                                    "subject_token": identity,
                                    "grant_id": grant_id,
                                    "audience": audience,
                                },
                            )
                            result.raise_for_status()
                            tokens[audience] = result.json()["access_token"]
                        evidence = run_storage_execution(
                            runtime,
                            loom,
                            client=client,
                            config=config,
                            storage=storage,
                            loom_token=tokens["loom"],
                            iris_token=tokens["iris"],
                            identity_file=args.identity_token_file,
                            result_file=result_file,
                        )
                        report = args.output_dir / "storage-execution-proof.json"
                        report.write_text(json.dumps(evidence, indent=2) + "\n")
                        print(json.dumps({"evidence_file": str(report)}))
                    finally:
                        client.post(
                            f"/v1/grants/{grant_id}/revoke",
                            headers={"Authorization": "Bearer " + args.identity_token_file.read_text().strip()},
                        ).raise_for_status()


if __name__ == "__main__":
    main()
