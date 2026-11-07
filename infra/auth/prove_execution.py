# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real Loom-to-Iris CPU execution in the isolated auth staging namespace.

The caller supplies short-lived synthetic grant credentials and a deployed,
digest-pinned non-root image. This module never prints or returns credentials.
It creates real Kubernetes Pods through the ordinary Iris controller/backend.
"""

import argparse
import base64
import json
import shlex
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import jwt
import yaml
from iris.cluster.backends.k8s.tasks import K8sTaskProvider, PodConfig, PodDnsConfig
from iris.cluster.config import AuthConfig, SharedAttemptConfig
from iris.cluster.controller.auth import create_controller_auth
from iris.cluster.controller.backend import BackendDescriptor, BackendKind
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.schema import jobs_table
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName
from iris.managed_thread import thread_container_scope
from iris.version import client_revision_date
from rigging.grant_auth import JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.shared_grants import SHARED_GRANT_TOKEN_SECRET, SharedGrantConfig
from rigging.timing import Duration, ExponentialBackoff
from sqlalchemy import select

from infra.auth.proof_fga import free_port
from infra.auth.proof_http import serve
from infra.auth.prove_loom import IsolatedLoom, isolated_loom, prepare_loom_profile
from infra.auth.service import LoomConfig, ServiceConfig, create_service

TASK_NAMESPACE = "marin-auth-task-staging"
PROOF_MARKER = "AUTH_ATTEMPT_BOUNDED"
PROOF_PYTHON = "/opt/marin-venv/bin/python"
RELEASE_FILE = "/app/auth-proof.release"


@dataclass(frozen=True)
class ExecutionConfig:
    trust: SharedGrantConfig
    controller_identity_file: Path
    task_image: str
    authority_ca_file: Path
    expected_canary_sha256: str
    permission_resource: str = "/secrets/task-canary/"
    local_queue: str = "auth-staging"
    cluster_name: str = "isolated-proof"


@dataclass(frozen=True)
class ExecutionRuntime:
    controller_url: str
    kubernetes: CloudK8sService
    database: ControllerDB


@contextmanager
def execution_controller(directory: Path, config: ExecutionConfig) -> Iterator[ExecutionRuntime]:
    """Start an isolated controller backed by the real Kubernetes API."""
    directory.mkdir(parents=True, exist_ok=False)
    if "@sha256:" not in config.task_image:
        raise ValueError("The execution proof requires a digest-pinned task image")
    auth = create_controller_auth(
        AuthConfig(
            shared_grants=config.trust,
            shared_attempts=SharedAttemptConfig(identity_token_file=config.controller_identity_file),
        ),
        cluster_name=config.cluster_name,
    )
    k8s = CloudK8sService(namespace=TASK_NAMESPACE)
    backend = K8sTaskProvider(
        descriptor=BackendDescriptor(backend_id=DEFAULT_BACKEND_ID, kind=BackendKind.KUBERNETES),
        kubectl=k8s,
        pods=PodConfig(
            namespace=TASK_NAMESPACE,
            default_image=config.task_image,
            local_queue=config.local_queue,
            sandbox_runtime_class="iris-ci-gvisor-cpu",
            dns_config=PodDnsConfig(
                nameservers=("10.16.0.30",),
                searches=(f"{TASK_NAMESPACE}.svc.cluster.local", "svc.cluster.local", "cluster.local"),
                options={"ndots": "5"},
            ),
        ),
        cluster_scan_interval=0.1,
    )
    controller_config = ControllerConfig(
        auth=auth,
        cluster_id=config.cluster_name,
        host="127.0.0.1",
        port=0,
        local_state_dir=directory,
        remote_state_dir=f"file://{directory / 'remote'}",
        poll_interval=Duration.from_seconds(0.2),
    )
    with thread_container_scope():
        log_stack = build_log_stack(
            log_service_address="",
            local_log_dir=directory / "log-server",
            host="127.0.0.1",
            worker_token_provider=auth.worker_token_provider,
        )
        database = ControllerDB(db_dir=directory / "db")
        controller = Controller(config=controller_config, log_stack=log_stack, db=database)
        controller.register_backend(backend)
        try:
            controller.start()
            yield ExecutionRuntime(controller.url, k8s, database)
        finally:
            controller.stop()


def execution_agent_command(
    directory: Path,
    *,
    fixture: Path,
    runtime: ExecutionRuntime,
    config: ExecutionConfig,
    job_name: str,
) -> tuple[str, Path]:
    """Wrap Loom's deterministic ACP fixture with one real Iris submission."""
    if not job_name.startswith("/proof/"):
        raise ValueError("The execution proof may submit only synthetic /proof/ jobs")
    result_file = directory / "execution-launch.json"
    wrapper = directory / "execution-agent.mjs"
    task_script = "\n".join(
        [
            "import os",
            "import hashlib",
            "import json",
            "from pathlib import Path",
            "import requests",
            "from rigging.secrets import require_secret, secret_path",
            "from rigging.timing import Duration, ExponentialBackoff",
            "token = require_secret('MARIN_GRANT_TOKEN')",
            "url = os.environ['MARIN_AUTH_URL'] + '/v1/authorize'",
            "headers = {'Authorization': 'Bearer ' + token}",
            "verify = str(secret_path('AUTH_PROOF_CA'))",
            f"payload = {{'audience':'openbao','action':'secret.read','resource':{config.permission_resource!r}}}",
            "allowed = requests.post(url, headers=headers, json=payload,",
            "    verify=verify, timeout=5, allow_redirects=False)",
            "assert allowed.status_code == 200, 'Bounded task authorization failed'",
            "assert allowed.json()['grant_id'] == os.environ['MARIN_GRANT_ID']",
            "secret = requests.post(os.environ['MARIN_AUTH_URL'] + '/v1/secrets/task-canary',",
            "    headers=headers, verify=verify, timeout=5, allow_redirects=False)",
            "assert secret.status_code == 200, 'The task could not read the actual canary secret'",
            "encoded = json.dumps(secret.json()['data'], sort_keys=True, separators=(',', ':')).encode()",
            "digest = hashlib.sha256(encoded).hexdigest()",
            f"assert digest == {config.expected_canary_sha256!r}, 'The actual secret differs from the expected canary'",
            "payload['action'] = 'secret.write'",
            "denied = requests.post(url, headers=headers, json=payload,",
            "    verify=verify, timeout=5, allow_redirects=False)",
            "assert denied.status_code == 403, 'Task exceeded its action bound'",
            "payload.update(action='secret.read', resource='/secrets/another-binding/')",
            "denied = requests.post(url, headers=headers, json=payload,",
            "    verify=verify, timeout=5, allow_redirects=False)",
            "assert denied.status_code == 403, 'Task exceeded its secret resource bound'",
            f"print({PROOF_MARKER!r}, flush=True)",
            f"ExponentialBackoff().wait_until_or_raise(lambda: Path({RELEASE_FILE!r}).exists(),",
            "    timeout=Duration.from_seconds(40), error_message='Proof observer did not release the task')",
        ]
    )
    payload = {
        "name": job_name,
        "clientRevisionDate": client_revision_date(),
        "containerProfile": "CONTAINER_PROFILE_SANDBOX",
        "taskImage": config.task_image,
        "resources": {"cpuMillicores": 1000, "memoryBytes": "536870912"},
        "entrypoint": {"runCommand": {"argv": [PROOF_PYTHON, "-c", task_script]}},
        "environment": {"secrets": {"AUTH_PROOF_CA": config.authority_ca_file.read_text()}},
        "replicas": 1,
        "maxRetriesFailure": 0,
        "maxRetriesPreemption": 0,
        "timeout": {"milliseconds": 55000},
        "existingJobPolicy": "EXISTING_JOB_POLICY_KEEP",
    }
    launch_url = runtime.controller_url + "/iris.cluster.ControllerService/LaunchJob"
    wrapper.write_text(
        "import {spawn} from 'node:child_process';\n"
        "import {createInterface} from 'node:readline';\n"
        "import {writeFileSync} from 'node:fs';\n"
        f"const child = spawn('node', [{json.dumps(str(fixture))}], {{stdio:['pipe','inherit','inherit']}});\n"
        "child.on('exit', code => process.exit(code ?? 1));\n"
        "let submitted = false;\n"
        "for await (const line of createInterface({input:process.stdin})) {\n"
        "  const message = JSON.parse(line);\n"
        "  if (message.method === 'session/prompt' && !submitted) {\n"
        "    if (!process.env.MARIN_IRIS_TOKEN || !process.env.LOOM_SESSION_ID) "
        "throw new Error('Missing delegated Iris runtime authority');\n"
        f"    const response = await fetch({json.dumps(launch_url)}, {{\n"
        "      method:'POST', redirect:'error',\n"
        "      headers:{Authorization:'Bearer ' + process.env.MARIN_IRIS_TOKEN, 'Content-Type':'application/json'},\n"
        f"      body:JSON.stringify({json.dumps(payload)})\n"
        "    });\n"
        "    if (!response.ok) throw new Error('Iris submit failed: ' + response.status);\n"
        "    const job = await response.json();\n"
        f"    writeFileSync({json.dumps(str(result_file))}, JSON.stringify({{job_id:job.jobId, "
        "session_id:process.env.LOOM_SESSION_ID}), {mode:0o600});\n"
        "    submitted = true;\n"
        "  }\n"
        "  child.stdin.write(line + '\\n');\n"
        "}\n"
        "child.stdin.end();\n"
    )
    return f"node {shlex.quote(str(wrapper))}", result_file


def run_execution_proof(
    runtime: ExecutionRuntime,
    loom: IsolatedLoom,
    *,
    broker_client: httpx.Client,
    config: ExecutionConfig,
    loom_token: str,
    iris_token: str,
    template_name: str,
    result_file: Path,
    outcome: Literal["success", "cancel"],
) -> dict:
    """Observe real execution, retire its authority, and return only public evidence."""
    launched = broker_client.post(
        "/v1/loom/launch",
        headers={"Authorization": f"Bearer {loom_token}"},
        json={"template": template_name, "title": "Synthetic Iris execution proof", "goal": "say:task submitted"},
    )
    launched.raise_for_status()
    run = launched.json()
    try:
        ExponentialBackoff().wait_until_or_raise(
            result_file.exists,
            timeout=Duration.from_seconds(20),
            error_message="The actual Loom ACP runtime did not submit the Iris job",
        )
        submission = json.loads(result_file.read_text())
        if submission["session_id"] != run["session_id"]:
            raise RuntimeError("The Iris submission came from another Loom session")
        job_name = submission["job_id"]
        captured = {}

        def ready() -> bool:
            pods = runtime.kubernetes.list_json(K8sResource.PODS)
            matching = [
                pod
                for pod in pods
                if pod.get("metadata", {}).get("annotations", {}).get("iris.task_id") == job_name + "/0"
            ]
            if not matching:
                return False
            pod = matching[0]
            if pod.get("status", {}).get("phase") != "Running":
                return False
            pod_name = pod["metadata"]["name"]
            if PROOF_MARKER not in runtime.kubernetes.logs(pod_name, container="task", tail=20):
                return False
            secret = runtime.kubernetes.get_json(K8sResource.SECRETS, pod_name + "-secrets")
            if secret is None:
                raise RuntimeError("The task's credential Secret disappeared before observation")
            captured.update(
                pod_name=pod_name,
                attempt_uid=pod["metadata"]["labels"]["iris.attempt_uid"],
                runtime_class=pod["spec"]["runtimeClassName"],
                token=base64.b64decode(secret["data"][SHARED_GRANT_TOKEN_SECRET]).decode(),
            )
            return True

        ExponentialBackoff().wait_until_or_raise(
            ready, timeout=Duration.from_seconds(35), error_message="The real CPU task did not prove its grant bounds"
        )
        payload = {"audience": "openbao", "action": "secret.read", "resource": config.permission_resource}
        headers = {"Authorization": f"Bearer {captured['token']}"}
        active = broker_client.post("/v1/authorize", headers=headers, json=payload)
        if active.status_code != 200:
            raise RuntimeError("The observed running attempt lost its bounded authority")
        grant_id = active.json()["grant_id"]
        attempt_expiry = active.json()["expires_at"]
        claims = jwt.decode(captured["token"], options={"verify_signature": False})
        issued_at = claims["iat"]
        token_expiry = claims["exp"]
        if not time.time() < attempt_expiry <= issued_at + 61:
            raise RuntimeError("The actual attempt grant exceeds its fixed one-minute lifetime")
        with runtime.database.read_snapshot() as snapshot:
            job_grant_id = snapshot.execute(
                select(jobs_table.c.shared_grant_id).where(jobs_table.c.job_id == JobName.from_wire(job_name))
            ).scalar_one()
        if not job_grant_id or job_grant_id == run["grant_id"]:
            raise RuntimeError("The job did not persist its delegated Loom runtime grant")
        with httpx.Client(base_url=runtime.controller_url, headers={"Authorization": f"Bearer {iris_token}"}) as iris:
            if outcome == "cancel":
                iris.post("/iris.cluster.ControllerService/TerminateJob", json={"jobId": job_name}).raise_for_status()
            else:
                released = runtime.kubernetes.exec(
                    captured["pod_name"], ["touch", RELEASE_FILE], container="task", timeout=5
                )
                if released.returncode != 0:
                    raise RuntimeError("The proof observer could not release its CPU task")

            def finished() -> bool:
                response = iris.post("/iris.cluster.ControllerService/GetJobStatus", json={"jobId": job_name})
                response.raise_for_status()
                expected = "JOB_STATE_KILLED" if outcome == "cancel" else "JOB_STATE_SUCCEEDED"
                return response.json().get("job", {}).get("state") == expected

            ExponentialBackoff().wait_until_or_raise(
                finished, timeout=Duration.from_seconds(15), error_message="The actual Iris job did not terminate"
            )
        ExponentialBackoff().wait_until_or_raise(
            lambda: broker_client.post("/v1/authorize", headers=headers, json=payload).status_code == 403,
            timeout=Duration.from_seconds(10),
            error_message="The terminated attempt retained authority",
        )
        revocation_observed_at = time.time()
        if revocation_observed_at >= min(token_expiry, attempt_expiry):
            raise RuntimeError("The terminal denial was observed after credential expiry")
        return {
            "job_id": job_name,
            "session_id": run["session_id"],
            "attempt_uid": captured["attempt_uid"],
            "grant_id": grant_id,
            "job_grant_id": job_grant_id,
            "loom_grant_id": run["grant_id"],
            "attempt_expires_at": attempt_expiry,
            "token_expires_at": token_expiry,
            "revocation_observed_at": revocation_observed_at,
            "namespace": TASK_NAMESPACE,
            "runtime_class": captured["runtime_class"],
            "outcome": outcome,
            "executed_cpu_task": True,
            "outside_action_denied": True,
            "outside_secret_denied": True,
            "actual_canary_read": True,
            "terminal_credential_denied": True,
        }
    finally:
        loom.post("sessions/archive", {"session": run["session_id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, required=True)
    parser.add_argument("--identity-token-file", type=Path, required=True)
    parser.add_argument("--controller-identity-token-file", type=Path, required=True)
    parser.add_argument("--public-key-file", type=Path, action="append", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--authority-ca-file", type=Path, required=True)
    parser.add_argument("--expected-canary-sha256", required=True)
    parser.add_argument("--loom-executable", type=Path, required=True)
    parser.add_argument("--loom-source", type=Path, required=True)
    parser.add_argument("--template-name", default="task-proof")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    service_config = ServiceConfig.model_validate(yaml.safe_load(args.service_config.read_text()))
    config = ExecutionConfig(
        trust=SharedGrantConfig(
            issuer=service_config.issuer,
            public_keys=tuple(path.read_text() for path in args.public_key_file),
            authority_url=service_config.issuer,
            ca_bundle=str(args.authority_ca_file),
        ),
        controller_identity_file=args.controller_identity_token_file,
        task_image=args.task_image,
        authority_ca_file=args.authority_ca_file,
        expected_canary_sha256=args.expected_canary_sha256,
    )
    evidence = []
    invocation_id = uuid.uuid4().hex[:12]
    for outcome in ("success", "cancel"):
        case_directory = args.output_dir / outcome
        case_directory.mkdir()
        with execution_controller(case_directory / "controller", config) as runtime:
            command, result_file = execution_agent_command(
                case_directory,
                fixture=args.loom_source / "crates/loom/tests/fixtures/fake-acp-agent.mjs",
                runtime=runtime,
                config=config,
                job_name=f"/proof/execution-{outcome}-{invocation_id}",
            )
            with isolated_loom(
                case_directory / "loom",
                executable=args.loom_executable,
                source=args.loom_source,
                agent_command=command,
            ) as loom:
                local_config = service_config.model_copy(
                    update={
                        "loom": LoomConfig(
                            base_url=loom.base_url,
                            observer_token_file=loom.observer_token_file,
                            callback_token_file=loom.callback_token_file,
                        ),
                        "iris": None,
                    }
                )
                app, authority = create_service(local_config)
                with serve(app, free_port()) as broker_url:
                    prepare_loom_profile(
                        loom,
                        trust=config.trust,
                        template_name=args.template_name,
                        credentials_url=broker_url + "/v1/loom/credentials",
                    )
                    identity = args.identity_token_file.read_text().strip()
                    principal = authority.identity_verifier.verify(identity)
                    with httpx.Client(base_url=broker_url, timeout=15) as client:
                        grant_response = client.post(
                            "/v1/grants",
                            headers={"Authorization": f"Bearer {identity}"},
                            json={
                                "actor": principal,
                                "permissions": [
                                    {
                                        "service": "loom",
                                        "actions": ["session.create"],
                                        "resource": f"/{args.template_name}/",
                                    },
                                    {
                                        "service": "iris",
                                        "actions": ["job.submit", "job.read", "job.cancel"],
                                        "resource": f"/clusters/{config.cluster_name}/jobs/proof/",
                                    },
                                    {
                                        "service": "openbao",
                                        "actions": ["secret.read"],
                                        "resource": config.permission_resource,
                                    },
                                ],
                                "expires_at": time.time() + 300,
                            },
                        )
                        grant_response.raise_for_status()
                        grant_id = grant_response.json()["id"]
                        tokens = {}
                        for audience in ("loom", "iris"):
                            token_response = client.post(
                                "/oauth/token",
                                data={
                                    "grant_type": TOKEN_EXCHANGE_GRANT,
                                    "subject_token_type": JWT_TOKEN_TYPE,
                                    "subject_token": identity,
                                    "grant_id": grant_id,
                                    "audience": audience,
                                },
                            )
                            token_response.raise_for_status()
                            tokens[audience] = token_response.json()["access_token"]
                        evidence.append(
                            run_execution_proof(
                                runtime,
                                loom,
                                broker_client=client,
                                config=config,
                                loom_token=tokens["loom"],
                                iris_token=tokens["iris"],
                                template_name=args.template_name,
                                result_file=result_file,
                                outcome=outcome,
                            )
                        )
                        client.post(
                            f"/v1/grants/{grant_id}/revoke", headers={"Authorization": f"Bearer {identity}"}
                        ).raise_for_status()
    report = args.output_dir / "execution-proof.json"
    report.write_text(json.dumps({"cases": evidence}, indent=2) + "\n")
    print(json.dumps({"executed_cpu_cases": len(evidence), "evidence_file": str(report)}))


if __name__ == "__main__":
    main()
