# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared grant flow against a real TLS, Raft-backed OpenBao.

This proof never contacts CoreWeave object storage. The separately invoked live storage
check is required to establish provider enforcement after WIF registration.
"""

import argparse
import datetime
import ipaddress
import json
import secrets
import socket
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx
import jwt
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from rigging.grant_auth import ACCESS_TOKEN_TYPE, JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.openbao import OpenBaoClient, OpenBaoTransitSigner

from infra.auth.bootstrap import OperatorClient, configure
from infra.auth.proof_fga import configure_proof, isolated_openfga
from infra.auth.proof_http import serve
from infra.auth.prove_loom import isolated_loom, run_loom_proof
from infra.auth.service import ServiceConfig, create_service


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _certificate(directory: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "tls.crt", directory / "tls.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.touch(mode=0o600)
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return certificate_path, key_path


@contextmanager
def _openbao(executable: Path, directory: Path):
    certificate, key = _certificate(directory)
    api_port, cluster_port = _free_port(), _free_port()
    address = f"https://127.0.0.1:{api_port}"
    (directory / "raft").mkdir()
    configuration = {
        "disable_mlock": True,
        "api_addr": address,
        "cluster_addr": f"https://127.0.0.1:{cluster_port}",
        "storage": {"raft": {"path": str(directory / "raft"), "node_id": "proof"}},
        "listener": {
            "tcp": {
                "address": f"127.0.0.1:{api_port}",
                "cluster_address": f"127.0.0.1:{cluster_port}",
                "tls_cert_file": str(certificate),
                "tls_key_file": str(key),
            }
        },
    }
    config_file = directory / "bao.json"
    config_file.write_text(json.dumps(configuration))
    with (directory / "bao.log").open("w") as log:
        process = subprocess.Popen([str(executable), "server", "-config", str(config_file)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("OpenBao exited before initialization")
                try:
                    response = requests.get(address + "/v1/sys/health", verify=str(certificate), timeout=1)
                    if response.status_code == 501:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("OpenBao did not become reachable")
                time.sleep(0.1)
            response = requests.post(
                address + "/v1/sys/init",
                json={"secret_shares": 1, "secret_threshold": 1},
                verify=str(certificate),
                timeout=10,
            )
            response.raise_for_status()
            initialized = response.json()
            response = requests.post(
                address + "/v1/sys/unseal",
                json={"key": initialized["keys"][0]},
                verify=str(certificate),
                timeout=10,
            )
            response.raise_for_status()
            yield OperatorClient(address, initialized["root_token"], str(certificate)), certificate
        finally:
            process.terminate()
            process.wait(timeout=15)


def _identity(operator: OperatorClient, name: str, policy: str, directory: Path) -> tuple[str, Path]:
    password = secrets.token_urlsafe(32)
    operator.request("POST", f"auth/proof/users/{name}", {"password": password, "token_policies": [policy]})
    login = operator.request("POST", f"auth/proof/login/{name}", {"password": password})["auth"]
    token_file = directory / f"{name}.token"
    token_file.touch(mode=0o600)
    token_file.write_text(login["client_token"])
    return login["entity_id"], token_file


def _identity_jwt(operator: OperatorClient, token_file: Path, role: str, ca_file: Path) -> str:
    client = OperatorClient(operator.address, token_file.read_text(), str(ca_file))
    return client.request("GET", f"identity/oidc/token/{role}")["data"]["token"]


def prove(
    executable: Path,
    *,
    openfga_executable: Path,
    loom_executable: Path | None = None,
    loom_source: Path | None = None,
    postgres_bin: Path | None = None,
    finelog_executable: Path | None = None,
) -> dict:
    """Return evidence without exposing credentials or initialization secrets."""
    with tempfile.TemporaryDirectory(prefix="marin-auth-proof-") as scratch:
        directory = Path(scratch)
        with _openbao(executable, directory) as (operator, certificate), ExitStack() as resources:
            fga = resources.enter_context(isolated_openfga(openfga_executable, directory))
            fga_config = configure_proof(fga)
            issuer = operator.address + "/v1/identity/oidc"
            configure(operator, issuer)
            operator.request("POST", "sys/auth/proof", {"type": "userpass"})
            subject, caller_file = _identity(operator, "launcher", "marin-auth-identity", directory)
            other_subject, other_file = _identity(operator, "other-session", "marin-auth-identity", directory)
            storage_subject, storage_file = _identity(operator, "storage", "marin-coreweave-identity", directory)
            broker_token = operator.request(
                "POST",
                "auth/token/create",
                {"policies": ["marin-auth-broker"], "no_default_policy": True, "ttl": "1h", "renewable": False},
            )["auth"]["client_token"]
            broker_file = directory / "broker.token"
            broker_file.touch(mode=0o600)
            broker_file.write_text(broker_token)
            storage_permission = {
                "service": "cw_s3",
                "actions": ["s3:GetObject"],
                "resource": "s3://auth-proof/incidents/",
            }
            permissions = [
                {"service": "loom", "actions": ["session.create"], "resource": "/auth-proof/"},
                {"service": "iris", "actions": ["job.submit"], "resource": "/proof/"},
                storage_permission,
            ]
            auth_port = _free_port()
            auth_url = f"http://127.0.0.1:{auth_port}"
            loom = None
            if loom_executable is not None:
                assert loom_source is not None
                loom = resources.enter_context(
                    isolated_loom(
                        directory / "loom",
                        executable=loom_executable,
                        source=loom_source,
                        verify_storage_binding="diagnostics",
                    )
                )
            config = ServiceConfig.model_validate(
                {
                    "bao_address": operator.address,
                    "ca_bundle": str(certificate),
                    "bao_issuer": issuer,
                    "issuer": auth_url,
                    "broker_token_file": broker_file,
                    "signing_key_version": 1,
                    "database": directory / "grants.db",
                    "subjects": {subject: "launcher", other_subject: "other-session"},
                    "openfga": fga_config,
                    "storage": [
                        {
                            "name": "diagnostics",
                            "role": "marin-coreweave",
                            "subject": storage_subject,
                            "token_file": storage_file,
                            "permissions": [storage_permission],
                        }
                    ],
                    "max_token_ttl": 60,
                    "loom": (
                        {
                            "base_url": loom.base_url,
                            "observer_token_file": loom.observer_token_file,
                        }
                        if loom is not None
                        else None
                    ),
                }
            )
            app, service = create_service(config)
            caller = _identity_jwt(operator, caller_file, "marin-auth", certificate)
            other = _identity_jwt(operator, other_file, "marin-auth", certificate)
            headers = {"Authorization": f"Bearer {caller}"}
            evidence = {
                "openbao": "real TLS + Raft, non-dev",
                "openfga": "real authenticated engine; disposable memory datastore",
                "coreweave_storage": "not contacted",
                "checks": [],
            }
            with serve(app, auth_port), httpx.Client(base_url=auth_url, timeout=30) as client:
                response = client.post(
                    "/v1/grants",
                    headers=headers,
                    json={
                        "actor": "launcher",
                        "permissions": permissions,
                        "expires_at": time.time() + 1800,
                    },
                )
                assert response.status_code == 201, response.text
                grant_id = response.json()["id"]
                assert response.json()["owner"] == "proof-owner"
                revision = client.get("/healthz").json()["policy_revision"]
                assert response.json()["policy_revision"] == revision
                evidence["authorization"] = {"engine": "OpenFGA", "grant_version": 1, "policy_revision": revision}
                evidence["checks"].append("authenticated identity + policy-owned owner + durable bundled grant")

                widened = [{**storage_permission, "actions": ["s3:GetObject", "s3:PutObject"]}]
                assert (
                    client.post(
                        "/v1/grants",
                        headers=headers,
                        json={
                            "actor": "launcher",
                            "permissions": widened,
                            "expires_at": time.time() + 1800,
                        },
                    ).status_code
                    == 403
                )
                assert (
                    client.post(
                        "/v1/grants",
                        headers={"Authorization": f"Bearer {other}"},
                        json={
                            "actor": "other-session",
                            "permissions": permissions,
                            "expires_at": time.time() + 900,
                            "parent_id": grant_id,
                        },
                    ).status_code
                    == 403
                )

                def exchange(identity: str, audience: str, *, token_type: str = JWT_TOKEN_TYPE):
                    return client.post(
                        "/oauth/token",
                        data={
                            "grant_type": TOKEN_EXCHANGE_GRANT,
                            "subject_token_type": token_type,
                            "subject_token": identity,
                            "audience": audience,
                            "grant_id": grant_id,
                        },
                    )

                assert exchange(other, "iris").status_code == 403
                iris = exchange(caller, "iris").json()["access_token"]
                decision = client.post(
                    "/v1/authorize",
                    headers={"Authorization": f"Bearer {iris}"},
                    json={"audience": "iris", "action": "job.submit", "resource": "/proof/job"},
                )
                assert decision.status_code == 200 and decision.json()["grant_id"] == grant_id
                for audience, action, resource in [
                    ("loom", "session.create", "/auth-proof/"),
                    ("iris", "job.submit", "/other/job"),
                    ("iris", "job.submit", "/proof/private/job"),
                ]:
                    assert (
                        client.post(
                            "/v1/authorize",
                            headers={"Authorization": f"Bearer {iris}"},
                            json={"audience": audience, "action": action, "resource": resource},
                        ).status_code
                        == 403
                    )
                evidence["checks"].append(
                    "OpenFGA HTTP decisions: cross-session, audience, namespace and explicit-forbid denials"
                )
                membership = {"user": "group:marin-team#member", "relation": "member", "object": "group:marin-research"}
                fga.write(fga_config["store_id"], fga_config["model_id"], writes=[], deletes=[membership])
                assert (
                    client.post(
                        "/v1/authorize",
                        headers={"Authorization": f"Bearer {iris}"},
                        json={"audience": "iris", "action": "job.submit", "resource": "/proof/job"},
                    ).status_code
                    == 403
                )
                fga.write(fga_config["store_id"], fga_config["model_id"], writes=[membership], deletes=[])
                assert (
                    client.post(
                        "/v1/authorize",
                        headers={"Authorization": f"Bearer {iris}"},
                        json={"audience": "iris", "action": "job.submit", "resource": "/proof/job"},
                    ).status_code
                    == 200
                )
                assert client.get("/healthz").json()["policy_revision"] == revision
                evidence["checks"].append(
                    "nested marin-team -> marin-research membership; group removal denies an existing JWT "
                    "without changing the model revision"
                )
                native_grant = exchange(caller, "cw_s3").json()["access_token"]
                response = client.post(
                    "/v1/storage/diagnostics/token", headers={"Authorization": f"Bearer {native_grant}"}
                )
                assert response.status_code == 200, response.text
                native = jwt.decode(response.json()["token"], options={"verify_signature": False})
                assert native["sub"] == storage_subject and native["aud"] == "https://coreweave.com/iam"
                assert exchange(native_grant, "cw_s3", token_type=ACCESS_TOKEN_TYPE).status_code == 200
                evidence["checks"].append("real Transit signing + real provider OIDC identity + bounded refresh")

                narrower = client.post(
                    "/v1/grants",
                    headers=headers,
                    json={
                        "actor": "launcher",
                        "permissions": [{**storage_permission, "resource": "s3://auth-proof/incidents/narrow/"}],
                        "expires_at": time.time() + 1800,
                    },
                )
                assert narrower.status_code == 403
                evidence["checks"].append("widened grants and broader fixed provider identities rejected")

                operator.request(
                    "POST",
                    "identity/oidc/role/marin-coreweave",
                    {
                        "key": "marin-coreweave",
                        "client_id": "https://coreweave.com/iam",
                        "ttl": "2s",
                    },
                )
                short_response = client.post(
                    "/v1/storage/diagnostics/token", headers={"Authorization": f"Bearer {native_grant}"}
                )
                assert short_response.status_code == 200
                short = jwt.decode(short_response.json()["token"], options={"verify_signature": False})
                time.sleep(max(0, short["exp"] - time.time() + 0.1))
                renewed_response = client.post(
                    "/v1/storage/diagnostics/token", headers={"Authorization": f"Bearer {native_grant}"}
                )
                assert renewed_response.status_code == 200
                renewed = jwt.decode(renewed_response.json()["token"], options={"verify_signature": False})
                assert renewed["iat"] >= short["exp"]
                evidence["checks"].append("native OpenBao OIDC issuance crosses an actual token expiry")
                operator.request("POST", "transit/keys/marin-grants/rotate")
                time.sleep(1.05)
                pinned = exchange(caller, "iris").json()["access_token"]
                assert jwt.get_unverified_header(pinned)["kid"] == jwt.get_unverified_header(iris)["kid"]
                next_signer = OpenBaoTransitSigner(
                    OpenBaoClient(operator.address, broker_file, ca_bundle=str(certificate)),
                    issuer=auth_url,
                    key_version=2,
                )
                claims = jwt.decode(iris, options={"verify_signature": False})
                rotated = next_signer.mint(
                    {key: value for key, value in claims.items() if key not in {"iss", "aud", "iat", "exp"}},
                    audience="iris",
                    ttl_seconds=60,
                )
                assert jwt.get_unverified_header(rotated)["kid"] != jwt.get_unverified_header(iris)["kid"]
                service.authorize(rotated, audience="iris", action="job.submit", resource="/proof/job")
                evidence["checks"].append("Transit signing-key rotation with retained old token verification")
                service.authorize(iris, audience="iris", action="job.submit", resource="/proof/job")
                if loom is not None:
                    evidence["loom"] = run_loom_proof(
                        loom,
                        client=client,
                        grants=service,
                        loom_token=exchange(caller, "loom").json()["access_token"],
                        grant=service.store.active(grant_id),
                        auth_url=auth_url,
                        public_keys=next_signer.public_keys(),
                        template_name="auth-proof",
                    )
                else:
                    service.revoke(grant_id, "launcher")
                if postgres_bin is not None:
                    from infra.auth.prove_services import (  # noqa: PLC0415 - optional application dependencies
                        run_service_proof,
                    )

                    assert finelog_executable is not None and loom is not None
                    evidence["services"] = run_service_proof(
                        directory / "applications",
                        trust_issuer=auth_url,
                        public_keys=next_signer.public_keys(),
                        client=client,
                        caller=caller,
                        fga=fga,
                        fga_config=fga_config,
                        bao_address=operator.address,
                        bao_token_file=caller_file,
                        bao_ca=certificate,
                        postgres_bin=postgres_bin,
                        finelog_executable=finelog_executable,
                        loom=loom,
                    )
                assert exchange(native_grant, "cw_s3", token_type=ACCESS_TOKEN_TYPE).status_code == 403
                assert (
                    client.post(
                        "/v1/storage/diagnostics/token", headers={"Authorization": f"Bearer {native_grant}"}
                    ).status_code
                    == 403
                )
                evidence["checks"].append("completion/revocation stops service refresh and native token issuance")
            return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bao", type=Path, required=True)
    parser.add_argument("--openfga", type=Path, required=True)
    parser.add_argument("--loom", type=Path)
    parser.add_argument("--loom-source", type=Path)
    parser.add_argument("--postgres-bin", type=Path)
    parser.add_argument("--finelog", type=Path)
    args = parser.parse_args()
    if bool(args.loom) != bool(args.loom_source):
        parser.error("--loom and --loom-source must be supplied together")
    if args.postgres_bin and (not args.finelog or not args.loom):
        parser.error("--postgres-bin requires --finelog and --loom")
    print(
        json.dumps(
            prove(
                args.bao.resolve(),
                openfga_executable=args.openfga.resolve(),
                loom_executable=args.loom,
                loom_source=args.loom_source,
                postgres_bin=args.postgres_bin,
                finelog_executable=args.finelog,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
