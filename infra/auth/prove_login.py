# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise JWT enrollment against a real Bao plugin and a private OIDC issuer."""

import argparse
import json
import ssl
import tempfile
import threading
import time
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from iris.version import client_revision_date
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT
from rigging.openbao import OpenBaoClient, OpenBaoTransitSigner
from rigging.shared_grants import SharedGrantClient, SharedGrantConfig

from infra.auth.bootstrap import OperatorClient, configure, configure_google
from infra.auth.login import close_session, enroll, read_private, verify_google_identity, write_private
from infra.auth.login import login as login_session
from infra.auth.proof_fga import configure_proof, configure_runtime, isolated_openfga
from infra.auth.proof_http import serve
from infra.auth.proof_iris import iris
from infra.auth.prove import _certificate, _free_port, isolated_openbao
from infra.auth.service import ServiceConfig, create_service

AUDIENCE = "private-desktop-client"
SUBJECT = "private-google-subject"
EMAIL = "trial@example.test"
NESTED_MEMBERSHIP = {"user": "group:marin-team#member", "relation": "member", "object": "group:marin-research"}


@contextmanager
def private_issuer(directory: Path):
    directory.mkdir()
    certificate, key_file = _certificate(directory)
    key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
    jwk = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())), "kid": "trial-key", "alg": "RS256"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/.well-known/openid-configuration":
                data = {
                    "issuer": address,
                    "jwks_uri": address + "/keys",
                    "authorization_endpoint": address + "/authorize",
                    "token_endpoint": address + "/token",
                    "response_types_supported": ["code"],
                    "subject_types_supported": ["public"],
                    "id_token_signing_alg_values_supported": ["RS256"],
                }
            elif self.path == "/keys":
                data = {"keys": [jwk]}
            else:
                self.send_error(404)
                return
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        address = f"https://127.0.0.1:{server.server_port}"
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key_file)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield address, certificate, key
        finally:
            server.shutdown()
            thread.join()


def prove(
    bao: Path,
    openfga: Path,
    *,
    google_directory: Path | None = None,
    expected_email: str | None = None,
    hosted_domain: str | None = None,
) -> dict:
    if google_directory is not None:
        if expected_email is None:
            raise ValueError("The real Google proof requires the independently selected --email")
    with tempfile.TemporaryDirectory(prefix="marin-login-proof-") as scratch:
        directory = Path(scratch)
        if google_directory is not None:
            google_token = read_private(google_directory / "google.jwt")
            verify_google_identity(
                google_token,
                audience=MARIN_DESKTOP_OAUTH_CLIENT.client_id,
                expected_email=expected_email,
                hosted_domain=hosted_domain,
            )
            # Enrollment and Bao sessions belong to this disposable foundation.
            # Preserve the caller's discovery files across failed trials.
            google_directory = directory / "login"
            google_directory.mkdir(mode=0o700)
            write_private(google_directory / "google.jwt", google_token)
        with (
            isolated_openbao(bao, directory) as (operator, ca_file),
            private_issuer(directory / "issuer") as (issuer, issuer_ca, signing_key),
            isolated_openfga(openfga, directory) as fga,
        ):
            bao_issuer = operator.address + "/v1/identity/oidc"
            configure(operator, bao_issuer)
            enrollment_parameters = {
                "subject": SUBJECT,
                "audience": AUDIENCE,
                "email": EMAIL,
                "principal": "launcher",
                "hosted_domain": "example.test",
                "issuer": issuer,
                "issuer_ca_pem": issuer_ca.read_text(),
            }
            if google_directory is None:
                enrollment = configure_google(operator, **enrollment_parameters)
            else:
                enroll(
                    google_directory,
                    operator,
                    principal="launcher",
                    audience=MARIN_DESKTOP_OAUTH_CLIENT.client_id,
                    expected_email=expected_email,
                    hosted_domain=hosted_domain,
                )
                enrollment = json.loads(read_private(google_directory / "enrollment.json"))

            def upstream_token(**overrides):
                now = int(time.time())
                return jwt.encode(
                    {
                        "iss": issuer,
                        "sub": SUBJECT,
                        "aud": AUDIENCE,
                        "iat": now,
                        "exp": now + 300,
                        "email": EMAIL,
                        "email_verified": True,
                        "hd": "example.test",
                        **overrides,
                    },
                    signing_key,
                    algorithm="RS256",
                    headers={"kid": "trial-key"},
                )

            anonymous = OperatorClient(operator.address, "", str(ca_file))

            def login(token: str, role: str = enrollment["role"]) -> dict:
                return anonymous.request("POST", "auth/google/login", {"role": role, "jwt": token})["auth"]

            if google_directory is None:
                upstream = upstream_token()
                authenticated = login(upstream)
            else:
                upstream = read_private(google_directory / "google.jwt")
                login_result = login_session(
                    google_directory, address=operator.address, issuer=bao_issuer, ca_file=str(ca_file)
                )
                authenticated = {
                    **login_result,
                    "client_token": read_private(google_directory / "bao.token"),
                }
            assert authenticated["entity_id"] == enrollment["entity_id"]
            assert set(authenticated["policies"]) == {"marin-auth-identity", "marin-auth-session"}
            denied = []
            invalid_claims = (
                (
                    ("iss", "https://wrong.example"),
                    ("aud", "another-desktop-client"),
                    ("sub", "another-subject"),
                    ("email", "another@example.test"),
                    ("email_verified", False),
                    ("hd", "another.example"),
                    ("exp", int(time.time()) - 120),
                )
                if google_directory is None
                else ()
            )
            for claim, value in invalid_claims:
                try:
                    login(upstream_token(**{claim: value}))
                except urllib.error.HTTPError as error:
                    assert error.code in (400, 403)
                    denied.append(claim)
                else:
                    raise AssertionError(f"Bao accepted an invalid {claim} claim")
            try:
                login(upstream, "")
            except urllib.error.HTTPError as error:
                assert error.code in (400, 403)
                denied.append("missing-role")
            else:
                raise AssertionError("Bao accepted a login without the explicit role")
            if google_directory is None:
                repeated = configure_google(operator, **enrollment_parameters)
            else:
                enroll(
                    google_directory,
                    operator,
                    principal="launcher",
                    audience=MARIN_DESKTOP_OAUTH_CLIENT.client_id,
                    expected_email=expected_email,
                    hosted_domain=hosted_domain,
                )
                repeated = json.loads(read_private(google_directory / "enrollment.json"))
            assert repeated == enrollment
            if google_directory is None:
                assert login(upstream)["entity_id"] == authenticated["entity_id"]
            session = OperatorClient(operator.address, authenticated["client_token"], str(ca_file))
            identity = (
                session.request("GET", "identity/oidc/token/marin-auth")["data"]["token"]
                if google_directory is None
                else read_private(google_directory / "identity.jwt")
            )
            fga_config = configure_proof(fga)
            fga.write(fga_config["store_id"], fga_config["model_id"], writes=[], deletes=[NESTED_MEMBERSHIP])
            broker_file = directory / "broker.token"
            broker_file.touch(mode=0o600)
            broker_file.write_text(
                operator.request(
                    "POST", "auth/token/create", {"policies": ["marin-auth-broker"], "no_default_policy": True}
                )["auth"]["client_token"]
            )
            auth_port = _free_port()
            config = ServiceConfig.model_validate(
                {
                    "bao_address": operator.address,
                    "bao_issuer": bao_issuer,
                    "ca_bundle": str(ca_file),
                    "issuer": f"http://127.0.0.1:{auth_port}",
                    "broker_token_file": broker_file,
                    "signing_key_version": 1,
                    "database": {"backend": "sqlite", "path": directory / "grants.db"},
                    "subjects": {enrollment["entity_id"]: "launcher"},
                    "openfga": fga_config,
                    "runtime_grants": configure_runtime(fga),
                    "storage": [],
                    "max_token_ttl": 60,
                    "loom": None,
                }
            )
            app, service = create_service(config)
            try:
                with serve(app, auth_port) as auth_url, httpx.Client(base_url=auth_url) as client:
                    headers = {"Authorization": f"Bearer {identity}"}
                    request = {
                        "actor": "launcher",
                        "permissions": [
                            {
                                "service": "iris",
                                "actions": ["job.submit", "job.read", "job.cancel"],
                                "resource": "/clusters/isolated-proof/jobs/proof/",
                            }
                        ],
                        "expires_at": time.time() + 240,
                    }
                    assert client.post("/v1/grants", headers=headers, json=request).status_code == 403
                    fga.write(fga_config["store_id"], fga_config["model_id"], writes=[NESTED_MEMBERSHIP], deletes=[])
                    created = client.post("/v1/grants", headers=headers, json=request)
                    assert created.status_code == 201
                    grant_id = created.json()["id"]
                    exchange = client.post(
                        "/oauth/token",
                        data={
                            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                            "subject_token": identity,
                            "grant_id": grant_id,
                            "audience": "iris",
                        },
                    )
                    assert exchange.status_code == 200
                    bearer = {"Authorization": f"Bearer {exchange.json()['access_token']}"}
                    operation = {
                        "audience": "iris",
                        "action": "job.submit",
                        "resource": "/clusters/isolated-proof/jobs/proof/one/",
                    }
                    assert client.post("/v1/authorize", headers=bearer, json=operation).status_code == 200
                    assert (
                        client.post("/v1/authorize", headers=bearer, json={**operation, "audience": "loom"}).status_code
                        == 403
                    )
                    trust = SharedGrantConfig(
                        issuer=auth_url,
                        authority_url=auth_url,
                        public_keys=OpenBaoTransitSigner(
                            OpenBaoClient(operator.address, broker_file, ca_bundle=str(ca_file)),
                            issuer=auth_url,
                            key_version=1,
                        ).public_keys(),
                    )
                    with iris(directory / "iris", trust) as iris_url:
                        job = {
                            "name": "/proof/google-login",
                            "clientRevisionDate": client_revision_date(),
                            "containerProfile": "CONTAINER_PROFILE_SANDBOX",
                            "taskImage": "registry.invalid/auth-proof:synthetic",
                            "resources": {"cpuMillicores": 1000, "memoryBytes": "134217728"},
                            "entrypoint": {"runCommand": {"argv": ["true"]}},
                            "replicas": 1,
                        }
                        token = exchange.json()["access_token"]
                        SharedGrantClient(trust, audience="iris").access(token)
                        iris_headers = {"Authorization": f"Bearer {token}"}
                        launch = client.post(
                            iris_url + "/iris.cluster.ControllerService/LaunchJob", json=job, headers=iris_headers
                        )
                        assert launch.status_code == 200, launch.text
                        status_url = iris_url + "/iris.cluster.ControllerService/GetJobStatus"
                        assert (
                            client.post(status_url, json={"jobId": job["name"]}, headers=iris_headers).status_code == 200
                        )
                        assert client.post(f"/v1/grants/{grant_id}/revoke", headers=headers).status_code == 200
                        assert (
                            client.post(status_url, json={"jobId": job["name"]}, headers=iris_headers).status_code == 403
                        )
                    assert client.post("/v1/authorize", headers=bearer, json=operation).status_code == 403
                    assert (
                        jwt.decode(exchange.json()["access_token"], options={"verify_signature": False})["exp"]
                        > time.time()
                    )
                    if google_directory is None:
                        session.request("POST", "auth/token/revoke-self")
                    else:
                        close_session(
                            google_directory, address=operator.address, ca_file=str(ca_file), auth_url=None, grants=[]
                        )
                    try:
                        session.request("GET", "identity/oidc/token/marin-auth")
                    except urllib.error.HTTPError as error:
                        assert error.code == 403
                    else:
                        raise AssertionError("The revoked Bao session still mints identities")
                    if google_directory is None:
                        assert login(upstream)["entity_id"] == enrollment["entity_id"]
                    else:
                        write_private(google_directory / "google.jwt", upstream)
                        relogin = login_session(
                            google_directory, address=operator.address, issuer=bao_issuer, ca_file=str(ca_file)
                        )
                        assert relogin["principal"] == enrollment["principal"]
                        close_session(
                            google_directory, address=operator.address, ca_file=str(ca_file), auth_url=None, grants=[]
                        )
            finally:
                service.store.close()
            return {
                "identity": (
                    "real Google identity through discover, operator enroll and login"
                    if google_directory is not None
                    else "real Bao JWT plugin, TLS discovery and JWKS, signed synthetic upstream identity"
                ),
                "denied_claims": denied,
                "checks": [
                    "stable entity across enrollment and fresh login",
                    "deny before nested membership, allow after membership",
                    "audience replay denied",
                    "root revocation denies a still-unexpired service JWT",
                    "Iris native admission and status use the authenticated user's grant",
                    "Bao session revocation blocks new identity minting",
                ],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bao", type=Path, required=True)
    parser.add_argument("--openfga", type=Path, required=True)
    parser.add_argument("--google-directory", type=Path)
    parser.add_argument("--email")
    parser.add_argument("--hosted-domain")
    args = parser.parse_args()
    print(
        json.dumps(
            prove(
                args.bao.resolve(),
                args.openfga.resolve(),
                google_directory=args.google_directory,
                expected_email=args.email,
                hosted_domain=args.hosted_domain,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
