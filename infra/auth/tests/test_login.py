# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import ssl
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import google.oauth2.id_token
import jwt
import pytest

from infra.auth import login
from infra.auth.bootstrap import OperatorClient
from infra.auth.prove import _certificate


@pytest.fixture
def google_token(tmp_path, monkeypatch):
    certificate, key = _certificate(tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"trial-key": certificate.read_text()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        monkeypatch.setattr(
            google.oauth2.id_token, "_GOOGLE_OAUTH2_CERTS_URL", f"http://127.0.0.1:{server.server_port}/certs"
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()

        def issue(**overrides):
            now = int(time.time())
            return jwt.encode(
                {
                    "iss": "https://accounts.google.com",
                    "aud": "reviewed-client",
                    "sub": "stable-google-subject",
                    "iat": now,
                    "exp": now + 300,
                    "email": "trial@example.test",
                    "email_verified": True,
                    "hd": "example.test",
                    **overrides,
                },
                key.read_text(),
                algorithm="RS256",
                headers={"kid": "trial-key"},
            )

        try:
            yield issue
        finally:
            server.shutdown()
            thread.join()


def test_google_discovery_verifies_signed_subject_and_workspace(google_token):
    identity = login.verify_google_identity(
        google_token(), audience="reviewed-client", expected_email="trial@example.test", hosted_domain="example.test"
    )
    assert identity == login.GoogleIdentity(
        "stable-google-subject", "reviewed-client", "trial@example.test", "example.test"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://another.example"},
        {"aud": "another-client"},
        {"exp": 1},
        {"email": "another@example.test"},
        {"email_verified": False},
        {"hd": "another.example"},
    ],
)
def test_google_discovery_rejects_wrong_signed_claims(google_token, overrides):
    with pytest.raises(ValueError):
        login.verify_google_identity(
            google_token(**overrides),
            audience="reviewed-client",
            expected_email="trial@example.test",
            hosted_domain="example.test",
        )


@pytest.mark.parametrize("overrides", [{"aud": "attacker-client"}, {"email": "attacker@example.test"}])
def test_enrollment_checks_operator_account_and_audience_independently_of_discovery_record(
    google_token, tmp_path, overrides
):
    login.write_private(tmp_path / "google.jwt", google_token(**overrides))
    login.write_private(tmp_path / "google-identity.json", json.dumps(overrides))
    with pytest.raises(ValueError):
        login.enroll(
            tmp_path,
            OperatorClient("https://localhost:1", "unused-operator", None),
            principal="trial-principal",
            audience="reviewed-client",
            expected_email="trial@example.test",
            hosted_domain="example.test",
        )


def test_login_credentials_replace_atomically_with_private_permissions(tmp_path):
    target = tmp_path / "identity.jwt"
    login.write_private(target, "previous-token")
    login.write_private(target, "fresh-token")
    assert login.read_private(target) == "fresh-token"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("inherited", [True, False])
def test_login_rejects_and_revokes_unexpected_bao_policy(google_token, tmp_path, inherited):
    login.write_private(tmp_path / "google.jwt", google_token())
    login.write_private(
        tmp_path / "enrollment.json",
        json.dumps(
            {
                "subject": "stable-google-subject",
                "audience": "reviewed-client",
                "email": "trial@example.test",
                "hosted_domain": "example.test",
                "entity_id": "enrolled-entity",
                "role": "trial-role",
                "principal": "trial-principal",
            }
        ),
    )
    bao_directory = tmp_path / "bao"
    bao_directory.mkdir()
    certificate, key = _certificate(bao_directory)
    revoked = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/v1/auth/google/login":
                policies = ["marin-auth-identity", "marin-auth-session"]
                result = {
                    "auth": {
                        "client_token": "cleanup-session",
                        "entity_id": "enrolled-entity",
                        "policies": policies if inherited else [*policies, "elevated"],
                        "identity_policies": ["elevated"] if inherited else [],
                    }
                }
            elif self.path == "/v1/auth/token/revoke-self":
                revoked.append(self.headers.get("X-Vault-Token"))
                result = {}
            else:
                self.send_error(404)
                return
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        address = f"https://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with pytest.raises(ValueError, match="authority beyond"):
                login.login(tmp_path, address=address, issuer=address + "/v1/identity/oidc", ca_file=str(certificate))
            assert not (tmp_path / "identity.jwt").exists()
            assert revoked == ["cleanup-session"]
            assert not (tmp_path / "bao.token").exists()
        finally:
            server.shutdown()
            thread.join()
