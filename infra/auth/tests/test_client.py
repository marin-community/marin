# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import jwt
import pytest
import requests

from infra.auth.client import StorageCredentials, refreshing_storage_credentials, storage_environment


def test_grant_renewal_rotates_private_broker_authorization_file(tmp_path):
    clock = [1000]
    renewed = threading.Event()
    deny = [False]
    signing_key = "synthetic-auth-service-signing-key"

    def token(audience, lifetime):
        return jwt.encode(
            {
                "iss": "test",
                "sub": "workload",
                "aud": audience,
                "iat": clock[0],
                "exp": clock[0] + lifetime,
                "grant_id": "one",
            },
            signing_key,
            algorithm="HS256",
        )

    def authenticated(value):
        claims = jwt.decode(
            value,
            signing_key,
            algorithms=["HS256"],
            audience="cw_s3",
            options={"verify_exp": False, "verify_iat": False},
        )
        return claims["exp"] > clock[0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/oauth/token"
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            credential = form["subject_token"][0]
            result = {"access_token": token("cw_s3", 300)}
            status = 200 if authenticated(credential) and not deny[0] else 401
            body = json.dumps(result if status == 200 else {"error": "expired_subject"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            renewed.set()

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            auth_url = f"http://127.0.0.1:{server.server_port}"
            original_grant = token("cw_s3", 300)
            token_file = tmp_path / "native.jwt"
            credentials = StorageCredentials(
                auth_url=auth_url,
                grant_token=original_grant,
                binding="read",
                token_file=token_file,
                now=lambda: clock[0],
            )
            original_authorization = token_file.read_text()
            assert original_authorization == "Bearer " + original_grant
            assert token_file.stat().st_mode & 0o777 == 0o600
            clock[0] = 1290
            with refreshing_storage_credentials(credentials):
                assert renewed.wait(timeout=2)
            clock[0] = 1400
            credentials.refresh()
            assert token_file.read_text() != original_authorization
            assert (
                requests.post(auth_url + "/oauth/token", data={"subject_token": original_grant}, timeout=5).status_code
                == 401
            )
            clock[0] = 1580
            credentials.refresh()
            renewed_token = token_file.read_text().removeprefix("Bearer ")
            assert authenticated(renewed_token)
            assert token_file.stat().st_mode & 0o777 == 0o600
            assert jwt.decode(renewed_token, options={"verify_signature": False})["exp"] == 1880
            assert credentials.credentials_url == auth_url + "/v1/storage/read/credentials"
            clock[0] = 1880
            deny[0] = True
            renewed.clear()
            with pytest.raises(requests.HTTPError):
                with refreshing_storage_credentials(credentials):
                    assert renewed.wait(timeout=2)
        finally:
            server.shutdown()
            thread.join()


def test_wrapped_native_sdk_ignores_ambient_profile_role_and_static_credentials(tmp_path):
    grant = jwt.encode(
        {"grant_id": "scoped", "exp": time.time() + 300}, "synthetic-signing-key-for-storage-test", algorithm="HS256"
    )
    requested = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/storage/read/credentials"
            assert self.headers["Authorization"] == "Bearer " + grant
            body = json.dumps(
                {
                    "AccessKeyId": "scoped-native-key",
                    "SecretAccessKey": "scoped-native-secret",
                    "Token": "",
                    "Expiration": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            requested.set()

        def log_message(self, *args):
            pass

    profile = tmp_path / "broad-profile"
    profile.write_text("[default]\naws_access_key_id=broad-profile-key\naws_secret_access_key=broad-profile-secret\n")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            credentials = StorageCredentials(
                auth_url=f"http://127.0.0.1:{server.server_port}",
                grant_token=grant,
                binding="read",
                token_file=tmp_path / "authorization",
            )
            ambient = {
                **os.environ,
                "AWS_ACCESS_KEY_ID": "broad-env-key",
                "AWS_SECRET_ACCESS_KEY": "broad-env-secret",
                "AWS_PROFILE": "broad",
                "AWS_DEFAULT_PROFILE": "broad",
                "AWS_CONFIG_FILE": str(profile),
                "AWS_SHARED_CREDENTIALS_FILE": str(profile),
                "BOTO_CONFIG": str(profile),
                "AWS_WEB_IDENTITY_TOKEN_FILE": "/missing/broad-token",
                "AWS_ROLE_ARN": "arn:aws:iam::123:role/broad",
                "AWS_ROLE_SESSION_NAME": "broad",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/broad",
                "AWS_CONTAINER_AUTHORIZATION_TOKEN": "broad-token",
                "AWS_CREDENTIAL_FILE": str(profile),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from botocore.session import Session; c=Session().get_credentials(); "
                    "assert c.method == 'container-role'; "
                    "assert c.get_frozen_credentials().access_key == 'scoped-native-key'",
                ],
                env=storage_environment(credentials, ambient, None),
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0, result.stderr
            assert requested.is_set()
        finally:
            server.shutdown()
            thread.join()
