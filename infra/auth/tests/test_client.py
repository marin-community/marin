# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import jwt
import requests

from infra.auth.client import StorageCredentials


def test_grant_renews_before_native_file_needs_refresh(tmp_path):
    clock = [1000]
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
            if self.path == "/oauth/token":
                form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                credential = form["subject_token"][0]
                result = {"access_token": token("cw_s3", 300)}
            else:
                credential = self.headers["Authorization"].removeprefix("Bearer ")
                result = {"token": token("https://coreweave.com/iam", 600)}
            status = 200 if authenticated(credential) else 401
            body = json.dumps(result if status == 200 else {"error": "expired_subject"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
            original_native = token_file.read_text()
            clock[0] = 1270
            credentials.refresh()
            clock[0] = 1400
            credentials.refresh()
            assert token_file.read_text() == original_native
            assert (
                requests.post(auth_url + "/oauth/token", data={"subject_token": original_grant}, timeout=5).status_code
                == 401
            )
            clock[0] = 1540
            credentials.refresh()
            renewed_native = token_file.read_text()
            assert renewed_native != original_native
            assert jwt.decode(renewed_native, options={"verify_signature": False})["exp"] == 2140
        finally:
            server.shutdown()
            thread.join()
