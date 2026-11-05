# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the pilot's actual SDK provider and S3 signing without live services."""

import json
import re
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError
from botocore.httpsession import URLLib3Session
from urllib3.response import HTTPResponse

from infra.buckets.openbao_pilot import pilot


@pytest.mark.parametrize("denial", ["AccessDenied", "ExpiredToken"])
def test_probe_uses_fresh_credentials_on_the_same_client(tmp_path, monkeypatch, denial):
    class Clock(datetime):
        current = datetime.now(UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

        @classmethod
        def advance(cls, seconds):
            cls.current += timedelta(seconds=seconds)

    first_expiration = Clock.current + timedelta(minutes=5)
    role = "https://pilot.example.net/v1/identity/oidc:example-entity"
    bucket = "marin-openbao-pilot-test"
    keys = {}
    reads = []
    denied = []

    def fake_bao(path, *, token=None, data=None):
        if path == "auth/approle/login":
            return {"auth": {"client_token": "fake-approle-token"}}
        assert path == pilot.TOKEN_PATH
        assert token == "fake-approle-token"
        return {"data": {"token": "jwt-1" if Clock.current < first_expiration else "jwt-2"}}

    def response(request, status, body):
        headers = {"content-length": str(len(body))}
        return AWSResponse(request.url, status, headers, HTTPResponse(body=BytesIO(body), preload_content=False))

    def send(session, request):
        url = urlsplit(request.url)
        authorization = request.headers["Authorization"]
        if url.hostname == "api.coreweave.com":
            key = {"jwt-1": "key-1", "jwt-2": "key-2"}[authorization]
            expires = first_expiration if key == "key-1" else Clock.current + timedelta(hours=1)
            keys[key] = expires
            credentials = {
                "AccessKeyId": key,
                "SecretAccessKey": "synthetic-credential",
                "Token": "",
                "Expiration": expires.isoformat(),
                "attributes": {"cw-role": role},
            }
            return response(request, 200, json.dumps(credentials).encode())

        assert url.hostname == f"{bucket}.cwobject.com"
        signed_key = re.search(rb"Credential=([^/]+)/", authorization).group(1).decode()
        assert keys[signed_key] > Clock.current
        if request.method == "GET" and url.path == "/allowed/read.txt":
            reads.append((session, signed_key))
            return response(request, 200, pilot.SENTINEL)
        denied.append((request.method, url.path))
        return response(request, 403, f"<Error><Code>{denial}</Code></Error>".encode())

    credentials_file = tmp_path / "approle.json"
    credentials_file.write_text(
        json.dumps(
            {
                "approle": {"role_id": "synthetic-role", "secret_id": "synthetic-secret"},
                "bucket": bucket,
                "role": role,
            }
        )
    )
    monkeypatch.setenv("AWS_PROFILE", "missing")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unrelated-key")
    monkeypatch.setattr(pilot, "datetime", Clock)
    monkeypatch.setattr(pilot, "time", SimpleNamespace(sleep=Clock.advance))
    monkeypatch.setattr(pilot, "READ_INTERVAL", 301)
    monkeypatch.setattr(pilot, "bao", fake_bao)
    monkeypatch.setattr(URLLib3Session, "send", send)

    if denial == "ExpiredToken":
        with pytest.raises(ClientError, match="ExpiredToken"):
            pilot.probe(credentials_file, "123456", "US-EAST-02A", 10)
        assert [key for _, key in reads] == ["key-1"]
        return

    pilot.probe(credentials_file, "123456", "US-EAST-02A", 10)
    assert [key for _, key in reads] == ["key-1", "key-2"]
    assert reads[0][0] is reads[1][0]
    assert [method for method, _ in denied] == ["GET", "PUT", "DELETE"]
    assert denied[0][1] == "/outside/read.txt"
    assert denied[1][1].startswith("/allowed/put-")
    assert denied[2][1].startswith("/allowed/delete-")
