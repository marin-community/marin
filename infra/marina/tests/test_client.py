# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator

import pytest
import requests
from marina.client import MarinaRequestError, marina_request


class RotatingTokens:
    def __init__(self, tokens: Iterator[str]):
        self.tokens = tokens

    def get_token(self) -> str:
        return next(self.tokens)


def test_shared_transport_preserves_auth_rotation_payload_and_error_detail(monkeypatch):
    observed = []

    def send(_session, request, **_kwargs):
        observed.append(request)
        response = requests.Response()
        response.status_code = 422 if request.url.endswith("/failure") else 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(
            {"detail": "invalid query"} if response.status_code == 422 else {"row_count": 1}
        ).encode()
        return response

    # Capture the prepared HTTP request at the network boundary.
    monkeypatch.setattr(requests.Session, "send", send)
    provider = RotatingTokens(iter(("first-token", "renewed-token", "renewed-token")))
    with requests.Session() as session:
        first = marina_request(
            "https://marina.example",
            "POST",
            "/query",
            json_body={"sql": "SELECT :value", "parameters": {"value": 7}},
            session=session,
            token_provider=provider,
        )
        second = marina_request("https://marina.example", "GET", "/overview", session=session, token_provider=provider)
        with pytest.raises(MarinaRequestError, match="invalid query") as error:
            marina_request("https://marina.example", "GET", "/failure", session=session, token_provider=provider)
    assert first == second == {"row_count": 1}
    assert [request.headers["Authorization"] for request in observed] == [
        "Bearer first-token",
        "Bearer renewed-token",
        "Bearer renewed-token",
    ]
    assert json.loads(observed[0].body) == {"sql": "SELECT :value", "parameters": {"value": 7}}
    assert error.value.status_code == 422


def test_local_request_does_not_send_shared_credentials(monkeypatch):
    observed = []

    def send(_session, request, **_kwargs):
        observed.append(request)
        response = requests.Response()
        response.status_code = 204
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    with requests.Session() as session:
        result = marina_request(
            "http://127.0.0.1:8080",
            "POST",
            "/upload",
            payload=b"package",
            session=session,
            token_provider=RotatingTokens(iter(())),
        )
    assert result is None
    assert "Authorization" not in observed[0].headers
    assert observed[0].headers["Content-Type"] == "application/gzip"
    assert observed[0].body == b"package"
