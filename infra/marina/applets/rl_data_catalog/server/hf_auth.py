# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scope ephemeral HF authentication to HF hosts; resolve a runtime secret on rate limits."""

import base64
from collections.abc import Generator

import google.auth
import httpx
from google.auth.transport.requests import AuthorizedSession
from requests import RequestException

HF_HOSTS = {"huggingface.co", "datasets-server.huggingface.co"}
HF_SECRET_VERSION = "projects/hai-gcp-models/secrets/HF_TOKEN_READONLY/versions/1"


class HFCredentialError(RuntimeError):
    """Runtime credentials could not be resolved for an authenticated HF retry."""


def runtime_hf_token() -> str:
    """Return the Hugging Face token from the configured runtime secret."""
    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        with AuthorizedSession(credentials) as session:
            response = session.get(f"https://secretmanager.googleapis.com/v1/{HF_SECRET_VERSION}:access", timeout=15)
            response.raise_for_status()
            return base64.b64decode(response.json()["payload"]["data"]).decode().strip()
    except (google.auth.exceptions.GoogleAuthError, RequestException, ValueError, KeyError) as error:
        raise HFCredentialError(
            "HF rate limited the refresh and Marina could not read its configured Hugging Face secret. "
            "Use an authenticated refresh or configure Secret Manager access."
        ) from error


class HuggingFaceAuth(httpx.Auth):
    def __init__(self, token: str | None) -> None:
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if request.url.host not in HF_HOSTS:
            yield request
            return
        if self.token:
            request.headers["Authorization"] = f"Bearer {self.token}"
        response = yield request
        if response.status_code == 429 and self.token is None:
            self.token = runtime_hf_token()
            request.headers["Authorization"] = f"Bearer {self.token}"
            yield request
