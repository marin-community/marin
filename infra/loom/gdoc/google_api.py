# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Call the Google Docs and Drive APIs with a token from gcloud.

In a Loom session, gcloud's account is the Loom VM's service account, whose metadata token has only the
cloud-platform scope. The account impersonates itself for the Docs and Drive scopes and to sign diagram
URLs, which `roles/iam.serviceAccountTokenCreator` on itself allows. A person's gcloud login is used as is,
so it needs Drive access (`gcloud auth login --enable-gdrive-access`) and cannot sign diagram URLs.

This needs only the standard library and gcloud, so it runs with the session image's system Python.
"""

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DOCS_API = "https://docs.googleapis.com/v1"
DRIVE_API = "https://www.googleapis.com/drive/v3"
SCOPES = ("https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive")
SERVICE_ACCOUNT_SUFFIX = ".gserviceaccount.com"


def gcloud(*args: str, stdin: bytes | None = None) -> str:
    result = subprocess.run(["gcloud", *args], input=stdin, check=True, capture_output=True)
    return result.stdout.decode().strip()


def gcloud_account() -> str:
    account = gcloud("config", "get", "account")
    if not account:
        raise ValueError("gcloud has no active account; run `gcloud auth login --enable-gdrive-access`")
    return account


def is_service_account(account: str) -> bool:
    return account.endswith(SERVICE_ACCOUNT_SUFFIX)


def access_token(account: str) -> str:
    if is_service_account(account):
        return gcloud(
            "auth", "print-access-token", f"--impersonate-service-account={account}", f"--scopes={','.join(SCOPES)}"
        )
    return gcloud("auth", "print-access-token", account)


@dataclass(frozen=True)
class GoogleApi:
    """JSON calls to Google APIs with one bearer token."""

    token: str

    def request(self, method: str, url: str, *, params: dict | None = None, body: dict | None = None) -> bytes:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # Google puts the reason in the body, for example a missing scope or a bad request index.
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"{method} {url} failed with HTTP {error.code}: {detail}") from error

    def get(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.request("GET", url, params=params))

    def post(self, url: str, body: dict, params: dict | None = None) -> dict:
        return json.loads(self.request("POST", url, params=params, body=body))
