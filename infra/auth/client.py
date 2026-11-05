# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep an SDK identity-token file fresh for one granted workload command."""

import argparse
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import jwt
import requests
from rigging.auth import RefreshingTokenProvider
from rigging.grant_auth import ACCESS_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT


class StorageCredentials:
    """Renew the service grant before obtaining each native provider token."""

    def __init__(
        self,
        *,
        auth_url: str,
        grant_token: str,
        binding: str,
        token_file: Path,
        now: Callable[[], float] = time.time,
    ):
        parsed = urlsplit(auth_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("The grant service requires HTTPS outside loopback")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("The grant service URL must be an origin")
        if not binding or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in binding):
            raise ValueError("invalid storage binding name")
        self.auth_url = auth_url.rstrip("/")
        self.binding = binding
        self.token_file = token_file
        self.grant_id = jwt.decode(grant_token, options={"verify_signature": False})["grant_id"]
        self.grant = RefreshingTokenProvider(grant_token, self._refresh_grant, refresh_margin=30, now=now)
        initial = self._storage_token()
        self.native = RefreshingTokenProvider(
            initial, lambda _previous: self._storage_token(), refresh_margin=60, cache_path=token_file, now=now
        )
        # The SDK needs the initial token before its first request, even though
        # RefreshingTokenProvider only writes its cache after a renewal.
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(initial)

    def _post(self, path: str, *, data: dict | None = None, token: str | None = None) -> dict:
        response = requests.post(
            self.auth_url + path,
            data=data,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=10,
            allow_redirects=False,
        )
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("The auth service redirected a credential-bearing request")
        return response.json()

    def _refresh_grant(self, previous: str) -> str:
        return self._post(
            "/oauth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "subject_token": previous,
                "audience": "cw_s3",
                "grant_id": self.grant_id,
            },
        )["access_token"]

    def _storage_token(self) -> str:
        return self._post(f"/v1/storage/{self.binding}/token", token=self.grant.get_token())["token"]

    def refresh(self) -> None:
        # The service JWT usually expires before the native OIDC token. Renew
        # it while the native token is cached so later file renewal can succeed.
        self.grant.get_token()
        self.native.get_token()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-url", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required")
    args.token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    credentials = StorageCredentials(
        auth_url=args.auth_url,
        grant_token=os.environ["MARIN_GRANT_TOKEN"],
        binding=args.binding,
        token_file=args.token_file,
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MARIN_GRANT_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
    }
    environment.update(
        AWS_CONTAINER_CREDENTIALS_FULL_URI=(
            "https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/" + args.organization
        ),
        AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE=str(args.token_file.resolve()),
    )
    try:
        with subprocess.Popen(command, env=environment) as process:
            try:
                while True:
                    try:
                        result = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        credentials.refresh()
            finally:
                if process.poll() is None:
                    process.terminate()
        raise SystemExit(result)
    finally:
        args.token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
