# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a private broker authorization file fresh for one granted SDK workload."""

import argparse
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import jwt
import requests
from rigging.auth import RefreshingTokenProvider
from rigging.grant_auth import ACCESS_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.secrets import require_secret
from rigging.shared_grants import authority_origin


class StorageCredentials:
    """Refresh a Marin grant; native SDKs obtain bounded provider keys from the broker."""

    def __init__(
        self,
        *,
        auth_url: str,
        grant_token: str,
        binding: str,
        token_file: Path,
        ca_bundle: Path | None = None,
        now: Callable[[], float] = time.time,
    ):
        if not binding or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in binding):
            raise ValueError("invalid storage binding name")
        self.auth_url = authority_origin(auth_url)
        self.credentials_url = self.auth_url + f"/v1/storage/{binding}/credentials"
        self.token_file = token_file
        self.verify = str(ca_bundle) if ca_bundle is not None else True
        self.grant_id = jwt.decode(grant_token, options={"verify_signature": False})["grant_id"]
        self.grant = RefreshingTokenProvider(grant_token, self._refresh_grant, refresh_margin=10, now=now)
        self._current = self.grant.get_token()
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write("Bearer " + self._current)

    def _refresh_grant(self, previous: str) -> str:
        response = requests.post(
            self.auth_url + "/oauth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "subject_token": previous,
                "audience": "cw_s3",
                "grant_id": self.grant_id,
            },
            timeout=10,
            allow_redirects=False,
            verify=self.verify,
        )
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("The auth service redirected a credential-bearing request")
        return response.json()["access_token"]

    def refresh(self) -> None:
        token = self.grant.get_token()
        if token == self._current:
            return
        with tempfile.NamedTemporaryFile(mode="w", dir=self.token_file.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write("Bearer " + token)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(self.token_file)
            finally:
                temporary.unlink(missing_ok=True)
        self._current = token


@contextmanager
def refreshing_storage_credentials(credentials: StorageCredentials) -> Iterator[Callable[[], None]]:
    """Renew independently of workload latency and propagate renewal failures."""
    stop = threading.Event()

    def renew() -> None:
        while not stop.is_set():
            credentials.refresh()
            stop.wait(2)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="storage-credentials") as executor:
        future = executor.submit(renew)

        def check() -> None:
            if future.done():
                future.result()

        try:
            yield check
        finally:
            stop.set()
            future.result()


def storage_environment(
    credentials: StorageCredentials, source: Mapping[str, str], ca_bundle: Path | None
) -> dict[str, str]:
    """Select only broker-provided credentials, excluding ambient SDK identity sources."""
    excluded = {
        "MARIN_GRANT_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CREDENTIAL_FILE",
    }
    environment = {key: value for key, value in source.items() if key not in excluded}
    environment.update(
        MARIN_AUTH_URL=credentials.auth_url,
        AWS_CONTAINER_CREDENTIALS_FULL_URI=credentials.credentials_url,
        AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE=str(credentials.token_file.resolve()),
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        AWS_CONFIG_FILE=os.devnull,
        BOTO_CONFIG=os.devnull,
        AWS_EC2_METADATA_DISABLED="true",
    )
    if ca_bundle is not None:
        environment["MARIN_AUTH_CA_BUNDLE"] = str(ca_bundle)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-url", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required")
    args.token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    credentials = StorageCredentials(
        auth_url=args.auth_url,
        grant_token=require_secret("MARIN_GRANT_TOKEN"),
        binding=args.binding,
        token_file=args.token_file,
        ca_bundle=args.ca_bundle,
    )
    environment = storage_environment(credentials, os.environ, args.ca_bundle)
    try:
        with refreshing_storage_credentials(credentials) as check, subprocess.Popen(command, env=environment) as process:
            try:
                while True:
                    try:
                        result = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        check()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        raise SystemExit(result)
    finally:
        args.token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
