# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["botocore==1.42.5"]
# ///

"""Bootstrap and run the disposable CoreWeave OIDC credential probe."""

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple, TextIO
from urllib.parse import urlsplit
from uuid import uuid4

from botocore.config import Config
from botocore.credentials import ContainerProvider, CredentialResolver
from botocore.exceptions import ClientError, MetadataRetrievalError
from botocore.session import Session
from botocore.utils import ContainerMetadataFetcher

ROLE = "coreweave-pilot"
APPROLE_PATH = f"auth/approle/role/{ROLE}"
TOKEN_PATH = f"identity/oidc/token/{ROLE}"
AUDIENCE = "https://coreweave.com/iam"
BUCKET_PATTERN = r"marin-openbao-pilot-[a-z0-9][a-z0-9-]{0,41}[a-z0-9]"
READ_PREFIX = "allowed"
SENTINEL = b"openbao-pilot\n"
READ_INTERVAL = 30


def bao(path: str, *, token: str | None = None, data: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Vault-Token"] = token
    method = "GET" if data is None else "POST"
    body = None if data is None else json.dumps(data)
    # Use a direct loopback connection that cannot follow redirects or honor proxy settings.
    with closing(HTTPConnection("127.0.0.1", 18200, timeout=10)) as connection:
        connection.request(method, f"/v1/{path}", body=body, headers=headers)
        with connection.getresponse() as response:
            if response.status == 204:
                return {}
            if response.status != 200:
                raise RuntimeError(f"OpenBao {method} {path} returned HTTP {response.status}")
            return json.load(response)


def login(approle: dict) -> dict:
    return bao("auth/approle/login", data=approle)["auth"]


def bootstrap(origin: str, bucket: str, credentials_file: Path, root_token: str) -> dict:
    parsed = urlsplit(origin)
    if not parsed.hostname or origin != f"https://{parsed.hostname}":
        raise ValueError("Public origin must be https:// followed by a lowercase DNS hostname")

    with create_private_file(credentials_file) as output:
        bao("identity/oidc/config", token=root_token, data={"issuer": origin})
        bao(f"identity/oidc/key/{ROLE}", token=root_token, data={"allowed_client_ids": [AUDIENCE]})
        bao(f"identity/oidc/role/{ROLE}", token=root_token, data={"key": ROLE, "client_id": AUDIENCE, "ttl": "5m"})
        bao(
            f"sys/policies/acl/{ROLE}",
            token=root_token,
            data={"policy": f'path "{TOKEN_PATH}" {{ capabilities = ["read"] }}'},
        )
        bao("sys/auth/approle", token=root_token, data={"type": "approle"})
        bao(
            APPROLE_PATH,
            token=root_token,
            data={
                "token_policies": [ROLE],
                "token_no_default_policy": True,
                "token_type": "batch",
                "token_ttl": "2m",
                "secret_id_ttl": "4h",
            },
        )
        approle = {
            "role_id": bao(f"{APPROLE_PATH}/role-id", token=root_token)["data"]["role_id"],
            "secret_id": bao(f"{APPROLE_PATH}/secret-id", token=root_token, data={})["data"]["secret_id"],
        }
        auth = login(approle)
        bao(TOKEN_PATH, token=auth["client_token"])  # Generate the signing key and check the workload ACL.
        discovery = bao("identity/oidc/.well-known/openid-configuration")
        issuer = discovery["issuer"]
        role = f"{issuer}:{auth['entity_id']}"
        json.dump({"approle": approle, "bucket": bucket, "role": role}, output)

    return {
        "issuer": issuer,
        "jwks_uri": discovery["jwks_uri"],
        "audience": AUDIENCE,
        "policy": {
            "version": "v1alpha1",
            "name": bucket,
            "statements": [
                {
                    "name": "exchange-oidc",
                    "effect": "Allow",
                    "actions": ["cwobject:CreateAccessKeyOIDC"],
                    "resources": ["*"],
                    "principals": [f"role/{role}"],
                },
                {
                    "name": "read-pilot-prefix",
                    "effect": "Allow",
                    "actions": ["s3:GetObject"],
                    "resources": [f"{bucket}/{READ_PREFIX}/*"],
                    "principals": [f"role/{role}"],
                },
            ],
        },
    }


class Exchange(NamedTuple):
    expires: datetime
    access_key_id: str
    bearer_hash: bytes


class ObservedCredentials(ContainerMetadataFetcher):
    """Record exchange expiration and rotation without emitting credentials."""

    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.first_exchange: Exchange | None = None
        self.last_exchange: Exchange | None = None

    def retrieve_full_uri(self, full_url, headers=None):
        try:
            metadata = super().retrieve_full_uri(full_url, headers)
        except MetadataRetrievalError:
            raise RuntimeError("CoreWeave OIDC credential exchange failed") from None
        if metadata.get("attributes", {}).get("cw-role") != self.role:
            raise RuntimeError("CoreWeave issued credentials for a different role")
        exchange = Exchange(
            datetime.fromisoformat(metadata["Expiration"]),
            metadata["AccessKeyId"],
            hashlib.sha256(headers["Authorization"].encode()).digest(),
        )
        if self.first_exchange is None:
            self.first_exchange = exchange
        self.last_exchange = exchange
        return metadata

    def rotated_after_expiry(self, read_started: datetime) -> bool:
        first = self.first_exchange
        last = self.last_exchange
        return (
            first is not None
            and last is not None
            and read_started >= first.expires
            and last.expires > first.expires
            and last.access_key_id != first.access_key_id
            and last.bearer_hash != first.bearer_hash
        )


def create_private_file(path: Path) -> TextIO:
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")


def refresh_token(approle: dict, token_file: Path) -> None:
    auth = login(approle)
    jwt = bao(TOKEN_PATH, token=auth["client_token"])["data"]["token"]
    with TemporaryDirectory(dir=token_file.parent) as directory:
        staging = Path(directory) / "oidc.jwt"
        with create_private_file(staging) as output:
            output.write(jwt)
        staging.replace(token_file)


def s3_client(region: str, token_file: Path, credentials_url: str, endpoint: str, fetcher: ObservedCredentials):
    # Install only the token-file provider so a local profile or worker key cannot satisfy the probe.
    provider = ContainerProvider(
        environ={
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": credentials_url,
            "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": str(token_file),
        },
        fetcher=fetcher,
    )
    session = Session(session_vars={"profile": (None, None, None, None)})
    session.set_config_variable("config_file", os.devnull)
    session.set_config_variable("credentials_file", os.devnull)
    session.register_component("credential_provider", CredentialResolver([provider]))
    return session.create_client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            region_name=region,
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            connect_timeout=10,
            read_timeout=10,
            retries={"total_max_attempts": 2, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def require_denied(description: str, action: Callable, **kwargs) -> None:
    try:
        response = action(**kwargs)
    except ClientError as error:
        response = error.response
        if response["ResponseMetadata"]["HTTPStatusCode"] == 403 and response["Error"]["Code"] == "AccessDenied":
            print(f"{description}: AccessDenied", flush=True)
            return
        raise
    if "Body" in response:
        response["Body"].close()
    raise RuntimeError(f"{description} unexpectedly succeeded")


def read_sentinel(client, bucket: str) -> None:
    response = client.get_object(Bucket=bucket, Key=f"{READ_PREFIX}/read.txt")
    with closing(response["Body"]) as body:
        if body.read(len(SENTINEL) + 1) != SENTINEL:
            raise RuntimeError("The allowed sentinel does not contain the expected pilot payload")


def probe(credentials_file: Path, organization: str, region: str, max_wait_minutes: int) -> None:
    config = json.loads(credentials_file.read_text())
    bucket = config["bucket"]
    if not re.fullmatch(BUCKET_PATTERN, bucket):
        raise ValueError("Use a disposable bucket named marin-openbao-pilot-<suffix>")
    if not organization.isascii() or not organization.isdecimal():
        raise ValueError("Expected a numeric CoreWeave organization ID")

    deadline = datetime.now(UTC) + timedelta(minutes=max_wait_minutes)
    fetcher = ObservedCredentials(config["role"])
    with TemporaryDirectory(prefix="openbao-pilot-") as directory:
        token_file = Path(directory) / "oidc.jwt"
        refresh_token(config["approle"], token_file)
        with closing(
            s3_client(
                region,
                token_file,
                f"https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/{organization}",
                "https://cwobject.com",
                fetcher,
            )
        ) as client:
            read_sentinel(client, bucket)
            print("Allowed sentinel read succeeded", flush=True)
            require_denied("Outside-prefix read", client.get_object, Bucket=bucket, Key="outside/read.txt")
            require_denied(
                "Write",
                client.put_object,
                Bucket=bucket,
                Key=f"{READ_PREFIX}/put-{uuid4()}",
                Body=SENTINEL,
                IfNoneMatch="*",
            )
            require_denied("Delete", client.delete_object, Bucket=bucket, Key=f"{READ_PREFIX}/delete-{uuid4()}")

            first = fetcher.first_exchange
            if first is None or first.expires >= deadline:
                raise RuntimeError("The first credential expiry is outside the probe window")
            print(f"Waiting past the first credential expiration ({first.expires.isoformat()})", flush=True)

            while datetime.now(UTC) < deadline:
                time.sleep(min(READ_INTERVAL, max(0, (deadline - datetime.now(UTC)).total_seconds())))
                refresh_token(config["approle"], token_file)
                started = datetime.now(UTC)
                read_sentinel(client, bucket)
                if fetcher.rotated_after_expiry(started):
                    print("The same S3 client read successfully after rotating its credentials", flush=True)
                    return

    raise RuntimeError("The same S3 client did not demonstrate credential rotation before the deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("bootstrap")
    setup.add_argument("--public-origin", required=True)
    setup.add_argument("--bucket", required=True)
    setup.add_argument("--credentials-file", type=Path, required=True)
    setup.add_argument("--root-token-file", type=Path, required=True)
    issue = commands.add_parser("issue-token")
    issue.add_argument("--credentials-file", type=Path, required=True)
    issue.add_argument("--token-file", type=Path, required=True)
    run = commands.add_parser("probe")
    run.add_argument("--credentials-file", type=Path, required=True)
    run.add_argument("--organization", required=True)
    run.add_argument("--region", required=True)
    run.add_argument("--max-wait-minutes", type=int, default=90)
    args = parser.parse_args()
    if args.command == "bootstrap":
        if not re.fullmatch(BUCKET_PATTERN, args.bucket):
            parser.error("use a disposable bucket named marin-openbao-pilot-<suffix>")
        root_token = args.root_token_file.read_text().strip()
        metadata = bootstrap(args.public_origin, args.bucket, args.credentials_file, root_token)
        print(json.dumps(metadata, indent=2))
    elif args.command == "issue-token":
        config = json.loads(args.credentials_file.read_text())
        refresh_token(config["approle"], args.token_file)
    else:
        probe(args.credentials_file, args.organization, args.region, args.max_wait_minutes)


if __name__ == "__main__":
    main()
