# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded live CoreWeave policy probes against explicitly supplied test fixtures."""

import argparse
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from botocore.config import Config
from botocore.credentials import ContainerProvider, CredentialResolver
from botocore.exceptions import BotoCoreError, ClientError
from botocore.session import Session
from botocore.utils import ContainerMetadataFetcher
from rigging.filesystem.s3_compat import CoreWeaveFederation, coreweave_federation
from rigging.grants import Permission

from infra.auth.client import StorageCredentials


class CredentialExpirations(ContainerMetadataFetcher):
    """Observe native provider expiry without exposing credentials in the report."""

    def __init__(self):
        super().__init__()
        self.first: datetime | None = None
        self.latest: datetime | None = None
        self.rotated = False
        self._access_key: str | None = None

    def retrieve_full_uri(self, full_url, headers=None):
        metadata = super().retrieve_full_uri(full_url, headers=headers)
        expiry = datetime.fromisoformat(metadata["Expiration"].replace("Z", "+00:00"))
        self.first = self.first or expiry
        self.latest = expiry
        self.rotated = self.rotated or (self._access_key is not None and self._access_key != metadata["AccessKeyId"])
        self._access_key = metadata["AccessKeyId"]
        return metadata


def read_fixture(s3, bucket: str, key: str) -> None:
    """Read at most one byte from an existing, nonempty fixture."""
    with closing(s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-0")["Body"]) as body:
        body.read(1)


def _policy_denial(error: ClientError) -> bool:
    return (
        error.response["Error"]["Code"] == "AccessDenied" and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403
    )


def probe_access(s3, *, bucket: str, allowed_key: str, denied_key: str, prefix: str) -> dict:
    """Probe one inside/outside read and two fresh sentinel write operations."""
    Permission("cw_s3", frozenset({"s3:GetObject"}), f"s3://{bucket}/{prefix}")
    if not allowed_key.startswith(prefix) or denied_key.startswith(prefix):
        raise ValueError("The supplied fixtures must be inside and outside the approved prefix respectively")
    read_fixture(s3, bucket, allowed_key)
    results = {"inside_read": True}
    sentinel = prefix + "marin-auth-probe-" + uuid.uuid4().hex
    for name, operation in (
        ("outside_read_denied", lambda: read_fixture(s3, bucket, denied_key)),
        ("write_denied", lambda: s3.put_object(Bucket=bucket, Key=sentinel, Body=b"x", IfNoneMatch="*")),
        ("delete_denied", lambda: s3.delete_object(Bucket=bucket, Key=prefix + "marin-auth-probe-" + uuid.uuid4().hex)),
    ):
        try:
            operation()
        except ClientError as error:
            if not _policy_denial(error):
                raise
            results[name] = True
        else:
            results[name] = False
            if name == "write_denied":
                try:
                    s3.delete_object(Bucket=bucket, Key=sentinel)
                except (ClientError, BotoCoreError):
                    results["cleanup_required_key"] = sentinel
    results["passed"] = all(
        results.get(name) is True for name in ("outside_read_denied", "write_denied", "delete_denied")
    )
    return results


def native_session(federation: CoreWeaveFederation, metadata: ContainerMetadataFetcher) -> Session:
    """Use only the native container provider, regardless of ambient credentials."""
    session = Session(session_vars={"profile": (None, None, None, None)})
    provider = ContainerProvider(
        environ={
            "AWS_CONTAINER_CREDENTIALS_FULL_URI": federation.uri,
            "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": federation.token_file,
        },
        fetcher=metadata,
    )
    session.register_component("credential_provider", CredentialResolver([provider]))
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "auth-url",
        "binding",
        "organization",
        "endpoint",
        "region",
        "test-bucket",
        "allowed-key",
        "denied-key",
        "prefix",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--wait-up-to", type=float, default=0, help="Explicit maximum seconds to cross the first native key expiry"
    )
    args = parser.parse_args()
    if not 0 <= args.wait_up_to <= 7200:
        parser.error("--wait-up-to must be between 0 and 7200 seconds")
    endpoint = urlsplit(args.endpoint)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or not any(
            endpoint.hostname == domain or endpoint.hostname.endswith("." + domain)
            for domain in ("cwobject.com", "cwlota.com")
        )
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in ("", "/")
        or endpoint.port not in (None, 443)
    ):
        parser.error("--endpoint must be a CoreWeave object-storage HTTPS origin")
    with tempfile.TemporaryDirectory(prefix="marin-auth-storage-") as directory:
        token_file = Path(directory) / "native.jwt"
        credentials = StorageCredentials(
            auth_url=args.auth_url,
            grant_token=os.environ["MARIN_GRANT_TOKEN"],
            binding=args.binding,
            token_file=token_file,
        )
        os.environ["AWS_CONTAINER_CREDENTIALS_FULL_URI"] = (
            "https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/" + args.organization
        )
        os.environ["AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"] = str(token_file)
        federation = coreweave_federation()
        assert federation is not None
        metadata = CredentialExpirations()
        session = native_session(federation, metadata)
        with closing(
            session.create_client(
                "s3",
                endpoint_url=args.endpoint,
                region_name=args.region,
                config=Config(
                    connect_timeout=5,
                    read_timeout=10,
                    retries={"total_max_attempts": 1},
                    s3={"addressing_style": "virtual"},
                ),
            )
        ) as s3:
            report = probe_access(
                s3, bucket=args.test_bucket, allowed_key=args.allowed_key, denied_key=args.denied_key, prefix=args.prefix
            )
            assert metadata.first is not None
            report.update(
                test_bucket=args.test_bucket,
                provider_expires_at=metadata.first.isoformat(),
                crossed_initial_expiry=False,
            )
            if args.wait_up_to and report["passed"]:
                deadline = time.monotonic() + args.wait_up_to
                while datetime.now(UTC) <= metadata.first and time.monotonic() < deadline:
                    credentials.refresh()
                    threading.Event().wait(min(1, max(0, deadline - time.monotonic())))
                if datetime.now(UTC) > metadata.first:
                    credentials.refresh()
                    read_fixture(s3, args.test_bucket, args.allowed_key)
                    assert metadata.latest is not None
                    report["crossed_initial_expiry"] = metadata.rotated and metadata.latest > metadata.first
                report["passed"] = report["passed"] and report["crossed_initial_expiry"]
                assert metadata.latest is not None
                report["latest_provider_expires_at"] = metadata.latest.isoformat()
            print(json.dumps(report, sort_keys=True))
            raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
