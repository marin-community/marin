# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded live CoreWeave policy probes against explicitly supplied test fixtures."""

import argparse
import json
import re
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
from botocore.httpsession import URLLib3Session
from botocore.session import Session
from botocore.utils import ContainerMetadataFetcher
from rigging.filesystem.s3_compat import VIRTUAL_HOST_ONLY_S3_DOMAINS, CoreWeaveFederation
from rigging.filesystem.storage_path import prefix_join
from rigging.grants import Permission
from rigging.secrets import require_secret

from infra.auth.client import StorageCredentials, refreshing_storage_credentials


def validate_storage_endpoint(value: str) -> None:
    """Require a CoreWeave object-storage HTTPS origin before sending credentials."""
    endpoint = urlsplit(value)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or not any(
            endpoint.hostname == domain or endpoint.hostname.endswith("." + domain)
            for domain in VIRTUAL_HOST_ONLY_S3_DOMAINS
        )
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in ("", "/")
        or endpoint.port not in (None, 443)
    ):
        raise ValueError("The storage endpoint must be a CoreWeave object-storage HTTPS origin")


class CredentialExpirations(ContainerMetadataFetcher):
    """Observe native provider expiry without exposing credentials in the report."""

    def __init__(self, ca_bundle: Path | None = None):
        super().__init__(session=URLLib3Session(timeout=5, verify=str(ca_bundle) if ca_bundle else True))
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
    results: dict[str, bool | str] = {"inside_read": True}
    sentinel = prefix_join(prefix, "marin-auth-probe-" + uuid.uuid4().hex)
    for name, operation in (
        ("outside_read_denied", lambda: read_fixture(s3, bucket, denied_key)),
        ("write_denied", lambda: s3.put_object(Bucket=bucket, Key=sentinel, Body=b"x", IfNoneMatch="*")),
        ("delete_denied", lambda: s3.delete_object(Bucket=bucket, Key=sentinel + "-delete")),
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
                except (ClientError, BotoCoreError) as error:
                    results["cleanup_required_key"] = sentinel
                    code = error.response["Error"]["Code"] if isinstance(error, ClientError) else type(error).__name__
                    results["cleanup_error_code"] = (
                        code if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", code) else "ProviderError"
                    )
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
    parser.add_argument("--ca-bundle", type=Path)
    args = parser.parse_args()
    if not 0 <= args.wait_up_to <= 7200:
        parser.error("--wait-up-to must be between 0 and 7200 seconds")
    validate_storage_endpoint(args.endpoint)
    with tempfile.TemporaryDirectory(prefix="marin-auth-storage-") as directory:
        token_file = Path(directory) / "native.jwt"
        credentials = StorageCredentials(
            auth_url=args.auth_url,
            grant_token=require_secret("MARIN_GRANT_TOKEN"),
            binding=args.binding,
            token_file=token_file,
            ca_bundle=args.ca_bundle,
        )
        federation = CoreWeaveFederation(
            credentials.credentials_url, str(token_file), str(args.ca_bundle) if args.ca_bundle else None
        )
        metadata = CredentialExpirations(args.ca_bundle)
        session = native_session(federation, metadata)
        with (
            refreshing_storage_credentials(credentials) as check,
            closing(
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
            ) as s3,
        ):
            check()
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
                    check()
                    threading.Event().wait(min(1, max(0, deadline - time.monotonic())))
                if datetime.now(UTC) > metadata.first:
                    check()
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
