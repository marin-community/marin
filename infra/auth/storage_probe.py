# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native S3 probes run by a real Iris attempt, with public evidence only."""

import argparse
import json
import tempfile
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import jwt
from botocore.config import Config
from botocore.exceptions import ClientError
from rigging.filesystem.s3_compat import CoreWeaveFederation
from rigging.grants import Permission
from rigging.secrets import require_secret, secret_path

from infra.auth.client import StorageCredentials, refreshing_storage_credentials
from infra.auth.login import write_private
from infra.auth.verify_storage import CredentialExpirations, native_session, read_fixture

PROGRESS_MARKER = "AUTH_STORAGE_PROGRESS "
RESULT_MARKER = "AUTH_STORAGE_RESULT "
OBSERVER_TOKEN_FILE = "/tmp/auth-storage-observer.jwt"


@dataclass(frozen=True)
class StorageProbeConfig:
    binding: str
    endpoint: str
    region: str
    bucket: str
    prefix: str
    denied_key: str
    denied_write_prefix: str
    duration: float = 1050
    interval: float = 15

    def __post_init__(self):
        endpoint = urlsplit(self.endpoint)
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
            raise ValueError("The storage endpoint must be a CoreWeave object-storage HTTPS origin")
        Permission("cw_s3", frozenset({"s3:GetObject", "s3:PutObject"}), f"s3://{self.bucket}/{self.prefix}")
        if self.denied_key.startswith(self.prefix) or self.denied_write_prefix.startswith(self.prefix):
            raise ValueError("Denied fixtures must lie outside the granted prefix")
        if not self.prefix.endswith("/") or not self.denied_write_prefix.endswith("/"):
            raise ValueError("Probe prefixes must end in a slash")
        if not 900 <= self.duration <= 1200 or not 1 <= self.interval <= 30:
            raise ValueError("Proof requires 900-1200 seconds and a 1-30 second interval")


def storage_client(session, config: StorageProbeConfig):
    """Create a finite-request native S3 client with the supplied credential session."""
    return session.create_client(
        "s3",
        endpoint_url=config.endpoint,
        region_name=config.region,
        config=Config(
            connect_timeout=5,
            read_timeout=10,
            retries={"total_max_attempts": 1},
            s3={"addressing_style": "virtual"},
        ),
    )


def scope_denied(operation) -> bool:
    """Require an actual native policy denial; missing or expired keys do not count."""
    try:
        operation()
    except ClientError as error:
        if (
            error.response["Error"]["Code"] == "AccessDenied"
            and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403
        ):
            return True
        raise
    return False


def probe_cycle(s3, config: StorageProbeConfig, *, key: str, denied_write_key: str, payload: bytes) -> dict:
    """Verify read/write contents and native prefix/action limits on owned sentinels."""
    s3.put_object(Bucket=config.bucket, Key=key, Body=payload)
    with closing(s3.get_object(Bucket=config.bucket, Key=key)["Body"]) as body:
        if body.read(len(payload) + 1) != payload:
            raise RuntimeError("The native S3 round trip changed probe contents")
    checks = {
        "inside_read_write": True,
        "outside_read_denied": scope_denied(lambda: read_fixture(s3, config.bucket, config.denied_key)),
        "outside_write_denied": scope_denied(
            lambda: s3.put_object(Bucket=config.bucket, Key=denied_write_key, Body=b"probe", IfNoneMatch="*")
        ),
        "delete_denied": scope_denied(lambda: s3.delete_object(Bucket=config.bucket, Key=key)),
        "list_denied": scope_denied(lambda: s3.list_objects_v2(Bucket=config.bucket, Prefix=config.prefix, MaxKeys=1)),
    }
    if not all(checks.values()):
        raise RuntimeError("Native S3 exceeded its policy bounds: " + json.dumps(checks, sort_keys=True))
    return checks


def validate_progress(progress: dict, config: StorageProbeConfig) -> None:
    """Reject insufficient duration, missing boundaries, or unobserved credential renewal."""
    required = ("inside_read_write", "outside_read_denied", "outside_write_denied", "delete_denied", "list_denied")
    if not all(progress.get(name) is True for name in required):
        raise RuntimeError("The task did not prove every native storage boundary")
    if progress["elapsed"] < 900 or progress["cycles"] < 2:
        raise RuntimeError("The task did not sustain storage operations for 15 minutes")
    if progress["grant_rotations"] < 2 or not progress["native_rotated"]:
        raise RuntimeError("The task did not observe both grant and native credential renewal")
    if progress["observed_at"] <= progress["initial_provider_expires_at"]:
        raise RuntimeError("The task did not perform storage operations after its initial native expiry")
    if progress["elapsed"] > config.duration + 60:
        raise RuntimeError("The task exceeded its proof time bound")


def run_probe(config: StorageProbeConfig) -> None:
    """Run small native operations repeatedly, renewing credentials from the attempt grant."""
    identifier = uuid.uuid4().hex
    key = config.prefix + "probe-" + identifier
    denied_write_key = config.denied_write_prefix + "probe-" + identifier
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="auth-storage-probe-") as directory:
        token_file = Path(directory) / "native.jwt"
        credentials = StorageCredentials(
            auth_url=require_secret("MARIN_AUTH_URL"),
            grant_token=require_secret("MARIN_GRANT_TOKEN"),
            binding=config.binding,
            token_file=token_file,
            ca_bundle=secret_path("AUTH_PROOF_CA"),
        )
        metadata = CredentialExpirations(secret_path("AUTH_PROOF_CA"))
        federation = CoreWeaveFederation(
            credentials.credentials_url,
            str(token_file),
        )
        session = native_session(federation, metadata)
        previous = credentials.grant.get_token()
        rotations = 0
        cycles = 0
        with refreshing_storage_credentials(credentials) as check, closing(storage_client(session, config)) as s3:
            while time.monotonic() - started < config.duration:
                check()
                token = credentials.grant.get_token()
                write_private(Path(OBSERVER_TOKEN_FILE), token)
                if token != previous:
                    rotations += 1
                    previous = token
                checks = probe_cycle(
                    s3, config, key=key, denied_write_key=denied_write_key, payload=f"cycle-{cycles}".encode()
                )
                cycles += 1
                assert metadata.first is not None and metadata.latest is not None
                progress = {
                    **checks,
                    "key": key,
                    "denied_write_key": denied_write_key,
                    "elapsed": time.monotonic() - started,
                    "observed_at": time.time(),
                    "cycles": cycles,
                    "grant_rotations": rotations,
                    "grant_expires_at": jwt.decode(token, options={"verify_signature": False})["exp"],
                    "initial_provider_expires_at": metadata.first.timestamp(),
                    "latest_provider_expires_at": metadata.latest.timestamp(),
                    "native_rotated": metadata.rotated,
                }
                print(PROGRESS_MARKER + json.dumps(progress, sort_keys=True), flush=True)
                threading.Event().wait(min(config.interval, max(0, config.duration - (time.monotonic() - started))))
            validate_progress(progress, config)
            print(RESULT_MARKER + json.dumps(progress, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    run_probe(StorageProbeConfig(**json.loads(args.config_json)))


if __name__ == "__main__":
    main()
