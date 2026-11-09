# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
from datetime import UTC, datetime, timedelta

import pytest
from botocore.exceptions import ClientError
from botocore.utils import ContainerMetadataFetcher
from rigging.filesystem.s3_compat import CoreWeaveFederation

from infra.auth.verify_storage import native_session, probe_access


class StorageBoundary:
    def __init__(self, *, outside_error="AccessDenied", write_allowed=False, delete_allowed=False):
        self.objects = {"approved/fixture": b"inside", "outside/fixture": b"outside"}
        self.outside_error = outside_error
        self.write_allowed = write_allowed
        self.delete_allowed = delete_allowed

    def get_object(self, *, Bucket, Key, Range):
        assert Range == "bytes=0-0"
        if Key.startswith("outside/"):
            raise ClientError(
                {"Error": {"Code": self.outside_error}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.objects[Key][:1])}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch):
        if not self.write_allowed:
            raise ClientError(
                {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "PutObject"
            )
        assert IfNoneMatch == "*"
        assert Key not in self.objects
        self.objects[Key] = Body

    def delete_object(self, *, Bucket, Key):
        if not self.delete_allowed:
            raise ClientError(
                {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "DeleteObject"
            )
        self.objects.pop(Key, None)


def _probe(storage):
    return probe_access(
        storage, bucket="test-only", allowed_key="approved/fixture", denied_key="outside/fixture", prefix="approved/"
    )


def test_read_only_proof_uses_small_probes_and_preserves_existing_fixtures():
    storage = StorageBoundary()
    report = _probe(storage)
    assert report["passed"]
    assert storage.objects == {"approved/fixture": b"inside", "outside/fixture": b"outside"}


@pytest.mark.parametrize("error", ["ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch", "NoSuchKey"])
def test_authentication_and_missing_fixtures_do_not_count_as_policy_denials(error):
    with pytest.raises(ClientError):
        _probe(StorageBoundary(outside_error=error))


@pytest.mark.parametrize("delete_allowed", [False, True])
def test_unexpected_write_is_cleaned_up_or_reports_only_its_own_sentinel(delete_allowed):
    storage = StorageBoundary(write_allowed=True, delete_allowed=delete_allowed)
    report = _probe(storage)
    assert not report["passed"]
    assert not report["write_denied"]
    if delete_allowed:
        assert storage.objects == {"approved/fixture": b"inside", "outside/fixture": b"outside"}
        assert not report["delete_denied"]
    else:
        sentinel = report["cleanup_required_key"]
        assert sentinel.startswith("approved/marin-auth-probe-")
        assert storage.objects[sentinel] == b"x"
        assert storage.objects["approved/fixture"] == b"inside"


class CredentialBoundary(ContainerMetadataFetcher):
    def retrieve_full_uri(self, full_url, headers=None):
        return {
            "AccessKeyId": "native-temporary-key",
            "SecretAccessKey": "native-temporary-secret",
            "Token": "",
            "Expiration": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        }


def test_native_sdk_ignores_static_keys_and_accepts_coreweave_empty_token(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "ambient-token")
    token_file = tmp_path / "native.jwt"
    token_file.write_text("identity-jwt")
    session = native_session(
        CoreWeaveFederation("https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc/org", str(token_file)),
        CredentialBoundary(),
    )
    frozen = session.get_credentials().get_frozen_credentials()
    assert frozen.access_key == "native-temporary-key"
    assert frozen.secret_key == "native-temporary-secret"
    assert frozen.token == ""
