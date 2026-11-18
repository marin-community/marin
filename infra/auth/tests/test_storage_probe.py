# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io

import pytest
from botocore.exceptions import ClientError

from infra.auth.prove_storage_execution import latest_progress
from infra.auth.storage_probe import StorageProbeConfig, probe_cycle, validate_progress


@pytest.fixture
def config():
    return StorageProbeConfig(
        binding="proof",
        endpoint="https://cwobject.com",
        region="auto",
        bucket="test",
        prefix="owned/allowed/",
        denied_key="owned/sibling/fixture",
        denied_write_prefix="owned/sibling/",
    )


class NativeBoundary:
    def __init__(self, *, outside_error="AccessDenied", permitted=()):
        self.objects = {}
        self.outside_error = outside_error
        self.permitted = permitted

    def deny(self, operation, code="AccessDenied"):
        if operation not in self.permitted:
            raise ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": 403}}, operation)

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if Key.startswith("owned/sibling/"):
            self.deny("PutObject")
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key, **kwargs):
        if Key.startswith("owned/sibling/"):
            self.deny("GetObject", self.outside_error)
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, *, Bucket, Key):
        self.deny("DeleteObject")
        self.objects.pop(Key, None)

    def list_objects_v2(self, **kwargs):
        self.deny("ListBucket")
        return {"Contents": list(self.objects)}


def test_native_round_trip_checks_every_boundary_and_keeps_only_owned_sentinel(config):
    storage = NativeBoundary()
    report = probe_cycle(
        storage, config, key="owned/allowed/probe", denied_write_key="owned/sibling/probe", payload=b"content"
    )
    assert all(report.values())
    assert storage.objects == {"owned/allowed/probe": b"content"}


@pytest.mark.parametrize("operation", ["PutObject", "DeleteObject", "ListBucket"])
def test_native_proof_rejects_excess_permissions(config, operation):
    with pytest.raises(RuntimeError, match="policy bounds"):
        probe_cycle(
            NativeBoundary(permitted=(operation,)),
            config,
            key="owned/allowed/probe",
            denied_write_key="owned/sibling/probe",
            payload=b"content",
        )


@pytest.mark.parametrize("code", ["NoSuchKey", "ExpiredToken", "SignatureDoesNotMatch"])
def test_native_proof_never_counts_missing_or_expired_objects_as_scope_denials(config, code):
    with pytest.raises(ClientError):
        probe_cycle(
            NativeBoundary(outside_error=code),
            config,
            key="owned/allowed/probe",
            denied_write_key="owned/sibling/probe",
            payload=b"content",
        )


def test_evidence_requires_observed_renewal_and_sustained_success(config):
    evidence = {
        "inside_read_write": True,
        "outside_read_denied": True,
        "outside_write_denied": True,
        "delete_denied": True,
        "list_denied": True,
        "elapsed": 910,
        "cycles": 61,
        "grant_rotations": 4,
        "native_rotated": True,
        "observed_at": 2000,
        "initial_provider_expires_at": 1900,
    }
    validate_progress(evidence, config)
    for change in ({"elapsed": 100}, {"native_rotated": False}, {"initial_provider_expires_at": 2100}):
        with pytest.raises(RuntimeError):
            validate_progress({**evidence, **change}, config)
    assert latest_progress('noise\nAUTH_STORAGE_PROGRESS {"cycles": 2}\nnoise') == {"cycles": 2}
