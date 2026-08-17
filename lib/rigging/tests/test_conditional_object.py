# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from rigging.filesystem.conditional_object import (
    ConditionalWriteError,
    GcsConditionalObject,
    S3ConditionalObject,
    UnsupportedConditionalWrite,
    conditional_object,
)


def test_local_conditional_object_rejects_stale_version(tmp_path):
    path = str(tmp_path / "HEAD")
    first = conditional_object(path)
    second = conditional_object(path)

    initial_version = first.write(b"one", expected_version=None)
    assert first.version() == initial_version
    stale = second.read()
    assert stale is not None
    assert stale.version == initial_version
    first.write(b"two", expected_version=initial_version)

    with pytest.raises(ConditionalWriteError):
        second.write(b"three", expected_version=stale.version)
    assert conditional_object(path).read().data == b"two"


def test_conditional_object_rejects_backend_without_compare_and_swap():
    with pytest.raises(UnsupportedConditionalWrite):
        conditional_object("memory://archive/HEAD")


def test_gcs_conditional_object_uses_generation_preconditions():
    current = MagicMock(generation=7)
    current.download_as_bytes.return_value = b"one"
    written = MagicMock(generation=8)
    bucket = MagicMock()
    bucket.get_blob.return_value = current
    bucket.blob.return_value = written
    client = MagicMock()
    client.bucket.return_value = bucket

    with patch("rigging.filesystem.conditional_object._gcs_client", return_value=client):
        obj = GcsConditionalObject("gs://bucket/HEAD")
        assert obj.version() == "7"
        assert obj.read().version == "7"
        assert obj.write(b"two", expected_version="7") == "8"
    current.download_as_bytes.assert_called_once_with(if_generation_match=7)
    written.upload_from_string.assert_called_once_with(b"two", if_generation_match=7)


def test_s3_conditional_object_sends_the_etag_precondition(monkeypatch):
    client = MagicMock()
    client.head_object.return_value = {"ETag": '"v1"'}
    client.get_object.return_value = {"Body": BytesIO(b"one"), "ETag": '"v1"'}
    client.put_object.return_value = {"ETag": '"v2"'}
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    obj = S3ConditionalObject("s3://bucket/HEAD")

    assert obj.version() == '"v1"'
    assert obj.read().version == '"v1"'
    assert obj.write(b"two", expected_version='"v1"') == '"v2"'
    client.put_object.assert_called_once_with(Bucket="bucket", Key="HEAD", Body=b"two", IfMatch='"v1"')


def test_s3_conditional_object_requires_absence_for_creation(monkeypatch):
    client = MagicMock()
    client.put_object.return_value = {"ETag": '"v1"'}
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))

    S3ConditionalObject("s3://bucket/HEAD").write(b"one", expected_version=None)

    client.put_object.assert_called_once_with(Bucket="bucket", Key="HEAD", Body=b"one", IfNoneMatch="*")


def test_s3_conditional_object_maps_missing_head_to_no_version(monkeypatch):
    client = MagicMock()
    client.head_object.side_effect = ClientError({"Error": {"Code": "NotFound", "Message": "missing"}}, "HeadObject")
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))

    assert S3ConditionalObject("s3://bucket/HEAD").version() is None


def test_s3_conditional_object_maps_precondition_failure(monkeypatch):
    client = MagicMock()
    client.put_object.side_effect = ClientError(
        {"Error": {"Code": "PreconditionFailed", "Message": "stale"}}, "PutObject"
    )
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    with pytest.raises(ConditionalWriteError):
        S3ConditionalObject("s3://bucket/HEAD").write(b"two", expected_version='"v1"')
