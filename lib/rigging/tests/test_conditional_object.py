# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from unittest.mock import MagicMock, patch

import fsspec
import pytest
from botocore.exceptions import ClientError
from rigging.filesystem.conditional_object import (
    ConditionalWriteError,
    GcsConditionalObject,
    S3ConditionalObject,
    UnsupportedConditionalWrite,
    conditional_object,
)
from rigging.filesystem.factory import filesystem
from rigging.filesystem.s3_compat import configure_fsspec_s3


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


@pytest.mark.parametrize("competing_creator", [False, True])
def test_local_conditional_create_publishes_complete_object(tmp_path, monkeypatch, competing_creator):
    path = str(tmp_path / "HEAD")
    first = conditional_object(path)
    second = conditional_object(path)
    opened, resume = Event(), Event()
    real_open = open

    def pause_before_first_write(filename, mode):
        handle = real_open(filename, mode)
        if mode in ("a+b", "xb") and not opened.is_set():
            opened.set()
            assert resume.wait(10)
        return handle

    monkeypatch.setattr("rigging.filesystem.conditional_object.open", pause_before_first_write, raising=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(first.write, b"first", expected_version=None)
        assert opened.wait(10)
        try:
            # An unfinished first write must not expose an empty object to a reader.
            assert second.read() is None
            assert second.version() is None
            if competing_creator:
                second.write(b"winner", expected_version=None)
        finally:
            resume.set()
        if competing_creator:
            with pytest.raises(ConditionalWriteError):
                writer.result()
        else:
            writer.result()
    result = second.read()
    assert result.data == (b"winner" if competing_creator else b"first")


def test_local_conditional_update_does_not_recreate_missing_object(tmp_path):
    path = tmp_path / "HEAD"
    obj = conditional_object(str(path))
    version = obj.write(b"original", expected_version=None)
    path.unlink()
    with pytest.raises(ConditionalWriteError):
        obj.write(b"stale", expected_version=version)
    assert obj.read() is None


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


def test_configured_credentials_reach_both_s3_clients_without_env_export(monkeypatch, tmp_path):
    for name in ("AWS_ENDPOINT_URL", "AWS_REGION", "AWS_DEFAULT_REGION", "FSSPEC_S3"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "unrelated-session-token")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.setitem(fsspec.config.conf, "s3", {})
    configure_fsspec_s3("https://objects.example.com", key="configured-key", secret="configured-secret")
    assert os.environ["AWS_ACCESS_KEY_ID"] == "ambient-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "ambient-secret"
    client = MagicMock()
    client.head_object.return_value = {"ETag": '"v1"'}
    session = MagicMock()
    session.create_client.return_value = client
    captured = {}

    def create_filesystem(_protocol, **options):
        captured.update(options)
        return MagicMock()

    monkeypatch.setattr(fsspec, "filesystem", create_filesystem)
    S3ConditionalObject._client.cache_clear()
    try:
        with patch("botocore.session.get_session", return_value=session):
            assert conditional_object("s3://bucket/HEAD").version() == '"v1"'
        filesystem("s3")
        assert session.create_client.call_args.kwargs["aws_access_key_id"] == "configured-key"
        assert session.create_client.call_args.kwargs["aws_secret_access_key"] == "configured-secret"
        assert "aws_session_token" not in session.create_client.call_args.kwargs
        assert captured["key"] == "configured-key"
        assert captured["secret"] == "configured-secret"
        assert "token" not in captured
        monkeypatch.setitem(fsspec.config.conf, "s3", {})
        captured.clear()
        filesystem("s3", key="explicit-key", secret="explicit-secret")
        assert captured["key"] == "explicit-key"
        assert captured["secret"] == "explicit-secret"
        assert "token" not in captured
    finally:
        S3ConditionalObject._client.cache_clear()


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


@pytest.mark.parametrize("operation", ["read", "version"])
def test_s3_conditional_object_retries_spurious_path_style_response(monkeypatch, operation):
    client = MagicMock()
    error = ClientError(
        {"Error": {"Code": "PathStyleRequestNotAllowed", "Message": "use virtual-host addressing"}},
        "GetObject" if operation == "read" else "HeadObject",
    )
    if operation == "read":
        client.get_object.side_effect = [error, {"Body": BytesIO(b"one"), "ETag": '"v1"'}]
    else:
        client.head_object.side_effect = [error, {"ETag": '"v1"'}]
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _delay: None)

    result = getattr(S3ConditionalObject("s3://bucket/HEAD"), operation)()

    if operation == "read":
        assert result is not None
        assert result.data == b"one"
    else:
        assert result == '"v1"'


def test_s3_conditional_object_maps_precondition_failure(monkeypatch):
    client = MagicMock()
    client.put_object.side_effect = ClientError(
        {"Error": {"Code": "PreconditionFailed", "Message": "stale"}}, "PutObject"
    )
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    with pytest.raises(ConditionalWriteError):
        S3ConditionalObject("s3://bucket/HEAD").write(b"two", expected_version='"v1"')


def test_s3_conditional_object_retries_slow_down(monkeypatch):
    client = MagicMock()
    client.get_object.return_value = {"Body": BytesIO(b"one"), "ETag": '"v1"'}
    client.put_object.side_effect = [
        ClientError({"Error": {"Code": "SlowDown", "Message": "reduce request rate"}}, "PutObject"),
        {"ETag": '"v2"'},
    ]
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _delay: None)

    version = S3ConditionalObject("s3://bucket/HEAD").write(b"two", expected_version='"v1"')

    assert version == '"v2"'
    assert client.put_object.call_count == 2


def test_s3_conditional_object_reconciles_accepted_put_after_transport_failure(monkeypatch):
    client = MagicMock()
    client.put_object.side_effect = TimeoutError("response lost")
    client.get_object.return_value = {"Body": BytesIO(b"two"), "ETag": '"v2"'}
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _path: client))
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _delay: None)

    version = S3ConditionalObject("s3://bucket/HEAD").write(b"two", expected_version='"v1"')

    assert version == '"v2"'
    client.put_object.assert_called_once()
    client.get_object.assert_called_once_with(Bucket="bucket", Key="HEAD")
