# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filer's read-only API over real fixture files at the storage boundary."""

import gzip
import json
import zipfile

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from infra.marina.applets.filer.server import app as filer


@pytest.fixture
def storage_client(tmp_path, monkeypatch):
    fs = fsspec.filesystem("file")

    def route(url):
        return fs, str(tmp_path / url.split("/", 3)[3])

    monkeypatch.setattr(filer, "filesystem_for", route)
    monkeypatch.setattr("rigging.fsutil.listing.filesystem_for", route)
    return TestClient(filer.create_api(None)), tmp_path


def test_parquet_pages_preserve_nested_records_across_groups(storage_client):
    client, root = storage_client
    records = [{"id": i, "messages": [{"role": "user", "content": f"SELECT {i}"}]} for i in range(13)]
    pq.write_table(pa.Table.from_pylist(records), root / "rows.parquet", row_group_size=4)
    result = client.get("/inspect", params={"url": "s3://bucket/rows.parquet", "offset": 3, "limit": 6}).json()
    assert result["rows"] == records[3:9]
    assert result["total"] == 13
    assert result["next_offset"] == 9
    projected = client.get(
        "/inspect", params={"url": "s3://bucket/rows.parquet", "offset": 9, "columns": json.dumps(["id"])}
    ).json()
    assert projected["rows"] == [{"id": i} for i in range(9, 13)]
    assert projected["next_offset"] is None


def test_parquet_large_group_allows_schema_then_narrow_projection(storage_client, monkeypatch):
    client, root = storage_client
    pq.write_table(pa.table({"id": [1, 2], "body": ["x" * 10000, "y" * 10000]}), root / "large.parquet")
    monkeypatch.setattr(filer, "MAX_GROUP_BYTES", 1000)
    initial = client.get("/inspect", params={"url": "s3://bucket/large.parquet"}).json()
    assert initial["rows"] == []
    assert [column["name"] for column in initial["schema"]] == ["id", "body"]
    projected = client.get("/inspect", params={"url": "s3://bucket/large.parquet", "columns": '["id"]'}).json()
    assert projected["rows"] == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        (
            "rows.csv.gz",
            gzip.compress(b'name,body\na,"line one\nline two"\n'),
            [{"name": "a", "body": "line one\nline two"}],
        ),
        ("rows.jsonl", b'{"nested":{"a":[1,2]}}\n{"nested":null}\n', [{"nested": {"a": [1, 2]}}, {"nested": None}]),
        ("rows.tsv", b"name\tvalue\na\t7\n", [{"name": "a", "value": "7"}]),
    ],
)
def test_structured_text_preview_preserves_records(storage_client, filename, content, expected):
    client, root = storage_client
    (root / filename).write_bytes(content)
    result = client.get("/inspect", params={"url": f"gs://bucket/{filename}"}).json()
    assert result["rows"] == expected


def test_archive_inspection_lists_without_extracting(storage_client):
    client, root = storage_client
    with zipfile.ZipFile(root / "archive.zip", "w") as archive:
        archive.writestr("../../escape.txt", "secret")
    result = client.get("/inspect", params={"url": "s3://bucket/archive.zip"}).json()
    assert result["rows"] == [{"name": "../../escape.txt", "size": 6, "compressed_size": 6}]
    assert list(root.iterdir()) == [root / "archive.zip"]


def test_local_and_network_urls_cannot_expose_server_files(storage_client):
    client, root = storage_client
    (root / "credentials").write_text("private")
    for url in (str(root / "credentials"), "file:///etc/passwd", "http://metadata.google.internal/"):
        response = client.get("/inspect", params={"url": url})
        assert response.status_code == 400
        assert "private" not in response.text


def test_json_array_projection_pages_without_returning_whole_document(storage_client):
    client, root = storage_client
    (root / "rows.json").write_text(json.dumps([{"id": i, "text": "prompt"} for i in range(100)]))
    result = client.get(
        "/inspect", params={"url": "gs://bucket/rows.json", "offset": 50, "limit": 2, "columns": '["id"]'}
    ).json()
    assert result["rows"] == [{"id": 50}, {"id": 51}]
    assert result["columns"] == ["id"]
    assert "value" not in result


def test_media_preview_returns_bytes_and_enforces_size_limit(storage_client, monkeypatch):
    client, root = storage_client
    data = b"%PDF-1.7\nexample"
    (root / "document.pdf").write_bytes(data)
    inspected = client.get("/inspect", params={"url": "s3://bucket/document.pdf"}).json()
    assert inspected["kind"] == "pdf"
    response = client.get("/media", params={"url": "s3://bucket/document.pdf"})
    assert response.content == data
    assert response.headers["content-type"] == "application/pdf"
    monkeypatch.setattr(filer, "MAX_MEDIA_BYTES", 5)
    assert client.get("/media", params={"url": "s3://bucket/document.pdf"}).status_code == 413
