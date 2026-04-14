# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import zstandard as zstd

from marin.datakit.download.nemotron_v1 import NCC_PATHS_SUFFIX, download_nemotron_cc

SAMPLE_NEMOTRON_RECORDS = [
    {
        "warc_record_id": "record-001",
        "text": "This is the first document text.",
        "url": "https://example.com/page1",
        "language": "en",
        "score": 0.95,
    },
    {
        "warc_record_id": "record-002",
        "text": "This is the second document text.",
        "url": "https://example.com/page2",
        "language": "en",
        "score": 0.88,
    },
    {
        "warc_record_id": "record-003",
        "text": "This is the third document text.",
        "url": "https://example.com/page3",
        "language": "en",
        "score": 0.92,
    },
]


def _zstd_jsonl(records: list[dict]) -> bytes:
    jsonl = ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")
    return zstd.ZstdCompressor().compress(jsonl)


def _gzip_text(lines: list[str]) -> bytes:
    return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))


def _read_all_jsonl_zst(directory: Path, pattern: str = "*.jsonl.zst") -> list[dict]:
    records = []
    dctx = zstd.ZstdDecompressor()
    for file_path in sorted(directory.glob(pattern)):
        with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
            for line in reader.read().decode("utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


@pytest.fixture()
def local_ncc_server(tmp_path) -> Iterator[tuple[str, Path]]:
    """Serves `tmp_path / "server"` over HTTP as a stand-in for data.commoncrawl.org."""
    server_root = tmp_path / "server"
    server_root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_root), **kwargs)

        def log_message(self, format, *args):  # noqa: A002  # SimpleHTTPRequestHandler signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}", server_root
    finally:
        httpd.shutdown()
        thread.join()


def _publish(server_root: Path, relative_path: str, data: bytes) -> None:
    full = server_root / relative_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)


def test_download_nemotron_cc_pipeline(tmp_path, local_ncc_server):
    base_url, server_root = local_ncc_server
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    paths = ["contrib/Nemotron/file1.jsonl.zstd", "contrib/Nemotron/file2.jsonl.zstd"]
    _publish(server_root, NCC_PATHS_SUFFIX, _gzip_text(paths))
    _publish(server_root, paths[0], _zstd_jsonl([SAMPLE_NEMOTRON_RECORDS[0]]))
    _publish(server_root, paths[1], _zstd_jsonl(SAMPLE_NEMOTRON_RECORDS[1:]))

    download_nemotron_cc(str(output_dir), base_url=base_url)

    all_records = _read_all_jsonl_zst(output_dir / "contrib" / "Nemotron")

    assert len(all_records) == 3
    assert [r["id"] for r in all_records] == ["record-001", "record-002", "record-003"]
    for record in all_records:
        assert record["source"] == "nemotron"
        assert "metadata" in record


def test_download_nemotron_cc_dolma_format(tmp_path, local_ncc_server):
    base_url, server_root = local_ncc_server
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    nemotron_record = {
        "warc_record_id": "test-record-id",
        "text": "Document text content",
        "url": "https://example.com/doc",
        "language": "en",
        "language_score": 0.99,
        "perplexity": 245.3,
        "bucket": "high",
    }
    path = "contrib/Nemotron/test.jsonl.zstd"
    _publish(server_root, NCC_PATHS_SUFFIX, _gzip_text([path]))
    _publish(server_root, path, _zstd_jsonl([nemotron_record]))

    download_nemotron_cc(str(output_dir), base_url=base_url)

    records = _read_all_jsonl_zst(output_dir / "contrib" / "Nemotron")
    assert len(records) == 1

    dolma_record = records[0]
    assert dolma_record["id"] == "test-record-id"
    assert dolma_record["text"] == "Document text content"
    assert dolma_record["source"] == "nemotron"
    assert dolma_record["format"] == "text"

    metadata = dolma_record["metadata"]
    assert metadata["nemotron_url"] == "https://example.com/doc"
    assert metadata["nemotron_language"] == "en"
    assert metadata["nemotron_language_score"] == 0.99
    assert metadata["nemotron_perplexity"] == 245.3
    assert metadata["nemotron_bucket"] == "high"
    assert "nemotron_warc_record_id" not in metadata
    assert "nemotron_text" not in metadata


def test_download_nemotron_cc_skips_existing(tmp_path, local_ncc_server):
    base_url, server_root = local_ncc_server
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    path = "contrib/Nemotron/existing.jsonl.zstd"
    _publish(server_root, NCC_PATHS_SUFFIX, _gzip_text([path]))
    # Intentionally do NOT publish the data file — if the pipeline tries to fetch it,
    # the server will return 404 and the test will fail.

    existing_output = output_dir / "contrib" / "Nemotron" / "existing.jsonl.zst"
    existing_output.parent.mkdir(parents=True)
    existing_output.write_text("existing")

    download_nemotron_cc(str(output_dir), base_url=base_url)

    assert existing_output.read_text() == "existing"
