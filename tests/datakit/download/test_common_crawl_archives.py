# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from marin.datakit.download.common_crawl_archives import (
    COMMON_CRAWL_CRAWL_ID,
    COMMON_CRAWL_WARC_SOURCE,
    COMMON_CRAWL_WAT_SOURCE,
    CommonCrawlArchiveKind,
    download_common_crawl_sample,
)


@pytest.fixture()
def local_common_crawl_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    server_root = tmp_path / "server"
    server_root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_root), **kwargs)

        def log_message(self, format, *args):  # noqa: A002  # stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}/", server_root
    finally:
        httpd.shutdown()
        thread.join()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_gzip_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)


def _warc_record(headers: dict[str, str], body: bytes, *, version: str = "WARC/1.0") -> bytes:
    merged_headers = {"Content-Length": str(len(body)), **headers}
    header_block = "".join(f"{key}: {value}\r\n" for key, value in merged_headers.items())
    return (f"{version}\r\n{header_block}\r\n").encode() + body + b"\r\n\r\n"


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_download_common_crawl_warc_streams_selected_segments_and_skips_binary(
    tmp_path: Path,
    local_common_crawl_server,
) -> None:
    base_url, server_root = local_common_crawl_server
    warc_paths = [
        f"crawl-data/{COMMON_CRAWL_CRAWL_ID}/segments/100/warc/CC-MAIN-00000.warc.gz",
        f"crawl-data/{COMMON_CRAWL_CRAWL_ID}/segments/100/warc/CC-MAIN-00001.warc.gz",
        f"crawl-data/{COMMON_CRAWL_CRAWL_ID}/segments/200/warc/CC-MAIN-00002.warc.gz",
    ]
    _write_gzip_bytes(
        server_root / "crawl-data" / COMMON_CRAWL_CRAWL_ID / "warc.paths.gz",
        "\n".join(warc_paths).encode(),
    )

    first_file = b"".join(
        (
            _warc_record(
                {
                    "WARC-Type": "response",
                    "WARC-Target-URI": "https://example.com/",
                    "Content-Type": "application/http; msgtype=response",
                },
                (
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
                    b"<html><body>Hello Common Crawl</body></html>"
                ),
            ),
            _warc_record(
                {
                    "WARC-Type": "response",
                    "WARC-Target-URI": "https://example.com/logo.png",
                    "Content-Type": "application/http; msgtype=response",
                },
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\r\n\x89PNG\x00\x01\x02",
            ),
        )
    )
    third_file = _warc_record(
        {
            "WARC-Type": "request",
            "WARC-Target-URI": "https://other.example/robots.txt",
            "Content-Type": "application/http; msgtype=request",
        },
        b"GET /robots.txt HTTP/1.1\r\nHost: other.example\r\n\r\n",
    )

    _write_gzip_bytes(server_root / warc_paths[0], first_file)
    _write_gzip_bytes(server_root / warc_paths[1], first_file)
    _write_gzip_bytes(server_root / warc_paths[2], third_file)

    output_dir = tmp_path / "output"
    source = replace(COMMON_CRAWL_WARC_SOURCE, base_url=base_url, max_files=2)
    result = download_common_crawl_sample(source=source, output_path=str(output_dir))

    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert len(records) == 2
    assert [record["source_file"] for record in records] == [warc_paths[0], warc_paths[2]]
    assert "WARC-Target-URI: https://example.com/" in records[0]["text"]
    assert "<html><body>Hello Common Crawl</body></html>" in records[0]["text"]
    assert "GET /robots.txt HTTP/1.1" in records[1]["text"]
    assert result["selected_paths"] == [warc_paths[0], warc_paths[2]]
    assert metadata["materialized_output"]["record_count"] == 2
    assert metadata["materialized_output"]["metadata"]["counters"]["records_skipped_nontxt"] == 1


def test_download_common_crawl_wat_preserves_json_and_honors_output_cap(
    tmp_path: Path,
    local_common_crawl_server,
) -> None:
    base_url, server_root = local_common_crawl_server
    wat_paths = [
        f"crawl-data/{COMMON_CRAWL_CRAWL_ID}/segments/300/wat/CC-MAIN-10000.warc.wat.gz",
    ]
    _write_gzip_bytes(server_root / "crawl-data" / COMMON_CRAWL_CRAWL_ID / "wat.paths.gz", "\n".join(wat_paths).encode())

    wat_body = json.dumps(
        {
            "Envelope": {
                "WARC-Header-Metadata": {"WARC-Target-URI": "https://example.net/"},
                "Payload-Metadata": {
                    "HTTP-Response-Metadata": {
                        "HTML-Metadata": {
                            "Head": {"Title": "Example"},
                            "Extracted-Text": "lorem ipsum " * 60,
                        }
                    }
                },
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    wat_file = b"".join(
        _warc_record(
            {
                "WARC-Type": "metadata",
                "WARC-Target-URI": f"https://example.net/{index}",
                "Content-Type": "application/json",
            },
            wat_body,
        )
        for index in range(3)
    )
    _write_gzip_bytes(server_root / wat_paths[0], wat_file)

    output_dir = tmp_path / "output"
    source = replace(
        COMMON_CRAWL_WAT_SOURCE,
        base_url=base_url,
        max_files=1,
        max_output_bytes=1_500,
    )
    result = download_common_crawl_sample(source=source, output_path=str(output_dir))

    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert records
    assert len(records) < 3
    assert all(record["warc_type"] == "metadata" for record in records)
    assert '"Extracted-Text"' in records[0]["text"]
    assert '"Title":"Example"' in records[0]["text"]
    assert result["bytes_written"] <= 1_500
    assert metadata["materialized_output"]["metadata"]["archive_kind"] == CommonCrawlArchiveKind.WAT.value
    assert metadata["materialized_output"]["record_count"] == len(records)
