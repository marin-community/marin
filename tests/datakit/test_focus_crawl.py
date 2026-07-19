# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from marin.datakit.normalize import generate_id
from warcio.archiveiterator import ArchiveIterator
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from experiments.datakit.focus_crawl import (
    FOCUS_CRAWL,
    download_warc_text_records,
    extract_warc_paths,
    focus_crawl_warc_paths,
)


@pytest.fixture
def warc_server(tmp_path: Path) -> Iterator[tuple[str, Path, list[tuple[int, int]], dict[str, int]]]:
    server_root = tmp_path / "server"
    server_root.mkdir()
    requested_ranges: list[tuple[int, int]] = []
    forbidden_requests: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if forbidden_requests.get(self.path, 0):
                forbidden_requests[self.path] -= 1
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            payload = (server_root / self.path.lstrip("/")).read_bytes()
            range_header = self.headers.get("Range")
            if range_header:
                bounds = range_header.removeprefix("bytes=").split("-", 1)
                start, stop = int(bounds[0]), int(bounds[1]) + 1
                requested_ranges.append((start, stop))
            else:
                start, stop = 0, len(payload)
            body = payload[start:stop]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server_root, requested_ranges, forbidden_requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _warc_bytes() -> bytes:
    html = """<!doctype html>
<html>
  <body>
    <nav><a href=\"/\">Home</a> <a href=\"/login\">Sign in</a></nav>
    <article>
      <h1>Focused science report</h1>
      <p>A carefully controlled\x00 experiment is described in enough detail for other researchers.</p>
      <p>The apparatus costs £300, and repeated measurements support the published conclusion.</p>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Control</td><td>encoded&#1;character</td></tr>
        <tr><td>Sample</td><td>complete</td></tr>
      </table>
      <table>
        <tr><th>Day</th><th>Count</th></tr>
        <tr><td>&#9312;</td><td>2</td><td>3</td></tr>
        <tr><td>4</td><td>5</td><td>6</td></tr>
      </table>
    </article>
    <footer>Privacy policy and cookie settings</footer>
  </body>
</html>"""
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    html_headers = StatusAndHeaders(
        "200 OK",
        [("Content-Type", "text/html")],
        protocol="HTTP/1.1",
    )
    html_record = writer.create_warc_record(
        "https://example.com/report",
        "response",
        payload=io.BytesIO(html.encode("cp1252")),
        warc_headers_dict={
            "WARC-Record-ID": "<urn:uuid:report>",
            "WARC-Date": "2026-06-01T00:00:00Z",
        },
        http_headers=html_headers,
    )
    writer.write_record(html_record)

    malformed_record = writer.create_warc_record(
        "https://example.com/malformed",
        "response",
        payload=io.BytesIO(b"<button><form><p>Malformed document</p></form></button>"),
        warc_headers_dict={"WARC-Record-ID": "<urn:uuid:malformed>"},
        http_headers=html_headers,
    )
    writer.write_record(malformed_record)

    skipped_record = writer.create_warc_record(
        "https://ceres.larc.nasa.gov/science/publications/",
        "response",
        payload=io.BytesIO(b"<article><p>This record must not be extracted.</p></article>"),
        warc_headers_dict={"WARC-Record-ID": "<urn:uuid:88562b9a-a0da-40be-a939-330f53017c9d>"},
        http_headers=html_headers,
    )
    writer.write_record(skipped_record)

    pdf_headers = StatusAndHeaders("200 OK", [("Content-Type", "application/pdf")], protocol="HTTP/1.1")
    pdf_record = writer.create_warc_record(
        "https://example.com/report.pdf",
        "response",
        payload=io.BytesIO(b"not really a pdf"),
        http_headers=pdf_headers,
    )
    writer.write_record(pdf_record)
    return output.getvalue()


def _cdx_bytes(warc_path: str, warc: bytes) -> bytes:
    lines = []
    archive = ArchiveIterator(io.BytesIO(warc))
    for record in archive:
        content_type = record.http_headers.get_header("Content-Type")
        entry = {
            "url": record.rec_headers.get_header("WARC-Target-URI"),
            "mime": content_type,
            "mime-detected": content_type,
            "status": record.http_headers.get_statuscode(),
            "offset": str(archive.get_record_offset()),
            "length": str(archive.get_record_length()),
            "filename": warc_path,
        }
        lines.append(f"example,)/ 20260601000000 {json.dumps(entry)}\n")
    return gzip.compress("".join(lines).encode())


def test_extract_warc_paths_retries_transient_403_and_writes_clean_normalized_parquet(
    tmp_path: Path,
    warc_server: tuple[str, Path, list[tuple[int, int]], dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, server_root, requested_ranges, forbidden_requests = warc_server
    warc_path = "focus/warc/example.warc.gz"
    source_path = server_root / warc_path
    source_path.parent.mkdir(parents=True)
    warc = _warc_bytes()
    source_path.write_bytes(warc)
    cdx_path = server_root / "focus/cdx/warc/example.cdx.gz"
    cdx_path.parent.mkdir(parents=True)
    cdx_path.write_bytes(_cdx_bytes(warc_path, warc))
    index_path = server_root / "focus/index.parquet"
    pq.write_table(pa.table({"warc_filename": [warc_path]}), index_path)
    forbidden_requests["/focus/index.parquet"] = 1
    forbidden_requests["/focus/cdx/warc/example.cdx.gz"] = 1
    forbidden_requests["/focus/warc/example.warc.gz"] = 1
    monkeypatch.setenv("JUSTEXT_MODEL", "sklearn")
    monkeypatch.setenv("JUSTEXT_NO_DOWNLOAD", "1")

    assert focus_crawl_warc_paths(f"{base_url}/focus/index.parquet", 1) == [warc_path]
    retried_rows = list(download_warc_text_records(warc_path, base_url))
    assert len(retried_rows) == 1
    requested_ranges.clear()

    result = extract_warc_paths(
        str(tmp_path / "output"),
        [warc_path],
        base_url=base_url,
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        task_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=1,
    )

    output_files = list(Path(result.main_output_dir).glob("*.parquet"))
    rows = pq.read_table(output_files).to_pylist()
    assert result.main_output_dir == str(tmp_path / "output" / "outputs" / "main")
    assert result.dup_output_dir == ""
    assert len(rows) == 1

    row = rows[0]
    assert "carefully controlled experiment" in row["text"]
    assert "encodedcharacter" in row["text"]
    assert "£300" in row["text"]
    assert "Privacy policy" not in row["text"]
    assert row["id"] == generate_id(row["text"])
    assert row["source_id"] == "<urn:uuid:report>"
    assert row["source"] == FOCUS_CRAWL
    assert row["url"] == "https://example.com/report"
    assert row["warc_filename"] == warc_path
    assert row["content_encoding"] == "cp1252"
    assert result.counters["focus_crawl/documents"] == 1
    assert result.counters["focus_crawl/normalized_compatibility_digit"] == 1
    assert result.counters["focus_crawl/xml_invalid_character"] == 2
    assert result.counters["focus_crawl/justext_assertion"] == 1
    assert result.counters["focus_crawl/skipped_record"] == 1
    assert "focus_crawl/non_html_response" not in result.counters
    assert len(requested_ranges) == 1
    assert requested_ranges[0][0] == 0
    assert requested_ranges[0][1] < len(warc)
    assert not any(forbidden_requests.values())
