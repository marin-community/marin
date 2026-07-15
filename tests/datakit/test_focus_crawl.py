# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from marin.datakit.normalize import generate_id
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from experiments.datakit.focus_crawl import FOCUS_CRAWL, extract_warc_paths


@pytest.fixture
def warc_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    server_root = tmp_path / "server"
    server_root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_root), **kwargs)

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server_root
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
      <p>This scientific report explains a carefully controlled experiment in enough detail for other researchers.</p>
      <p>The apparatus costs £300, and repeated measurements support the published conclusion.</p>
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

    pdf_headers = StatusAndHeaders("200 OK", [("Content-Type", "application/pdf")], protocol="HTTP/1.1")
    pdf_record = writer.create_warc_record(
        "https://example.com/report.pdf",
        "response",
        payload=io.BytesIO(b"not really a pdf"),
        http_headers=pdf_headers,
    )
    writer.write_record(pdf_record)
    return output.getvalue()


def test_extract_warc_paths_writes_clean_normalized_parquet(
    tmp_path: Path, warc_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, server_root = warc_server
    warc_path = "focus/example.warc.gz"
    source_path = server_root / warc_path
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_warc_bytes())
    monkeypatch.setenv("JUSTEXT_MODEL", "sklearn")
    monkeypatch.setenv("JUSTEXT_NO_DOWNLOAD", "1")

    result = extract_warc_paths(
        str(tmp_path / "output"),
        [warc_path],
        base_url=base_url,
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=1,
    )

    output_files = list(Path(result.main_output_dir).glob("*.parquet"))
    rows = pq.read_table(output_files).to_pylist()
    assert result.main_output_dir == str(tmp_path / "output" / "outputs" / "main")
    assert result.dup_output_dir == ""
    assert len(rows) == 1

    row = rows[0]
    assert "carefully controlled experiment" in row["text"]
    assert "£300" in row["text"]
    assert "Privacy policy" not in row["text"]
    assert row["id"] == generate_id(row["text"])
    assert row["source_id"] == "<urn:uuid:report>"
    assert row["source"] == FOCUS_CRAWL
    assert row["url"] == "https://example.com/report"
    assert row["warc_filename"] == warc_path
    assert row["content_encoding"] == "cp1252"
    assert result.counters["focus_crawl/documents"] == 1
    assert result.counters["focus_crawl/non_html_response"] == 1
