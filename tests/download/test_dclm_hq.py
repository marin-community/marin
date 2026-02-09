# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DCLM HQ download script that fetches HTML from Common Crawl."""

import json
from unittest.mock import patch

import zstandard as zstd
from marin.download.dclm_hq.download_dclm_hq_html import DCLMHQDownloadConfig, extract_dclm_hq_dump

SAMPLE_DCLM_RECORDS = [
    {
        "id": "test-doc-001",
        "source": "common-crawl",
        "text": "This is the original text that should be removed.",
        "metadata": {
            "WARC-Record-ID": "<urn:uuid:12345678-1234-1234-1234-123456789abc>",
            "WARC-Target-URI": "http://example.com/test-page",
            "warcinfo": (
                "warc-type: WARC/1.1\nWARC-Date: 2024-01-15T10:30:00Z\nisPartOf: CC-MAIN-2024-01\ndescription: Test WARC"
            ),
        },
    },
    {
        "id": "test-doc-002",
        "source": "common-crawl",
        "text": "This is another original text.",
        "metadata": {
            "WARC-Record-ID": "<urn:uuid:87654321-4321-4321-4321-cba987654321>",
            "WARC-Target-URI": "http://example.com/another-page",
            "warcinfo": (
                "warc-type: WARC/1.1\nWARC-Date: 2024-01-15T11:30:00Z\nisPartOf: CC-MAIN-2024-01\ndescription: Test WARC"
            ),
        },
    },
    {
        "id": "test-doc-003",
        "source": "common-crawl",
        "text": "Third document text.",
        "metadata": {
            "WARC-Record-ID": "<urn:uuid:11111111-2222-3333-4444-555555555555>",
            "WARC-Target-URI": "http://example.com/third-page",
            "warcinfo": (
                "warc-type: WARC/1.1\nWARC-Date: 2024-02-10T09:00:00Z\nisPartOf: CC-MAIN-2024-10\ndescription: Test WARC"
            ),
        },
    },
]

SAMPLE_WARC_HTML = {
    "http://example.com/test-page": (
        """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Test Article</h1>
<p>This is test content from Common Crawl.</p>
</body>
</html>"""
    ),
    "http://example.com/another-page": (
        """<!DOCTYPE html>
<html>
<head><title>Another Page</title></head>
<body>
<h1>Another Article</h1>
<p>Different content here.</p>
</body>
</html>"""
    ),
    "http://example.com/third-page": (
        """<!DOCTYPE html>
<html>
<head><title>Third Page</title></head>
<body>
<h1>Third Article</h1>
<p>More content.</p>
</body>
</html>"""
    ),
}


def create_warc_bytes(html_content: str) -> bytes:
    """Create minimal WARC record bytes for testing."""
    http_response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        f"Content-Length: {len(html_content.encode())}\r\n"
        "\r\n"
        f"{html_content}"
    )

    warc_header = (
        "WARC/1.0\r\n"
        "WARC-Type: response\r\n"
        "WARC-Record-ID: <urn:uuid:test-record>\r\n"
        "WARC-Target-URI: http://example.com/test\r\n"
        "Content-Type: application/http; msgtype=response\r\n"
        f"Content-Length: {len(http_response.encode())}\r\n"
        "\r\n"
    )

    full_warc = warc_header + http_response + "\r\n\r\n"
    return full_warc.encode()


def create_zstd_compressed_jsonl(records: list[dict]) -> bytes:
    """Create zstd compressed JSONL content."""
    jsonl_content = "\n".join(json.dumps(record) for record in records) + "\n"
    jsonl_bytes = jsonl_content.encode("utf-8")
    cctx = zstd.ZstdCompressor()
    return cctx.compress(jsonl_bytes)


def test_extract_dclm_hq_pipeline(tmp_path, read_all_jsonl_gz):
    """Test full DCLM HQ download pipeline with zephyr integration."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create input files in nested structure
    shard1_dir = tmp_path / "input" / "shard1"
    shard2_dir = tmp_path / "input" / "shard2"
    shard1_dir.mkdir(parents=True)
    shard2_dir.mkdir(parents=True)

    file1_data = create_zstd_compressed_jsonl([SAMPLE_DCLM_RECORDS[0]])
    file2_data = create_zstd_compressed_jsonl(SAMPLE_DCLM_RECORDS[1:])

    file1_path = shard1_dir / "file1.json.zst"
    file2_path = shard2_dir / "file2.json.zst"

    file1_path.write_bytes(file1_data)
    file2_path.write_bytes(file2_data)

    def mock_requests_get(url, **kwargs):
        from unittest.mock import Mock

        # Mock CC index server responses
        if "CC-MAIN-2024-01-index" in url:
            response = Mock()
            response.status_code = 200
            if "test-page" in url:
                response.text = json.dumps({"filename": "test.warc.gz", "offset": "0", "length": "1000"})
            else:  # another-page
                response.text = json.dumps({"filename": "test2.warc.gz", "offset": "0", "length": "1000"})
            response.raise_for_status = Mock()
            return response
        elif "CC-MAIN-2024-10-index" in url:
            response = Mock()
            response.status_code = 200
            response.text = json.dumps({"filename": "test3.warc.gz", "offset": "0", "length": "1000"})
            response.raise_for_status = Mock()
            return response
        # Mock Common Crawl WARC fetches
        elif "data.commoncrawl.org" in url:
            response = Mock()
            response.status_code = 200
            # Determine which HTML to return based on the WARC file
            if "test.warc.gz" in url:
                html_content = SAMPLE_WARC_HTML["http://example.com/test-page"]
            elif "test2.warc.gz" in url:
                html_content = SAMPLE_WARC_HTML["http://example.com/another-page"]
            else:  # test3.warc.gz
                html_content = SAMPLE_WARC_HTML["http://example.com/third-page"]
            response.content = create_warc_bytes(html_content)
            response.raise_for_status = Mock()
            return response

        raise ValueError(f"Unexpected URL: {url}")

    with patch("marin.download.dclm_hq.download_dclm_hq_html.requests.get", side_effect=mock_requests_get):
        cfg = DCLMHQDownloadConfig(input_path=str(tmp_path / "input"), output_path=str(output_dir))
        extract_dclm_hq_dump(cfg)

    # Verify output files were created in nested structure
    shard1_output = output_dir / "shard1"
    shard2_output = output_dir / "shard2"

    assert shard1_output.exists()
    assert shard2_output.exists()

    # Read all records
    all_records = []
    all_records.extend(read_all_jsonl_gz(shard1_output, "*.jsonl.gz"))
    all_records.extend(read_all_jsonl_gz(shard2_output, "*.jsonl.gz"))

    assert len(all_records) == 3

    # Verify records have HTML and no text
    for record in all_records:
        assert "id" in record
        assert "html" in record
        assert "text" not in record
        assert "metadata" in record
        assert len(record["html"]) > 0
