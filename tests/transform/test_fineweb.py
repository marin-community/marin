# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FineWeb WARC processing."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter
from zephyr import create_backend, set_flow_backend

from marin.schemas.web.convert import HtmlToMarkdownConfig
from marin.transform.fineweb.process_parquet_fw import process_one_warc_file

SAMPLE_HTML_PAGES = {
    "http://example.com/page1": (
        """
        <!DOCTYPE html>
        <html>
        <head><title>Page 1</title></head>
        <body>
            <h1>Welcome to Page 1</h1>
            <p>This is the first test page with some content.</p>
            <ul>
                <li>Item A</li>
                <li>Item B</li>
            </ul>
        </body>
        </html>
    """
    ),
    "http://example.com/page2": (
        """
        <!DOCTYPE html>
        <html>
        <head><title>Page 2</title></head>
        <body>
            <h1>Page 2 Title</h1>
            <p>Another test page with different content.</p>
            <p>Multiple paragraphs here.</p>
        </body>
        </html>
    """
    ),
    "http://example.com/page3": (
        """
        <!DOCTYPE html>
        <html>
        <head><title>Page 3</title></head>
        <body>
            <h1>Third Page</h1>
            <p>Final test page for validation.</p>
        </body>
        </html>
    """
    ),
}


@pytest.fixture
def create_warc_file():
    """Create WARC files with HTML pages for testing."""

    def _create(warc_path: Path, html_pages: dict[str, str]) -> Path:
        warc_path.parent.mkdir(parents=True, exist_ok=True)

        # Create WARC file
        with open(warc_path, "wb") as output:
            writer = WARCWriter(output, gzip=False)

            for url, html_content in html_pages.items():
                # Create HTTP headers
                http_headers = StatusAndHeaders(
                    "200 OK",
                    [
                        ("Content-Type", "text/html"),
                        ("Content-Length", str(len(html_content))),
                    ],
                    protocol="HTTP/1.1",
                )

                # Create WARC record headers
                record_headers = {
                    "WARC-Target-URI": url,
                    "WARC-Date": "2024-01-01T00:00:00Z",
                }

                # Write response record
                record = writer.create_warc_record(
                    uri=url,
                    record_type="response",
                    payload=BytesIO(html_content.encode("utf-8")),
                    http_headers=http_headers,
                    warc_headers_dict=record_headers,
                )
                writer.write_record(record)

        return warc_path

    return _create


@pytest.fixture
def create_parquet_with_warc_refs(create_warc_file, tmp_path):
    """Create parquet files with WARC references for testing."""

    def _create(parquet_path: Path, warc_name: str, urls: list[str], html_pages: dict[str, str]) -> tuple[Path, Path]:
        # Create WARC file
        warc_path = tmp_path / "warcs" / warc_name
        create_warc_file(warc_path, html_pages)

        # Create parquet with references to WARC
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for idx, url in enumerate(urls):
            records.append(
                {
                    "id": f"fineweb_{idx}",
                    "url": url,
                    "file_path": f"file://{warc_path.resolve()}",  # Use file:// URL for local testing
                    "text": f"Original text {idx}",  # FineWeb includes original text
                    "token_count": 100 + idx,
                    "language": "en",
                    "language_score": 0.99,
                }
            )

        df = pd.DataFrame(records)
        df.to_parquet(parquet_path)

        return parquet_path, warc_path

    return _create


def test_process_one_warc_file_basic(tmp_path, create_parquet_with_warc_refs, read_jsonl_gz, validate_dolma_record):
    """Test basic WARC processing with URL matching."""
    # Use sync backend for testing
    set_flow_backend(create_backend("sync"))

    # Create test data
    urls = list(SAMPLE_HTML_PAGES.keys())
    parquet_path = tmp_path / "input" / "test.parquet"
    md_output_path = tmp_path / "output" / "md"
    md_output_path.mkdir(parents=True, exist_ok=True)

    parquet_path, _warc_path = create_parquet_with_warc_refs(parquet_path, "test.warc", urls, SAMPLE_HTML_PAGES)

    # Read the parquet file to get the DataFrame
    df = pd.read_parquet(parquet_path)

    # Process WARC file
    output_path = tmp_path / "output" / "processed.jsonl.gz"
    config = HtmlToMarkdownConfig.default_config()

    result = process_one_warc_file(
        df,  # df parameter
        str(parquet_path),  # input_path (for logging)
        str(output_path),  # output_path
        "readability",  # extract_method
        config,  # config
        str(md_output_path),  # md_output_path
    )

    assert result is True

    # Verify output - output file is at md_output_path/output/processed.jsonl.gz
    # because base_folder is extracted from parent of output_path
    output_file = md_output_path / "output" / "processed.jsonl.gz"
    assert output_file.exists()

    records = read_jsonl_gz(output_file)
    assert len(records) == len(urls)

    # Validate each record
    for record in records:
        validate_dolma_record(record)
        assert record["source"] == "fineweb"
        assert record["format"] == "md"
        assert "text" in record
        assert len(record["text"]) > 0  # Should have extracted content
        assert "metadata" in record
        # Verify FineWeb metadata is preserved with fw_ prefix
        assert "fw_url" in record["metadata"]
        assert "fw_token_count" in record["metadata"]
        assert "fw_language" in record["metadata"]


def test_process_one_warc_file_url_filtering(tmp_path, create_parquet_with_warc_refs, read_jsonl_gz):
    """Test that only URLs in parquet are processed from WARC."""
    set_flow_backend(create_backend("sync"))

    # Only include subset of URLs in parquet
    all_urls = list(SAMPLE_HTML_PAGES.keys())
    selected_urls = all_urls[:2]  # Only first 2 URLs

    parquet_path = tmp_path / "input" / "test.parquet"
    md_output_path = tmp_path / "output" / "md"
    md_output_path.mkdir(parents=True, exist_ok=True)

    # Create WARC with all pages but parquet with only subset
    parquet_path, _warc_path = create_parquet_with_warc_refs(parquet_path, "test.warc", selected_urls, SAMPLE_HTML_PAGES)

    # Read the parquet file to get the DataFrame
    df = pd.read_parquet(parquet_path)

    output_path = tmp_path / "output" / "processed.jsonl.gz"
    config = HtmlToMarkdownConfig.default_config()

    result = process_one_warc_file(
        df,
        str(parquet_path),
        str(output_path),
        "readability",
        config,
        str(md_output_path),
    )

    assert result is True

    # Verify only selected URLs were processed
    output_file = md_output_path / "output" / "processed.jsonl.gz"
    records = read_jsonl_gz(output_file)
    assert len(records) == len(selected_urls)

    # Verify correct URLs
    processed_urls = {record["metadata"]["fw_url"] for record in records}
    assert processed_urls == set(selected_urls)


def test_process_one_warc_file_metadata_preservation(tmp_path, create_parquet_with_warc_refs, read_jsonl_gz):
    """Test that FineWeb metadata is properly preserved with fw_ prefix."""
    set_flow_backend(create_backend("sync"))

    urls = list(SAMPLE_HTML_PAGES.keys())[:1]  # Just one URL for simplicity
    parquet_path = tmp_path / "input" / "test.parquet"
    md_output_path = tmp_path / "output" / "md"
    md_output_path.mkdir(parents=True, exist_ok=True)

    parquet_path, _warc_path = create_parquet_with_warc_refs(parquet_path, "test.warc", urls, SAMPLE_HTML_PAGES)

    # Read the original parquet to get expected metadata
    df = pd.read_parquet(parquet_path)
    original_record = df.iloc[0].to_dict()

    output_path = tmp_path / "output" / "processed.jsonl.gz"
    config = HtmlToMarkdownConfig.default_config()

    result = process_one_warc_file(
        df,
        str(parquet_path),
        str(output_path),
        "readability",
        config,
        str(md_output_path),
    )

    assert result is True

    output_file = md_output_path / "output" / "processed.jsonl.gz"
    records = read_jsonl_gz(output_file)
    assert len(records) == 1

    record = records[0]
    metadata = record["metadata"]

    # Verify all original fields (except md, html, text) are in metadata with fw_ prefix
    expected_metadata_keys = {f"fw_{key}" for key in original_record.keys() if key not in ("md", "html", "text")}
    actual_metadata_keys = set(metadata.keys())
    assert expected_metadata_keys == actual_metadata_keys

    # Verify specific metadata values
    assert metadata["fw_url"] == original_record["url"]
    assert metadata["fw_token_count"] == original_record["token_count"]
    assert metadata["fw_language"] == original_record["language"]
    assert metadata["fw_language_score"] == original_record["language_score"]
