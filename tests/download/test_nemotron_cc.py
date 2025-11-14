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

import gzip
import json
from io import BytesIO
from unittest.mock import patch

import zstandard as zstd
from marin.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc

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


def create_zstd_compressed_jsonl(records: list[dict]) -> bytes:
    jsonl_content = "\n".join(json.dumps(record) for record in records) + "\n"
    jsonl_bytes = jsonl_content.encode("utf-8")
    cctx = zstd.ZstdCompressor()
    return cctx.compress(jsonl_bytes)


def create_paths_file(paths: list[str]) -> bytes:
    content = "\n".join(paths) + "\n"
    return gzip.compress(content.encode("utf-8"))


def test_download_nemotron_cc_pipeline(tmp_path, read_all_jsonl_gz):
    """Test full Nemotron CC download pipeline with zephyr integration."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create real paths file in tmp_path to simulate remote download
    paths = ["contrib/Nemotron/file1.jsonl.zstd", "contrib/Nemotron/file2.jsonl.zstd"]
    paths_file = tmp_path / "data-jsonl.paths.gz"
    paths_file.write_bytes(create_paths_file(paths))

    # Create mock compressed data for each file
    file1_data = create_zstd_compressed_jsonl([SAMPLE_NEMOTRON_RECORDS[0]])
    file2_data = create_zstd_compressed_jsonl(SAMPLE_NEMOTRON_RECORDS[1:])

    def mock_requests_get(url, **kwargs):
        from unittest.mock import Mock

        response = Mock()
        response.status_code = 200
        if "file1" in url:
            response.headers = {"content-length": str(len(file1_data))}
            response.raw = BytesIO(file1_data)
        else:
            response.headers = {"content-length": str(len(file2_data))}
            response.raw = BytesIO(file2_data)
        return response

    # Mock only the remote HTTP request to download paths file
    import fsspec

    original_open = fsspec.open

    def mock_open_for_remote(path, mode="r", **kwargs):
        # Intercept only the remote paths file download
        if "data.commoncrawl.org" in path and "data-jsonl.paths.gz" in path:
            return paths_file.open("rb")
        return original_open(path, mode, **kwargs)

    with (
        patch("marin.download.nemotron_cc.download_nemotron_cc.fsspec.open", side_effect=mock_open_for_remote),
        patch("marin.download.nemotron_cc.download_nemotron_cc.requests.get", side_effect=mock_requests_get),
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir), chunk_size=1024)
        download_nemotron_cc(cfg)

    # Verify output files were created
    all_records = read_all_jsonl_gz(output_dir / "contrib" / "Nemotron", "*.jsonl.gz")

    assert len(all_records) == 3
    assert all_records[0]["id"] == "record-001"
    assert all_records[1]["id"] == "record-002"
    assert all_records[2]["id"] == "record-003"

    # Verify Dolma format
    for record in all_records:
        assert "id" in record
        assert "text" in record
        assert "source" in record
        assert record["source"] == "nemotron"
        assert "metadata" in record


def test_download_nemotron_cc_dolma_format(tmp_path, read_all_jsonl_gz):
    """Test Dolma format conversion in full pipeline."""
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

    # Create real paths file
    paths = ["contrib/Nemotron/test.jsonl.zstd"]
    paths_file = tmp_path / "data-jsonl.paths.gz"
    paths_file.write_bytes(create_paths_file(paths))

    compressed_data = create_zstd_compressed_jsonl([nemotron_record])

    def mock_requests_get(url, **kwargs):
        from unittest.mock import Mock

        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": str(len(compressed_data))}
        response.raw = BytesIO(compressed_data)
        return response

    # Mock only the remote HTTP request to download paths file
    import fsspec

    original_open = fsspec.open

    def mock_open_for_remote(path, mode="r", **kwargs):
        if "data.commoncrawl.org" in path and "data-jsonl.paths.gz" in path:
            return paths_file.open("rb")
        return original_open(path, mode, **kwargs)

    with (
        patch("marin.download.nemotron_cc.download_nemotron_cc.fsspec.open", side_effect=mock_open_for_remote),
        patch("marin.download.nemotron_cc.download_nemotron_cc.requests.get", side_effect=mock_requests_get),
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir))
        download_nemotron_cc(cfg)

    records = read_all_jsonl_gz(output_dir / "contrib" / "Nemotron", "*.jsonl.gz")
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


def test_download_nemotron_cc_skips_existing(tmp_path):
    """Test that pipeline skips files that already exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create real paths file
    paths = ["contrib/Nemotron/existing.jsonl.zstd"]
    paths_file = tmp_path / "data-jsonl.paths.gz"
    paths_file.write_bytes(create_paths_file(paths))

    # Pre-create the output file
    existing_output = output_dir / "contrib" / "Nemotron" / "existing.jsonl.gz"
    existing_output.parent.mkdir(parents=True)
    existing_output.write_text("existing")

    # Mock only the remote HTTP request to download paths file
    import fsspec

    original_open = fsspec.open

    def mock_open_for_remote(path, mode="r", **kwargs):
        if "data.commoncrawl.org" in path and "data-jsonl.paths.gz" in path:
            return paths_file.open("rb")
        return original_open(path, mode, **kwargs)

    with (
        patch("marin.download.nemotron_cc.download_nemotron_cc.fsspec.open", side_effect=mock_open_for_remote),
        patch("marin.download.nemotron_cc.download_nemotron_cc.requests.get") as mock_get,
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir))
        download_nemotron_cc(cfg)

    # Verify no HTTP requests were made (file was skipped)
    mock_get.assert_not_called()

    # Verify existing file was not modified
    assert existing_output.read_text() == "existing"
