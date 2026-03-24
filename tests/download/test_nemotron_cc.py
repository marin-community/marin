# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import zstandard as zstd
from iris.marin_fs import open_url as _real_open_url
from marin.download.nemotron_cc.download_nemotron_cc import NemotronIngressConfig, download_nemotron_cc

_OPEN_URL_TARGET = "marin.download.nemotron_cc.download_nemotron_cc.open_url"
_REQUESTS_SESSION_TARGET = "marin.download.nemotron_cc.download_nemotron_cc.requests.Session"

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
    import gzip

    content = "\n".join(paths) + "\n"
    return gzip.compress(content.encode("utf-8"))


def read_all_jsonl_zst(directory: Path, pattern: str = "*.jsonl.zst") -> list[dict]:
    records = []
    dctx = zstd.ZstdDecompressor()
    for file_path in sorted(directory.glob(pattern)):
        with open(file_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text = reader.read().decode("utf-8")
                for line in text.splitlines():
                    if line.strip():
                        records.append(json.loads(line))
    return records


@pytest.fixture()
def mock_paths_open(tmp_path):
    """Patches open_url to serve a local paths file for commoncrawl URLs."""
    paths_file = tmp_path / "data-jsonl.paths.gz"

    def _make_mock(paths: list[str]):
        paths_file.write_bytes(create_paths_file(paths))

        def _mock_open_url(path, mode="r", **kwargs):
            if "data.commoncrawl.org" in path and "data-jsonl.paths.gz" in path:
                return paths_file.open("rb")
            return _real_open_url(path, mode, **kwargs)

        return _mock_open_url

    return _make_mock


def _mock_session_for(url_to_data: dict[str, bytes]):
    """Build a mock requests.Session whose .get() dispatches on URL substrings."""
    session = Mock()

    def mock_get(url, **kwargs):
        response = Mock()
        response.raise_for_status = Mock()
        for key, data in url_to_data.items():
            if key in url:
                response.raw = BytesIO(data)
                return response
        raise ValueError(f"Unexpected URL: {url}")

    session.return_value.get = mock_get
    session.return_value.mount = Mock()
    return session


def test_download_nemotron_cc_pipeline(tmp_path, mock_paths_open):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    paths = ["contrib/Nemotron/file1.jsonl.zstd", "contrib/Nemotron/file2.jsonl.zstd"]
    file1_data = create_zstd_compressed_jsonl([SAMPLE_NEMOTRON_RECORDS[0]])
    file2_data = create_zstd_compressed_jsonl(SAMPLE_NEMOTRON_RECORDS[1:])

    with (
        patch(_OPEN_URL_TARGET, side_effect=mock_paths_open(paths)),
        patch(_REQUESTS_SESSION_TARGET, _mock_session_for({"file1": file1_data, "file2": file2_data})),
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir))
        download_nemotron_cc(cfg)

    all_records = read_all_jsonl_zst(output_dir / "contrib" / "Nemotron")

    assert len(all_records) == 3
    assert all_records[0]["id"] == "record-001"
    assert all_records[1]["id"] == "record-002"
    assert all_records[2]["id"] == "record-003"

    for record in all_records:
        assert "id" in record
        assert "text" in record
        assert record["source"] == "nemotron"
        assert "metadata" in record


def test_download_nemotron_cc_dolma_format(tmp_path, mock_paths_open):
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

    paths = ["contrib/Nemotron/test.jsonl.zstd"]
    compressed_data = create_zstd_compressed_jsonl([nemotron_record])

    with (
        patch(_OPEN_URL_TARGET, side_effect=mock_paths_open(paths)),
        patch(_REQUESTS_SESSION_TARGET, _mock_session_for({"test": compressed_data})),
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir))
        download_nemotron_cc(cfg)

    records = read_all_jsonl_zst(output_dir / "contrib" / "Nemotron")
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


def test_download_nemotron_cc_skips_existing(tmp_path, mock_paths_open):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    paths = ["contrib/Nemotron/existing.jsonl.zstd"]

    existing_output = output_dir / "contrib" / "Nemotron" / "existing.jsonl.zst"
    existing_output.parent.mkdir(parents=True)
    existing_output.write_text("existing")

    with (
        patch(_OPEN_URL_TARGET, side_effect=mock_paths_open(paths)),
        patch(_REQUESTS_SESSION_TARGET) as mock_session,
    ):
        cfg = NemotronIngressConfig(output_path=str(output_dir))
        download_nemotron_cc(cfg)

    mock_session.return_value.get.assert_not_called()
    assert existing_output.read_text() == "existing"
