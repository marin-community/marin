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
import zipfile

import pytest

from marin.download.ar5iv.download import DownloadConfig, download


@pytest.fixture
def test_zip_file(tmp_path):
    """Create a test zip file with ar5iv-like structure."""
    zip_path = tmp_path / "test-ar5iv.zip"

    # Create temporary directory structure
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()

    # Create test HTML files in shard001/papers/
    shard1_dir = test_data_dir / "shard001" / "papers"
    shard1_dir.mkdir(parents=True)

    (shard1_dir / "paper1.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>Test Paper 1</title></head>\n"
        "<body><h1>Test Paper 1</h1><p>This is a test paper for ar5iv.</p></body>\n"
        "</html>"
    )

    (shard1_dir / "paper2.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>Test Paper 2</title></head>\n"
        "<body><h1>Test Paper 2</h1><p>Another test paper.</p></body>\n"
        "</html>"
    )

    # Create test HTML files in shard002/papers/
    shard2_dir = test_data_dir / "shard002" / "papers"
    shard2_dir.mkdir(parents=True)

    (shard2_dir / "paper3.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>Test Paper 3</title></head>\n"
        "<body><h1>Test Paper 3</h1><p>Third test paper.</p></body>\n"
        "</html>"
    )

    # Create zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for html_file in test_data_dir.rglob("*.html"):
            arcname = html_file.relative_to(test_data_dir)
            zf.write(html_file, arcname)

    return zip_path


def test_ar5iv_download_sync(test_zip_file, tmp_path):
    """Test ar5iv download with sync backend."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Configure and run download
    cfg = DownloadConfig(input_path=test_zip_file, output_path=str(output_dir), max_files=2)

    download(cfg)

    # Verify output files were created
    output_files = list(output_dir.glob("*.jsonl.gz"))
    success_files = list(output_dir.glob("*.SUCCESS"))

    assert len(output_files) >= 1, "Should create at least one output file"
    assert len(success_files) >= 1, "Should create at least one SUCCESS file"

    # Verify content of output files
    all_records = []
    for output_file in output_files:
        with gzip.open(output_file, "rt") as f:
            for line in f:
                record = json.loads(line)
                all_records.append(record)

                # Verify record structure
                assert "filename" in record
                assert "format" in record
                assert "content" in record
                assert record["format"] == "html"

    # Should have 3 papers total
    assert len(all_records) == 3, f"Expected 3 records, got {len(all_records)}"

    # Verify all expected files are present
    filenames = {r["filename"] for r in all_records}
    expected_files = {"shard001/papers/paper1.html", "shard001/papers/paper2.html", "shard002/papers/paper3.html"}
    assert filenames == expected_files, f"Expected {expected_files}, got {filenames}"

    # Verify HTML content is preserved
    for record in all_records:
        assert "<!DOCTYPE html>" in record["content"]
        assert "Test Paper" in record["content"]


def test_ar5iv_download_skip_existing(test_zip_file, tmp_path):
    """Test that download skips existing shards."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DownloadConfig(input_path=test_zip_file, output_path=str(output_dir), max_files=2)

    # Run download first time
    download(cfg)

    # Count output files
    first_run_files = list(output_dir.glob("*.jsonl.gz"))
    first_run_count = len(first_run_files)

    # Run download second time
    download(cfg)

    # Should not create duplicate files
    second_run_files = list(output_dir.glob("*.jsonl.gz"))
    second_run_count = len(second_run_files)

    assert first_run_count == second_run_count, "Should skip existing shards"
