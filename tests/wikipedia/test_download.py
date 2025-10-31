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
import tarfile

import pytest
from zephyr import create_backend, set_flow_backend

from marin.download.wikipedia.download import DownloadConfig, download


@pytest.fixture
def test_tar_file(tmp_path):
    """Create a small test tar.gz file with sample content"""
    # Create a test .ndjson file
    test_content = b'{"id": 1, "text": "test article 1"}\n{"id": 2, "text": "test article 2"}\n'

    # Create a tar.gz file
    tar_path = tmp_path / "test_data.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # Create a temporary ndjson file to add to tar
        ndjson_path = tmp_path / "articles.ndjson"
        ndjson_path.write_bytes(test_content)
        tar.add(ndjson_path, arcname="articles.ndjson")
        ndjson_path.unlink()  # Clean up temp file

    return tar_path


def test_download_and_extract(tmp_path, test_tar_file):
    """Test download and extraction pipeline with pre-downloaded file"""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()

    # Copy the test tar file to the download location to simulate it already being downloaded
    # This way the download_tar function will skip it and we can test the extraction
    import shutil

    target_file = download_dir / test_tar_file.name
    shutil.copy(test_tar_file, target_file)

    # Now run the pipeline - it should skip download and extract the file
    cfg = DownloadConfig(
        input_urls=[f"http://example.com/{test_tar_file.name}"],  # Fake URL, will be skipped
        revision="test",
        output_path=str(download_dir),
    )

    backend = create_backend("threadpool")
    set_flow_backend(backend)
    download(cfg)

    # Verify extracted files exist in downloads/test/
    output_dir = download_dir / "test"
    assert output_dir.exists(), f"Output directory {output_dir} should exist"

    # Verify .gz files were created
    extracted_files = list(output_dir.glob("*.gz"))
    assert len(extracted_files) > 0, "Should have extracted at least one .gz file"

    # Verify files have content
    for gz_file in extracted_files:
        assert gz_file.stat().st_size > 0, f"File {gz_file} should have content"

        # Verify content is readable
        with gzip.open(gz_file, "rt") as f:
            content = f.read()
            assert len(content) > 0, "Extracted content should not be empty"
            assert "test article" in content, "Should contain test articles"


def test_download_skips_existing(tmp_path, test_tar_file):
    """Test that download skips files that already exist"""
    import shutil

    download_dir = tmp_path / "downloads"
    download_dir.mkdir()

    # Copy the test tar file to simulate an existing download
    target_path = download_dir / test_tar_file.name
    shutil.copy(test_tar_file, target_path)

    # Modify the file slightly to verify it's not overwritten
    original_content = target_path.read_bytes()
    target_path.write_bytes(original_content + b"__marker__")

    cfg = DownloadConfig(
        input_urls=[f"http://example.com/{test_tar_file.name}"],
        revision="test",
        output_path=str(download_dir),
    )

    backend = create_backend("threadpool")
    set_flow_backend(backend)
    download(cfg)

    # Verify the existing file was not overwritten (still has marker)
    assert target_path.read_bytes().endswith(b"__marker__"), "Existing file should not be overwritten"
