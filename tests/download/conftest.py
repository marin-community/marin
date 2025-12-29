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

"""Shared fixtures for download tests."""

import gzip
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from fray.job import create_job_ctx, fray_default_job_ctx


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    """Set up sync backend for all download tests."""
    with fray_default_job_ctx(create_job_ctx("sync")):
        yield


@pytest.fixture
def write_jsonl_gz():
    """Fixture to write JSONL gzipped files."""

    def _write(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")

    return _write


@pytest.fixture
def read_jsonl_gz():
    """Fixture to read JSONL gzipped files."""

    def _read(path: Path) -> list[dict]:
        records = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    return _read


@pytest.fixture
def read_all_jsonl_gz():
    """Fixture to read all JSONL gzipped files from a directory."""

    def _read_all(directory: Path, pattern: str = "*.jsonl.gz") -> list[dict]:
        records = []
        for file_path in sorted(directory.glob(pattern)):
            with gzip.open(file_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        return records

    return _read_all


@pytest.fixture
def verify_success_files():
    """Fixture to verify .SUCCESS files were created."""

    def _verify(output_dir: Path, min_count: int = 1) -> list[Path]:
        """Verify that SUCCESS files exist in output directory.

        Args:
            output_dir: Directory to check
            min_count: Minimum number of SUCCESS files expected

        Returns:
            List of SUCCESS file paths found
        """
        success_files = list(output_dir.glob("*.SUCCESS"))
        assert (
            len(success_files) >= min_count
        ), f"Expected at least {min_count} SUCCESS file(s), found {len(success_files)}"
        return success_files

    return _verify


@pytest.fixture
def create_test_zip():
    """Fixture to create test zip archives."""

    def _create(zip_path: Path, files_by_name: dict[str, str]) -> Path:
        """Create a zip archive.

        Args:
            zip_path: Path where zip should be created
            files_by_name: Dict mapping internal filenames to file contents

        Returns:
            Path to the created zip file
        """
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_by_name.items():
                zf.writestr(filename, content)
        return zip_path

    return _create


@pytest.fixture
def mock_http_response():
    """Fixture to create mock HTTP response objects."""

    def _create(
        content: bytes = b"",
        status_code: int = 200,
        headers: dict | None = None,
        stream_chunks: list[bytes] | None = None,
    ) -> Mock:
        """Create a mock HTTP response.

        Args:
            content: Response content as bytes
            status_code: HTTP status code
            headers: Response headers
            stream_chunks: If provided, makes response streamable with these chunks

        Returns:
            Mock response object
        """
        response = Mock()
        response.status_code = status_code
        response.content = content
        response.headers = headers or {}
        response.raise_for_status = Mock()

        if stream_chunks is not None:
            response.iter_content = Mock(return_value=iter(stream_chunks))

        return response

    return _create


@pytest.fixture
def mock_hf_filesystem():
    """Fixture to create mock HuggingFace filesystem."""

    def _create(files: dict[str, bytes]) -> MagicMock:
        """Create a mock HfFileSystem.

        Args:
            files: Dict mapping file paths to file contents

        Returns:
            Mock HfFileSystem object
        """
        fs = MagicMock()

        def mock_open(path, mode="rb"):
            if path in files:
                import io

                return io.BytesIO(files[path])
            raise FileNotFoundError(f"File not found: {path}")

        fs.open.side_effect = mock_open
        fs.exists = Mock(side_effect=lambda p: p in files)
        fs.ls = Mock(return_value=list(files.keys()))

        return fs

    return _create
