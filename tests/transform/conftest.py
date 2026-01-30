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

"""Shared fixtures for transform tests."""

import gzip
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from zephyr.context import create_backend_context, default_backend_context


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    """Set up sync backend for all transform tests."""
    with default_backend_context(create_backend_context("sync")):
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
def create_tar_gz():
    """Fixture to create tar.gz archives with JSON files."""

    def _create(tar_path: Path, records_by_filename: dict[str, list[dict]]) -> Path:
        """Create a tar.gz archive.

        Args:
            tar_path: Path where tar.gz should be created
            records_by_filename: Dict mapping internal filenames to list of records
                Example: {"2020/file1.json": [{"id": 1}], "2021/file2.json": [{"id": 2}]}

        Returns:
            Path to the created tar.gz file
        """
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "w:gz") as tar:
            for filename, records in records_by_filename.items():
                json_content = "\n".join(json.dumps(r) for r in records)
                json_bytes = json_content.encode("utf-8")

                import io
                import tarfile as tf

                tarinfo = tf.TarInfo(name=filename)
                tarinfo.size = len(json_bytes)
                tar.addfile(tarinfo, io.BytesIO(json_bytes))
        return tar_path

    return _create


@pytest.fixture
def create_zip():
    """Fixture to create zip archives with HTML or other files."""

    def _create(zip_path: Path, files_by_name: dict[str, str]) -> Path:
        """Create a zip archive.

        Args:
            zip_path: Path where zip should be created
            files_by_name: Dict mapping internal filenames to file contents
                Example: {"shard001/papers/paper1.html": "<html>...</html>"}

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
def validate_dolma_record():
    """Fixture to validate that a record matches Dolma schema."""

    def _validate(record: dict) -> None:
        """Validate Dolma record has required fields."""
        assert "id" in record, "Dolma record missing 'id' field"
        assert "text" in record, "Dolma record missing 'text' field"
        assert "source" in record, "Dolma record missing 'source' field"
        # metadata and created are optional but commonly used
        if "metadata" in record:
            assert isinstance(record["metadata"], dict), "metadata must be a dict"

    return _validate


@pytest.fixture
def sample_html_with_latex():
    """Fixture providing sample HTML with LaTeX equations."""
    return """<!DOCTYPE html>
<html>
<head><title>Test Paper</title></head>
<body>
<h1>Introduction</h1>
<p>This is a test paper with equations.</p>
<p>Inline equation: $E = mc^2$</p>
<p>Display equation:</p>
<div class="ltx_equation">
$$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$
</div>
<h2>Methods</h2>
<p>We use the method described in <a href="#ref1">Reference 1</a>.</p>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>alpha</td><td>0.05</td></tr>
</table>
</body>
</html>"""


@pytest.fixture
def sample_html_simple():
    """Fixture providing simple HTML without LaTeX."""
    return """<!DOCTYPE html>
<html>
<head><title>Simple Page</title></head>
<body>
<h1>Simple Title</h1>
<p>This is a simple paragraph.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
</ul>
</body>
</html>"""
