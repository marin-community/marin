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

"""Tests for vortex file format support."""

import pytest
from zephyr.readers import InputFileSpec, load_vortex
from zephyr.writers import write_vortex_file


@pytest.fixture
def vortex_file(tmp_path):
    """Create a test vortex file with sample data."""
    records = [{"id": i, "name": f"item_{i}", "score": i * 10} for i in range(100)]
    path = tmp_path / "test.vortex"
    write_vortex_file(records, str(path))
    return path


# ============================================================================
# Reader Tests (pure I/O, no backends needed)
# ============================================================================


def test_load_vortex_basic(vortex_file):
    """Test basic vortex file reading."""
    records = list(load_vortex(str(vortex_file)))
    assert len(records) == 100
    assert records[0]["id"] == 0
    assert records[0]["name"] == "item_0"
    assert records[0]["score"] == 0


def test_load_vortex_column_projection(vortex_file):
    """Test column selection (projection)."""
    spec = InputFileSpec(path=str(vortex_file), columns=["id", "name"])
    records = list(load_vortex(spec))
    assert len(records) == 100
    assert set(records[0].keys()) == {"id", "name"}


def test_load_vortex_empty_file(tmp_path):
    """Test loading an empty vortex file."""
    empty_path = tmp_path / "empty.vortex"
    write_vortex_file([], str(empty_path))

    records = list(load_vortex(str(empty_path)))
    assert records == []


# ============================================================================
# Writer Tests (pure I/O, no backends needed)
# ============================================================================


def test_write_vortex_basic(tmp_path):
    """Test basic vortex file writing and roundtrip."""
    records = [{"id": i, "value": i * 2} for i in range(10)]
    output_path = tmp_path / "output.vortex"

    result = write_vortex_file(records, str(output_path))

    assert result["count"] == 10
    assert output_path.exists()

    # Verify roundtrip
    loaded = list(load_vortex(str(output_path)))
    assert loaded == records


def test_write_vortex_empty(tmp_path):
    """Test writing empty dataset."""
    output_path = tmp_path / "empty.vortex"
    result = write_vortex_file([], str(output_path))

    assert result["count"] == 0
    assert output_path.exists()


def test_write_vortex_single_record(tmp_path):
    """Test writing single record."""
    records = [{"key": "value", "number": 42}]
    output_path = tmp_path / "single.vortex"

    result = write_vortex_file(records, str(output_path))
    assert result["count"] == 1

    loaded = list(load_vortex(str(output_path)))
    assert loaded == records
