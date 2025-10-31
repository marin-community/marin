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

"""Tests for Dataset API."""

import gzip
import json
from pathlib import Path

from ml_flow import Dataset
from ml_flow.testing import get_test_pipeline_options


def test_dataset_from_text_files(tmp_path: Path):
    """Test reading text files."""
    # Create test files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    (input_dir / "file1.txt").write_text("line1\nline2\n")
    (input_dir / "file2.txt").write_text("line3\nline4\n")

    # Read with Dataset
    ds = Dataset.from_text_files(str(input_dir / "*.txt"), get_test_pipeline_options())

    # Collect results
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_text_files(str(output_dir / "output"))
    result = ds.run_and_wait()

    # Beam's state might be a string or object depending on runner
    state = result.state if isinstance(result.state, str) else result.state.name
    assert state == "DONE"

    # Verify output
    output_files = list(output_dir.glob("output-*"))
    assert len(output_files) > 0

    lines = []
    for f in output_files:
        lines.extend(f.read_text().strip().split("\n"))

    assert set(lines) == {"line1", "line2", "line3", "line4"}


def test_dataset_from_jsonl_files(tmp_path: Path):
    """Test reading JSONL files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test JSONL
    with open(input_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"id": 1, "text": "hello"}) + "\n")
        f.write(json.dumps({"id": 2, "text": "world"}) + "\n")

    # Read with Dataset
    ds = Dataset.from_jsonl_files(str(input_dir / "*.jsonl"), get_test_pipeline_options())

    # Write output
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_jsonl_gz(str(output_dir / "output"))
    result = ds.run_and_wait()

    # Beam's state might be a string or object depending on runner
    state = result.state if isinstance(result.state, str) else result.state.name
    assert state == "DONE"

    # Verify output
    output_files = list(output_dir.glob("output-*.jsonl.gz"))
    assert len(output_files) > 0

    records = []
    for f in output_files:
        with gzip.open(f, "rt") as gf:
            for line in gf:
                records.append(json.loads(line))

    assert len(records) == 2
    assert any(r["id"] == 1 and r["text"] == "hello" for r in records)
    assert any(r["id"] == 2 and r["text"] == "world" for r in records)


def test_dataset_map(tmp_path: Path):
    """Test map transformation."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test data
    with open(input_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"value": 1}) + "\n")
        f.write(json.dumps({"value": 2}) + "\n")

    # Apply map
    ds = Dataset.from_jsonl_files(str(input_dir / "*.jsonl"), get_test_pipeline_options())
    ds = ds.map(lambda x: {"value": x["value"] * 2})

    # Write and verify
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_jsonl_gz(str(output_dir / "output"))
    ds.run_and_wait()

    # Check results
    output_file = next(output_dir.glob("output-*.jsonl.gz"))
    with gzip.open(output_file, "rt") as f:
        records = [json.loads(line) for line in f]

    assert set(r["value"] for r in records) == {2, 4}


def test_dataset_filter(tmp_path: Path):
    """Test filter transformation."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test data
    with open(input_dir / "data.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"id": i}) + "\n")

    # Apply filter
    ds = Dataset.from_jsonl_files(str(input_dir / "*.jsonl"), get_test_pipeline_options())
    ds = ds.filter(lambda x: x["id"] % 2 == 0)  # Keep even only

    # Write and verify
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_jsonl_gz(str(output_dir / "output"))
    ds.run_and_wait()

    # Check results
    output_file = next(output_dir.glob("output-*.jsonl.gz"))
    with gzip.open(output_file, "rt") as f:
        records = [json.loads(line) for line in f]

    ids = set(r["id"] for r in records)
    assert ids == {0, 2, 4, 6, 8}


def test_dataset_flat_map(tmp_path: Path):
    """Test flat_map transformation."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test data
    with open(input_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"text": "a b c"}) + "\n")
        f.write(json.dumps({"text": "x y"}) + "\n")

    # Apply flat_map (split words)
    ds = Dataset.from_jsonl_files(str(input_dir / "*.jsonl"), get_test_pipeline_options())
    ds = ds.flat_map(lambda x: [{"word": w} for w in x["text"].split()])

    # Write and verify
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_jsonl_gz(str(output_dir / "output"))
    ds.run_and_wait()

    # Check results
    output_file = next(output_dir.glob("output-*.jsonl.gz"))
    with gzip.open(output_file, "rt") as f:
        records = [json.loads(line) for line in f]

    words = set(r["word"] for r in records)
    assert words == {"a", "b", "c", "x", "y"}


def test_dataset_chaining(tmp_path: Path):
    """Test chaining multiple transformations."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test data
    with open(input_dir / "data.jsonl", "w") as f:
        for i in range(20):
            f.write(json.dumps({"value": i}) + "\n")

    # Chain operations with unique names
    ds = Dataset.from_jsonl_files(str(input_dir / "*.jsonl"), get_test_pipeline_options())
    ds = (
        ds.filter(lambda x: x["value"] > 5, name="FilterGT5")  # Keep > 5
        .map(lambda x: {"value": x["value"] * 2}, name="Double")  # Double
        .filter(lambda x: x["value"] < 30, name="FilterLT30")  # Keep < 30
    )

    # Write and verify
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ds.write_jsonl_gz(str(output_dir / "output"))
    ds.run_and_wait()

    # Check results
    output_file = next(output_dir.glob("output-*.jsonl.gz"))
    with gzip.open(output_file, "rt") as f:
        records = [json.loads(line) for line in f]

    values = sorted(r["value"] for r in records)
    # Original: 6-14 pass first filter
    # After doubling: 12-28
    # All < 30, so all pass
    assert values == [12, 14, 16, 18, 20, 22, 24, 26, 28]
