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
from pathlib import Path

from marin.transform.dolmino.filter_dolmino import FilterDolminoConfig, filter_dolmino


def test_filter_dolmino_with_min_length(tmp_path: Path, write_jsonl_gz) -> None:
    """Test filtering dolmino records by minimum length."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create input files with records of varying lengths
    split_dir = input_dir / "data" / "wiki"
    first_file = split_dir / "00000.json.gz"
    second_file = split_dir / "00001.json.gz"

    write_jsonl_gz(
        first_file,
        [
            {
                "id": "long_doc_1",
                "text": "This is a long document with sufficient content.",
                "metadata": {"length": 1500},
            },
            {"id": "short_doc_1", "text": "Too short.", "metadata": {"length": 500}},
        ],
    )
    write_jsonl_gz(
        second_file,
        [
            {
                "id": "long_doc_2",
                "text": "Another long document that meets the threshold.",
                "metadata": {"length": 2000},
            },
            {"id": "short_doc_2", "text": "Also too short.", "metadata": {"length": 800}},
            {"id": "exact_threshold", "text": "Exactly at threshold.", "metadata": {"length": 1000}},
        ],
    )

    config = FilterDolminoConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        split="wiki",
        min_length=1000,
    )

    filter_dolmino(config)

    # Verify that output files were created
    output_files = list(output_dir.glob("*.jsonl.gz"))
    assert len(output_files) > 0, "Expected at least one output file"

    # Collect all IDs from output files
    observed_ids: set[str] = set()
    for output_file in output_files:
        with gzip.open(output_file, "rt", encoding="utf-8") as handle:
            observed_ids.update(json.loads(line)["id"] for line in handle if line.strip())

    # Should only include documents with length >= 1000
    assert observed_ids == {"long_doc_1", "long_doc_2", "exact_threshold"}


def test_filter_dolmino_no_min_length(tmp_path: Path, write_jsonl_gz) -> None:
    """Test that all records pass when min_length is None."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    split_dir = input_dir / "data" / "stackexchange"
    input_file = split_dir / "00000.json.gz"

    write_jsonl_gz(
        input_file,
        [
            {"id": "record_1", "text": "First record", "metadata": {"length": 100}},
            {"id": "record_2", "text": "Second record", "metadata": {"length": 5000}},
            {"id": "record_3", "text": "Third record", "metadata": {"length": 1}},
        ],
    )

    config = FilterDolminoConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        split="stackexchange",
        min_length=None,
    )

    filter_dolmino(config)

    # Verify that output files were created
    output_files = list(output_dir.glob("*.jsonl.gz"))
    assert len(output_files) > 0, "Expected at least one output file"

    # Collect all IDs from output files
    observed_ids: set[str] = set()
    for output_file in output_files:
        with gzip.open(output_file, "rt", encoding="utf-8") as handle:
            observed_ids.update(json.loads(line)["id"] for line in handle if line.strip())

    # All records should pass when min_length is None
    assert observed_ids == {"record_1", "record_2", "record_3"}


def test_filter_dolmino_missing_metadata(tmp_path: Path, write_jsonl_gz) -> None:
    """Test handling of records with missing or incomplete metadata."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    split_dir = input_dir / "data" / "pes2o"
    input_file = split_dir / "00000.json.gz"

    write_jsonl_gz(
        input_file,
        [
            {"id": "has_length", "text": "Has proper metadata", "metadata": {"length": 2000}},
            {"id": "no_metadata", "text": "Missing metadata entirely"},
            {"id": "no_length", "text": "Has metadata but no length field", "metadata": {"other_field": "value"}},
            {"id": "zero_length", "text": "Has zero length", "metadata": {"length": 0}},
        ],
    )

    config = FilterDolminoConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        split="pes2o",
        min_length=1000,
    )

    filter_dolmino(config)

    # Verify that output files were created
    output_files = list(output_dir.glob("*.jsonl.gz"))
    assert len(output_files) > 0, "Expected at least one output file"

    # Collect all IDs from output files
    observed_ids: set[str] = set()
    for output_file in output_files:
        with gzip.open(output_file, "rt", encoding="utf-8") as handle:
            observed_ids.update(json.loads(line)["id"] for line in handle if line.strip())

    # Only records with length >= 1000 should pass
    # Records with missing metadata or missing length field get 0 as default
    assert observed_ids == {"has_length"}
