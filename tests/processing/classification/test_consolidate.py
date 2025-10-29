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

from ddsketch import DDSketch
import pytest

from marin.processing.classification.consolidate import (
    ConsolidateConfig,
    FilterConfig,
    FilterType,
    calculate_percentile_threshold,
    consolidate,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_calculate_percentile_threshold_without_ray(tmp_path, ray_tpu_cluster):

    documents_dir = tmp_path / "documents"
    attributes_dir = tmp_path / "attributes"
    documents_dir.mkdir()
    attributes_dir.mkdir()

    attribute_rows = [
        [
            {"id": "doc-0", "attributes": {"quality": {"good": 0.1}}},
            {"id": "doc-1", "attributes": {"quality": {"good": 0.4}}},
        ],
        [
            {"id": "doc-2", "attributes": {"quality": {"good": 0.7}}},
            {"id": "doc-3", "attributes": {"quality": {"good": 0.9}}},
        ],
    ]

    input_paths: list[str] = []
    for shard_index, rows in enumerate(attribute_rows):
        doc_path = documents_dir / f"part-{shard_index}.jsonl"
        doc_path.write_text("{}", encoding="utf-8")
        input_paths.append(str(doc_path))
        attr_path = attributes_dir / f"part-{shard_index}.jsonl"
        _write_jsonl(attr_path, rows)

    keep_fraction = 0.5
    threshold = calculate_percentile_threshold(
        base_input_path=str(documents_dir),
        input_paths=input_paths,
        attribute_path=str(attributes_dir),
        attribute_name="quality",
        label="good",
        keep_fraction=keep_fraction,
        use_ray=False,
    )

    expected_sketch = DDSketch()
    for shard in attribute_rows:
        for row in shard:
            expected_sketch.add(row["attributes"]["quality"]["good"])
    expected_threshold = expected_sketch.get_quantile_value(1 - keep_fraction)

    assert threshold == pytest.approx(expected_threshold, rel=1e-6)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_consolidate_filters_and_writes_output(tmp_path):
    input_root = tmp_path / "input"
    attributes_root = tmp_path / "attributes"
    output_root = tmp_path / "output"
    input_root.mkdir()
    attributes_root.mkdir()
    output_root.mkdir()

    input_rows = [
        {"id": "doc-0", "text": "first"},
        {"id": "doc-1", "text": "second"},
        {"id": "doc-2", "text": "third"},
    ]
    attribute_rows = [
        {"id": "doc-0", "attributes": {"quality": {"good": 0.1}}},
        {"id": "doc-1", "attributes": {"quality": {"good": 0.6}}},
        {"id": "doc-2", "attributes": {"quality": {"good": 0.8}}},
    ]

    input_file = input_root / "part-0000.jsonl.gz"
    attribute_file = attributes_root / "part-0000.jsonl.gz"
    _write_jsonl_gz(input_file, input_rows)
    _write_jsonl_gz(attribute_file, attribute_rows)

    config = ConsolidateConfig(
        input_path=str(input_root),
        output_path=str(output_root),
        filters=[
            FilterConfig(
                type=FilterType.CLASSIFY,
                attribute_path=str(attributes_root),
                name="quality",
                label="good",
                lower_threshold=0.5,
            )
        ],
        max_tasks_in_flight=1,
    )

    consolidate(config)

    output_file = output_root / "part-0000.jsonl.gz"
    assert output_file.exists(), "Expected consolidated output file to be written."

    with gzip.open(output_file, "rt", encoding="utf-8") as handle:
        output_rows = [json.loads(line) for line in handle if line.strip()]

    kept_ids = {row["id"] for row in output_rows}
    assert kept_ids == {"doc-1", "doc-2"}

    success_file = Path(f"{output_file}.SUCCESS")
    assert success_file.exists(), "Expected consolidate to write success ledger."
