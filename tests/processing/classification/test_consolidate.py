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

import json
from pathlib import Path

import pytest

from marin.processing.classification.consolidate import calculate_percentile_threshold


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_calculate_percentile_threshold_without_ray(tmp_path):
    ddsketch = pytest.importorskip("ddsketch")
    DDSketch = ddsketch.DDSketch

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
