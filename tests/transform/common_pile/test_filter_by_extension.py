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

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)
from zephyr import create_backend, set_flow_backend


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_filter_by_metadata_extension(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    first_file = input_dir / "stack-edu-0000.jsonl.gz"
    second_file = input_dir / "stack-edu-0001.jsonl.gz"
    unmatched_file = input_dir / "stack-edu-0002.jsonl.gz"

    _write_jsonl_gz(
        first_file,
        [
            {"id": "python", "text": "print('hello')", "metadata": {"extension": "py"}},
            {"id": "javascript", "text": "console.log('hi')", "metadata": {"extension": ".js"}},
        ],
    )
    _write_jsonl_gz(
        second_file,
        [
            {"id": "markdown", "text": "# Notes", "metadata": {"extension": ".MD"}},
            {"id": "list_based", "text": "pass", "metadata": {"extension": ["Py"]}},
            {"id": "missing", "text": "", "metadata": {}},
        ],
    )
    _write_jsonl_gz(
        unmatched_file,
        [
            {"id": "drop_me", "text": "n/a", "metadata": {"extension": "txt"}},
        ],
    )

    config = FilterByMetadataExtensionConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        allowed_extensions=(".py", ".md"),
        input_glob="*.jsonl.gz",
    )

    set_flow_backend(create_backend("sync"))
    filter_dataset_by_metadata_extension(config)

    # Verify that output files were created
    output_files = list(output_dir.glob("*.jsonl.gz"))
    assert len(output_files) > 0, "Expected at least one output file"

    # Collect all IDs from output files
    observed_ids: set[str] = set()
    for output_file in output_files:
        with gzip.open(output_file, "rt", encoding="utf-8") as handle:
            observed_ids.update(json.loads(line)["id"] for line in handle if line.strip())

    assert observed_ids == {"python", "markdown", "list_based"}
