# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The summaries, templates, and converted stages over a small source parquet built from fixtures.

The parquet stores every task in one row group, the shape that made single-row-group sources one
worker's job before rows were shuffled into even shards.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.tasktrove.convert import convert_tasks
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.fingerprint import EXEMPLAR_MIN_TASKS, build_template_index, summarize_templates
from experiments.post_training.tasktrove.taskbinary import INSTRUCTION, TaskFiles, read_task_binary, write_task_binary

FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"


def _write_raw(root: Path, blobs: dict[str, bytes]) -> Path:
    source = root / "raw" / SOURCE
    source.mkdir(parents=True)
    table = pa.table({"path": list(blobs), "task_binary": list(blobs.values())})
    pq.write_table(table, source / "tasks.parquet", row_group_size=len(blobs))
    return root / "raw"


def test_summaries_templates_and_convert_over_one_row_group(tmp_path: Path):
    blob = (FIXTURES / "nemotron_mcqa.tar.gz").read_bytes()
    task = read_task_binary(blob)
    reworded = write_task_binary(TaskFiles({**task.files, INSTRUCTION: b"Which option is right? A or B?"}))
    blobs = {f"task-{i:02d}.tar.gz": blob for i in range(EXEMPLAR_MIN_TASKS)}
    blobs["reworded.tar.gz"] = reworded
    raw = _write_raw(tmp_path, blobs)
    summaries, templates, converted = (str(tmp_path / name) for name in ("summaries", "templates", "converted"))

    summarize_templates(str(raw), summaries)
    build_template_index(str(raw), summaries, templates)

    index = json.loads((tmp_path / "templates" / "templates.json").read_text())
    assert [(t["tasks"], t["sources"]) for t in index] == [(EXEMPLAR_MIN_TASKS + 1, {SOURCE: EXEMPLAR_MIN_TASKS + 1})]
    assert index[0]["exemplar_path"] == "reworded.tar.gz"
    exemplar = tmp_path / "templates" / "templates" / index[0]["template_id"] / "exemplar"
    assert (exemplar / INSTRUCTION).read_bytes() == b"Which option is right? A or B?"
    coverage = json.loads((tmp_path / "templates" / "coverage.json").read_text())
    assert [c["converter"] for c in coverage] == ["nemotron_mcqa"]

    convert_tasks(str(raw), templates, converted, tool_ref="ref")

    rows = [
        row
        for f in sorted((tmp_path / "converted" / "converted").glob("*.parquet"))
        for row in pq.read_table(f).to_pylist()
    ]
    assert sorted(row["path"] for row in rows) == sorted(blobs)
    assert {row["status"] for row in rows} == {ConvertStatus.CONVERTED}
    assert len({row["instruction_key"] for row in rows}) == 2
    assert len(list((tmp_path / "converted" / "converted").glob("*.parquet"))) > 1
