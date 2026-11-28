# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-source deduplication: later sources' copies of a contract become typed drops."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.context import ZephyrContext

from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.cross_source import (
    CROSS_SOURCE_DUPLICATE,
    RetainedSource,
    deduplicate_across_sources,
)
from taskcompendium.pipeline.models import RawRow


def write_final(path: Path, tasks: list[TaskSpec]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


def arithmetic_task(task_id: str, prompt: str, answer: str) -> TaskSpec:
    row = RawRow(task_id, Source(dataset="fixture", revision="1", row=task_id, importer_revision="1"), {})
    task = numeric_answer_task(row, prompt=prompt, answer=answer, tolerance_abs=0, tolerance_rel=0)
    assert isinstance(task, TaskSpec)
    return task.model_copy(update={"id": task_id})


def replace_order(source: RetainedSource, order: int) -> RetainedSource:
    return RetainedSource(source_id=source.source_id, final_path=source.final_path, order=order)


def final_source(tmp_path: Path, name: str, tasks: list[TaskSpec]) -> RetainedSource:
    final = tmp_path / name / "final"
    write_final(final, tasks)
    return RetainedSource(source_id=name, final_path=str(final), order=0)


def test_later_sources_duplicates_become_typed_drops_with_first_source_wins(tmp_path):
    shared = arithmetic_task("shared-task", "What is 7 + 5?", "12")
    first_only = arithmetic_task("first-only", "What is 3 + 4?", "7")
    second_only = arithmetic_task("second-only", "What is 10 - 2?", "8")
    # The same contract in both sources with different task IDs: the first source wins.
    later_copy = shared.model_copy(update={"id": "shared-task-copy"})
    first = replace_order(final_source(tmp_path, "alpha", [shared, first_only]), 0)
    second = replace_order(final_source(tmp_path, "beta", [later_copy, second_only]), 1)

    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        evidence = deduplicate_across_sources([first, second], str(tmp_path / "cross-source"), context=context)

    assert evidence.cross_source_duplicates == 1
    alpha = evidence.sources["alpha"]
    assert alpha.retained_rows == 2
    assert alpha.cross_source_duplicates == 0
    beta = evidence.sources["beta"]
    assert beta.retained_rows == 2
    assert beta.cross_source_duplicates == 1
    drops = pq.read_table(Path(beta.drop_manifest_uri)).to_pylist()
    assert drops == [
        {
            "task_id": "shared-task-copy",
            "kept_id": "shared-task",
            "kept_source_id": "alpha",
            "reason": CROSS_SOURCE_DUPLICATE,
        }
    ]
    manifest = json.loads((tmp_path / "cross-source" / "manifest.json").read_text())
    assert manifest["revision"] == "1"
    assert manifest["cross_source_duplicates"] == 1
    assert sorted(manifest["sources"]) == ["alpha", "beta"]


def test_sources_without_cross_source_duplicates_get_empty_manifests(tmp_path):
    only_a = arithmetic_task("a-task", "What is 1 + 1?", "2")
    only_b = arithmetic_task("b-task", "What is 2 + 2?", "4")
    first = replace_order(final_source(tmp_path, "alpha", [only_a]), 0)
    second = replace_order(final_source(tmp_path, "beta", [only_b]), 1)

    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        evidence = deduplicate_across_sources([first, second], str(tmp_path / "cross-source"), context=context)

    assert evidence.cross_source_duplicates == 0
    for source_evidence in evidence.sources.values():
        table = pq.read_table(Path(source_evidence.drop_manifest_uri))
        assert table.num_rows == 0


def test_campaign_order_breaks_ties_and_the_earliest_source_wins(tmp_path):
    shared_prompt = "What is 6 times 7?"
    first = replace_order(final_source(tmp_path, "alpha", [arithmetic_task("from-alpha", shared_prompt, "42")]), 0)
    second = replace_order(final_source(tmp_path, "beta", [arithmetic_task("from-beta", shared_prompt, "42")]), 1)
    third = replace_order(final_source(tmp_path, "gamma", [arithmetic_task("from-gamma", shared_prompt, "42")]), 2)

    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        evidence = deduplicate_across_sources([first, second, third], str(tmp_path / "cross-source"), context=context)

    assert evidence.cross_source_duplicates == 2
    gamma_drops = pq.read_table(Path(evidence.sources["gamma"].drop_manifest_uri)).to_pylist()
    beta_drops = pq.read_table(Path(evidence.sources["beta"].drop_manifest_uri)).to_pylist()
    assert gamma_drops == [
        {
            "task_id": "from-gamma",
            "kept_id": "from-alpha",
            "kept_source_id": "alpha",
            "reason": CROSS_SOURCE_DUPLICATE,
        }
    ]
    assert beta_drops == [
        {
            "task_id": "from-beta",
            "kept_id": "from-alpha",
            "kept_source_id": "alpha",
            "reason": CROSS_SOURCE_DUPLICATE,
        }
    ]
