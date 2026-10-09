# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sqlite3
import tarfile
import zlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taskcompendium.harbor.compare import ParityReport, Tolerance, compare_tasks
from taskcompendium.harbor.export import archive_bytes
from taskcompendium.harbor.records import NormalizedIndex
from taskcompendium.harbor.snapshots import task_snapshot


@pytest.mark.parametrize(
    "path,category",
    [
        ("instruction.md", "instruction"),
        ("tests/test_hidden.py", "private_test"),
        ("environment/Dockerfile", "actor_environment"),
        ("setup_files/source.py", "public_source"),
    ],
)
def test_content_changes_are_reported_for_every_task_role(path, category):
    before = task_snapshot("source", "task", "converted", task_binary=archive_bytes({path: b"original"}, {}))
    after = task_snapshot("source", "task", "converted", task_binary=archive_bytes({path: b"changed"}, {}))
    differences = compare_tasks(before, after)
    assert [(difference.category, difference.kind, difference.path) for difference in differences] == [
        (category, "bytes", "task/" + path)
    ]


def test_oracle_files_are_compared_separately_from_private_tests():
    before = task_snapshot(
        "source", "task", "converted", solution_binary=archive_bytes({"solution/solve.sh": b"gold"}, {})
    )
    after = task_snapshot(
        "source", "task", "converted", solution_binary=archive_bytes({"solution/solve.sh": b"wrong"}, {})
    )
    (difference,) = compare_tasks(before, after)
    assert (difference.category, difference.kind, difference.path) == ("oracle", "bytes", "oracle/solution/solve.sh")


def link_archive(target, *, mode):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("source")
        entry.type = tarfile.SYMTYPE
        entry.linkname = target
        entry.mode = mode
        archive.addfile(entry)
    return buffer.getvalue()


def test_inventory_member_types_modes_and_link_targets_are_not_normalized_away():
    before = task_snapshot("source", "task", "converted", task_binary=link_archive("one", mode=0o755))
    after = task_snapshot("source", "task", "converted", task_binary=link_archive("two", mode=0o644))
    assert {difference.kind for difference in compare_tasks(before, after)} == {"mode", "link"}
    regular = task_snapshot(
        "source", "task", "converted", task_binary=archive_bytes({"source": b"", "extra": b"new"}, {})
    )
    differences = compare_tasks(before, regular)
    assert ("task/source", "type") in {(difference.path, difference.kind) for difference in differences}
    assert ("task/extra", "inventory") in {(difference.path, difference.kind) for difference in differences}


def test_toml_formatting_is_explicit_and_does_not_tolerate_mode_or_value_changes():
    before = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"task.toml": b"answer = 1\n"}, {}))
    after = task_snapshot(
        "source",
        "task",
        "converted",
        task_binary=archive_bytes({"task.toml": b"answer=1 # comment\n"}, {"task.toml": "755"}),
    )
    strict = compare_tasks(before, after)
    assert {difference.kind for difference in strict} == {"mode", "bytes"}
    assert all(difference.tolerance is None for difference in strict)
    allowed = compare_tasks(before, after, tolerances=frozenset({Tolerance.TOML_FORMATTING}))
    assert [(difference.kind, difference.tolerance) for difference in allowed] == [
        ("mode", None),
        ("bytes", Tolerance.TOML_FORMATTING),
    ]
    changed = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"task.toml": b"answer=2\n"}, {}))
    differences = compare_tasks(before, changed, tolerances=frozenset({Tolerance.TOML_FORMATTING}))
    assert {difference.kind for difference in differences} == {"bytes", "parsed_toml"}
    assert all(difference.tolerance is None for difference in differences)


def test_report_retains_every_task_and_reuses_repeated_file_difference_details(tmp_path):
    baseline = task_snapshot("source", "one", "converted", task_binary=archive_bytes({"tests/helper.py": b"old"}, {}))
    candidate = task_snapshot("source", "one", "converted", task_binary=archive_bytes({"tests/helper.py": b"new"}, {}))
    path = tmp_path / "parity.sqlite"
    with ParityReport(path) as report:
        report.add(baseline, candidate)
        report.add(replace(baseline, path="two"), replace(candidate, path="two"))
        report.add(task_snapshot("source", "rejected", "reviewed_defect"), None)
        assert report.summary()["counts"]["different"] == 3
    with sqlite3.connect(path) as connection:
        tasks = connection.execute("SELECT path,groups_json FROM tasks ORDER BY path").fetchall()
        assert [row[0] for row in tasks] == ["one", "rejected", "two"]
        assert tasks[0][1] == tasks[2][1]
        group = json.loads(tasks[0][1])[0]
        details = json.loads(
            zlib.decompress(
                connection.execute("SELECT details FROM difference_groups WHERE id=?", (group,)).fetchone()[0]
            )
        )
        assert [(item["path"], item["kind"]) for item in details] == [("task/tests/helper.py", "bytes")]


def test_legacy_static_filter_is_separate_from_the_converted_payload():
    baseline = replace(
        task_snapshot("source", "task", "converted"),
        legacy_static_rejection="dockerfile",
        legacy_static_detail="missing dependency",
    )
    candidate = task_snapshot("source", "task", "converted")
    (difference,) = compare_tasks(baseline, candidate)
    assert (difference.category, difference.kind) == ("legacy_static_filter", "dockerfile")
    rejected = task_snapshot("source", "task", "source_defect", detail="bad environment")
    assert any(difference.kind == "conversion" for difference in compare_tasks(baseline, rejected))


def test_disk_index_matches_reordered_source_paths_and_reports_unmatched_rows(tmp_path):
    rows = [{"original_path": str(index), "task_json": str(index)} for index in range(65)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet", row_group_size=25)
    with NormalizedIndex(tmp_path, tmp_path / "index.sqlite") as index:
        for position in [64, 0, 24, 25, 16, 40]:
            assert index.get(str(position)) == rows[position]
        assert index.get("absent") is None
        remaining = list(index.unmatched())
    assert remaining == [row for position, row in enumerate(rows) if position not in {64, 0, 24, 25, 16, 40}]
