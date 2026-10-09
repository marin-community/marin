# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import json
import sqlite3
import tarfile
import zlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.compare import ParityReport, compare_tasks, write_archive_diff
from taskcompendium.harbor.export import archive_bytes, archive_file_mode, harbor_payload, harbor_record
from taskcompendium.harbor.records import NormalizedIndex
from taskcompendium.harbor.snapshots import file_map_snapshot, task_snapshot
from taskcompendium.models import ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource


@pytest.mark.parametrize(
    "archive,path,category",
    [
        ("task", "instruction.md", "instruction"),
        ("task", "tests/test_hidden.py", "private_test"),
        ("task", "environment/Dockerfile", "actor_environment"),
        ("task", "setup_files/source.py", "public_source"),
        ("solution", "solution/solve.sh", "oracle"),
    ],
)
def test_content_changes_are_reported_for_every_task_role(archive, path, category):
    snapshots = []
    for content in (b"original", b"changed"):
        blob = archive_bytes({path: content}, {})
        snapshots.append(
            task_snapshot(
                "source",
                "task",
                "converted",
                task_binary=blob if archive == "task" else None,
                solution_binary=blob if archive == "solution" else None,
            )
        )
    before, after = snapshots
    differences = compare_tasks(before, after)
    namespace = "oracle" if archive == "solution" else "task"
    assert [(difference.category, difference.kind, difference.path) for difference in differences] == [
        (category, "bytes", f"{namespace}/{path}")
    ]


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


def test_toml_formatting_modes_and_parsed_values_are_reported_separately():
    before = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"task.toml": b"answer = 1\n"}, {}))
    after = task_snapshot(
        "source",
        "task",
        "converted",
        task_binary=archive_bytes({"task.toml": b"answer=1 # comment\n"}, {"task.toml": "755"}),
    )
    assert {difference.kind for difference in compare_tasks(before, after)} == {"mode", "bytes"}
    changed = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"task.toml": b"answer=2\n"}, {}))
    assert {difference.kind for difference in compare_tasks(before, changed)} == {"bytes", "parsed_toml"}


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


def test_report_does_not_persist_large_reference_bodies_from_renamed_specs(tmp_path):
    reference = "private reference answer " * 12000
    spec = f'reference = "{reference}"\n'.encode()
    baseline = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"tests/verifier.toml": spec}, {}))
    candidate = task_snapshot("source", "task", "converted", task_binary=archive_bytes({"tests/spec.toml": spec}, {}))
    path = tmp_path / "parity.sqlite"
    with ParityReport(path) as report:
        report.add(baseline, candidate)
    with sqlite3.connect(path) as connection:
        details = [zlib.decompress(row[0]) for row in connection.execute("SELECT details FROM difference_groups")]
    assert sum(map(len, details)) < 4096
    assert all(reference.encode() not in detail for detail in details)
    assert {item["path"] for detail in details for item in json.loads(detail)} == {
        "task/tests/verifier.toml",
        "task/tests/spec.toml",
    }


def test_selected_archive_diff_exposes_changed_text_missing_files_and_binary_hashes():
    before = archive_bytes(
        {
            "instruction.md": b"Solve old question\n",
            "environment/Dockerfile": b"FROM old\n",
            "tests/test.sh": b"old check\n",
            "source.py": b"old public source\n",
            "removed.py": b"deleted content\n",
            "binary": b"\xffold",
        },
        {},
    )
    after = archive_bytes(
        {
            "instruction.md": b"Solve new question\n",
            "environment/Dockerfile": b"FROM new\n",
            "tests/test.sh": b"new check\n",
            "source.py": b"new public source\n",
            "binary": b"\xffnew",
        },
        {},
    )
    output = io.StringIO()
    write_archive_diff(before, after, "task", output)
    text = output.getvalue()
    for old, new in [
        ("Solve old question", "Solve new question"),
        ("FROM old", "FROM new"),
        ("old check", "new check"),
        ("old public source", "new public source"),
    ]:
        assert f"-{old}\n" in text and f"+{new}\n" in text
    assert "-deleted content\n" in text
    assert '"candidate": null' in text
    assert "Binary content differs" in text
    assert '"sha256"' in text


def test_disk_index_matches_reordered_source_paths_and_reports_unmatched_rows(tmp_path):
    rows = [{"original_path": str(index), "task_json": str(index)} for index in range(65)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet", row_group_size=25)
    with NormalizedIndex(tmp_path, tmp_path / "index.sqlite") as index:
        for position in [64, 0, 24, 25, 16, 40]:
            assert index.get(str(position)) == rows[position]
        assert index.get("absent") is None
        remaining = list(index.unmatched())
    assert remaining == [row for position, row in enumerate(rows) if position not in {64, 0, 24, 25, 16, 40}]


def test_file_map_snapshot_matches_actual_export_archives():
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("fixture", source, {}), prompt="Name a color", answers=("red",), ignore_case=False)
    assert isinstance(task, TaskSpec)
    task = task.model_copy(
        update={
            "resources": ResourceGroups(
                worker=(inline_resource("app/input.txt", b"public input").model_copy(update={"mode": "600"}),),
                verifier=(inline_resource("helper.sh", b"private helper").model_copy(update={"mode": "700"}),),
                oracle=(inline_resource("solution/solve.sh", b"echo red").model_copy(update={"mode": "750"}),),
            )
        }
    )
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "fixture.tar.gz"}
    options = {
        "grader_image": "example.test/grader@sha256:" + "a" * 64,
        "family": "fixture",
        "fallback_actor_image": "python:3.12",
    }
    payload = harbor_payload(row, **options)
    record = harbor_record(row, **options)
    direct = {
        **file_map_snapshot(
            payload.files, {name: archive_file_mode(name, payload.modes) for name in payload.files}, "task"
        ),
        **file_map_snapshot(
            payload.solution,
            {name: archive_file_mode(name, payload.solution_modes) for name in payload.solution},
            "oracle",
        ),
    }
    archived = task_snapshot(
        record.source,
        record.path,
        "converted",
        task_binary=gzip.compress(gzip.decompress(record.task_binary), compresslevel=9, mtime=0),
        solution_binary=record.solution_binary,
    )
    assert direct == archived.files
    assert direct["task/environment/files/app/input.txt"].mode == 0o600
    assert direct["task/tests/helper.sh"].mode == 0o700
    assert direct["oracle/solution/solve.sh"].mode == 0o750
    assert direct["task/tests/test.sh"].mode == 0o755
