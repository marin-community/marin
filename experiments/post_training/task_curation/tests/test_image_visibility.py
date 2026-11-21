# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the built-image registry map and its GHCR visibility check."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation import image_registry
from experiments.post_training.task_curation.check_image_visibility import (
    CheckOutcome,
    ImageVisibility,
    MissingPackage,
    PackageVisibilityError,
    check_registry,
    ghcr_package,
)
from experiments.post_training.task_curation.image_registry import (
    MAP_COLUMNS,
    payload_task_rows,
    pinned_digest,
    source_map_rows,
)

DIGEST = "sha256:" + "1" * 64
ROLLOUT = f"ghcr.io/marin-community/rollout-images@{DIGEST}"


def source_columns(
    source_id: str = "TaskTrove:demo",
    *,
    data_restriction: str = DataRestriction.OPEN.value,
    publication_scope: str = "",
) -> dict[str, str]:
    """The source-identity columns of one map row."""
    return {
        "source_id": source_id,
        "dataset_id": "open-athena/demo",
        "dataset_revision": "rev1",
        "data_restriction": data_restriction,
        "publication_scope": publication_scope,
    }


def task(task_id: str = "task-1", *, reference: str = ROLLOUT) -> dict:
    """One frozen task environment shape the map reads."""
    return {
        "id": task_id,
        "source": {"row": f"row-{task_id}"},
        "environment_requirements": {"docker_image": reference, "command_semantics": "LINUX_PROCESS"},
    }


def map_row(source: dict[str, str], task_id: str = "task-1", reference: str = ROLLOUT) -> dict[str, str]:
    """One complete map row for the visibility check."""
    row = {
        **source,
        "task_id": task_id,
        "task_row": f"row-{task_id}",
        "image_reference": reference,
        "image_digest": pinned_digest(reference),
    }
    return {column: row[column] for column in MAP_COLUMNS}


def checker(visibilities: dict[str, str]):
    """A fake package-visibility reader standing in for the gh CLI."""

    def visibility(package: str) -> str:
        return visibilities[package]

    return visibility


def payload_file(tmp_path: Path, tasks: list[dict], *, encoding: str = "canonical", tamper_hash: bool = False) -> Path:
    """One published payload parquet in either encoding."""
    if encoding == "canonical":
        rows = [
            {
                "task_spec": json.dumps(task),
                "task_sha256": "0" * 64 if tamper_hash else hashlib.sha256(json.dumps(task).encode()).hexdigest(),
            }
            for task in tasks
        ]
    else:
        rows = [{"lowered_task_spec": json.dumps({"task": task, "runtime": {"cpus": 1}})} for task in tasks]
    path = tmp_path / f"tasks-{encoding}-{len(tasks)}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_over_exposed_public_package_with_internal_allowance_is_reported():
    check = check_registry(
        [map_row(source_columns())],
        visibility=checker({"marin-community/rollout-images": "public"}),
    )
    (verdict,) = check.verdicts
    assert verdict.outcome is CheckOutcome.OVER_EXPOSED
    assert verdict.allowance is ImageVisibility.INTERNAL
    assert verdict.actual is ImageVisibility.PUBLIC
    assert verdict.sources == ("TaskTrove:demo",)
    assert check.failures == check.verdicts


def test_matching_visibility_passes():
    rows = [
        map_row(source_columns("TaskTrove:display", publication_scope="public_display")),
        map_row(
            source_columns("TaskTrove:plain"),
            task_id="task-2",
            reference="ghcr.io/marin-community/other-images@sha256:" + "3" * 64,
        ),
    ]
    check = check_registry(
        rows,
        visibility=checker(
            {
                "marin-community/rollout-images": "public",
                "marin-community/other-images": "private",
            }
        ),
    )
    assert [verdict.outcome for verdict in check.verdicts] == [CheckOutcome.MATCH, CheckOutcome.MATCH]
    assert check.failures == ()


def test_shared_package_uses_the_most_restrictive_source_allowance():
    """One package shared by a public-display source and a plain open source."""
    rows = [
        map_row(source_columns("TaskTrove:display", publication_scope="public_display"), task_id="task-a"),
        map_row(source_columns("TaskTrove:plain"), task_id="task-b"),
    ]
    allowed = check_registry(rows, visibility=checker({"marin-community/rollout-images": "private"}))
    assert allowed.verdicts[0].outcome is CheckOutcome.MATCH
    assert allowed.verdicts[0].allowance is ImageVisibility.INTERNAL

    over_exposed = check_registry(rows, visibility=checker({"marin-community/rollout-images": "public"}))
    assert over_exposed.verdicts[0].outcome is CheckOutcome.OVER_EXPOSED
    assert over_exposed.verdicts[0].sources == ("TaskTrove:display", "TaskTrove:plain")


def test_private_always_source_refuses_even_an_org_private_package():
    """A private_always source's image must not sit in any GHCR package."""
    check = check_registry(
        [map_row(source_columns(data_restriction=DataRestriction.PRIVATE_ALWAYS.value))],
        visibility=checker({"marin-community/rollout-images": "private"}),
    )
    verdict = check.verdicts[0]
    assert verdict.outcome is CheckOutcome.OVER_EXPOSED
    assert verdict.allowance is ImageVisibility.PRIVATE
    assert verdict.actual is ImageVisibility.INTERNAL


def test_missing_package_fails_closed():
    def visibility(package: str) -> str:
        raise MissingPackage(package)

    check = check_registry([map_row(source_columns())], visibility=visibility)
    assert check.verdicts[0].outcome is CheckOutcome.MISSING
    assert check.failures == check.verdicts


def test_failed_query_fails_closed():
    def visibility(package: str) -> str:
        raise PackageVisibilityError("gh is not authenticated")

    check = check_registry([map_row(source_columns())], visibility=visibility)
    assert check.verdicts[0].outcome is CheckOutcome.QUERY_ERROR
    assert check.verdicts[0].actual is None
    assert check.failures == check.verdicts


def test_references_outside_the_audited_org_are_not_checked():
    rows = [
        map_row(source_columns("TaskTrove:base"), task_id="task-1", reference="python:3.12"),
        map_row(
            source_columns("TaskTrove:other-org"), task_id="task-2", reference="ghcr.io/other-org/img@sha256:" + "2" * 64
        ),
        map_row(source_columns("TaskTrove:demo"), task_id="task-3"),
    ]
    check = check_registry(rows, visibility=checker({"marin-community/rollout-images": "public"}))
    assert [verdict.outcome for verdict in check.verdicts] == [CheckOutcome.OVER_EXPOSED]
    assert check.out_of_scope == (
        "ghcr.io/other-org/img@sha256:" + "2" * 64,
        "python:3.12",
    )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (f"ghcr.io/marin-community/rollout-images@{DIGEST}", "marin-community/rollout-images"),
        ("ghcr.io/marin-community/rollout-images:task-curation-env-abc123", "marin-community/rollout-images"),
        ("ghcr.io/marin-community/marin/tpu-ci:latest", "marin-community/marin/tpu-ci"),
        (f"ghcr.io/Marin-Community/Rollout-Images@{DIGEST}", "marin-community/rollout-images"),
        ("docker.io/library/python:3.12", None),
        ("iris-task", None),
        ("ghcr.io/other-org/img:tag", "other-org/img"),
    ],
)
def test_package_extraction_from_image_references(reference: str, expected: str | None):
    assert ghcr_package(reference) == expected


def test_source_map_rows_record_one_row_per_task_image(tmp_path: Path):
    payload = payload_file(
        tmp_path,
        [
            task("task-1"),
            task("task-2"),
            {"id": "task-3", "source": {"row": "row-3"}, "environment_requirements": {"packages_lock": "s3://lock"}},
        ],
    )
    rows, without_image = source_map_rows(source_columns(), payload)
    assert [row["task_id"] for row in rows] == ["task-1", "task-2"]
    assert without_image == 1
    assert rows[0]["image_reference"] == ROLLOUT
    assert rows[0]["image_digest"] == DIGEST
    assert rows[0]["source_id"] == "TaskTrove:demo"
    assert rows[0]["task_row"] == "row-task-1"


def test_source_map_rows_refuse_a_repeated_task(tmp_path: Path):
    payload = payload_file(tmp_path, [task("task-1"), task("task-1")])
    with pytest.raises(ValueError, match="repeats task task-1"):
        source_map_rows(source_columns(), payload)


def test_payload_rows_read_the_lowered_envelope(tmp_path: Path):
    payload = payload_file(tmp_path, [task("task-1"), task("task-2")], encoding="lowered")
    tasks = list(payload_task_rows(payload))
    assert [task["id"] for task in tasks] == ["task-1", "task-2"]


def test_payload_rows_verify_the_recorded_task_hash(tmp_path: Path):
    payload = payload_file(tmp_path, [task("task-1")], tamper_hash=True)
    with pytest.raises(ValueError, match="recorded hash"):
        list(payload_task_rows(payload))


def test_payload_rows_refuse_an_unknown_layout(tmp_path: Path):
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.table({"path": ["a"]}), path)
    with pytest.raises(ValueError, match="neither task_spec nor lowered_task_spec"):
        list(payload_task_rows(path))


def test_build_map_rows_counts_sources_without_a_payload(monkeypatch, tmp_path: Path):
    """A source pinning an upstream dataset has no tasks.parquet; it is counted, not read."""
    payload = payload_file(tmp_path, [task("task-1")])
    pins = [
        (source_columns("TaskTrove:cohort"), ("open-athena/cohort", "rev1")),
        (source_columns("TaskTrove:upstream"), ("allenai/ai2_arc", "rev2")),
        (source_columns("TaskTrove:no-pin"), None),
    ]
    monkeypatch.setattr(image_registry, "served_source_columns", lambda root: iter(pins))
    monkeypatch.setattr(
        image_registry,
        "download_payload",
        lambda dataset_id, revision: payload if dataset_id.endswith("cohort") else None,
    )
    rows, summary = image_registry.build_map_rows("s3://catalog")
    assert [row["source_id"] for row in rows] == ["TaskTrove:cohort"]
    assert "1 without a payload pin" in summary
    assert "1 whose pin carries no payload" in summary
