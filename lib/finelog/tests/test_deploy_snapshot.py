# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import subprocess
import tarfile
from pathlib import Path

import pytest
from finelog.deploy.snapshot import (
    STORE_DIR,
    CatalogSegment,
    extract_tar,
    namespaces_in_catalog,
    plan_snapshot,
    read_catalog_segments,
    store_relative,
    tar_command,
)


def _segment(namespace: str, seq: int, *, byte_size: int = 1) -> CatalogSegment:
    return CatalogSegment(
        namespace=namespace,
        path=f"{STORE_DIR}/{namespace}/seg-{seq:06d}.parquet",
        max_seq=seq,
        byte_size=byte_size,
    )


def _catalog(path: Path, segments: list[CatalogSegment], locations: list[str], namespaces: list[str]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE namespaces (namespace TEXT PRIMARY KEY, schema_json TEXT)")
    connection.execute(
        "CREATE TABLE segments (" "namespace TEXT, path TEXT, max_seq INTEGER, byte_size INTEGER, location TEXT)"
    )
    connection.executemany("INSERT INTO namespaces VALUES (?, '{}')", [(name,) for name in namespaces])
    connection.executemany(
        "INSERT INTO segments VALUES (?, ?, ?, ?, ?)",
        [
            (segment.namespace, segment.path, segment.max_seq, segment.byte_size, location)
            for segment, location in zip(segments, locations, strict=True)
        ],
    )
    connection.commit()
    connection.close()


def test_a_snapshot_takes_the_newest_segments_of_each_namespace() -> None:
    segments = [_segment("telemetry_v1", seq) for seq in range(10)] + [_segment("log", seq) for seq in range(3)]

    plan = plan_snapshot(segments, segments_per_namespace=2)

    assert plan.selected_per_namespace == {"telemetry_v1": 2, "log": 2}
    assert plan.skipped_per_namespace == {"telemetry_v1": 8, "log": 1}
    assert "log/seg-000002.parquet" in plan.patterns
    assert "telemetry_v1/seg-000009.parquet" in plan.patterns
    assert "telemetry_v1/seg-000007.parquet" not in plan.patterns


def test_every_selected_segment_brings_its_index_sidecars() -> None:
    plan = plan_snapshot([_segment("log", 1)], segments_per_namespace=1)

    # The `.fidx` bundle and each covering projection's Parquet share the
    # segment's name as a prefix; without them a boot re-derives indexes and
    # rehearses the wrong startup path.
    assert "log/seg-000001.parquet" in plan.patterns
    assert "log/seg-000001.parquet.fidx*" in plan.patterns


def test_the_catalog_is_always_copied_even_when_no_segment_fits() -> None:
    plan = plan_snapshot([_segment("log", 1, byte_size=10**9)], max_bytes=1)

    # The first segment is taken regardless — a plan that copied nothing but the
    # catalog would rehearse adoption over an empty store.
    assert plan.selected_per_namespace == {"log": 1}
    assert plan.patterns[:2] == ("_finelog_catalog.sqlite*", ".finelog-rust-catalog")


def test_a_byte_budget_is_not_spent_entirely_on_the_busiest_namespace() -> None:
    segments = [_segment("telemetry_v1", seq, byte_size=40) for seq in range(8)]
    segments += [_segment("log", seq, byte_size=40) for seq in range(8)]

    plan = plan_snapshot(segments, segments_per_namespace=8, max_bytes=100)

    assert plan.selected_bytes <= 120
    assert set(plan.selected_per_namespace) == {"log", "telemetry_v1"}


def test_a_store_mounted_somewhere_else_is_refused_rather_than_copied_wrong() -> None:
    # Adoption matches catalog rows to files by exact absolute path, so a
    # snapshot from an unexpected mount point would silently adopt nothing.
    with pytest.raises(ValueError, match="not under /var/cache/finelog"):
        store_relative("/srv/finelog/log/seg-000001.parquet")


def test_the_tar_stream_carries_the_matched_paths_and_skips_the_rest(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "log").mkdir(parents=True)
    (store / "_finelog_catalog.sqlite").write_text("catalog")
    (store / "log" / "seg-1.parquet").write_text("rows")
    (store / "log" / "seg-1.parquet.fidx").write_text("index")
    (store / "log" / "seg-2.parquet").write_text("not selected")

    # A segment with no index bundle contributes a pattern that matches nothing;
    # the archive must still carry everything else rather than abort.
    stream = subprocess.run(
        ["sh", "-c", tar_command(str(store), ("_finelog_catalog.sqlite*", "log/seg-1.parquet*", "log/seg-9.parquet*"))],
        capture_output=True,
        check=True,
    ).stdout

    destination = tmp_path / "snapshot"
    archive = tmp_path / "store.tar"
    archive.write_bytes(stream)
    extract_tar(archive, destination)

    assert sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()) == [
        "_finelog_catalog.sqlite",
        "log/seg-1.parquet",
        "log/seg-1.parquet.fidx",
    ]


def test_only_segments_still_on_the_host_are_planned(tmp_path: Path) -> None:
    catalog = tmp_path / "_finelog_catalog.sqlite"
    segments = [_segment("log", 1), _segment("log", 2), _segment("log", 3)]
    _catalog(catalog, segments, ["LOCAL", "BOTH", "REMOTE"], ["log"])

    read = read_catalog_segments(catalog)

    # A `REMOTE` row's local Parquet has already been unlinked by maintenance.
    assert {segment.max_seq for segment in read} == {1, 2}
    assert namespaces_in_catalog(catalog) == {"log"}


def test_an_archive_cannot_name_where_it_lands(tmp_path: Path) -> None:
    # The archive is built by a shell on a production host; it is data, not a
    # trusted instruction to write outside the snapshot directory.
    payload = tmp_path / "payload"
    payload.write_text("x")
    archive = tmp_path / "store.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="../escaped")

    destination = tmp_path / "snapshot"
    with pytest.raises(tarfile.OutsideDestinationError):
        extract_tar(archive, destination)
    assert not (tmp_path / "escaped").exists()
