# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finelog.client import ObjectQueryClient
from finelog.errors import StatsError


def _write_catalog(
    root: Path,
    *,
    catalog_bytes_override: bytes | None = None,
    active_version: int = 1,
    l0_mode: str = "L0_MODE_OBJECT_NATIVE",
) -> Path:
    namespace = "iris.worker"
    native = root / "_native" / "namespaces" / namespace
    object_key = "objects/v1/l0/content/seg_L0_0000000000000000001.parquet"
    object_path = native / object_key
    object_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "seq": [1, 2],
                "worker_id": ["w-1", "w-2"],
                "mem_bytes": [10, 20],
                "cluster": [None, None],
            }
        ),
        object_path,
    )
    catalog = {
        "formatVersion": "1",
        "namespace": namespace,
        "catalogGeneration": "7",
        "activeTableSpecVersion": str(active_version),
        "desiredTableSpecVersion": "0",
        "retainedTableSpecs": [
            {
                "version": "1",
                "logicalSchema": {
                    "columns": [
                        {"name": "worker_id", "type": "COLUMN_TYPE_STRING"},
                        {"name": "mem_bytes", "type": "COLUMN_TYPE_INT64"},
                    ]
                },
                "operatingPolicy": {"l0Mode": l0_mode},
            }
        ],
        "versionSegments": [
            {
                "tableSpecVersion": str(active_version),
                "liveSegments": [
                    {
                        "segmentId": object_path.name,
                        "source": {"uri": object_key},
                        "rowCount": "2",
                    }
                ],
            }
        ],
    }
    catalog_bytes = catalog_bytes_override or json.dumps(catalog, separators=(",", ":")).encode()
    catalog_key = "catalogs/00000000000000000007-test.json"
    catalog_path = native / catalog_key
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(catalog_bytes)
    head = {
        "formatVersion": "1",
        "namespace": namespace,
        "catalogGeneration": "7",
        "activeTableSpecVersion": str(active_version),
        "catalog": {
            "uri": catalog_key,
            "sha256": base64.b64encode(hashlib.sha256(catalog_bytes).digest()).decode(),
        },
    }
    (native / "HEAD.json").write_text(json.dumps(head))
    return catalog_path


def test_object_query_reads_the_pinned_active_catalog_and_reports(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    starts: list[tuple] = []
    finishes: list[tuple] = []
    client = ObjectQueryClient(
        str(tmp_path),
        report_start=lambda *args: starts.append(args),
        report_finish=lambda *args: finishes.append(args),
    )

    result = client.query(
        'SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY seq',
        namespaces=["iris.worker"],
    )

    assert result.to_pydict() == {
        "worker_id": ["w-1", "w-2"],
        "mem_bytes": [10, 20],
    }
    assert len(starts) == 1
    assert starts[0][2][0].catalog_generation == 7
    assert starts[0][2][0].table_spec_version == 1
    assert len(finishes) == 1
    assert finishes[0][3] is True
    assert finishes[0][2] == 2


def test_object_query_rejects_catalog_bytes_that_do_not_match_head(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog_path.write_bytes(b"{}")

    with pytest.raises(StatsError):
        ObjectQueryClient(str(tmp_path)).query(
            'SELECT * FROM "iris.worker"',
            namespaces=["iris.worker"],
        )


def test_object_query_reporting_is_best_effort(tmp_path: Path) -> None:
    _write_catalog(tmp_path)

    def unavailable(*_args) -> None:
        raise ConnectionError("server unavailable")

    result = ObjectQueryClient(
        str(tmp_path),
        report_start=unavailable,
        report_finish=unavailable,
    ).query('SELECT count(*) AS rows FROM "iris.worker"', namespaces=["iris.worker"])

    assert result.column("rows").to_pylist() == [2]


@pytest.mark.parametrize(
    ("active_version", "l0_mode"),
    [
        (0, "L0_MODE_OBJECT_NATIVE"),
        (1, "L0_MODE_LEGACY_LOCAL"),
    ],
)
def test_object_query_rejects_catalog_without_an_active_native_version(
    tmp_path: Path,
    active_version: int,
    l0_mode: str,
) -> None:
    _write_catalog(tmp_path, active_version=active_version, l0_mode=l0_mode)

    with pytest.raises(StatsError):
        ObjectQueryClient(str(tmp_path)).query(
            'SELECT * FROM "iris.worker"',
            namespaces=["iris.worker"],
        )
