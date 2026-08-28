# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path

import duckdb
import finelog.client.object_query_client as object_query_client_mod
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finelog.client import ObjectQueryClient
from finelog.errors import QueryTimeoutError, StatsError


def _write_catalog(
    root: Path,
    *,
    catalog_bytes_override: bytes | None = None,
    active_version: int = 1,
    l0_mode: str = "L0_MODE_OBJECT_STORE",
    max_query_time_ms: int = 600_000,
    object_id_override: str | None = None,
) -> Path:
    namespace = "iris.worker"
    table_root = root / "_finelog" / "tables" / namespace
    canonical_object_id = f"_finelog/tables/{namespace}/objects/v1/l1/content/seg_L1_0000000000000000001.parquet"
    object_id = object_id_override or canonical_object_id
    object_path = root / canonical_object_id
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
        "maxQueryTimeMs": str(max_query_time_ms),
        "directQueryHighWater": "2",
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
                        "source": {"objectId": object_id},
                        "level": 1,
                        "rowCount": "2",
                    }
                ],
            }
        ],
        "directQuerySegments": [
            {
                "segmentId": object_path.name,
                "source": {"objectId": object_id},
                "level": 1,
                "minSeq": "1",
                "maxSeq": "2",
                "rowCount": "2",
            }
        ],
    }
    catalog_bytes = catalog_bytes_override or json.dumps(catalog, separators=(",", ":")).encode()
    catalog_key = "catalogs/00000000000000000007-test.json"
    catalog_id = f"_finelog/tables/{namespace}/{catalog_key}"
    catalog_path = root / catalog_id
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(catalog_bytes)
    head = {
        "formatVersion": "1",
        "namespace": namespace,
        "catalogGeneration": "7",
        "activeTableSpecVersion": str(active_version),
        "catalog": {
            "objectId": catalog_id,
            "sha256": base64.b64encode(hashlib.sha256(catalog_bytes).digest()).decode(),
        },
    }
    (table_root / "HEAD.json").write_text(json.dumps(head))
    return catalog_path


def test_object_query_reads_the_stable_catalog_projection(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    client = ObjectQueryClient(str(tmp_path))
    pin = client.pin_catalog("iris.worker")

    result = client.query(
        'SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY seq',
        namespaces=["iris.worker"],
    )

    assert result.to_pydict() == {
        "worker_id": ["w-1", "w-2"],
        "mem_bytes": [10, 20],
    }
    assert pin.high_water == 2


def test_object_query_rejects_catalog_bytes_that_do_not_match_head(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog_path.write_bytes(b"{}")

    with pytest.raises(StatsError):
        ObjectQueryClient(str(tmp_path)).query(
            'SELECT * FROM "iris.worker"',
            namespaces=["iris.worker"],
        )


def test_object_query_rejects_noncanonical_object_ids(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        object_id_override="_finelog/tables/iris.worker/objects/v1/../escaped.parquet",
    )

    with pytest.raises(StatsError, match="canonical relative object ID"):
        ObjectQueryClient(str(tmp_path)).pin_catalog("iris.worker")


def test_object_query_stops_at_the_catalog_lifetime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_catalog(tmp_path)

    class ImmediateTimer:
        def __init__(self, _interval: float, callback: Callable[[], None]) -> None:
            self._callback = callback

        def start(self) -> None:
            self._callback()

        def cancel(self) -> None:
            return None

        def join(self) -> None:
            return None

    class BlockingConnection:
        def __init__(self) -> None:
            self.interrupted = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def register_filesystem(self, _filesystem) -> None:
            pass

        def execute(self, _sql: str):
            return self

        def fetch_arrow_table(self) -> pa.Table:
            assert self.interrupted.is_set()
            raise duckdb.InterruptException("interrupted")

        def interrupt(self) -> None:
            self.interrupted.set()

    connection = BlockingConnection()
    monkeypatch.setattr(duckdb, "connect", lambda: connection)
    monkeypatch.setattr(object_query_client_mod.threading, "Timer", ImmediateTimer)

    with pytest.raises(QueryTimeoutError):
        ObjectQueryClient(str(tmp_path)).query(
            'SELECT * FROM "iris.worker"',
            namespaces=["iris.worker"],
        )
    assert connection.interrupted.is_set()


@pytest.mark.parametrize(
    ("active_version", "l0_mode"),
    [
        (0, "L0_MODE_OBJECT_STORE"),
        (1, "L0_MODE_LEGACY_LOCAL"),
    ],
)
def test_object_query_rejects_catalog_without_an_active_object_version(
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
