# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis.export_decon_contaminated_docs import export_contaminated_docs


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_export_contaminated_docs_records_source_parquet_and_eval_ids(tmp_path: Path) -> None:
    normalized_root = tmp_path / "normalized"
    attrs_root = tmp_path / "attrs"
    output_root = tmp_path / "out"
    part = "part-00000-of-00002.parquet"
    other_part = "part-00001-of-00002.parquet"

    _write_parquet(
        normalized_root / part,
        [
            {"id": "clean-doc", "text": "clean"},
            {"id": "contam-doc-a", "text": "hit a"},
            {"id": "contam-doc-b", "text": "hit b"},
        ],
    )
    _write_parquet(normalized_root / other_part, [{"id": "clean-other", "text": "other"}])
    _write_parquet(
        attrs_root / part,
        [
            {
                "id": "clean-doc",
                "partition_id": 0,
                "attributes": {"contaminated": False, "max_overlap": 0.25, "matched_hashes": []},
            },
            {
                "id": "contam-doc-a",
                "partition_id": 0,
                "attributes": {"contaminated": True, "max_overlap": 1.0, "matched_hashes": [11, 22]},
            },
            {
                "id": "contam-doc-b",
                "partition_id": 0,
                "attributes": {"contaminated": True, "max_overlap": 0.5, "matched_hashes": [33]},
            },
        ],
        schema=pa.schema(
            [
                ("id", pa.string()),
                ("partition_id", pa.int64()),
                (
                    "attributes",
                    pa.struct(
                        [
                            pa.field("contaminated", pa.bool_()),
                            pa.field("max_overlap", pa.float64()),
                            pa.field("matched_hashes", pa.list_(pa.uint64())),
                        ]
                    ),
                ),
            ]
        ),
    )
    _write_parquet(
        attrs_root / other_part,
        [{"id": "clean-other", "partition_id": 1, "contaminated": False, "max_overlap": 0.0, "matched_hashes": []}],
    )
    _write_parquet(
        attrs_root / "_bloom" / "eval_hash_index.parquet",
        [
            {"hash": 11, "eval_id": "math500/test:0"},
            {"hash": 22, "eval_id": "math500/test:1"},
            {"hash": 33, "eval_id": "math500/test:0"},
        ],
        schema=pa.schema([("hash", pa.uint64()), ("eval_id", pa.string())]),
    )

    summary = export_contaminated_docs(
        decon_attrs=str(attrs_root),
        normalized_root=str(normalized_root),
        output_root=str(output_root),
        resume=False,
        force=False,
    )

    assert summary["total_attr_rows"] == 4
    assert summary["contaminated_docs"] == 2
    assert summary["matched_eval_records"] == 2
    rows = pq.read_table(output_root / "contaminated_docs.parquet").to_pylist()
    assert rows == [
        {
            "id": "contam-doc-a",
            "partition_id": 0,
            "row_index_in_partition": 1,
            "source_parquet": str(normalized_root / part),
            "attr_parquet": str(attrs_root / part),
            "max_overlap": 1.0,
            "matched_hashes": [11, 22],
            "matched_eval_ids": ["math500/test:0", "math500/test:1"],
            "matched_eval_id_count": 2,
        },
        {
            "id": "contam-doc-b",
            "partition_id": 0,
            "row_index_in_partition": 2,
            "source_parquet": str(normalized_root / part),
            "attr_parquet": str(attrs_root / part),
            "max_overlap": 0.5,
            "matched_hashes": [33],
            "matched_eval_ids": ["math500/test:0"],
            "matched_eval_id_count": 1,
        },
    ]
    assert (output_root / "contaminated_docs.csv").exists()
    assert json.loads((output_root / "summary.json").read_text())["output_parquet"] == str(
        output_root / "contaminated_docs.parquet"
    )
