# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis.localize_decon_contaminated_docs import localize_overlaps


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_localize_overlaps_records_exact_source_and_eval_paragraphs(tmp_path: Path) -> None:
    source_parquet = tmp_path / "normalized" / "part-00000-of-00001.parquet"
    manifest = tmp_path / "contaminated_docs.parquet"
    eval_data = tmp_path / "math500" / "test" / "data.jsonl.gz"
    output_root = tmp_path / "localized"

    shared_tokens = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    source_text = f"short intro\n{shared_tokens}"
    eval_text = f"Problem:\n{shared_tokens} eval tail\n\nAnswer:\n42"

    _write_parquet(
        source_parquet,
        [
            {"id": "clean-doc", "text": "not relevant"},
            {"id": "contam-doc", "text": source_text},
        ],
    )
    _write_parquet(
        manifest,
        [
            {
                "id": "contam-doc",
                "partition_id": 0,
                "row_index_in_partition": 1,
                "source_parquet": str(source_parquet),
                "attr_parquet": str(tmp_path / "attrs" / "part-00000-of-00001.parquet"),
                "max_overlap": 1.0,
                "matched_hashes": [1],
                "matched_eval_ids": ["math500:test:0"],
                "matched_eval_id_count": 1,
            }
        ],
        schema=pa.schema(
            [
                ("id", pa.string()),
                ("partition_id", pa.int64()),
                ("row_index_in_partition", pa.int64()),
                ("source_parquet", pa.string()),
                ("attr_parquet", pa.string()),
                ("max_overlap", pa.float64()),
                ("matched_hashes", pa.list_(pa.uint64())),
                ("matched_eval_ids", pa.list_(pa.string())),
                ("matched_eval_id_count", pa.int64()),
            ]
        ),
    )
    _write_jsonl_gz(eval_data, [{"id": "math500:test:0", "text": eval_text}])

    summary = localize_overlaps(
        manifest_path=str(manifest),
        eval_data=str(eval_data),
        output_root=str(output_root),
        ngram_length=13,
        stride=0,
        min_source_containment=0.5,
        min_jaccard=0.0,
        max_snippet_chars=200,
        max_shared_ngrams=5,
        max_workers=2,
        worker_cpu=1,
        worker_ram="512m",
        worker_disk="1g",
        resume=False,
        force=False,
    )

    assert summary["manifest_docs"] == 1
    assert summary["docs_with_localized_overlap"] == 1
    assert summary["docs_without_localized_overlap"] == 0
    rows = pq.read_table(output_root / "localized_overlaps.parquet").to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == "contam-doc"
    assert row["eval_id"] == "math500:test:0"
    assert row["source_paragraph_index"] == 1
    assert row["source_char_start"] == len("short intro\n")
    assert row["record_source_containment"] == 1.0
    assert row["record_jaccard"] == 1 / 3
    assert row["best_eval_paragraph_index"] == 1
    assert row["best_eval_source_containment"] == 1.0
    assert row["shared_ngrams"] == [shared_tokens]
    assert pq.read_table(output_root / "localization_misses.parquet").num_rows == 0
    assert (output_root / "localized_overlaps.csv").exists()
    assert json.loads((output_root / "summary.json").read_text())["localized_rows"] == 1
