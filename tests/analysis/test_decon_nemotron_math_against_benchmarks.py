# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis.decon_nemotron_math_against_benchmarks import (
    EvalSource,
    discover_eval_sources,
    summarize_decon_output,
)


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_parquet(path: Path, records: list[dict], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


def test_discover_eval_sources_from_manifest(tmp_path: Path) -> None:
    output_file = tmp_path / "gsm8k" / "main" / "test" / "data.jsonl.gz"
    _write_jsonl_gz(output_file, [{"id": "x", "text": "Question"}])
    (tmp_path / ".manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "output_file": str(output_file),
                        "dataset_id": "openai/gsm8k",
                        "revision": "rev",
                        "config": "main",
                        "split": "test",
                        "record_count": 1,
                    }
                ]
            }
        )
    )

    sources = discover_eval_sources(str(tmp_path))

    assert sources == [
        EvalSource(
            key="gsm8k__main__test",
            source_dir=str(output_file.parent),
            data_file=str(output_file),
            dataset_id="openai/gsm8k",
            revision="rev",
            config="main",
            split="test",
            record_count_hint=1,
        )
    ]


def test_summarize_decon_output_counts_docs_and_eval_hits(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval" / "data.jsonl.gz"
    _write_jsonl_gz(
        eval_file,
        [
            {"id": "eval:a", "text": "A"},
            {"id": "eval:b", "text": "B"},
            {"id": "eval:c", "text": "C"},
        ],
    )
    attrs_dir = tmp_path / "attrs"
    _write_parquet(
        attrs_dir / "part-00000-of-00001.parquet",
        [
            {
                "id": "doc1",
                "partition_id": 0,
                "contaminated": True,
                "max_overlap": 1.0,
                "matched_hashes": [1, 2],
            },
            {
                "id": "doc2",
                "partition_id": 0,
                "contaminated": False,
                "max_overlap": 0.25,
                "matched_hashes": [],
            },
            {
                "id": "doc3",
                "partition_id": 0,
                "contaminated": True,
                "max_overlap": 0.5,
                "matched_hashes": [2, 3],
            },
        ],
    )
    _write_parquet(
        attrs_dir / "_bloom" / "eval_hash_index.parquet",
        [
            {"hash": 1, "eval_id": "eval:a"},
            {"hash": 2, "eval_id": "eval:b"},
            {"hash": 4, "eval_id": "eval:c"},
        ],
        schema=pa.schema([("hash", pa.uint64()), ("eval_id", pa.string())]),
    )
    source = EvalSource(
        key="toy__test",
        source_dir=str(eval_file.parent),
        data_file=str(eval_file),
        dataset_id="toy/eval",
        revision="rev",
        config=None,
        split="test",
    )

    report = summarize_decon_output(
        source=source,
        decon_output=str(attrs_dir),
        eval_record_count=3,
        top_k_eval_ids=10,
    )

    assert report["total_docs"] == 3
    assert report["contaminated_docs"] == 2
    assert report["clean_docs"] == 1
    assert report["contamination_rate"] == 2 / 3
    assert report["eval_records_with_feature_hit"] == 2
    assert report["eval_record_hit_rate"] == 2 / 3
    assert sorted(report["top_eval_ids_by_matched_features"], key=lambda item: item["eval_id"]) == [
        {"eval_id": "eval:a", "matched_feature_rows": 1},
        {"eval_id": "eval:b", "matched_feature_rows": 1},
    ]


def test_summarize_decon_output_accepts_nested_datakit_attributes(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval" / "data.jsonl.gz"
    _write_jsonl_gz(eval_file, [{"id": "eval:a", "text": "A"}])
    attrs_dir = tmp_path / "attrs"
    _write_parquet(
        attrs_dir / "part-00000-of-00001.parquet",
        [
            {
                "id": "doc1",
                "partition_id": 0,
                "attributes": {
                    "contaminated": True,
                    "max_overlap": 0.75,
                    "matched_hashes": [1],
                },
            }
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
        attrs_dir / "_bloom" / "eval_hash_index.parquet",
        [{"hash": 1, "eval_id": "eval:a"}],
        schema=pa.schema([("hash", pa.uint64()), ("eval_id", pa.string())]),
    )
    source = EvalSource(
        key="math500__test",
        source_dir=str(eval_file.parent),
        data_file=str(eval_file),
        dataset_id="toy/eval",
        revision="rev",
        config=None,
        split="test",
    )

    report = summarize_decon_output(
        source=source,
        decon_output=str(attrs_dir),
        eval_record_count=1,
        top_k_eval_ids=10,
    )

    assert report["contaminated_docs"] == 1
    assert report["mean_contaminated_max_overlap"] == 0.75
    assert report["eval_records_with_feature_hit"] == 1
