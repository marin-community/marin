# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analysis.build_localized_overlap_viewer import DatasetSpec, main, multi_viewer_payload, viewer_payload


def _write_localized_rows(path: Path) -> None:
    rows = [
        {
            "doc_id": "doc-a",
            "source_parquet": "gs://bucket/source/part-00000.parquet",
            "partition_id": 0,
            "row_index_in_partition": 12,
            "datakit_max_overlap": 1.0,
            "eval_id": "math500:test:algebra/0.json",
            "source_paragraph_index": 1,
            "source_char_start": 7,
            "source_char_end": 104,
            "source_token_count": 18,
            "source_ngram_count": 6,
            "record_intersection_count": 4,
            "record_source_containment": 0.75,
            "record_source_unique_containment": 0.8,
            "record_eval_containment": 0.5,
            "record_jaccard": 0.4,
            "best_eval_paragraph_index": 2,
            "best_eval_char_start": 11,
            "best_eval_char_end": 92,
            "best_eval_source_containment": 0.75,
            "best_eval_jaccard": 0.4,
            "source_snippet": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu source",
            "eval_snippet": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu eval",
            "shared_ngrams": ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"],
        },
        {
            "doc_id": "doc-b",
            "source_parquet": "gs://bucket/source/part-00001.parquet",
            "partition_id": 1,
            "row_index_in_partition": 3,
            "datakit_max_overlap": 0.5,
            "eval_id": "math500:test:number_theory/1.json",
            "source_paragraph_index": 0,
            "source_char_start": 0,
            "source_char_end": 90,
            "source_token_count": 14,
            "source_ngram_count": 2,
            "record_intersection_count": 1,
            "record_source_containment": 0.5,
            "record_source_unique_containment": 0.5,
            "record_eval_containment": 0.2,
            "record_jaccard": 0.125,
            "best_eval_paragraph_index": 0,
            "best_eval_char_start": 0,
            "best_eval_char_end": 70,
            "best_eval_source_containment": 0.5,
            "best_eval_jaccard": 0.125,
            "source_snippet": "second source",
            "eval_snippet": "second eval",
            "shared_ngrams": [],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_viewer_payload_compacts_localized_overlap_rows(tmp_path: Path) -> None:
    path = tmp_path / "localized_overlaps.parquet"
    _write_localized_rows(path)

    payload = viewer_payload(str(path), max_rows=None)

    assert payload["summary"]["rows"] == 2
    assert payload["summary"]["docs"] == 2
    assert payload["summary"]["evalRecords"] == 2
    assert payload["summary"]["maxJaccard"] == 0.4
    first = payload["rows"][0]
    assert first["doc"] == "doc-a"
    assert first["sourceShard"] == "part-00000.parquet"
    assert first["row"] == 12
    assert first["sourceSpan"] == [7, 104]
    assert first["shared"] == ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"]


def test_viewer_main_writes_html(tmp_path: Path) -> None:
    input_path = tmp_path / "localized_overlaps.parquet"
    output_path = tmp_path / "viewer.html"
    _write_localized_rows(input_path)

    main(
        [
            "--input",
            str(input_path),
            "--local-output",
            str(output_path),
            "--max-rows",
            "1",
            "--no-gcs-upload",
        ]
    )

    html_text = output_path.read_text(encoding="utf-8")
    assert "__PAYLOAD_JSON__" not in html_text
    assert "Benchmark Localized Overlaps" in html_text
    assert "Localized overlaps" in html_text
    assert "doc-a" in html_text
    assert "doc-b" not in html_text
    assert "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu" in html_text


def test_multi_viewer_payload_includes_dataset_metadata(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "localized_overlaps.parquet"
    second_path = tmp_path / "second" / "localized_overlaps.parquet"
    _write_localized_rows(first_path)
    _write_localized_rows(second_path)

    payload = multi_viewer_payload(
        [
            DatasetSpec("first", "First split", "First eval paragraph", str(first_path)),
            DatasetSpec("second", "Second split", "Second eval paragraph", str(second_path)),
        ],
        max_rows=1,
    )

    assert payload["defaultDataset"] == "first"
    assert [dataset["key"] for dataset in payload["datasets"]] == ["first", "second"]
    assert payload["datasets"][0]["label"] == "First split"
    assert payload["datasets"][1]["evalLabel"] == "Second eval paragraph"
    assert payload["datasets"][0]["summary"]["rows"] == 1
    assert payload["datasets"][1]["rows"][0]["doc"] == "doc-a"
