# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName

from experiments.datakit.embeddings import harrier
from experiments.datakit.embeddings.harrier import (
    InputPart,
    SourceFile,
    allocate_source_quotas,
    assigned_parts,
    inference_groups,
    resolve_shard_index,
)


def test_source_inventory_uses_canonical_nested_names(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = {"family/nested": object(), "standalone": object()}
    monkeypatch.setattr(harrier, "all_sources", lambda: sources)
    monkeypatch.setattr(
        harrier,
        "_source_files",
        lambda source: (SourceFile(input_url=f"s3://input/{source}/part.parquet", row_count=1),),
    )

    inventory = harrier._source_inventory()

    assert set(inventory) == set(sources)


def test_source_inventory_rejects_missing_canonical_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harrier, "all_sources", lambda: {"complete": object(), "missing/nested": object()})
    monkeypatch.setattr(
        harrier,
        "_source_files",
        lambda source: () if source == "missing/nested" else (SourceFile("s3://input/part.parquet", 1),),
    )

    with pytest.raises(ValueError, match="missing/nested"):
        harrier._source_inventory()


def test_resolve_shard_index_uses_iris_replica_index(monkeypatch: pytest.MonkeyPatch) -> None:
    job_info = JobInfo(task_id=JobName.from_wire("/held/harrier/37"))
    monkeypatch.setattr(harrier, "get_job_info", lambda: job_info)

    assert resolve_shard_index(None) == 37
    assert resolve_shard_index(12) == 12


def test_allocate_source_quotas_hits_exact_proportional_target() -> None:
    quotas = allocate_source_quotas({"large": 30, "medium": 15, "small": 5}, 17)

    assert quotas == {"large": 10, "medium": 5, "small": 2}
    assert sum(quotas.values()) == 17


def test_assigned_parts_covers_every_part_once() -> None:
    parts = tuple(
        InputPart(source="source", input_url=f"s3://input/{index}", row_count=rows, output_url=f"s3://output/{index}")
        for index, rows in enumerate((9, 8, 7, 6, 5))
    )

    assignments = assigned_parts(parts, 3)

    assert {part.input_url for assignment in assignments for part in assignment} == {part.input_url for part in parts}
    assert sum(part.row_count for assignment in assignments for part in assignment) == 35
    assert max(sum(part.row_count for part in assignment) for assignment in assignments) == 13


def test_inference_groups_respects_padded_token_budget() -> None:
    lengths = [8_192, 8_192, 4_000, 1_000, 900, 100]

    groups = inference_groups(lengths, max_batch_tokens=8_192, max_batch_size=4)

    assert sorted(index for group in groups for index in group) == list(range(len(lengths)))
    assert all(max(lengths[index] for index in group) * len(group) <= 8_192 for group in groups)
    assert all(len(group) <= 4 for group in groups)


def test_input_batches_truncate_large_text_before_embedding(tmp_path) -> None:
    input_path = tmp_path / "input.parquet"
    oversized_text = "x" * 1_048_577
    pq.write_table(pa.table({"id": ["large", "small"], "text": [oversized_text, "short"]}), input_path)
    part = InputPart(source="source", input_url=str(input_path), row_count=2, output_url="unused")

    batches = list(harrier._input_batches(part))
    texts = [text.as_py() for batch in batches for text in batch.column("text")]

    assert texts == [oversized_text[:1_048_576], "short"]


def test_embedding_sanity_metrics_measure_norms_and_reembedding_alignment() -> None:
    stored = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float16)
    recomputed = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    metrics = harrier.embedding_sanity_metrics(stored, recomputed)

    assert metrics["nonfinite_value_count"] == 0
    assert metrics["norm_min"] == pytest.approx(1.0)
    assert metrics["norm_mean"] == pytest.approx(1.5)
    assert metrics["norm_max"] == pytest.approx(2.0)
    assert metrics["norm_max_error"] == pytest.approx(1.0)
    assert metrics["reembed_cosine_min"] == pytest.approx(1.0)
    assert metrics["reembed_cosine_mean"] == pytest.approx(1.0)
