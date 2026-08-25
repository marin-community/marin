# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pytest
from marin.datakit.ingestion_manifest import (
    IngestionPolicy,
    IngestionSourceManifest,
    StagingMetadata,
    UsagePolicy,
)
from marin.transform.evaluation.continuation_records import (
    ContinuationStagingConfig,
    stage_continuation_slice,
)

RECORDS = [
    {"id": "task/template/ex-01", "input": "Q: 2 + 2\nA: ", "target": "4"},
    {"id": "task/template/ex-02", "input": "Q: 3 + 3\nA: ", "target": "6"},
]


def _manifest() -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key="static/continuation_probe",
        slice_key="test/task/template",
        source_label="continuation_probe",
        source_urls=("https://example.invalid/continuation_probe",),
        source_license="test-only",
        source_format="static_generator",
        surface_form="fewshot_continuation",
        policy=IngestionPolicy(usage_policy=UsagePolicy.EVAL_ONLY, use_policy="Eval-only probe."),
        staging=StagingMetadata(transform_name="stage_continuation_slice"),
    )


def test_staged_slice_metadata_describes_the_written_file(tmp_path: Path) -> None:
    manifest = _manifest()

    result = stage_continuation_slice(
        ContinuationStagingConfig(
            output_path=str(tmp_path),
            task_key="task",
            template_key="template",
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        ),
        records=RECORDS,
        source_id="continuation_static",
        metadata={"task_key": "task", "heldout_examples": len(RECORDS)},
    )

    with gzip.open(result["output_file"], "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle if line.strip()] == RECORDS
    assert result["record_count"] == len(RECORDS)
    assert result["bytes_written"] == Path(result["output_file"]).stat().st_size

    metadata = json.loads(Path(result["metadata_file"]).read_text())
    assert metadata["content_fingerprint"] == manifest.fingerprint()
    materialized = metadata["materialized_output"]
    assert materialized["input_path"] == "continuation_static"
    assert materialized["output_file"] == result["output_file"]
    assert materialized["record_count"] == len(RECORDS)
    assert materialized["bytes_written"] == result["bytes_written"]
    assert materialized["metadata"]["heldout_examples"] == len(RECORDS)


def test_uncompressed_filename_writes_plain_jsonl(tmp_path: Path) -> None:
    result = stage_continuation_slice(
        ContinuationStagingConfig(
            output_path=str(tmp_path),
            task_key="task",
            template_key="template",
            output_filename="staged.jsonl",
        ),
        records=RECORDS,
        source_id="continuation_static",
        metadata={},
    )

    lines = Path(result["output_file"]).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == RECORDS
    assert "metadata_file" not in result


def test_fingerprint_mismatch_stages_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content_fingerprint mismatch"):
        stage_continuation_slice(
            ContinuationStagingConfig(
                output_path=str(tmp_path),
                task_key="task",
                template_key="template",
                source_manifest=_manifest(),
                content_fingerprint="stale-fingerprint",
            ),
            records=RECORDS,
            source_id="continuation_static",
            metadata={},
        )

    assert list(tmp_path.iterdir()) == []
