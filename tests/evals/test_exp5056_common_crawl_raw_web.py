# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from experiments.evals.exp5056_common_crawl_raw_web import common_crawl_raw_validation_sets
from marin.datakit.download.common_crawl_archives import (
    COMMON_CRAWL_OUTPUT_FILENAME,
    COMMON_CRAWL_WARC_SLICE_KEY,
    COMMON_CRAWL_WAT_SLICE_KEY,
    CommonCrawlSampleArtifact,
)
from marin.execution.artifact import Artifact
from marin.execution.executor import InputName
from marin.execution.step_spec import StepSpec


def test_common_crawl_raw_validation_sets_render_expected_raw_root_paths() -> None:
    datasets = common_crawl_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")

    warc_dataset = datasets[COMMON_CRAWL_WARC_SLICE_KEY]
    wat_dataset = datasets[COMMON_CRAWL_WAT_SLICE_KEY]

    assert warc_dataset.input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/warc.jsonl.gz"
    assert wat_dataset.input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/wat.jsonl.gz"


def test_common_crawl_raw_validation_sets_use_materialized_artifact_paths(tmp_path: Path) -> None:
    warc_output = tmp_path / "warc"
    wat_output = tmp_path / "wat"
    warc_output.mkdir()
    wat_output.mkdir()
    warc_step = StepSpec(
        name="raw/common_crawl/warc", override_output_path=str(warc_output), fn=lambda output_path: None
    )
    wat_step = StepSpec(name="raw/common_crawl/wat", override_output_path=str(wat_output), fn=lambda output_path: None)

    Artifact.save(
        CommonCrawlSampleArtifact(
            output_file=str(warc_output / "warc-output.jsonl.gz"),
            metadata_file=str(warc_output / "metadata.json"),
            selected_paths=("crawl-data/example.warc.gz",),
            bytes_written=123,
            record_count=4,
            counters={"records_seen": 5, "records_kept": 4},
            record_type_counts={"response": 4},
        ),
        str(warc_output),
    )
    Artifact.save(
        CommonCrawlSampleArtifact(
            output_file=str(wat_output / "wat-output.jsonl.gz"),
            metadata_file=str(wat_output / "metadata.json"),
            selected_paths=("crawl-data/example.wat.gz",),
            bytes_written=456,
            record_count=7,
            counters={"records_seen": 9, "records_kept": 7},
            record_type_counts={"metadata": 7},
        ),
        str(wat_output),
    )

    datasets = common_crawl_raw_validation_sets(warc_raw=warc_step, wat_raw=wat_step)

    assert datasets[COMMON_CRAWL_WARC_SLICE_KEY].input_path == str(warc_output / "warc-output.jsonl.gz")
    assert datasets[COMMON_CRAWL_WAT_SLICE_KEY].input_path == str(wat_output / "wat-output.jsonl.gz")


def test_common_crawl_raw_validation_sets_fall_back_to_step_relative_paths_when_unmaterialized() -> None:
    warc_step = StepSpec(
        name="raw/common_crawl/warc",
        override_output_path="gs://example-bucket/raw/common_crawl/warc",
        fn=lambda output_path: None,
    )
    wat_step = StepSpec(
        name="raw/common_crawl/wat",
        override_output_path="gs://example-bucket/raw/common_crawl/wat",
        fn=lambda output_path: None,
    )

    datasets = common_crawl_raw_validation_sets(warc_raw=warc_step, wat_raw=wat_step)

    warc_input = datasets[COMMON_CRAWL_WARC_SLICE_KEY].input_path
    wat_input = datasets[COMMON_CRAWL_WAT_SLICE_KEY].input_path
    assert isinstance(warc_input, InputName)
    assert isinstance(wat_input, InputName)
    assert warc_input.name == COMMON_CRAWL_OUTPUT_FILENAME
    assert wat_input.name == COMMON_CRAWL_OUTPUT_FILENAME
