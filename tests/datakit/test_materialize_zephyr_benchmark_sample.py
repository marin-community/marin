# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Zephyr benchmark sample materialization."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import write_artifact

from experiments.datakit.materialize_zephyr_benchmark_sample import copy_sample_steps


def test_copy_sample_steps_uses_source_step_path_when_artifact_path_is_rebased(tmp_path: Path):
    source_prefix = tmp_path / "us-central1"
    source_root = source_prefix / "example"
    source_main = source_root / "outputs" / "main"
    source_main.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["1"], "text": ["source"]}), source_main / "part-00000.parquet")

    # NormalizedData artifacts serialize paths relative to MARIN_PREFIX. Loading
    # one from another GCS region therefore points this field at the destination.
    write_artifact(
        NormalizedData(
            main_output_dir=str(tmp_path / "europe-west4" / "example" / "outputs" / "main"),
            dup_output_dir="",
            counters={},
        ),
        str(source_root),
    )

    destination_root = tmp_path / "europe-west4" / "example"
    [step] = copy_sample_steps(str(source_prefix), str(tmp_path / "europe-west4"))
    result = step.fn(str(destination_root))

    assert pq.read_table(Path(result.main_output_dir) / "part-00000-of-00001.parquet").to_pylist() == [
        {"id": "1", "text": "source"}
    ]
