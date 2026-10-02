# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact, write_artifact

from experiments.datakit.materialize_zephyr_benchmark_sample import main
from experiments.datakit.testbed.sampler import SampleManifest


def test_copy_sample_publishes_data_and_completion_record(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    data_dir = source / "first" / "outputs" / "main"
    data_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["document"], "text": ["Original sample text."]}), data_dir / "part.parquet")
    write_artifact(
        NormalizedData(main_output_dir=str(data_dir), dup_output_dir="", counters={}),
        output_path=str(source / "first"),
    )
    (source / "first" / ".executor_status").write_text("SUCCESS")
    monkeypatch.setattr(
        sys,
        "argv",
        ["materialize", "--mode", "copy", "--source-prefix", str(source), "--destination-prefix", str(destination)],
    )

    main()

    manifest = read_artifact(str(destination), SampleManifest)
    assert manifest.source_paths == {"first": str(source / "first")}
    sampled = read_artifact(str(destination / "first"), NormalizedData)
    assert pq.read_table(sampled.main_output_dir).to_pylist() == [{"id": "document", "text": "Original sample text."}]
