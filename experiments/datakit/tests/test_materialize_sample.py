# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import DatakitSource
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

from experiments.datakit.materialize_zephyr_benchmark_sample import main, regenerate_sample_steps
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
    write_artifact(
        NormalizedData(main_output_dir=str(data_dir), dup_output_dir="", counters={}),
        output_path=str(source / "excluded"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize",
            "--mode",
            "copy",
            "--source-prefix",
            str(source),
            "--destination-prefix",
            str(destination),
            "--sources",
            "first",
        ],
    )

    main()

    manifest = read_artifact(str(destination), SampleManifest)
    assert manifest.source_paths == {"first": str(source / "first")}
    assert not (destination / "excluded").exists()
    sampled = read_artifact(str(destination / "first"), NormalizedData)
    assert pq.read_table(sampled.main_output_dir).to_pylist() == [{"id": "document", "text": "Original sample text."}]


def test_regenerate_sample_uses_requested_mixture_instead_of_corpus_sizes(tmp_path):
    sources = []
    for name, rows, tokens_b in [("small", 20, 10.0), ("large", 180, 90.0)]:
        root = tmp_path / "normalized" / name
        data_dir = root / "outputs" / "main"
        data_dir.mkdir(parents=True)
        pq.write_table(pa.table({"id": list(range(rows)), "text": [name] * rows}), data_dir / "part.parquet")
        write_artifact(
            NormalizedData(main_output_dir=str(data_dir), dup_output_dir="", counters={}),
            output_path=str(root),
        )
        (root / ".executor_status").write_text("SUCCESS")
        sources.append(
            DatakitSource(
                name=name,
                normalize_steps=(StepSpec(name=f"normalized/{name}", override_output_path=str(root)),),
                rough_token_count_b=tokens_b,
            )
        )

    destination = str(tmp_path / "sample")
    steps = regenerate_sample_steps(sources, destination, 10, source_weights={"small": 3, "large": 1})
    StepRunner().run(steps, max_concurrent=2)

    for name, expected_rows in [("small", 15), ("large", 5)]:
        sampled = read_artifact(f"{destination}/{name}", NormalizedData)
        assert pq.read_table(sampled.main_output_dir).to_pylist() == [
            {"id": i, "text": name} for i in range(expected_rows)
        ]
