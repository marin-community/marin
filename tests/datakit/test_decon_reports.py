# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from marin.datakit.decon import DeconAttributes, NGramConfig, _bloom_hash
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.step_runner import step_is_built
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS, StatusFile

from experiments.datakit import reference_pipeline
from experiments.datakit.decontam.viewer.export_reference_run import _wait_for_marks
from experiments.datakit.decontam.viewer.export_run import _overlapping_ngrams
from experiments.datakit.reports.common import StageReport


@pytest.fixture(autouse=True)
def unavailable_workers(fray_client, monkeypatch):
    def reject_allocation(*args, **kwargs):
        pytest.fail("worker allocation is unavailable")

    monkeypatch.setattr(fray_client, "create_actor_group", reject_allocation)


def test_reference_export_requires_success_status(tmp_path: Path):
    step = StepSpec(name="mark", override_output_path=str(tmp_path))
    write_artifact({"counters": {}}, step.output_path)
    status = StatusFile(step.output_path, "test")

    with pytest.raises(TimeoutError):
        _wait_for_marks({"source": step}, minimum_sources=1, timeout=0)
    status.write_status(STATUS_FAILED)
    with pytest.raises(TimeoutError):
        _wait_for_marks({"source": step}, minimum_sources=1, timeout=0)

    status.write_status(STATUS_SUCCESS)
    assert _wait_for_marks({"source": step}, minimum_sources=1, timeout=0) == {"source": step}


def test_report_target_renders_completed_marks_without_workers(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    samples = tmp_path / "samples"
    write_artifact({}, str(samples / "source"))
    sources = reference_pipeline.sample_sources(str(samples), ["source"])
    steps = reference_pipeline.decontamination_steps(sources, scale=reference_pipeline.SMOKE_SCALE)
    mark = steps.marks["source"]
    attrs = DeconAttributes(
        main_output_dir=str(tmp_path / "main"),
        flagged_output_dir=str(tmp_path / "flagged"),
        num_partitions=1,
        eval_hash_index_path=str(tmp_path / "index.parquet"),
        counters={"decon/clean": 3, "decon/contaminated": 0},
    )
    write_artifact(attrs, mark.output_path)
    StatusFile(mark.output_path, "test").write_status(STATUS_SUCCESS)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_pipeline",
            "--mode",
            "sample",
            "--sample-prefix",
            str(samples),
            "--sources",
            "source",
            "--target",
            "decon-report",
        ],
    )

    reference_pipeline.main()

    report = read_artifact(steps.report.output_path, StageReport)
    assert report.stats["total_docs"] == 3
    assert report.stats["contaminated_docs"] == 0
    assert Path(str(report.html_path)).is_file()
    assert step_is_built(steps.report)


@pytest.mark.parametrize(
    ("text", "features"),
    [
        ("Distinctive alpha phrase", ["Distinctive alpha phrase"]),
        (
            "one two three four\n\nfive six seven eight\n\nnine ten eleven twelve\n\nthirteen fourteen",
            [
                "one two three four five six seven eight nine ten eleven twelve thirteen",
                "two three four five six seven eight nine ten eleven twelve thirteen fourteen",
            ],
        ),
    ],
)
def test_gallery_includes_short_and_whole_record_matches(text: str, features: list[str]):
    matched = {_bloom_hash(feature) for feature in features}
    assert _overlapping_ngrams(text, matched, NGramConfig(ngram_length=13, min_matched_features=2)) == features


def test_mark_target_rejects_missing_preparation_before_worker_allocation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    samples = tmp_path / "samples"
    write_artifact({}, str(samples / "source"))
    StatusFile(str(samples / "source"), "test").write_status(STATUS_SUCCESS)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_pipeline",
            "--mode",
            "sample",
            "--sample-prefix",
            str(samples),
            "--sources",
            "source",
            "--target",
            "decon-mark",
        ],
    )

    with pytest.raises(RuntimeError):
        reference_pipeline.main()
