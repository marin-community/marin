# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.decon import FLAGGED_SCHEMA, DeconAttributes
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.step_runner import step_is_built
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import StatusFile
from rigging.filesystem.step_status import STATUS_FAILED, STATUS_SUCCESS

from experiments.datakit import reference_pipeline
from experiments.datakit.decontam.viewer.export_reference_run import _wait_for_marks
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
    flagged = tmp_path / "flagged"
    flagged.mkdir()
    row = {
        "id": "d1",
        "text": "a copied question",
        "max_score": 0.9,
        "matched_items": ["gsm8k-0"],
        "matched_evals": ["gsm8k"],
    }
    pq.write_table(pa.Table.from_pylist([row], schema=FLAGGED_SCHEMA), str(flagged / "part-00000-of-00001.parquet"))
    attrs = DeconAttributes(
        main_output_dir=str(tmp_path / "main"),
        flagged_output_dir=str(flagged),
        num_partitions=1,
        counters={"decon/clean": 3, "decon/contaminated": 1},
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
    assert report.stats["total_docs"] == 4
    assert report.stats["contaminated_docs"] == 1
    assert report.stats["n_evals_hit"] == 1
    assert Path(str(report.html_path)).is_file()
    assert step_is_built(steps.report)


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
