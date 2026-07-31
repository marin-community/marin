# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from marin.datakit.sources import DatakitSource
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import STATUS_SUCCESS, StatusFile

from experiments.datakit.scripts import trigger_sources


def test_list_pending_lists_missing_sources_without_execution(monkeypatch, capsys, tmp_path):
    root = f"memory://trigger-sources-{uuid4().hex}"
    execution_marker = tmp_path / "executed"

    def record_execution(_output_path: str) -> None:
        execution_marker.write_text("executed")

    cached = StepSpec(
        name="cached",
        override_output_path=f"{root}/cached",
        fn=record_execution,
    )
    missing = StepSpec(
        name="missing",
        override_output_path=f"{root}/missing",
        fn=record_execution,
    )
    StatusFile(cached.output_path, worker_id="test").write_status(STATUS_SUCCESS)
    registry = {
        "cached-source": DatakitSource(name="cached-source", normalize_steps=(cached,), rough_token_count_b=1),
        "missing-source": DatakitSource(name="missing-source", normalize_steps=(missing,), rough_token_count_b=1),
    }
    monkeypatch.setattr(trigger_sources, "all_sources", lambda: registry)

    trigger_sources.main(["--list-pending"])

    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines[:-1]] == ["missing-source"]
    assert lines[-1] == "1/2 source(s) would run."
    assert not execution_marker.exists()
