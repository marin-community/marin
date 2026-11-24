# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.cluster.client.job_info import get_job_info, set_job_info
import marin.execution.remote as remote_execution
from marin.execution.lazy import StepContext
from marin.experiment import cli

from experiments.post_training.long_loop.sft_epoch_capacity import main


@pytest.fixture(autouse=True)
def isolated_job_info():
    previous = get_job_info()
    set_job_info(None)
    yield
    set_job_info(previous)


@pytest.mark.parametrize("nested", [False, True])
def test_capacity_cli_keeps_nested_children_local_to_the_regional_coordinator(monkeypatch, nested):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a/marin/users/test/capacity")
    if nested:
        monkeypatch.setenv("IRIS_TASK_ID", "/test/capacity/0")
    else:
        monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    handles = []
    monkeypatch.setattr(cli, "run", lambda *steps, **kwargs: handles.extend(steps))
    submitted = []

    def submit(request):
        submitted.append(request)
        return SimpleNamespace(wait=lambda **kwargs: None)

    monkeypatch.setattr(remote_execution, "current_client", lambda: SimpleNamespace(submit=submit))
    result = CliRunner().invoke(
        main,
        [
            "--data", "s3://marin-us-east-08a/marin/users/test/data.json",
            "--sha256", "0" * 64,
            "--output-prefix", "s3://marin-us-east-08a/marin/users/test/capacity",
            "--packing-cap", "262144", "--lookahead", "0", "--version", "2026.11.24.1", "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(handles) == 1
    context = StepContext.for_run("s3://marin-us-east-08a/marin/users/test/capacity/result", "s3://marin-us-east-08a/marin/users/test/capacity")
    handles[0].run(handles[0].build_config(context))
    assert len(submitted) == 1
    assert submitted[0].resources.target_cluster == (None if nested else "cw-us-east-08a")
