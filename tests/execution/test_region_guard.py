# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The region guard that keeps a step graph from silently running against another region."""

import pytest
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.region_guard import (
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    CrossRegionAccessError,
    region_guard,
)
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec

from tests.execution.conftest import recording_step

CENTRAL1_PREFIX = "gs://marin-us-central1/infra/datakit/ferry_perf"


@pytest.fixture(autouse=True)
def _no_ambient_placement(monkeypatch):
    """Neither the developer's shell nor a leaked Iris identity may decide the outcome."""
    monkeypatch.delenv(MARIN_CROSS_REGION_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    set_job_info(None)
    yield
    set_job_info(None)


def runs_in(region: str | None) -> None:
    """Run as an Iris task placed on a worker in *region* (``None``: not an Iris task)."""
    if region is None:
        return
    set_job_info(JobInfo(task_id=JobName.from_wire("/test/ferry/0"), worker_region=region))


# ---------------------------------------------------------------------------
# Guard behavior
# ---------------------------------------------------------------------------


def test_regional_bucket_outside_the_runtime_region_fails():
    runs_in("us-central2")

    with pytest.raises(CrossRegionAccessError) as exc_info:
        region_guard(allow_cross_region=False).check("The run's storage prefix", [("MARIN_PREFIX", CENTRAL1_PREFIX)])

    error = exc_info.value
    assert error.runtime_region == "us-central2"
    assert [(path.label, path.path, path.region) for path in error.paths] == [
        ("MARIN_PREFIX", CENTRAL1_PREFIX, "us-central1")
    ]
    # The message is the whole point of the guard: it has to name where the run is and
    # which path is somewhere else, or the operator cannot act on it.
    assert "us-central2" in str(error)
    assert CENTRAL1_PREFIX in str(error)


@pytest.mark.parametrize(
    "path",
    [
        "gs://marin-us-central1/data",  # the run's own region
        "gs://marin-data/raw/shard",  # a marin bucket no config places in a region
        "gs://someone-elses-bucket/data",  # governed by the read-time transfer budget
        "s3://marin-na/marin/data",  # not GCS
        "/tmp/marin/data",  # not remote at all
    ],
)
def test_paths_outside_the_declared_regional_buckets_are_allowed(path):
    """Only buckets a cluster config places in a region are compared.

    Everything else either matches, or has no declared placement to compare against —
    blocking those would stop every pipeline that reads a shared bucket like marin-data.
    """
    runs_in("us-central1")

    region_guard(allow_cross_region=False).check("The run's storage prefix", [("MARIN_PREFIX", path)])


def test_check_is_inert_outside_iris():
    """A driver on a laptop or dev VM pins nothing: its own placement decides nothing.

    Its own reads stay governed by the cross-region transfer budget.
    """
    runs_in(None)

    region_guard(allow_cross_region=False).check("The run's storage prefix", [("MARIN_PREFIX", CENTRAL1_PREFIX)])


def test_coreweave_worker_region_is_not_compared_to_gcs_placement():
    """A CoreWeave region names another cloud; comparing it to a GCS bucket is meaningless."""
    runs_in("us-east-02a")

    assert region_guard(allow_cross_region=False).runtime_region is None


# ---------------------------------------------------------------------------
# StepRunner integration
# ---------------------------------------------------------------------------


def test_runner_runs_a_graph_whose_prefix_matches_the_runtime_region(monkeypatch, tmp_path):
    runs_in("us-central1")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    ran: list[str] = []
    output = (tmp_path / "step").as_posix()

    StepRunner().run([recording_step("in_region", output, ran)])

    assert ran == ["in_region"]
    assert read_artifact(output, Artifact).path == output


def test_runner_refuses_a_prefix_in_another_region_before_running_anything(monkeypatch, tmp_path):
    runs_in("us-central2")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    ran: list[str] = []

    with pytest.raises(CrossRegionAccessError) as exc_info:
        StepRunner().run([recording_step("cross_region", (tmp_path / "step").as_posix(), ran)])

    assert ran == []
    assert exc_info.value.runtime_region == "us-central2"


def test_runner_refuses_a_step_writing_to_another_region(monkeypatch, tmp_path):
    """The prefix can be fine while an absolutely-pinned step output is not."""
    runs_in("us-central1")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    ran: list[str] = []
    step = recording_step("pinned_elsewhere", "gs://marin-us-west4/pinned", ran)

    with pytest.raises(CrossRegionAccessError) as exc_info:
        StepRunner().run([step])

    # Raised before the runner reads or writes the foreign-region output path.
    assert ran == []
    assert [path.path for path in exc_info.value.paths] == ["gs://marin-us-west4/pinned"]


def test_runner_refuses_a_dependency_in_another_region(monkeypatch, tmp_path):
    runs_in("us-central1")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    ran: list[str] = []
    dep = recording_step("foreign_dep", "gs://marin-eu-west4/raw", ran)
    step = StepSpec(
        name="consumer",
        override_output_path=(tmp_path / "consumer").as_posix(),
        deps=[dep],
        fn=lambda path: Artifact(path=path),
    )

    with pytest.raises(CrossRegionAccessError) as exc_info:
        StepRunner().run([step])

    assert ran == []
    assert [path.region for path in exc_info.value.paths] == ["europe-west4"]


def test_override_env_var_lets_a_cross_region_graph_run(monkeypatch, tmp_path):
    runs_in("us-central2")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setenv(MARIN_CROSS_REGION_OVERRIDE_ENV, "1")
    ran: list[str] = []
    output = (tmp_path / "step").as_posix()

    StepRunner().run([recording_step("deliberate", output, ran)])

    assert ran == ["deliberate"]
    assert read_artifact(output, Artifact).path == output


def test_allow_cross_region_argument_lets_a_cross_region_graph_run(monkeypatch, tmp_path):
    runs_in("us-central2")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    ran: list[str] = []

    StepRunner().run([recording_step("deliberate", (tmp_path / "step").as_posix(), ran)], allow_cross_region=True)

    assert ran == ["deliberate"]
