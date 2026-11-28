# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The campaign cross-source step binds source artifacts in campaign order."""

import pyarrow as pa
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from taskcompendium.pipeline.cross_source import RetainedSource

from experiments.post_training.task_curation.cross_source_step import cross_source_dedup_step

RESOURCES = ResourceConfig(cpu=1, ram="1g")


def _fake_step(source_id: str, version: str = "2026.11.27.1"):
    return ArtifactStep(
        name=f"source/{source_id}",
        version=version,
        artifact_type=pa.Table,
        deps=[],
        run=lambda: None,
        build_config=lambda ctx: None,
    )


def test_step_binds_final_views_in_campaign_order_and_changes_identity_with_dependency(tmp_path):
    alpha = _fake_step("alpha")
    beta = _fake_step("beta")
    step = cross_source_dedup_step(
        {"alpha": alpha, "beta": beta},
        order=("beta", "alpha"),
        max_workers=1,
        worker_resources=RESOURCES,
    )
    assert step is not None
    prefix = str(tmp_path)
    context = StepContext.for_run(f"{prefix}/output", prefix, deps=step.deps)
    request = step.build_config(context)
    assert request.sources == (
        RetainedSource(source_id="beta", final_path=f"{beta.path(prefix)}/final", order=0),
        RetainedSource(source_id="alpha", final_path=f"{alpha.path(prefix)}/final", order=1),
    )

    rebuilt = cross_source_dedup_step(
        {"alpha": _fake_step("alpha", version="2026.11.28.1"), "beta": beta},
        order=("beta", "alpha"),
        max_workers=1,
        worker_resources=RESOURCES,
    )
    assert rebuilt is not None
    assert step.name != rebuilt.name


def test_step_rejects_an_order_that_is_not_exactly_the_source_set():
    with pytest.raises(ValueError):
        cross_source_dedup_step(
            {"alpha": _fake_step("alpha"), "beta": _fake_step("beta")},
            order=("alpha",),
            max_workers=1,
            worker_resources=RESOURCES,
        )
    with pytest.raises(ValueError):
        cross_source_dedup_step(
            {"alpha": _fake_step("alpha"), "beta": _fake_step("beta")},
            order=("alpha", "phantom"),
            max_workers=1,
            worker_resources=RESOURCES,
        )
