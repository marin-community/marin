# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint evaluations honor the experiment CLI's independent artifact version."""

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context

from experiments.evals.run_key_evals import build


@pytest.mark.parametrize("version", ["dev", "2026.07.16"])
def test_checkpoint_evaluation_and_report_use_requested_version(version):
    with build_context(BuildContext(versions=VersionCodex(default=version))):
        report = build(limit=5)
    assert report.version == version
    assert {step.version for step in report.deps} == {version}
    assert {producer.version for step in report.deps for producer in step.deps} == {"2026.06.30"}
