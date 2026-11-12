# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute local artifact graphs without retrying failed submissions."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run


def execute_local_graph(output: ArtifactStep) -> list[Artifact]:
    """Run a bound graph without automatic retries of failed submissions."""
    return run(output, max_concurrent=1, force_run_failed=False)
