# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full marin integration pipeline running on Iris.

Ports the pipeline from tests/integration_test.py to dispatch through an Iris
cluster via FrayIrisClient instead of Ray.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fray import set_current_client
from fray.v2.iris_backend import FrayIrisClient
from marin.execution.executor import ExecutorMainConfig, executor_main
from tests.integration_test import create_steps

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH_DATA = str(REPO_ROOT / "tests" / "quickstart-data")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.timeout(600)
def test_marin_pipeline_on_iris(integration_cluster):
    """Run the full marin data pipeline dispatched through Iris."""
    prefix = tempfile.mkdtemp(prefix="iris-marin-itest-")
    try:
        os.environ["MARIN_PREFIX"] = prefix
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_API_KEY", "")
        os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

        iris_client = FrayIrisClient(
            controller_address=integration_cluster.url,
            workspace=REPO_ROOT / "lib" / "iris",
        )

        config = ExecutorMainConfig(
            prefix=prefix,
            executor_info_base_path=os.path.join(prefix, "experiments"),
        )

        experiment_prefix = "quickstart-tests"
        steps = create_steps(experiment_prefix, SYNTH_DATA)

        with set_current_client(iris_client):
            executor_main(config, steps=steps)
    finally:
        shutil.rmtree(prefix, ignore_errors=True)
