# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin pipeline integration test running on Iris.

A variation of tests/integration_test.py that submits the pipeline as an Iris
job rather than using Ray directly. Validates that Marin executor steps run
correctly when dispatched through the Iris cluster.
"""

import logging
import os

import pytest
from iris.rpc import cluster_pb2

from .conftest import IrisIntegrationCluster
from .jobs import IntegrationJobs

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _marin_data_pipeline_step():
    """A lightweight Marin-like pipeline step that validates data processing works.

    This is intentionally simpler than the full integration_test.py pipeline
    since we are testing Iris job submission, not the full Marin pipeline.
    The full pipeline test remains at tests/integration_test.py.
    """
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write synthetic input
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        for i in range(5):
            doc = {"text": f"Document {i} with some content for processing.", "url": f"http://example.com/{i}"}
            with open(os.path.join(input_dir, f"doc_{i}.jsonl"), "w") as f:
                f.write(json.dumps(doc) + "\n")

        # Process: read, transform, write output
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        processed = 0
        for fname in os.listdir(input_dir):
            with open(os.path.join(input_dir, fname)) as f:
                for line in f:
                    doc = json.loads(line)
                    doc["processed"] = True
                    doc["text_length"] = len(doc["text"])
                    with open(os.path.join(output_dir, fname), "a") as out:
                        out.write(json.dumps(doc) + "\n")
                    processed += 1

        assert processed == 5, f"Expected 5 documents, got {processed}"

        # Verify output
        output_files = os.listdir(output_dir)
        assert len(output_files) == 5, f"Expected 5 output files, got {len(output_files)}"

        for fname in output_files:
            with open(os.path.join(output_dir, fname)) as f:
                for line in f:
                    doc = json.loads(line)
                    assert doc["processed"] is True
                    assert doc["text_length"] > 0

    return "pipeline_complete"


def test_marin_data_pipeline_on_iris(integration_cluster):
    """Run a lightweight Marin-like data processing pipeline as an Iris job."""
    job = integration_cluster.submit(
        _marin_data_pipeline_step,
        "itest-marin-pipeline",
        cpu=1,
    )
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_multi_step_pipeline_on_iris(integration_cluster):
    """Submit multiple sequential jobs simulating a multi-step Marin pipeline."""
    # Step 1: data prep
    job1 = integration_cluster.submit(IntegrationJobs.quick, "itest-pipeline-step1")
    status1 = integration_cluster.wait(job1, timeout=integration_cluster.job_timeout)
    assert status1.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Step 2: processing (depends on step 1 success)
    job2 = integration_cluster.submit(_marin_data_pipeline_step, "itest-pipeline-step2")
    status2 = integration_cluster.wait(job2, timeout=integration_cluster.job_timeout)
    assert status2.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Step 3: validation
    job3 = integration_cluster.submit(IntegrationJobs.quick, "itest-pipeline-step3")
    status3 = integration_cluster.wait(job3, timeout=integration_cluster.job_timeout)
    assert status3.state == cluster_pb2.JOB_STATE_SUCCEEDED
