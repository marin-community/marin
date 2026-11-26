# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the postprocessing chain's retention-sink handling."""

import pytest
import yaml

from experiments.post_training.mochi_loop_guards.postprocess import retention_source


def launch_document(output_path):
    return {"config": {"skyrl": {"generator": {"trajectory_retention": {"output_path": output_path}}}}}


def write_launch(tmp_path, output_path):
    (tmp_path / "resolved-launch.yaml").write_text(yaml.safe_dump(launch_document(output_path)))
    return str(tmp_path)


def test_retention_source_reads_the_run_shared_trajectory_sink(tmp_path):
    run_output = write_launch(tmp_path, "s3://marin-us-east-08a/tmp/ttl=3d/marin/run/traces")
    assert retention_source(run_output) == "s3://marin-us-east-08a/tmp/ttl=3d/marin/run/traces"


def test_retention_source_refuses_sinks_outside_the_execution_region(tmp_path):
    run_output = write_launch(tmp_path, "gs://marin-eu-west-4/run/traces")
    with pytest.raises(ValueError, match="outside the execution region"):
        retention_source(run_output)
