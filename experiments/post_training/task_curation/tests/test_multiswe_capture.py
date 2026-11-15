# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy

from marin.execution.artifact import ArtifactRecord, result_type_name, write_record

from experiments.post_training.task_curation.datasets.prime_intellect.capture_models import CapturedArtifacts


def test_capture_artifact_remains_loadable_when_producer_runs_as_an_entrypoint(tmp_path):
    producer = runpy.run_module(
        "experiments.post_training.task_curation.datasets.prime_intellect.capture_artifacts",
        run_name="capture_cli_alias",
    )
    artifact = producer["CapturedArtifacts"](
        path=str(tmp_path / "capture"),
        source_id="fixture:multi",
        dataset_id="fixture/multi",
        revision="a" * 40,
        input_sha256="b" * 64,
        images=2231,
        tasks=2232,
        manifest_uri="s3://fixture/captures.json",
        manifest_sha256="c" * 64,
        placement_uri="s3://fixture/placement.json",
    )
    write_record(
        ArtifactRecord(
            output_path=artifact.path,
            result_type=result_type_name(type(artifact)),
            result=artifact.result_payload(),
        )
    )
    assert CapturedArtifacts.raw_load(artifact.path) == artifact
