# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy

from marin.execution.artifact import ArtifactRecord, result_type_name, write_record

from experiments.post_training.task_curation.datasets.facet.artifacts import PreservedImage


def test_image_proof_remains_loadable_when_producer_runs_as_an_entrypoint(tmp_path):
    producer = runpy.run_module(
        "experiments.post_training.task_curation.datasets.facet.images", run_name="facet_images_cli_alias"
    )
    image = producer["PreservedImage"](
        path=str(tmp_path / "image"),
        bundle_id="a" * 64,
        source_repo="fixture/facet",
        source_revision="b" * 40,
        source_filename="image.sif/image.sif",
        sif_sha256="c" * 64,
        sif_bytes=10,
        partition_sha256="d" * 64,
        manifest_digest="sha256:" + "e" * 64,
        config_digest="sha256:" + "f" * 64,
        diff_id="sha256:" + "0" * 64,
        tree_sha256="1" * 64,
        tree_entries=5,
        provenance_uri="s3://fixture/image/preservation.json",
        provenance_sha256="2" * 64,
    )
    write_record(
        ArtifactRecord(output_path=image.path, result_type=result_type_name(type(image)), result=image.result_payload())
    )
    assert PreservedImage.raw_load(image.path) == image
