# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Endless combined original-pair cohort declaration identity and binding shape."""

from finestore.rl.curation import DatasetRelease, SourceUse
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.endless.tasks import (
    COHORT_CENSUS,
    COHORT_IMAGES,
    COHORT_LOCATORS,
    COHORT_SOURCE_ID,
    VERSION,
    cohort_source,
)
from experiments.post_training.task_curation.pipeline import process_rows
from experiments.post_training.task_curation.source import TaskFormat

SEALED_FILTRATION_ORIGINAL_RELEASE = DatasetRelease(
    repo="FWeindel/validated-tasks",
    revision="317659ac88c8b706d318020c5fc3813a931fb6ec",
    url="https://huggingface.co/datasets/FWeindel/validated-tasks/tree/317659ac88c8b706d318020c5fc3813a931fb6ec",
)


def test_combined_cohort_declaration_binds_the_upstream_pin_as_the_filtration_original():
    (source,) = cohort_source()
    assert source.info.id == COHORT_SOURCE_ID
    assert source.info.subset == "endless"
    assert source.info.count == 2069
    assert source.info.task_format is TaskFormat.TASKSPEC
    assert source.intended_use is SourceUse.TRAIN
    assert source.config.intended_use is IntendedUse.TRAIN
    assert source.pipeline is process_rows
    # The task ids embed the recipe version, so the combined declaration keeps the
    # endless converter's version exactly; a bump would change every derived identity.
    assert source.version == VERSION == "2026.11.20.1"
    assert source.config.source.files == ("endless_v2/tasks.parquet", "endless_v3/tasks.parquet")
    assert source.config.source.select is not None
    assert source.config.source.read is not None
    # bind_oracle_admitted_release requires the declared pin to equal the sealed
    # filtration evidence's original release exactly.
    assert source.current_release == SEALED_FILTRATION_ORIGINAL_RELEASE
    assert source.pipeline_reference.revision == "317659ac88c8b706d318020c5fc3813a931fb6ec"


def test_cohort_reference_pins_are_content_addressed():
    for reference in (COHORT_IMAGES, COHORT_CENSUS, COHORT_LOCATORS):
        assert reference.sha256 and len(reference.sha256) == 64
        assert reference.uri.startswith("s3://")
