# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census records for local synthetic source populations."""

from dataclasses import replace

import pytest
from finestore.rl.curation import ActorCensusInput

from experiments.post_training.task_curation.admission_binding import AdmissionCensusReceipt
from experiments.post_training.task_curation.datasets.admission_censuses import ADMISSION_CENSUS_RECEIPTS
from experiments.post_training.task_curation.source import RlDataSource


@pytest.fixture
def isolated_admission_census():
    original = dict(ADMISSION_CENSUS_RECEIPTS)
    yield
    ADMISSION_CENSUS_RECEIPTS.clear()
    ADMISSION_CENSUS_RECEIPTS.update(original)


def register_admission_census(source_id: str, dataset: str, revision: str, version: str, count: int) -> ActorCensusInput:
    binding = ActorCensusInput(
        dataset_id=dataset,
        dataset_revision=revision,
        converter_version=version,
        converter_revision="fixture",
        measured_tasks=count,
        task_specs_sha256="a" * 64,
        evidence_url="https://example.test/census",
    )
    ADMISSION_CENSUS_RECEIPTS[source_id] = AdmissionCensusReceipt(
        source_id=source_id,
        input=binding,
        population_path="https://example.test/population",
        scanned_task_specs_sha256="a" * 64,
        retained_tasks=count,
        runnable_grader_tasks=count,
        no_grader_tasks=0,
        grader_kinds={"verifyit": count},
        missing_grader_reasons={},
        evidence_url="https://example.test/admission",
    )
    return binding


def with_admission_census(source: RlDataSource) -> RlDataSource:
    assert source.dataset is not None and source.info.within_source_dedup is not None
    binding = register_admission_census(
        source.info.id,
        source.dataset.name,
        source.dataset.revision,
        source.version,
        source.info.within_source_dedup.unique_count,
    )
    return replace(source, info=replace(source.info, actor_census_input=binding))
