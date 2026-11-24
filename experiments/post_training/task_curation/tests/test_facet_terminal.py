# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACET-Terminal-Tasks-6k declaration identity and admission binding shape."""

from finestore.rl.curation import DatasetRelease, SourceUse
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.facet_terminal import (
    CONVERTER_VERSION,
    PREPARED_REPO,
    PREPARED_REVISION,
    REPAIR_PROTOCOL,
    SOURCE_ID,
    UPSTREAM,
    UPSTREAM_REVISION,
    sources,
)
from experiments.post_training.task_curation.prebuilt import import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import TaskFormat

FILTRATION_ORIGINAL_RELEASE = DatasetRelease(
    repo=PREPARED_REPO,
    revision=PREPARED_REVISION,
    url=f"https://huggingface.co/datasets/{PREPARED_REPO}/tree/{PREPARED_REVISION}",
)


def test_declaration_binds_the_prepared_pin_as_the_oracle_filtration_original():
    (source,) = sources()
    assert source.info.id == SOURCE_ID
    assert source.info.upstream_dataset == UPSTREAM
    assert source.info.upstream_url == f"https://huggingface.co/datasets/{UPSTREAM}/tree/{UPSTREAM_REVISION}"
    assert source.info.count == 6010
    assert source.info.task_format is TaskFormat.TASKSPEC
    assert source.intended_use is SourceUse.TRAIN
    assert source.config.intended_use is IntendedUse.TRAIN
    assert source.pipeline is import_taskcompendium_prebuilt
    assert source.version == CONVERTER_VERSION
    assert source.config.expected_rows == 6010
    # bind_oracle_admitted_release requires the declared pin to equal the sealed
    # filtration evidence's original release exactly (the contract-repair pin).
    assert source.current_release == FILTRATION_ORIGINAL_RELEASE
    assert source.pipeline_reference.revision == PREPARED_REVISION
    assert source.pipeline_reference.version == CONVERTER_VERSION
    assert REPAIR_PROTOCOL in source.info.build_path
