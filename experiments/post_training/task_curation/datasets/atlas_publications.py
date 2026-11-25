# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the Atlas's sealed publications, including chained prebuilt lineages.

The pinned document records final publications, not instructions to rerun their
upstream converters. A chained release's immediate input is often another
published cohort. Reading its sealed final bytes preserves that lineage without
attributing the release to a newer raw converter.
"""

import json
from functools import cache

from finestore.rl.curation import ActorCensusInput, SourceUse, require_lexical_qualification
from finestore.rl.pinned_file import PinnedHfFile, read_pinned_hf_file
from pydantic import BaseModel, ConfigDict, model_validator

from experiments.post_training.task_curation.frozen_atlas import FrozenAtlasConversion, import_frozen_atlas
from experiments.post_training.task_curation.retained_release import RetainedCohort, import_retained
from experiments.post_training.task_curation.source import PENDING_TAG, RlDataSource, SourceInfo

PUBLICATIONS_REFERENCE = PinnedHfFile(
    repo="open-athena/atlas-source-pins",
    revision="e4f09fcf0e20ffd36f189dd7c267e652167e925c",
    filename="atlas_publications.json",
    sha256="082676b2b0f4518a92d7f97b0b64a175c950e241c64cdb14dad6015680a92c35",
    byte_size=1884753,
)


class AtlasPublication(BaseModel):
    """A final cohort and the measurement that certifies its exact import pin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    info: SourceInfo
    cohort: RetainedCohort
    census: ActorCensusInput

    @model_validator(mode="after")
    def sealed_population(self) -> "AtlasPublication":
        cohort = self.cohort
        observed = (cohort.release.repo, cohort.release.revision, cohort.version, cohort.expected_rows)
        measured = (
            self.census.dataset_id,
            self.census.dataset_revision,
            self.census.converter_version,
            self.census.measured_tasks,
        )
        if observed != measured:
            raise ValueError(f"Publication differs from its measured conversion: {self.info.id}")
        if self.info.current_release != cohort.release:
            raise ValueError(f"Publication differs from its declared release: {self.info.id}")
        require_lexical_qualification(
            self.info.lexical_contamination,
            cohort.release.repo,
            cohort.release.revision,
            SourceUse(cohort.intended_use.value),
        )
        self.source().require_license()
        return self

    def source(self) -> RlDataSource[RetainedCohort]:
        return RlDataSource(info=self.info, config=self.cohort, pipeline=import_retained)


class AtlasEvaluation(BaseModel):
    """An evaluation whose exact converter executes in the pinned Atlas tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    info: SourceInfo
    conversion: FrozenAtlasConversion
    census: ActorCensusInput

    @model_validator(mode="after")
    def measured_conversion(self) -> "AtlasEvaluation":
        config = self.conversion
        if (config.dataset.name, config.dataset.revision, config.version) != (
            self.census.dataset_id,
            self.census.dataset_revision,
            self.census.converter_version,
        ):
            raise ValueError(f"Evaluation differs from its measured conversion: {self.info.id}")
        if config.source_id != self.info.id:
            raise ValueError(f"Evaluation converter belongs to another source: {self.info.id}")
        self.source().require_license()
        return self

    def source(self) -> RlDataSource[FrozenAtlasConversion]:
        return RlDataSource(info=self.info, config=self.conversion, pipeline=import_frozen_atlas)


@cache
def publication_catalog() -> tuple[dict[str, AtlasPublication | AtlasEvaluation], frozenset[str]]:
    """Read and validate the frozen publication and retirement cut."""
    document = json.loads(read_pinned_hf_file(PUBLICATIONS_REFERENCE))
    publications: dict[str, AtlasPublication | AtlasEvaluation] = {
        source_id: AtlasPublication.model_validate(record) for source_id, record in document["publications"].items()
    }
    publications.update(
        {source_id: AtlasEvaluation.model_validate(record) for source_id, record in document["evaluations"].items()}
    )
    for source_id, record in publications.items():
        if record.info.id != source_id:
            raise ValueError(f"Publication key differs from its generated identity: {source_id}")
    return publications, frozenset(document["excluded_or_deprecated"])


def bind_publications(sources: list[RlDataSource]) -> list[RlDataSource]:
    """Select sealed final populations and omit Atlas-excluded or deprecated IDs."""
    identifiers = [source.info.id for source in sources]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate RL source identifiers before Atlas publication binding")
    publications, retired = publication_catalog()
    selected = {
        source.info.id: source for source in sources if source.info.id not in retired or PENDING_TAG in source.info.tags
    }
    selected.update({source_id: publication.source() for source_id, publication in publications.items()})
    return list(selected.values())
