# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind frozen lexical measurements to the currently configured TaskSpec population."""

from dataclasses import replace

from finestore.rl.curation import ActorCensusInput, DeconReceiptStatus, LexicalContaminationEvidence, LexicalUse
from finestore.rl.source_identity import SourceId
from pydantic import BaseModel, ConfigDict, model_validator

from experiments.post_training.task_curation.source import RlDataSource


class LexicalMeasurementReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    input: ActorCensusInput
    use: LexicalUse
    measurement: LexicalContaminationEvidence

    @model_validator(mode="after")
    def _same_population(self) -> "LexicalMeasurementReceipt":
        evidence = self.measurement
        if (
            self.input.dataset_id,
            self.input.dataset_revision,
            self.input.converter_revision,
            self.input.measured_tasks,
        ) != (
            evidence.input_dataset_id,
            evidence.input_dataset_revision,
            evidence.converter_revision,
            evidence.measured_tasks,
        ):
            raise ValueError("Lexical measurement belongs to another converted population")
        receipt = evidence.receipt
        if self.use is LexicalUse.TRAIN:
            if receipt is not None:
                raise ValueError("Raw training measurements cannot certify a release")
        elif (
            receipt is None or receipt.use is not self.use or receipt.status is not DeconReceiptStatus.BENCHMARK_MEASURED
        ):
            raise ValueError("Evaluation measurements require a matching measurement-only receipt")
        return self


def bind_lexical_measurement(
    source: RlDataSource, receipt: LexicalMeasurementReceipt | None, use: LexicalUse
) -> RlDataSource:
    """Attach matching raw metrics while preserving a valid current release proof."""
    dataset = source.dataset
    dataset_id = dataset.name if dataset else None
    dataset_revision = dataset.revision if dataset else None
    current = source.info.lexical_contamination
    if current is not None and current.admission_blocker(dataset_id, dataset_revision) is None:
        return source

    actor_input = source.info.actor_census_input
    expected = (
        source.info.id,
        dataset_id,
        dataset_revision,
        source.version,
        use,
        actor_input.model_dump(exclude={"evidence_url"}) if actor_input is not None else None,
    )
    measurement = None
    if receipt is not None:
        observed = (
            receipt.source_id,
            receipt.input.dataset_id,
            receipt.input.dataset_revision,
            receipt.input.converter_version,
            receipt.use,
            receipt.input.model_dump(exclude={"evidence_url"}),
        )
        if observed == expected:
            measurement = receipt.measurement
    return replace(source, info=replace(source.info, lexical_contamination=measurement))
