# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind actor censuses to the configured input and converter before declaration."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import ActorCensusInput, DaytonaMeasurement, IntegrityContractSummary
from pydantic import BaseModel, ConfigDict, model_validator

from experiments.post_training.task_curation.source import RlDataSource


class ActorCensusReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    input: ActorCensusInput
    measurement: DaytonaMeasurement
    integrity_contracts: IntegrityContractSummary | None

    @model_validator(mode="after")
    def _same_population(self) -> "ActorCensusReceipt":
        contract = self.integrity_contracts
        if (self.input.dataset_id, self.input.dataset_revision) != (
            self.measurement.dataset_id,
            self.measurement.dataset_revision,
        ):
            raise ValueError("Census input differs from measured dataset pin")
        if contract is not None and (contract.dataset_id, contract.dataset_revision, contract.task_specs_sha256) != (
            self.measurement.dataset_id,
            self.measurement.dataset_revision,
            self.input.task_specs_sha256,
        ):
            raise ValueError("Integrity contract summary belongs to another census population")
        if self.measurement.daytona_machine_slots > self.input.measured_tasks:
            raise ValueError("Actor slots exceed the measured task population")
        return self


def bind_actor_census(source: RlDataSource, receipt: ActorCensusReceipt | None) -> RlDataSource:
    """Keep absent or stale census evidence pending instead of preserving inline estimates."""
    dataset = source.dataset
    expected = (source.info.id, dataset.name if dataset else None, dataset.revision if dataset else None, source.version)
    observed = (
        None
        if receipt is None
        else (
            receipt.source_id,
            receipt.measurement.dataset_id,
            receipt.measurement.dataset_revision,
            receipt.input.converter_version,
        )
    )
    if observed != expected:
        reason = (
            "not measured"
            if receipt is None
            else f"input or converter mismatch: expected {expected}, measured {observed}"
        )
        info = replace(
            source.info,
            daytona_snapshot_count=None,
            daytona_machine_slots=None,
            daytona_execution_failures=None,
            daytona_count_basis=None,
            daytona_unsafe_reasons=(),
            daytona_unsafe_details=None,
            daytona_unsafe_counts=None,
            daytona_roles=None,
            integrity_contracts=None,
            actor_census_input=None,
            notes=f"{source.info.notes} Actor census pending: {reason}.".strip(),
        )
        return replace(source, info=info)
    assert receipt is not None
    measurement = receipt.measurement
    fields = measurement.model_dump(
        exclude={"dataset_id", "dataset_revision", "daytona_unsafe_reasons", "daytona_roles"}
    )
    fields["daytona_unsafe_reasons"] = tuple(measurement.daytona_unsafe_reasons)
    fields["daytona_roles"] = measurement.daytona_roles
    return replace(
        source,
        info=replace(
            source.info, integrity_contracts=receipt.integrity_contracts, actor_census_input=receipt.input, **fields
        ),
    )


def receipt_from_census(census: Mapping, summary: IntegrityContractSummary | None) -> ActorCensusReceipt:
    """Bind an actor measurement; contract summaries require their own refreshed evidence."""
    return ActorCensusReceipt(
        source_id=census["source_id"],
        input=ActorCensusInput(
            dataset_id=census["dataset_id"],
            dataset_revision=census["dataset_revision"],
            converter_version=census["converter_version"],
            converter_revision=census["converter_revision"],
            measured_tasks=census["task_count"],
            task_specs_sha256=census["task_specs_sha256"],
            evidence_url=census["path"] + "/.artifact.json",
        ),
        measurement=DaytonaMeasurement.model_validate(
            {name: census[name] for name in DaytonaMeasurement.model_fields if name in census}
        ),
        integrity_contracts=summary,
    )


def read_census_receipts(directory: Path) -> dict[str, ActorCensusReceipt]:
    receipts = {}
    for path in sorted(directory.glob("part-*.json")):
        for source_id, record in json.loads(path.read_text()).items():
            receipt = ActorCensusReceipt.model_validate(record)
            if source_id != receipt.source_id or source_id in receipts:
                raise ValueError(f"Duplicate or mismatched census source identity: {source_id}")
            receipts[source_id] = receipt
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censuses", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipts = {}
    for path in args.censuses:
        for record in json.loads(path.read_text()):
            receipt = receipt_from_census(record, None)
            if receipt.source_id in receipts:
                raise ValueError(f"Duplicate actor census: {receipt.source_id}")
            receipts[receipt.source_id] = receipt.model_dump(mode="json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(receipts.items())
    for old in args.output_dir.glob("part-*.json"):
        old.unlink()
    for part, offset in enumerate(range(0, len(entries), 64)):
        (args.output_dir / f"part-{part:03d}.json").write_text(
            json.dumps(dict(entries[offset : offset + 64]), indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
