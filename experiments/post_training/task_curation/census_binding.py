# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind actor censuses to the configured input and converter before declaration."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import ActorCensusInput, DaytonaMeasurement, IntegrityContractSummary
from finestore.rl.source_identity import SourceId, canonical_source_id
from pydantic import BaseModel, ConfigDict, model_validator
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.receipt_shards import write_receipt_shards
from experiments.post_training.task_curation.source import RlDataSource


class ActorCensusReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
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
            evidence_url=str(StoragePath(census["path"]) / ".artifact.json"),
        ),
        measurement=DaytonaMeasurement.model_validate(
            {name: census[name] for name in DaytonaMeasurement.model_fields if name in census}
        ),
        integrity_contracts=summary,
    )


def integrity_summary_for_actor(actor: Mapping, integrity: Mapping) -> IntegrityContractSummary:
    """Join refreshed contract evidence only when it measured the actor census's exact input."""
    fields = ("input_artifact", "converter_version", "converter_revision", "task_count")
    if canonical_source_id(actor["source_id"]) != canonical_source_id(integrity["source_id"]) or any(
        actor[name] != integrity[name] for name in fields
    ):
        raise ValueError(f"Integrity census input differs from actor census: {actor['source_id']}")
    raw = integrity["summary"]
    if raw["schema_version"] != 2:
        raise ValueError("Integrity census requires corrected family and SHARED-missing metadata")
    groups = [
        {
            **{
                name: group.get(name)
                for name in ("id", "version", "dependency_state", "native_family", "present", "count")
            },
            "harness_policy_sha256": canonical_sha256(group["harness_policy"]) if group["present"] else None,
        }
        for group in raw["groups"]
    ]
    example = raw["example"]
    return IntegrityContractSummary.model_validate(
        {
            **{
                name: raw[name]
                for name in (
                    "distinct_groups",
                    "absent_contract_tasks",
                    "required_shared_tasks",
                    "shared_missing_contract_tasks",
                    "evidence_url",
                    "dataset_id",
                    "dataset_revision",
                    "task_specs_sha256",
                )
            },
            "groups": groups,
            "example": (
                None
                if example is None
                else {"task_id": example["task_id"], "contract_json": json.dumps(example["contract"], sort_keys=True)}
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censuses", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integrity-censuses", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    integrities = {}
    for path in args.integrity_censuses:
        for record in json.loads(path.read_text()):
            source_id = canonical_source_id(record["source_id"])
            if source_id in integrities:
                raise ValueError(f"Duplicate integrity census: {source_id}")
            integrities[source_id] = record
    receipts = {}
    for path in args.censuses:
        for record in json.loads(path.read_text()):
            integrity = integrities.get(canonical_source_id(record["source_id"]))
            summary = None if integrity is None else integrity_summary_for_actor(record, integrity)
            receipt = receipt_from_census(record, summary)
            if receipt.source_id in receipts:
                raise ValueError(f"Duplicate actor census: {receipt.source_id}")
            receipts[receipt.source_id] = receipt.model_dump(mode="json")
    if unknown := set(integrities) - set(receipts):
        raise ValueError(f"Integrity censuses have no actor receipt: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, receipts)


if __name__ == "__main__":
    main()
