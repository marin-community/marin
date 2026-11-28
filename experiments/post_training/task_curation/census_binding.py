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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.receipt_shards import (
    read_receipt_shards,
    receipts_from_documents,
    write_receipt_shards,
)
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId, canonical_source_id


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


EVIDENCE_FIELDS = (
    # The measurement-defining fields a receipt may never claim beyond its evidence.
    ("dataset_id", lambda receipt: receipt.measurement.dataset_id),
    ("dataset_revision", lambda receipt: receipt.measurement.dataset_revision),
    ("task_count", lambda receipt: receipt.input.measured_tasks),
    ("task_specs_sha256", lambda receipt: receipt.input.task_specs_sha256),
    ("converter_version", lambda receipt: receipt.input.converter_version),
    ("converter_revision", lambda receipt: receipt.input.converter_revision),
)


def evidence_record(evidence_url: str) -> Mapping:
    """Read a census evidence artifact's embedded measurement record."""
    document = json.loads(StoragePath(evidence_url).read_bytes())
    record = document.get("result")
    if record is None:
        raise ValueError(f"Census evidence artifact carries no embedded result: {evidence_url}")
    return record


def check_receipt_evidence(receipt: ActorCensusReceipt) -> None:
    """Refuse a census receipt that claims a measurement its evidence does not support.

    The receipt binds an input pin, content hash and converter identity; the evidence
    artifact at ``input.evidence_url`` carries the census those claims must describe. A
    receipt disagreeing with its own evidence was rewritten without re-measurement (the
    re-pin-without-census defect), and re-measuring at the declared pin is the only repair.
    """
    record = evidence_record(receipt.input.evidence_url)
    mismatches = [name for name, claimed in EVIDENCE_FIELDS if record.get(name) != claimed(receipt)]
    if mismatches:
        raise ValueError(
            f"Census receipt evidence disagrees with the receipt binding for {receipt.source_id} "
            f"({', '.join(sorted(mismatches))}): re-measure the census at the declared pin instead of "
            f"rebinding ({receipt.input.evidence_url})"
        )


def audit_receipt_evidence(receipts: Mapping[str, ActorCensusReceipt]) -> None:
    """Audit every receipt against its own evidence, naming every mismatched source."""
    failures = []
    for source_id in sorted(receipts):
        try:
            check_receipt_evidence(receipts[source_id])
        except ValueError as error:
            failures.append(f"{source_id}: {error}")
    if failures:
        raise ValueError("Census receipt evidence audit refused re-pin-without-census bindings:\n" + "\n".join(failures))


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
    evidence_url = str(StoragePath(census["path"]) / ".artifact.json")
    if evidence_url.startswith("/workspace/"):
        raise ValueError(
            f"census evidence for {census['source_id']} records an ephemeral in-region workspace path;"
            " bind the durable actor-artifact path the census stage sealed instead"
        )
    return ActorCensusReceipt(
        source_id=census["source_id"],
        input=ActorCensusInput(
            dataset_id=census["dataset_id"],
            dataset_revision=census["dataset_revision"],
            converter_version=census["converter_version"],
            converter_revision=census["converter_revision"],
            measured_tasks=census["task_count"],
            task_specs_sha256=census["task_specs_sha256"],
            evidence_url=evidence_url,
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
    parser.add_argument("--censuses", type=Path, nargs="+", required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--integrity-censuses", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--audit-receipt-evidence",
        action="store_true",
        help="Audit committed receipt shards against their evidence artifacts instead of binding",
    )
    parser.add_argument("--shard-dir", type=Path, help="Receipt shard directory for --audit-receipt-evidence")
    parser.add_argument(
        "--shard-files",
        type=Path,
        nargs="*",
        default=[],
        help="Audit only these shard files (CI scopes to the diff); default audits the whole --shard-dir",
    )
    parser.add_argument(
        "--source-ids",
        type=Path,
        help="Audit only these sources (the fingerprint CI's changed.json), regardless of shard file",
    )
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.audit_receipt_evidence:
        if args.shard_dir is None:
            parser.error("--audit-receipt-evidence requires --shard-dir")
        load_json = read_registry_json
        if args.source_ids is not None:
            wanted = {canonical_source_id(value) for value in json.loads(args.source_ids.read_text())}
            receipts = read_receipt_shards(args.shard_dir, ActorCensusReceipt, "census", load_json=load_json)
            receipts = {key: receipt for key, receipt in receipts.items() if key in wanted}
        elif args.shard_files:
            documents = {path.name: load_json(path) for path in args.shard_files}
            receipts = receipts_from_documents(documents, ActorCensusReceipt, "census")
        else:
            receipts = read_receipt_shards(args.shard_dir, ActorCensusReceipt, "census", load_json=load_json)
        audit_receipt_evidence(receipts)
        print(json.dumps({"audited_sources": len(receipts), "state": "ok"}))
        return
    if args.censuses is None or args.output_dir is None:
        parser.error("binding requires --censuses and --output-dir")
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
            check_receipt_evidence(receipt)
            if receipt.source_id in receipts:
                raise ValueError(f"Duplicate actor census: {receipt.source_id}")
            receipts[receipt.source_id] = receipt.model_dump(mode="json")
    if unknown := set(integrities) - set(receipts):
        raise ValueError(f"Integrity censuses have no actor receipt: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_receipt_shards(args.output_dir, receipts)


if __name__ == "__main__":
    main()
