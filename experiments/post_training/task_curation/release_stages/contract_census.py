# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive target contract receipts from the release's frozen population census."""

from dataclasses import dataclass

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.datasets.derived_contract_measurements import derived_contract_receipts
from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
    DerivedContractReceipt,
    census_source,
)
from experiments.post_training.task_curation.release_stages.census import seal
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


@dataclass(frozen=True)
class ContractCensusInputs:
    """The frozen census and target set consumed by the regional contract scan."""

    release_id: str
    source_revision: str
    census: ArtifactRef
    targets: tuple[str, ...]
    output_root: str


def target_census_rows(census: dict, targets: tuple[str, ...]) -> dict[str, dict]:
    """Select every target from the digest-bound population census."""
    rows = {row["declaration"]["id"]: row for row in census["results"]}
    missing = sorted(set(targets) - rows.keys())
    if missing:
        raise ValueError(f"targets missing from the census: {missing}")
    return {source_id: rows[source_id] for source_id in targets}


def target_contract_tasks(census: dict, targets: tuple[str, ...]) -> int:
    """The target task total from frozen actor bindings, never stale contract receipts."""
    return sum(
        row["declaration"]["actor_census_input"]["measured_tasks"]
        for row in target_census_rows(census, targets).values()
    )


def verify_contract(receipt: DerivedContractReceipt, config: DerivedContractCensusInput, scan_digest: str) -> None:
    """Refuse a contract that describes different inputs or scanned TaskSpec bytes."""
    if (
        receipt.source_id != config.population.source_id
        or receipt.input != config.input
        or receipt.population_path != config.population.path
        or receipt.scanned_task_specs_sha256 != scan_digest
    ):
        raise ValueError(f"derived contract does not cover the frozen census population: {config.population.source_id}")


def _census_contracts(inputs: ContractCensusInputs, *, dry_run: bool) -> dict:
    if not inputs.release_id or not inputs.source_revision or not inputs.targets or not inputs.output_root:
        raise ValueError("release_id, source_revision, targets and output_root are required")
    configure_coreweave_s3()
    census = read_ref(inputs.census)
    rows = target_census_rows(census, inputs.targets)
    prior = derived_contract_receipts()
    contracts = []
    reused = []
    rescanned = []
    for source_id, row in rows.items():
        config = DerivedContractCensusInput.model_validate(
            {"population": row["population"], "input": row["declaration"]["actor_census_input"]}
        )
        if config.population.source_id != source_id:
            raise ValueError(f"contract census population belongs to another source: {source_id}")
        scan_digest = row["inventory"]["scanned_task_specs_sha256"]
        receipt = prior.get(source_id)
        if (
            receipt is not None
            and receipt.input == config.input
            and receipt.population_path == config.population.path
            and receipt.scanned_task_specs_sha256 == scan_digest
        ):
            reused.append(source_id)
        else:
            rescanned.append(source_id)
            if dry_run:
                continue
            output = StoragePath(inputs.output_root) / "censuses" / canonical_sha256(config.population.source_id)[:20]
            path = output / "receipt.json"
            if path.exists():
                receipt = DerivedContractReceipt.model_validate_json(path.read_text())
            else:
                receipt = census_source(config, f"{inputs.output_root}/censuses")
                assert receipt is not None
        assert receipt is not None
        verify_contract(receipt, config, scan_digest)
        contradictions = receipt.contradictions(frozenset(row["declaration"].get("tags", ())))
        if contradictions:
            raise ValueError(
                f"declared contract tags contradict the measured population for {source_id}: {contradictions}"
            )
        contracts.append(receipt.model_dump(mode="json"))
    proof = {
        "kind": "atlas-derived-contract-census",
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "census": inputs.census.model_dump(mode="json"),
        "contract_count": len(rows),
        "contract_tasks": target_contract_tasks(census, inputs.targets),
        "reused_sources": reused,
        "rescanned_sources": rescanned,
        "dry_run": dry_run,
    }
    if dry_run:
        return proof
    contracts_ref = seal(f"{inputs.output_root}/fresh-contracts.json", {"contracts": contracts})
    proof["contracts"] = contracts_ref.model_dump(mode="json")
    receipt_ref = seal(f"{inputs.output_root}/receipt.json", proof)
    return {**proof, "receipt": receipt_ref.model_dump(mode="json")}


def run_contract_census(inputs: ContractCensusInputs, *, dry_run: bool) -> dict:
    """Reuse exact receipts and rescan changed target populations in-region."""
    alerts.stage_started(inputs.release_id, "contract_census", {"dry_run": dry_run})
    try:
        proof = _census_contracts(inputs, dry_run=dry_run)
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "contract_census", {"error": str(error)[:500]})
        raise
    alerts.stage_finished(
        inputs.release_id,
        "contract_census",
        {"contract_count": proof["contract_count"], "contract_tasks": proof["contract_tasks"]},
    )
    return proof
