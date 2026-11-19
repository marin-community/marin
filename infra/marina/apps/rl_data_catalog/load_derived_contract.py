# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive pinned derived-contract receipts for exact-input serving projections."""

import hashlib
import json
from pathlib import Path

import click
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from taskcompendium.grader_type import GraderType

from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from infra.marina.apps.rl_data_catalog.load import ensure_tables
from infra.marina.apps.rl_data_catalog.server.contract_selection import (
    INPUT_FIELDS,
    receipt_identity,
    select_contract_receipts,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


def load_contracts(records: list[DerivedContractReceipt], query: AppletQuery) -> dict[str, int]:
    for record in records:
        counts = record.summary.grader_type_counts
        if sum(counts.values()) != record.summary.task_count or any(value < 0 for value in counts.values()):
            raise ValueError(f"Grader types do not partition the population: {record.source_id}")
        for key in counts:
            GraderType(key)
    ensure_tables(query, ("atlas_derived_contracts",))
    archived = query("SELECT source_id,receipt FROM atlas_derived_contracts ORDER BY source_id,receipt_id", {})["rows"]
    populations = {}
    for source_id, payload in [
        *((row["source_id"], row["receipt"]) for row in archived),
        *((canonical_source_id(row.source_id), row.model_dump(mode="json")) for row in records),
    ]:
        identity = (source_id, *(payload["input"][field] for field in INPUT_FIELDS))
        populations.setdefault(identity, []).append(payload)
    for identity, payloads in populations.items():
        select_contract_receipts(identity[0], payloads)
    inserted = 0
    for record in records:
        payload = record.model_dump(mode="json")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        identity = receipt_identity(payload)
        params = {"source": canonical_source_id(record.source_id), "identity": identity, "receipt": serialized}
        inserted += query(
            (
                "INSERT INTO atlas_derived_contracts VALUES (:source,:identity,CAST(:receipt AS "
                "JSONB)) ON CONFLICT DO NOTHING"
            ),
            params,
        )["row_count"]
        stored = query(
            "SELECT receipt FROM atlas_derived_contracts WHERE source_id=:source AND receipt_id=:identity", params
        )["rows"]
        if stored != [{"receipt": payload}]:
            raise ValueError(f"Derived-contract receipt changed: {record.source_id}")
    return {"inserted": inserted, "records": len(records)}


@click.command(help=__doc__)
@click.option(
    "--input",
    "inputs",
    type=(click.Path(exists=True, path_type=Path), str),
    multiple=True,
    required=True,
    metavar="FILE SHA256",
)
@click.option("--app-id", required=True)
@click.option("--url", default=DEFAULT_MARINA_URL)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(inputs: tuple[tuple[Path, str], ...], app_id: str, url: str, receipt: Path) -> None:
    if receipt.exists():
        raise click.ClickException("Choose a new receipt path")
    records = []
    for path, sha256 in inputs:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise click.ClickException(f"Input hash differs: {path}")
        records.extend(DerivedContractReceipt.model_validate(row) for row in json.loads(raw))
    result = load_contracts(records, applet_query(url, app_id))
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
