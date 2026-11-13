# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare selected reference windows with Ensembl's GRCh38 sequence service."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

ENDPOINT = "https://rest.ensembl.org/sequence/region/homo_sapiens"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = json.loads(args.panels.read_text())
    rows = [row for panel in panels["panels"].values() for row in panel]
    expected = {}
    for row in rows:
        assert row["chrom"] in {str(i) for i in range(1, 23)}
        region = f"{row['chrom']}:{row['pos'] - 127}..{row['pos'] + 127}:1"
        sequence = row["sequences"]["+"]["reference"]
        assert region not in expected or expected[region] == sequence
        expected[region] = sequence
    regions = sorted(expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves failed-attempt evidence rather than overwriting it.
    with args.output.open("x") as output, requests.Session() as session:
        for offset in range(0, len(regions), 50):
            batch = regions[offset : offset + 50]
            response = session.post(
                ENDPOINT,
                params={"coord_system_version": "GRCh38"},
                json={"regions": batch},
                headers={"Accept": "application/json"},
                timeout=45,
            )
            response.raise_for_status()
            records = response.json()
            record = {
                "requested_at": datetime.now(UTC).isoformat(),
                "endpoint": ENDPOINT,
                "assembly": "GRCh38",
                "regions": batch,
                "response": records,
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            assert {item["query"] for item in records} == set(batch)
            assert len(records) == len(batch)
            for item in records:
                assert item["id"].startswith("chromosome:GRCh38:"), item["id"]
                assert item["seq"].upper() == expected[item["query"]], item["query"]
    result = {
        "variants": len(rows),
        "unique_windows": len(regions),
        "matching_windows": len(regions),
        "assembly": "GRCh38",
        "panels_sha256": hashlib.sha256(args.panels.read_bytes()).hexdigest(),
        "responses_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
