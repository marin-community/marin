# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain observed COX1 proteins and accession/version provenance from UniProt."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

SOURCE = "UniProt:metazoan-cox1-20260924"


def prepare(download: Path, manifest: Path, source_directory: Path) -> None:
    raw = download.read_bytes()
    metadata = json.loads(manifest.read_text())
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("UniProt response checksum changed")
    records = json.loads(raw)["results"]
    if len(records) != int(metadata["headers"]["X-Total-Results"]) or "Link" in metadata["headers"]:
        raise ValueError("Expected a complete single-page query response")
    proteins = {}
    excluded = {}
    for record in records:
        accession = record["primaryAccession"]
        sequence = record["sequence"]["value"]
        name = record["proteinDescription"]["recommendedName"]["fullName"]["value"]
        if record["entryType"] != "UniProtKB reviewed (Swiss-Prot)" or name != "Cytochrome c oxidase subunit 1":
            raise ValueError(f"Unexpected annotation for {accession}")
        if not 450 <= len(sequence) <= 650 or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
            excluded[accession] = "Outside the 450..650-residue range or contains noncanonical amino acids"
            continue
        proteins[accession] = {
            "sequence": sequence,
            "sequence_version": record["entryAudit"]["sequenceVersion"],
            "entry_version": record["entryAudit"]["entryVersion"],
            "taxon": record["organism"]["taxonId"],
            "organism": record["organism"]["scientificName"],
            "lineage": record["organism"]["lineage"],
            "name": name,
        }
    if len({protein["taxon"] for protein in proteins.values()}) != len(proteins):
        raise ValueError("Expected one curated protein per taxon")
    content = (json.dumps({"proteins": proteins, "excluded": excluded}, indent=2) + "\n").encode()
    compressed = gzip.compress(content, mtime=0)
    filename = "uniprot-metazoan-cox1.json.gz"
    (source_directory / "data" / filename).write_bytes(compressed)
    catalog_path = source_directory / "data_sources.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sources"][SOURCE] = {
        "landing_page": "https://www.uniprot.org/uniprotkb/P00395/entry",
        "lineage": "UniProt:metazoan-mitochondrial-COX1",
        "license": "CC-BY-4.0",
        "license_source": "https://www.uniprot.org/help/license",
        "citation": "UniProt Consortium; accession, sequence/entry versions and taxonomy retained per protein.",
        "retrieved_at": "2026-09-24",
        "release": metadata["headers"]["X-UniProt-Release"],
        "data_origin": "real",
        "benchmark_screening": "Public curated observations; exact benchmark-input lineage screening pending.",
        "accessions": sorted(proteins),
        "assets": {
            filename: {
                "url": metadata["url"],
                "query": metadata["query"],
                "download_sha256": metadata["sha256"],
                "download_bytes": len(raw),
                "download_records": len(records),
                "retained_records": len(proteins),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
                "content_bytes": len(content),
                "vendored_bytes": len(compressed),
                "transformation": (
                    "Retain every query result with 450..650 canonical amino acids, without trimming, "
                    "editing or generating residues. Record excluded accessions. Keep accession/version/"
                    "taxonomy metadata; discard unrelated database annotations. gzip mtime 0. "
                    "The curated taxon sample is not representative of Metazoa; a single mitochondrial "
                    "gene tree does not establish the species tree."
                ),
            }
        },
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.download, args.manifest, args.source_directory)


if __name__ == "__main__":
    main()
