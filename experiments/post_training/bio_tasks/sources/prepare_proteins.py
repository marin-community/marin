# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["biopython==1.86"]
# ///

"""Pin curated protein sequences and independently computed pairwise score bounds."""

import argparse
import gzip
import hashlib
import json
from itertools import combinations
from pathlib import Path

from Bio.Align import PairwiseAligner, substitution_matrices


def prepare(download: Path, source_directory: Path) -> None:
    records = json.loads(download.read_text())["results"]
    proteins = {}
    for record in records:
        if record["entryType"] != "UniProtKB reviewed (Swiss-Prot)":
            raise ValueError("Expected reviewed UniProt entries")
        proteins[record["primaryAccession"]] = {
            "sequence": record["sequence"]["value"],
            "sequence_version": record["entryAudit"]["sequenceVersion"],
            "entry_version": record["entryAudit"]["entryVersion"],
            "taxon": record["organism"]["taxonId"],
            "organism": record["organism"]["scientificName"],
            "name": record["proteinDescription"]["recommendedName"]["fullName"]["value"],
        }
    matrix = substitution_matrices.load("BLOSUM62") * 2
    aligner = PairwiseAligner(mode="global", substitution_matrix=matrix, open_gap_score=-20, extend_gap_score=-1)
    bounds = {
        f"{left}:{right}": int(aligner.score(proteins[left]["sequence"], proteins[right]["sequence"]))
        for left, right in combinations(sorted(proteins), 2)
    }
    content = (
        json.dumps(
            {
                "proteins": proteins,
                "scoring": {a + b: int(matrix[a, b]) for a in "ACDEFGHIKLMNPQRSTVWY" for b in "ACDEFGHIKLMNPQRSTVWY"},
                "gap_open": 20,
                "gap_extend": 1,
                "optimal_pair_scores": bounds,
                "reference_method": "Biopython 1.86 PairwiseAligner global; 2*BLOSUM62; gap open -20, extension -1",
            },
            indent=2,
        )
        + "\n"
    ).encode()
    compressed = gzip.compress(content, mtime=0)
    filename = "uniprot-globins.json.gz"
    (source_directory / "data" / filename).write_bytes(compressed)
    catalog_path = source_directory / "data_sources.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sources"]["UniProt:globins-20260923"] = {
        "landing_page": "https://www.uniprot.org/uniprotkb?query=family%3Aglobin",
        "lineage": "UniProt:vertebrate-globins",
        "license": "CC-BY-4.0",
        "license_source": "https://www.uniprot.org/help/license",
        "citation": "UniProt Consortium; accession, entry version and sequence version retained per protein.",
        "retrieved_at": "2026-09-23",
        "data_origin": "real",
        "benchmark_screening": "Candidate curated sequences; full benchmark-lineage screening pending.",
        "accessions": sorted(proteins),
        "assets": {
            filename: {
                "url": (
                    "https://rest.uniprot.org/uniprotkb/search?query="
                    + "%20OR%20".join(f"accession:{a}" for a in sorted(proteins))
                    + "&format=json&size=100"
                ),
                "download_sha256": hashlib.sha256(download.read_bytes()).hexdigest(),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
                "content_bytes": len(content),
                "vendored_bytes": len(compressed),
                "transformation": (
                    "Sequences unchanged. Selected accession/version/taxonomy/name metadata; added BLOSUM62 "
                    "and independent optimal pairwise scores. gzip mtime 0."
                ),
            }
        },
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.download, args.source_directory)


if __name__ == "__main__":
    main()
