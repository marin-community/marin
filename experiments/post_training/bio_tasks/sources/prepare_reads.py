# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendor bounded, unmodified paired FASTQ records with exact source intervals."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

RECORDS_PER_BLOCK = 2000
BLOCKS = 3


def prepare_reads(downloads: Path, source_directory: Path) -> None:
    metadata = json.loads((downloads / "ERR266411-metadata.json").read_text())[0]
    fetches = json.loads((downloads / "ena-reads-fetch.json").read_text())
    assets = {}
    names = []
    for mate in (1, 2):
        content = (downloads / f"ERR266411_{mate}.fastq").read_bytes()
        if hashlib.sha256(content).hexdigest() != fetches[mate - 1]["content_sha256"]:
            raise ValueError("Downloaded FASTQ changed")
        lines = content.splitlines(keepends=True)
        if len(lines) != RECORDS_PER_BLOCK * BLOCKS * 4:
            raise ValueError("Unexpected read-prefix size")
        identifiers = []
        for index in range(0, len(lines), 4):
            header, sequence, separator, qualities = lines[index : index + 4]
            if not header.startswith(b"@") or not separator.startswith(b"+") or len(sequence) != len(qualities):
                raise ValueError(f"Malformed FASTQ record at {index // 4}")
            identifiers.append(header.split()[0][1:])
        names.append(identifiers)
        for block in range(BLOCKS):
            first, last = block * RECORDS_PER_BLOCK, (block + 1) * RECORDS_PER_BLOCK
            subset = b"".join(lines[first * 4 : last * 4])
            compressed = gzip.compress(subset, compresslevel=9, mtime=0)
            filename = f"err266411-block{block + 1}-r{mate}.fastq.gz"
            (source_directory / "data" / filename).write_bytes(compressed)
            assets[filename] = {
                "url": fetches[mate - 1]["url"],
                "download_sha256": fetches[mate - 1]["prefix_sha256"],
                "download_scope": "First 2097152 compressed bytes, HTTP Range request; not the complete remote archive.",
                "content_sha256": hashlib.sha256(subset).hexdigest(),
                "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
                "content_bytes": len(subset),
                "vendored_bytes": len(compressed),
                "first_record_1based": first + 1,
                "last_record_1based": last,
                "transformation": (
                    "Consecutive original complete FASTQ records; no changes to names, sequences or "
                    "qualities. The three blocks are disjoint technical subsets, not biological replicates. "
                    "Gzip level 9, mtime 0."
                ),
            }
    if names[0] != names[1] or len(set(names[0])) != len(names[0]):
        raise ValueError("FASTQ pair IDs are duplicated or mismatched")
    catalog_path = source_directory / "data_sources.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sources"]["ENA:ERR266411"] = {
        "landing_page": "https://www.ebi.ac.uk/ena/browser/view/ERR266411",
        "lineage": "BioSample:SAMEA1879783",
        "study": "BioProject:PRJEB1861",
        "license": "EMBL-EBI Terms of Use",
        "license_source": "https://www.ebi.ac.uk/about/terms-of-use/",
        "citation": "ERX240907 / ERR266411: observed PhiX sequencing, Illumina HiSeq 2000",
        "retrieved_at": "2026-09-23",
        "data_origin": "real",
        "source_role": "observed paired-end sequencing reads from a reference sequencing sample",
        "sampling_limits": (
            "First 6000 pairs in archive order, split into three disjoint 2000-pair blocks. "
            "Preserves observed bases/qualities; not a random sample and not biological replication."
        ),
        "benchmark_screening": "Independent candidate sequencing run; full biological-lineage screening pending.",
        "run_metadata": metadata,
        "assets": assets,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    args = parser.parse_args()
    prepare_reads(args.downloads, args.source_directory)


if __name__ == "__main__":
    main()
