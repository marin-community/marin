# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["biopython==1.86"]
# ///

"""Materialize pinned GenBank observations and independent CDS references.

Run with uv run --no-project on already downloaded records. This preparation step
uses Biopython; task generation and isolated solvers do not depend on it.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from Bio import SeqIO
from Bio.Data.CodonTable import TranslationError


def prepare_record(path: Path, destination: Path) -> dict:
    """Retain source bytes, exact coordinates, and reviewed coding-record exclusions."""
    record = SeqIO.read(path, "genbank")
    accession = record.id
    sequence = str(record.seq).upper()
    topology = record.annotations["topology"]
    genes = []
    excluded = []
    gff = ["##gff-version 3", f"##sequence-region {accession} 1 {len(sequence)}"]
    for feature in record.features:
        if feature.type != "CDS":
            continue
        identifier = feature.qualifiers.get("protein_id", feature.qualifiers.get("locus_tag", ["unidentified"]))[0]
        if "translation" not in feature.qualifiers or "transl_except" in feature.qualifiers:
            excluded.append({"id": identifier, "reason": "missing translation or recoded CDS"})
            continue
        coding = str(feature.extract(record.seq)).upper()
        table = int(feature.qualifiers.get("transl_table", ["1"])[0])
        if table not in (1, 11) or int(feature.qualifiers.get("codon_start", ["1"])[0]) != 1:
            excluded.append({"id": identifier, "reason": "unsupported genetic code or partial codon"})
            continue
        try:
            translated = str(feature.extract(record.seq).translate(table=table, cds=True))
        except TranslationError as error:
            excluded.append({"id": identifier, "reason": str(error)})
            continue
        if translated != feature.qualifiers["translation"][0]:
            raise ValueError(f"Deposited protein disagrees with Biopython translation: {accession}/{identifier}")
        parts = [{"start": int(part.start), "end": int(part.end)} for part in feature.location.parts]
        strand = int(feature.location.strand)
        genes.append(
            {
                "id": identifier,
                "gene": feature.qualifiers.get("gene", [identifier])[0],
                "strand": strand,
                "parts": parts,
                "coding_sequence": coding,
                "protein": translated,
                "genetic_code": table,
            }
        )
        consumed = 0
        for index, part in enumerate(parts):
            attributes = f"ID={quote(identifier)};part={index + 1}/{len(parts)};transl_table={table}"
            gff.append(
                "\t".join(
                    [
                        accession,
                        "RefSeq",
                        "CDS",
                        str(part["start"] + 1),
                        str(part["end"]),
                        ".",
                        "+" if strand == 1 else "-",
                        str((-consumed) % 3),
                        attributes,
                    ]
                )
            )
            consumed += part["end"] - part["start"]
    if not genes:
        raise ValueError(f"No complete coding sequences: {accession}")
    stem = accession.lower().replace(".", "-")
    assets = {
        f"{stem}.gb.gz": path.read_bytes(),
        f"{stem}.fa.gz": (
            (f">{accession}\n" + "\n".join(sequence[i : i + 70] for i in range(0, len(sequence), 70)) + "\n").encode()
        ),
        f"{stem}.gff3.gz": ("\n".join(gff) + "\n").encode(),
        f"{stem}-reference.json.gz": (
            (
                json.dumps(
                    {
                        "accession": accession,
                        "topology": topology,
                        "length": len(sequence),
                        "genes": genes,
                        "excluded_cds": excluded,
                    },
                    indent=2,
                )
                + "\n"
            ).encode()
        ),
    }
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=gbwithparts&retmode=text"
    metadata = {}
    for name, content in assets.items():
        compressed = gzip.compress(content, compresslevel=9, mtime=0)
        (destination / name).write_bytes(compressed)
        metadata[name] = {
            "url": url,
            "download_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
            "content_bytes": len(content),
            "vendored_bytes": len(compressed),
            "transformation": (
                "Original GenBank retained; FASTA sequence unchanged. GFF3 and private CDS references "
                "derived with sources/prepare_genbank.py and Biopython 1.86. Complete CDSs must reproduce deposited "
                "translations; excluded annotations are recorded in the private reference. Gzip level 9, mtime 0."
            ),
        }
    return {
        "landing_page": f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}",
        "lineage": f"RefSeq:{accession}",
        "license": "NCBI molecular-data reuse policy",
        "license_source": "https://www.ncbi.nlm.nih.gov/home/about/policies/#data",
        "citation": record.description,
        "retrieved_at": "2026-09-23",
        "data_origin": "real",
        "source_role": "reference genome and deposited annotations",
        "benchmark_screening": "Candidate reference data; full biological-lineage screening pending.",
        "assets": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    args = parser.parse_args()
    catalog_path = args.source_directory / "data_sources.json"
    catalog = json.loads(catalog_path.read_text())
    for path in args.records:
        source = prepare_record(path, args.source_directory / "data")
        catalog["sources"][source["lineage"]] = source
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
