# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the complete chromosome 4 Ensembl gene models and matched assembly sequence."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

GTF_SHA256 = "c87b75e6d522033c0912a5ce7742d56b43228a92d3a7316905eff9b94c30087f"
FASTA_SHA256 = "78622035d6f3ec532f165b792ba21c8e0dbf4fceac3a8dfd3d5968ae33f4d0da"
CHROMOSOME = "4"
CHROMOSOME_BASES = 1_348_131
CHROMOSOME_GTF_ROWS = 9_582


def freeze(gtf_path: Path, fasta_path: Path, output: Path) -> dict:
    """Preserve every chromosome 4 feature row and the matching FASTA record."""
    gtf_compressed = gtf_path.read_bytes()
    fasta_compressed = fasta_path.read_bytes()
    if hashlib.sha256(gtf_compressed).hexdigest() != GTF_SHA256:
        raise ValueError("Unexpected Ensembl release 115 GTF")
    if hashlib.sha256(fasta_compressed).hexdigest() != FASTA_SHA256:
        raise ValueError("Unexpected BDGP6.54 chromosome 4 FASTA")

    retained = []
    features = Counter()
    for line in gzip.decompress(gtf_compressed).splitlines(keepends=True):
        if line.startswith(b"#"):
            continue
        fields = line.rstrip(b"\r\n").split(b"\t")
        if len(fields) != 9:
            raise ValueError("Malformed Ensembl GTF")
        if fields[0] == CHROMOSOME.encode():
            retained.append(line)
            features[fields[2].decode()] += 1
    if len(retained) != CHROMOSOME_GTF_ROWS:
        raise ValueError("Incomplete chromosome 4 annotation")

    fasta = gzip.decompress(fasta_compressed)
    lines = fasta.splitlines(keepends=True)
    if not lines or not lines[0].startswith(b">4 ") or len(lines) < 2:
        raise ValueError("Unexpected chromosome 4 FASTA header")
    sequence = b"".join(line.strip() for line in lines[1:])
    if len(sequence) != CHROMOSOME_BASES or not set(sequence.upper()) <= set(b"ACGTN"):
        raise ValueError("Unexpected chromosome 4 sequence")
    line_bases = len(lines[1].rstrip(b"\r\n"))
    line_bytes = len(lines[1])
    if any(len(line.rstrip(b"\r\n")) != line_bases or len(line) != line_bytes for line in lines[1:-1]):
        raise ValueError("Irregular FASTA line width")
    fai = f"4\t{len(sequence)}\t{len(lines[0])}\t{line_bases}\t{line_bytes}\n".encode()

    output.mkdir(parents=True, exist_ok=True)
    assets = {
        "dmel-bdgp6.54-r115-chr4.gtf.gz": b"".join(retained),
        "dmel-bdgp6.54-r115-chr4.fa.gz": fasta,
        "dmel-bdgp6.54-r115-chr4.fa.fai.gz": fai,
    }
    result = {"chromosome": CHROMOSOME, "bases": len(sequence), "gtf_rows": len(retained), "features": features}
    for name, content in assets.items():
        compressed = gzip.compress(content, compresslevel=9, mtime=0)
        (output / name).write_bytes(compressed)
        result[name] = {
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "content_bytes": len(content),
            "vendored_sha256": hashlib.sha256(compressed).hexdigest(),
            "vendored_bytes": len(compressed),
        }
    (output / "gene-model-preparation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.gtf, args.fasta, args.output)


if __name__ == "__main__":
    main()
