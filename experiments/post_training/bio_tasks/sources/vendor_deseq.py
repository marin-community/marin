# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compress checksum-verified DESeq2 references and frozen GO annotations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def vendor_references(source: Path, output: Path) -> dict:
    """Preserve fitted references and select biological-process GO memberships."""
    record = json.loads((source / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, expected in sorted(record["file_hashes"].items()):
        path = source / name
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Changed preparation output: {name}")
        filename = "gse60450-" + path.name + ".gz"
        role = "private-reference"
        transformation = (
            "Unchanged native R preparation output on observed GSE60450 counts; "
            "the source script is recorded with the preparation evidence."
        )
        if path.name == "mouse-go-membership.tsv":
            filename = "mouse-go-bp-3.22.0.tsv.gz"
            role = "public-annotation"
            transformation = (
                "GOALL propagated membership for observed GSE60450 Entrez IDs; unique gene/term/ontology "
                "triples, biological process only, with evidence codes collapsed."
            )
        elif path.name == "go-terms.tsv":
            filename = "go-terms-3.22.0.tsv.gz"
            role = "public-annotation"
        digest = hashlib.sha256()
        size = 0
        target = output / filename
        with path.open("rb") as stream, target.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                for line in stream:
                    if path.name == "mouse-go-membership.tsv" and not (
                        line.startswith(b"gene\t") or line.rstrip().endswith(b"\tBP")
                    ):
                        continue
                    compressed.write(line)
                    digest.update(line)
                    size += len(line)
        with target.open("rb") as stream:
            vendored_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        assets[filename] = {
            "role": role,
            "content_sha256": digest.hexdigest(),
            "vendored_sha256": vendored_hash,
            "content_bytes": size,
            "vendored_bytes": target.stat().st_size,
            "transformation": transformation,
            "preparation_output_sha256": expected,
        }
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="verified worker outputs, including result.json")
    parser.add_argument("--output", type=Path, required=True, help="destination for compressed assets")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    assets = vendor_references(args.source, args.output)
    args.manifest.write_text(json.dumps(assets, indent=2) + "\n")


if __name__ == "__main__":
    main()
