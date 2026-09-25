# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded offline CD14 task inputs from pinned public source bytes."""

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import gate_cd14_gene_length as gate
import gate_gene_length_expression as base

INPUT_NAMES = {
    "counts": "matrix.mtx.gz",
    "genes": "genes.csv",
    "cells": "cells.csv",
    "gtf": "annotations.tsv.gz",
}


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def compress_matrix(source: Path, destination: Path) -> None:
    with source.open("rb") as raw, destination.open("xb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6) as compressed:
            shutil.copyfileobj(raw, compressed, length=1024 * 1024)


def compact_annotation(source: Path, destination: Path) -> int:
    count = 0
    with gzip.open(source, "rt") as gtf, destination.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t", lineterminator="\n")
                writer.writerow(("gene_id", "chromosome", "start", "end", "gene_biotype"))
                for line in gtf:
                    if line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if fields[2] != "gene":
                        continue
                    gene_id = base.GENE_ID.search(fields[8])
                    biotype = base.GENE_BIOTYPE.search(fields[8])
                    if gene_id is None or biotype is None:
                        raise ValueError("Malformed Ensembl gene feature")
                    writer.writerow((gene_id.group(1), fields[0], fields[3], fields[4], biotype.group(1)))
                    count += 1
    return count


def prepare(directory: Path) -> dict:
    """Download original bytes once, verify hashes, and cache deterministic inputs."""
    source = directory / "source"
    source.mkdir()
    cache = directory / "source_cache"
    cache.mkdir()
    inputs = directory / "inputs"
    inputs.mkdir()
    downloads = {}
    for key in ("counts", "genes", "cells", "gtf"):
        item = base.download(key, source)
        if item["sha256"] != gate.SOURCE_SHA256[key]:
            raise ValueError(f"Changed source SHA256: {key}")
        downloads[key] = item
    prepared = {}
    for key, name in INPUT_NAMES.items():
        original = Path(downloads[key]["path"])
        if key == "counts":
            asset = directory / name
            compress_matrix(original, asset)
        elif key == "gtf":
            asset = directory / name
            annotation_rows = compact_annotation(original, asset)
        else:
            asset = original
        identity = file_identity(asset)
        cached = cache / identity["sha256"]
        os.link(asset, cached)
        (inputs / name).symlink_to(cached)
        prepared[name] = identity
    result = {
        "source_downloads": downloads,
        "file_inputs": prepared,
        "annotation_gene_features": annotation_rows,
        "matrix_compression": "gzip level 6, empty filename, mtime 0; original MatrixMarket bytes unchanged",
        "annotation_transformation": (
            "Ensembl release-111 GTF gene features to gene_id/chromosome/start/end/biotype TSV; gzip mtime 0"
        ),
        "prep_script_sha256": file_identity(Path(__file__))["sha256"],
    }
    (directory / "prepared.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    directory = Path(os.environ["CD14_PREP_DIR"])
    directory.mkdir(exist_ok=False)
    report = prepare(directory)
    print(
        json.dumps(
            {"file_inputs": report["file_inputs"], "annotation_gene_features": report["annotation_gene_features"]}
        )
    )


if __name__ == "__main__":
    main()
