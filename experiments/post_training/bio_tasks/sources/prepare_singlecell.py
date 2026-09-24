# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert observed GSE81682 counts to sparse native formats without changing counts."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


def prepare(counts: Path, metadata: Path, output: Path) -> None:
    """Preserve all cells, features and mapping metrics with their source identities."""
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(metadata, "rt") as handle:
        fields = {
            row[0]: row[1:]
            for row in csv.reader(handle, delimiter="\t")
            if row and row[0] in {"!Sample_title", "!Sample_geo_accession"}
        }
    sample_ids = {}
    for title, accession in zip(fields["!Sample_title"], fields["!Sample_geo_accession"], strict=True):
        cell = title.rsplit(" ", 1)[-1]
        if cell in sample_ids:
            raise ValueError(f"Repeated GEO cell identity: {cell}")
        sample_ids[cell] = accession

    with TemporaryDirectory(prefix="matrix-body-", dir=output) as temporary:
        body = Path(temporary) / "coordinates.txt"
        with gzip.open(counts, "rt") as handle, body.open("w") as coordinates:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            cells = header[1:]
            if header[0] != "ID" or len(set(cells)) != len(cells) or len(cells) != 1920:
                raise ValueError("Unexpected GSE81682 count columns")
            if set(cells) - sample_ids.keys():
                raise ValueError("Count columns missing from GEO metadata")
            features = []
            seen = set()
            metrics = {}
            nonzero = 0
            total_counts = 0
            for row in reader:
                key = row[0]
                if key in seen or len(row) != len(header):
                    raise ValueError(f"Duplicate or malformed feature: {key}")
                seen.add(key)
                values = [int(value) for value in row[1:]]
                if any(value < 0 for value in values):
                    raise ValueError(f"Negative observed count: {key}")
                if key.startswith("__"):
                    metrics[key] = values
                    continue
                if key.startswith("ERCC-"):
                    kind = "ERCC"
                elif key.startswith("ENSMUSG"):
                    kind = "endogenous"
                else:
                    raise ValueError(f"Unrecognized source feature: {key}")
                features.append((key, kind))
                for column, value in enumerate(values, 1):
                    if value:
                        coordinates.write(f"{len(features)} {column} {value}\n")
                        nonzero += 1
                        total_counts += value
        with (output / "matrix.mtx.gz").open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as compressed:
                compressed.write(
                    (
                        "%%MatrixMarket matrix coordinate integer general\n"
                        "% GSE81682 original counts; features by cells; HTSeq metrics stored separately\n"
                        f"{len(features)} {len(cells)} {nonzero}\n"
                    ).encode()
                )
                with body.open("rb") as source:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)

    (output / "features.tsv").write_text("id\tfeature_type\n" + "".join(f"{key}\t{kind}\n" for key, kind in features))
    with (output / "cells.tsv").open("w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "geo_accession", "sorting_gate", *metrics])
        for index, cell in enumerate(cells):
            gate = cell.rsplit("_", 1)[0]
            if gate not in {"HSPC", "LT-HSC", "Prog"}:
                raise ValueError(f"Unexpected sorting gate: {gate}")
            writer.writerow([cell, sample_ids[cell], gate, *(values[index] for values in metrics.values())])
    sources = {}
    for name, path in {"counts": counts, "series_metadata": metadata}.items():
        with path.open("rb") as handle:
            sources[name] = {"sha256": hashlib.file_digest(handle, "sha256").hexdigest(), "bytes": path.stat().st_size}
    artifacts = {}
    for path in sorted(output.iterdir()):
        with path.open("rb") as handle:
            artifacts[path.name] = {
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                "bytes": path.stat().st_size,
            }
    record = {
        "source": "GSE81682",
        "cells": len(cells),
        "features": len(features),
        "feature_types": dict(Counter(kind for _, kind in features)),
        "nonzero_counts": nonzero,
        "total_counts": total_counts,
        "htseq_metrics": list(metrics),
        "sources": sources,
        "artifacts": artifacts,
        "transformation": (
            "All original count values and cell/feature order preserved; HTSeq metrics moved to cell metadata."
        ),
        "limits": (
            "Broad sorting gates are not fine cell annotations or biological replicate identifiers. "
            "No FACS measurements, embeddings or gene symbols inferred."
        ),
    }
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.counts, args.metadata, args.output)


if __name__ == "__main__":
    main()
