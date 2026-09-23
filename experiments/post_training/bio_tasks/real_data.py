# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read pinned, vendored biological observations without network access."""

import csv
import gzip
import hashlib
import io
import json
from functools import cache
from pathlib import Path

SOURCE_DIR = Path(__file__).parent


@cache
def source_catalog() -> dict:
    return json.loads((SOURCE_DIR / "data_sources.json").read_text())["sources"]


@cache
def source_text(source_id: str, filename: str) -> str:
    asset = source_catalog()[source_id]["assets"][filename]
    compressed = (SOURCE_DIR / "data" / filename).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != asset["vendored_sha256"]:
        raise ValueError(f"Changed biological source asset: {filename}")
    content = gzip.decompress(compressed)
    if hashlib.sha256(content).hexdigest() != asset["content_sha256"]:
        raise ValueError(f"Changed biological observations: {filename}")
    return content.decode()


def mammary_samples() -> list[dict[str, str]]:
    """Join GEO sample accessions to count columns through supplementary filenames."""
    metadata = {}
    for row in csv.reader(io.StringIO(source_text("GSE60450", "gse60450-series.txt.gz")), delimiter="\t"):
        if row and row[0] in {"!Sample_title", "!Sample_geo_accession", "!Sample_supplementary_file_1"}:
            metadata[row[0]] = row[1:]
    rows = []
    for accession, title, url in zip(
        metadata["!Sample_geo_accession"],
        metadata["!Sample_title"],
        metadata["!Sample_supplementary_file_1"],
        strict=True,
    ):
        count_column = url.rsplit("/", 1)[1].removeprefix(accession + "_").removesuffix(".txt.gz")
        population, rest = title.split(" ", 1)
        stage, replicate = rest.rsplit(" #", 1)
        rows.append(
            {
                "sample": count_column,
                "accession": accession,
                "population": population.lower(),
                "stage": stage,
                "replicate": replicate,
            }
        )
    return rows


def tsv_text(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
