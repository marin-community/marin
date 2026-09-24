# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for the explicitly declared small input profiles used by the oracles."""

import csv
from collections.abc import Iterator
from itertools import product
from pathlib import Path

CODONS = dict(
    zip(
        ("".join(x) for x in product("TCAG", repeat=3)),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        strict=True,
    )
)


def table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def fasta(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = ""
        elif line:
            records[name] += line.strip()
    return records


def fastq(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield read ID, bases and qualities from the declared four-line profile."""
    with path.open() as handle:
        while header := handle.readline():
            sequence, separator, qualities = (handle.readline().rstrip("\r\n") for _ in range(3))
            if not header.startswith("@") or not separator.startswith("+") or len(sequence) != len(qualities):
                raise ValueError("Malformed FASTQ input")
            yield header[1:].split()[0], sequence, qualities


def tab_rows(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line and not line.startswith("#")]


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTRYMKSWBDHVN", "TGCAYRKMSWVHDBN"))[::-1]


def translate(sequence: str) -> str:
    return "".join(CODONS[sequence[i : i + 3]] for i in range(0, len(sequence) - 2, 3))
