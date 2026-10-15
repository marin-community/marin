# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recursive scan of a build tree for strings that must not leave the lab.

`uv run python -m experiments.domain_phase_mix.release.scan_upload release_build` prints the first 200 hits and exits 1
on any hit.

Every file is read by its structure: a CSV cell by cell, header included; a Parquet file's column names and the cells of
its non-numeric columns; a JSON file's keys and values; any other file line by line as UTF-8 text, so a binary or
compressed file, whose strings no pattern could see, stops the scan. Each file's own path is scanned too. A few fields
hold hex digests by design: `ALLOWED` exempts each of them from the hash pattern alone, so the same digest under any
other field, or any other pattern in an allowed field, is still a hit.
"""

import argparse
import dataclasses
import fnmatch
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import tables

PATTERNS = {
    "gcs_path": re.compile(r"gs://|marin-(?:us|eu)-(?:east|central|west)\d"),
    # Not the bare word: a public source file name ends in `_wandb.csv`.
    "wandb": re.compile(r"wandb\.ai|wandb_url|wandb_run|wandb:", re.I),
    "home_path": re.compile(r"/Users/|/home/\w+/"),
    # Fieldbook ids are ULIDs; a public exposure id (`exp_` and 16 hex digits) may begin `exp_01` but is too short.
    "fieldbook": re.compile(r"(?:exp|run|job)_01[a-z0-9]{24}|(?i:fieldbook)"),
    "checkpoint": re.compile(r"/checkpoints/|hf/step-"),
    "run_hash": re.compile(r"[A-Za-z0-9_]+-[0-9a-f]{6}(?![0-9A-Za-z])"),  # run names end in a six-hex-digit hash
    "long_hash": re.compile(r"[0-9a-f]{40,}"),  # git commits, SHA-256 digests, policy hashes
    "personal": re.compile(r"pinlin_calvin_xu|calvinxu", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"https?://", re.I),
}
EXCERPT = 30  # characters of context on each side of a match
SHOWN = 200  # hits the command prints


@dataclasses.dataclass(frozen=True)
class Allowance:
    """A field whose values may match one pattern by design."""

    path_glob: str  # the file, relative to the scanned root
    field: str  # a table column, or a JSON key path with `[*]` standing for every list position
    pattern: str


ALLOWED = (
    Allowance("MANIFEST.json", "$.commit", "long_hash"),  # the git commit of the build
    Allowance("MANIFEST.json", "$.tables[*].sha256", "long_hash"),  # each file's SHA-256
    Allowance("data/tasks/tasks.csv", "source_revision", "long_hash"),  # the evaluations' upstream git revisions
)


@dataclasses.dataclass(frozen=True)
class Hit:
    """One pattern matching one string of a file.

    `line_or_row` is the line of a text file, the row of a table (0 for the header, 1 for the first data row) or 0 in
    a JSON file; `field` is the table column or the JSON key path (`$.tables[3].sha256`). A hit in the file's own path
    has row 0 and an empty field.
    """

    path: Path
    line_or_row: int
    field: str
    pattern: str
    excerpt: str


def scan(root: Path) -> list[Hit]:
    """Every hit in every file under `root`, file by file in path order.

    A file the scan cannot parse (binary, compressed, a ragged CSV, invalid JSON) raises its parser's ValueError.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"no build tree to scan at {root}")
    hits = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        try:
            strings = [((0,), "", relative), *_strings(path)]
        except ValueError as error:  # the parsers say what is malformed but not in which file
            error.add_note(f"scan_upload cannot read {path}; its strings would leave unscanned")
            raise
        for rows, field, text in strings:
            for pattern, excerpt in _matches(text, _exempt(relative, field)):
                hits.extend(Hit(path, row, field, pattern, excerpt) for row in rows)
    return hits


def _strings(path: Path) -> Iterator[tuple[Sequence[int], str, str]]:
    """(rows, field, text) for every string of a file; a table yields each distinct value of a column once."""
    if path.suffix == ".csv":
        # Read without a header row, pandas neither makes a column the index nor drops a field: a ragged row raises.
        frame = tables.read_csv(path, header=None, dtype=str, keep_default_na=False)
        for column in frame:
            yield from _column_strings(frame[column], frame[column].iat[0], first_row=0)
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        for column in frame:
            yield (0,), str(column), str(column)
            if not pd.api.types.is_numeric_dtype(frame[column]):
                yield from _column_strings(frame[column], str(column), first_row=1)
    elif path.suffix == ".json":
        yield from _json_strings(json.loads(path.read_text(encoding="utf-8")), "$")
    else:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            yield (number,), "", line


def _column_strings(values: pd.Series, field: str, first_row: int) -> Iterator[tuple[Sequence[int], str, str]]:
    """Each distinct value of a column once, with the rows that hold it; the column's first value is `first_row`."""
    rows: dict[str, list[int]] = {}
    for row, value in enumerate(values.astype(str), start=first_row):
        rows.setdefault(value, []).append(row)
    for value, value_rows in rows.items():
        yield value_rows, field, value


def _json_strings(value: object, field: str) -> Iterator[tuple[Sequence[int], str, str]]:
    """Every key and scalar of a JSON value, each under its key path."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield (0,), f"{field}.{key}", key
            yield from _json_strings(child, f"{field}.{key}")
    elif isinstance(value, list):
        for position, child in enumerate(value):
            yield from _json_strings(child, f"{field}[{position}]")
    else:
        yield (0,), field, str(value)


def _exempt(relative: str, field: str) -> frozenset[str]:
    """The patterns `ALLOWED` lets this field of this file match."""
    shape = re.sub(r"\[\d+\]", "[*]", field)
    return frozenset(a.pattern for a in ALLOWED if a.field == shape and fnmatch.fnmatchcase(relative, a.path_glob))


def _matches(text: str, exempt: frozenset[str]) -> Iterator[tuple[str, str]]:
    """(pattern, excerpt around its first match) for each pattern the text matches."""
    for name, pattern in PATTERNS.items():
        match = None if name in exempt else pattern.search(text)
        if match:
            yield name, " ".join(text[max(0, match.start() - EXCERPT) : match.end() + EXCERPT].split())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the build tree, such as release_build")
    args = parser.parse_args(argv)
    hits = scan(args.root)
    for hit in hits[:SHOWN]:
        field = f" [{hit.field}]" if hit.field else ""
        print(f"{hit.path.relative_to(args.root)}:{hit.line_or_row}{field} {hit.pattern}: {hit.excerpt}")
    print(f"{len(hits)} hits")
    raise SystemExit(1 if hits else 0)


if __name__ == "__main__":
    main()
