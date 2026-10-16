# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recursive scan of a build tree for strings that must not leave the lab.

`uv run python -m experiments.domain_phase_mix.release.scan_upload release_build` prints the first 200 hits and exits 1
on any hit. `scan(root, include)` reads only the files `include` names, so an upload scans exactly the files it sends.

The patterns catch storage paths and bucket names, local and checkpoint paths, the author's user names, e-mail
addresses, Fieldbook ids, long hex hashes and URLs. Training run names and W&B run ids pass: the observation ids are the
runs' ids in the public W&B project marin-community/marin.

Every file is read by its structure: a CSV cell by cell, header included; a Parquet file's column names, the keys and
values of its schema and field metadata, and the cells of its string columns, a stored index among them; a JSON file's
keys and values; any other file line by line as UTF-8 text. Each file's own path is scanned too. Whatever would hide a
string from the patterns stops the scan: a binary or compressed file, a ragged CSV row, a repeated JSON key, and a
Parquet column of any type but string, number, boolean or time. A few fields hold hex digests by design: `ALLOWED`
exempts each of them from the hash pattern alone, so the same digest under any other field, or any other pattern in an
allowed field, is still a hit. `ALLOWED` also lets the dataset card at the root link to `LINK_HOSTS`: the `url` pattern
skips the `https://` that begins an inline Markdown link or an autolink to one of them under its path, so any other URL
of the card, or any other pattern inside such a link, is still a hit.
"""

import argparse
import collections
import dataclasses
import fnmatch
import json
import re
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.domain_phase_mix.release import tables

PATTERNS = {
    # GCS paths and Marin's bucket names on GCS and S3-compatible stores (marin-us-east5, marin-us-east-02a, marin-na).
    "bucket": re.compile(r"gs://|marin-(?:us|eu)-[a-z]+-?\d|marin-(?:na|eu|cw)\b"),
    "home_path": re.compile(r"/Users/|/home/\w+/"),
    # Fieldbook ids are ULIDs after a lowercase prefix (`exp_`, `run_`, `note_`, ...); a public id continues its prefix
    # with at most 16 characters and may begin `_01`, but it is too short.
    "fieldbook": re.compile(r"(?<![a-z0-9])[a-z]+_01[a-z0-9]{24}(?![a-z0-9])|(?i:fieldbook)"),
    "checkpoint": re.compile(r"/checkpoints/|hf/step-"),
    "long_hash": re.compile(r"[0-9a-f]{40,}"),  # git commits, SHA-256 digests, policy hashes
    "personal": re.compile(r"pinlin_calvin_xu|calvinxu", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"[a-z][a-z0-9+.-]*://", re.I),  # any scheme (http, s3, gcs, ...), whatever precedes it
}
# Arrow types that cannot hold text; a string column is scanned, and a column of any other type raises.
TEXTLESS = (
    pa.types.is_null,
    pa.types.is_boolean,
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_decimal,
    pa.types.is_temporal,
)
EXCERPT = 30  # characters of context on each side of a match
SHOWN = 200  # hits the command prints
# Hosts the dataset card may link to, each with the path its links must begin with: the paper, its code and data, the
# sources of its evaluations and buckets, and the public W&B project that logged the runs.
LINK_HOSTS = {
    "huggingface.co": "",
    "arxiv.org": "",
    "github.com": "",
    "allenai.org": "",
    "marin.community": "",
    "wandb.ai": "/marin-community/",
}
# An inline link `[text](url)` or an autolink `<url>`. A title, a bracket inside the text, or a space, parenthesis or
# angle bracket inside the URL leaves a link unrecognized, so its URL stays a hit.
MARKDOWN_LINK = re.compile(r"\[[^\[\]]*\]\((?P<inline>[^\s()<>]+)\)|<(?P<auto>[^\s<>]+)>")


@dataclasses.dataclass(frozen=True)
class Allowance:
    """A field whose values may match one pattern by design."""

    path_glob: str  # the file, relative to the scanned root
    field: str  # a table column, or a JSON key path with `[*]` standing for every list position
    pattern: str

    def exempts(self, relative: str, field: str, pattern: str, match: re.Match[str]) -> bool:
        """Whether this allows `match`, a match of `pattern` in `field` of the file at `relative`."""
        shape = re.sub(r"\[\d+\]", "[*]", field)
        return (pattern, shape) == (self.pattern, self.field) and fnmatch.fnmatchcase(relative, self.path_glob)


@dataclasses.dataclass(frozen=True)
class LinkAllowance:
    """Markdown links of a text file to `hosts`: the `url` pattern may match the `https://` that begins each one.

    A link counts when it is an inline link `[text](url)` or an autolink `<url>` whose URL has the scheme https, one of
    `hosts` as its whole network location, without a user, password or port, and a path that begins with that host's
    path in `hosts`.
    """

    path_glob: str  # the file, relative to the scanned root
    hosts: Mapping[str, str]  # host -> the path its links must begin with

    def exempts(self, relative: str, field: str, pattern: str, match: re.Match[str]) -> bool:
        """Whether this allows `match`, a match of `pattern` in `field` of the file at `relative`."""
        if pattern != "url" or field or not fnmatch.fnmatchcase(relative, self.path_glob):
            return False
        return match.start() in _link_starts(match.string, self.hosts)


ALLOWED = (
    Allowance("MANIFEST.json", "$.commit", "long_hash"),  # the git commit of the build
    Allowance("MANIFEST.json", "$.tables[*].sha256", "long_hash"),  # each file's SHA-256
    Allowance("data/tasks/tasks.csv", "source_revision", "long_hash"),  # the evaluations' upstream git revisions
    LinkAllowance("README.md", LINK_HOSTS),  # the dataset card's links
)


@dataclasses.dataclass(frozen=True)
class Hit:
    """One pattern matching one string of a file.

    `line_or_row` is the line of a text file, the row of a table (0 for the header and for Parquet metadata, 1 for the
    first data row), 0 in a JSON file, or 0 for the file's own path. `field` is the table column, `schema metadata
    <key>` or `<column> metadata <key>` in Parquet, or the JSON key path (`$.tables[3].sha256`); it is empty for a text
    line and for the file's own path.
    """

    path: Path
    line_or_row: int
    pattern: str
    excerpt: str
    field: str


def scan(root: Path, include: Iterable[str] | None = None) -> list[Hit]:
    """Every hit in every file under `root`, or in the files `include` names relative to it, file by file in path order.

    A file `include` names that `root` lacks raises a FileNotFoundError. A file the scan cannot fully read raises a
    ValueError, its parser's or the scan's own.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"no build tree to scan at {root}")
    if include is None:
        files = sorted(p for p in root.rglob("*") if p.is_file())
    else:
        files = sorted({root / name for name in include})
    hits = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            strings = [((0,), "", relative), *_strings(path)]
        except ValueError as error:  # the parsers say what is malformed but not in which file
            error.add_note(f"scan_upload cannot read {path}; its strings would leave unscanned")
            raise
        for rows, field, text in strings:
            for pattern, excerpt in _matches(relative, field, text):
                hits.extend(Hit(path, row, pattern, excerpt, field) for row in rows)
    return hits


def hit_line(hit: Hit, root: Path) -> str:
    """A hit as `<path relative to root>:<line or row> [<field>] <pattern>: <excerpt>`."""
    field = f" [{hit.field}]" if hit.field else ""
    return f"{hit.path.relative_to(root)}:{hit.line_or_row}{field} {hit.pattern}: {hit.excerpt}"


def _strings(path: Path) -> Iterator[tuple[Sequence[int], str, str]]:
    """(rows, field, text) for every string of a file; a table yields each distinct value of a column once."""
    if path.suffix == ".csv":
        # Read without a header row, pandas neither makes a column the index nor drops a field: a ragged row raises.
        frame = tables.read_csv(path, header=None, dtype=str, keep_default_na=False)
        for column in frame:
            yield from _column_strings(frame[column], frame[column].iat[0], first_row=0)
    elif path.suffix == ".parquet":
        # The file alone: by default pyarrow turns a `key=value` directory above it into a partition column.
        yield from _parquet_strings(pq.read_table(path, partitioning=None))
    elif path.suffix == ".json":
        yield from _json_strings(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object), "$")
    else:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            yield (number,), "", line


def _parquet_strings(table: pa.Table) -> Iterator[tuple[Sequence[int], str, str]]:
    """Column names, schema and field metadata, and string cells; a stored pandas index is an ordinary column."""
    yield from _metadata_strings(table.schema.metadata, "schema metadata")
    for arrow_field, column in zip(table.schema, table.columns, strict=True):
        name, type_ = arrow_field.name, arrow_field.type
        yield (0,), name, name
        yield from _metadata_strings(arrow_field.metadata, f"{name} metadata")
        if pa.types.is_string(type_) or pa.types.is_large_string(type_):
            yield from _column_strings(column.to_pylist(), name, first_row=1)
        elif not any(is_type(type_) for is_type in TEXTLESS):
            raise ValueError(f"column {name!r} has Arrow type {type_}, whose strings the scan cannot read")


def _metadata_strings(metadata: dict[bytes, bytes] | None, label: str) -> Iterator[tuple[Sequence[int], str, str]]:
    for key, value in (metadata or {}).items():
        field = f"{label} {key.decode()}"
        yield (0,), field, key.decode()
        yield (0,), field, value.decode()


def _column_strings(
    values: Iterable[str | None], field: str, first_row: int
) -> Iterator[tuple[Sequence[int], str, str]]:
    """Each distinct value of a column once, with the rows that hold it; the column's first value is `first_row`."""
    rows: dict[str, list[int]] = {}
    for row, value in enumerate(values, start=first_row):
        if value is not None:
            rows.setdefault(value, []).append(row)
    for value, value_rows in rows.items():
        yield value_rows, field, value


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """A JSON object; `json.loads` would keep only a repeated key's last value, so a repeated key raises."""
    repeated = sorted(key for key, count in collections.Counter(key for key, _ in pairs).items() if count > 1)
    if repeated:
        raise ValueError(f"a JSON object repeats the keys {repeated}")
    return dict(pairs)


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


def _link_starts(text: str, hosts: Mapping[str, str]) -> set[int]:
    """The offsets in `text` at which the https URL of a Markdown link to one of `hosts`, under its path, begins."""
    starts = set()
    for link in MARKDOWN_LINK.finditer(text):
        group = "inline" if link.group("inline") is not None else "auto"
        parts = urllib.parse.urlsplit(link.group(group))
        if parts.scheme == "https" and parts.netloc in hosts and parts.path.startswith(hosts[parts.netloc]):
            starts.add(link.start(group))
    return starts


def _matches(relative: str, field: str, text: str) -> Iterator[tuple[str, str]]:
    """(pattern, excerpt around its first match that no allowance exempts) for each pattern the text matches."""
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if not any(allowance.exempts(relative, field, name, match) for allowance in ALLOWED):
                yield name, " ".join(text[max(0, match.start() - EXCERPT) : match.end() + EXCERPT].split())
                break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the build tree, such as release_build")
    args = parser.parse_args(argv)
    hits = scan(args.root)
    for hit in hits[:SHOWN]:
        print(hit_line(hit, args.root))
    print(f"{len(hits)} hits")
    raise SystemExit(1 if hits else 0)


if __name__ == "__main__":
    main()
