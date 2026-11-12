# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report which records of a dataset occur verbatim in the hero store's training documents.

The hero store ``store_4d2e363d`` holds tokens, so its text is read from each source's
normalized documents, and a document counts as kept unless the store dropped it: its
decontamination mark is set, or the exact or verified fuzzy dedup attributes list it.
``hero_store_sources.json`` holds those four directories for each of the store's sources.

A field of a record is found in a document when the field's text occurs in the document,
both lowercased with whitespace collapsed; the document is also searched after undoing
JSON and Python string escapes, since many copies sit inside JSON lines and code. Fields
shorter than ``--min-words`` words are not searched. A record is copied when one kept
document holds every searched field.

Input is a Hugging Face dataset (``--hf REPO --revision SHA --config NAME --split NAME``)
or Parquet files (``--parquet GLOB``), with the string fields to search (``--fields``).
Writes under OUT_PREFIX:

* ``targets.json``: the searched fields of each record;
* ``matches/<file>.parquet``: one row per normalized file, document and found field, with
  the store's drop flags;
* ``records.parquet``: per record, whether one document (kept or not) holds every searched
  field, and the kept documents and sources that do;
* ``report.md``: the summary.

Usage: store_copies.py OUT_PREFIX (--hf REPO --revision SHA --config NAME --split NAME
       | --parquet GLOB) --fields F [F ...] [--id-field F] [--min-words N]
       [--sources S [S ...]] [--every N] [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

logger = logging.getLogger(__name__)

STORE_ROOT = "s3://marin-us-east-02a/marin"
SOURCES_FILE = Path(__file__).with_name("hero_store_sources.json")
DEFAULT_MIN_WORDS = 8
READ_BATCH_ROWS = 64
SAMPLE_DOCS = 3
TOP_SOURCES = 10

MATCH_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("path", pa.string()),
        ("doc_id", pa.string()),
        ("record_id", pa.string()),
        ("field", pa.string()),
        ("contaminated", pa.bool_()),
        ("exact_dup", pa.bool_()),
        ("fuzzy_dup", pa.bool_()),
        ("kept", pa.bool_()),
    ]
)

_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|[nrt\"'\\/])")
_SIMPLE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "\\": "\\", "/": "/"}


_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\S+")


def normalized(text: str) -> str:
    """Lowercase and collapse whitespace runs to one space.

    A regex substitution, not ``" ".join(text.split())``: repository documents reach
    hundreds of MB, and a list of every word costs several times the text's size.
    """
    return _WHITESPACE.sub(" ", text.lower()).strip()


def unescaped(text: str) -> str:
    """Undo JSON and Python string escapes, leaving any other backslash as is."""

    def replace(match: re.Match) -> str:
        escape = match.group(1)
        return chr(int(escape[1:], 16)) if escape[0] == "u" else _SIMPLE_ESCAPES[escape]

    # Escaped surrogate pairs (emoji in JSON) decode to two code points; join them.
    return _ESCAPE.sub(replace, text).encode("utf-16", "surrogatepass").decode("utf-16", "replace")


@dataclass(frozen=True)
class Target:
    record_id: str
    field: str
    text: str
    """The field's normalized text."""
    anchor: str
    """The text's longest inner word. An occurrence of the text keeps its inner words whole,
    so every occurrence holds this word as a whole word of the document."""
    anchor_offset: int
    """Where ``anchor`` starts in ``text``."""


def make_target(record_id: str, field: str, value: str) -> Target:
    text = normalized(value)
    words = text.split(" ")
    starts = [0]
    for word in words[:-1]:
        starts.append(starts[-1] + len(word) + 1)
    index = max(range(1, len(words) - 1), key=lambda i: (len(words[i]), -i))
    return Target(record_id, field, text, words[index], starts[index])


def targets_for(records: Iterable[dict], id_field: str | None, fields: list[str], min_words: int) -> list[Target]:
    """Build the targets of every record's fields that have at least ``min_words`` words."""
    if min_words < 3:
        raise ValueError("min_words must be at least 3, so every target has inner words")
    out, seen = [], set()
    for row_number, record in enumerate(records):
        record_id = str(record[id_field]) if id_field else str(row_number)
        if record_id in seen:
            raise ValueError(f"record id {record_id!r} occurs twice; pass a unique --id-field")
        seen.add(record_id)
        for field in fields:
            value = record[field]
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"field {field!r} of record {record_id!r} is {type(value).__name__}, not a string")
            if len(value.split()) >= min_words:
                out.append(make_target(record_id, field, value))
    return out


class Matcher:
    """Find which targets occur in a document."""

    def __init__(self, targets: list[Target]):
        self.targets = targets
        self.by_anchor: dict[str, list[int]] = defaultdict(list)
        for i, target in enumerate(targets):
            self.by_anchor[target.anchor].append(i)

    def found(self, text: str) -> list[Target]:
        variants = [normalized(text)]
        if "\\" in text:
            variants.append(normalized(unescaped(text)))
        hits: dict[int, Target] = {}
        for variant in variants:
            # Compare each target only where its anchor occurs, so the cost grows with the
            # document's length, not with its length times the number of targets.
            anchor_starts: dict[str, list[int]] = defaultdict(list)
            for match in _WORD.finditer(variant):
                if match.group() in self.by_anchor:
                    anchor_starts[match.group()].append(match.start())
            for anchor, starts in anchor_starts.items():
                for i in self.by_anchor[anchor]:
                    target = self.targets[i]
                    if i not in hits and any(
                        variant.startswith(target.text, start - target.anchor_offset)
                        for start in starts
                        if start >= target.anchor_offset
                    ):
                        hits[i] = target
        return list(hits.values())


@dataclass(frozen=True)
class StoreSource:
    """A store source's normalized directory and the attribute directories the store joined, under ``STORE_ROOT``."""

    dir: str
    decon: str
    exact: str
    fuzzy: str | None
    """None for sources the store exempts from fuzzy dedup."""


def store_sources() -> dict[str, StoreSource]:
    return {name: StoreSource(**dirs) for name, dirs in json.loads(SOURCES_FILE.read_text()).items()}


@dataclass(frozen=True)
class SourceFile:
    source: str
    path: str
    decon: str
    exact: str
    fuzzy: str | None
    """None for sources the store exempts from fuzzy dedup."""


def store_files(sources: dict[str, StoreSource], names: list[str] | None, every: int) -> list[SourceFile]:
    """List the normalized files of the chosen sources, every ``every``-th one of each."""
    unknown = sorted(set(names or []) - sources.keys())
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    files = []
    for name in sorted(names or sources):
        source = sources[name]
        root = StoragePath(STORE_ROOT)
        paths = sorted(str(p) for p in (root / source.dir / "*.parquet").glob())
        if not paths:
            raise ValueError(f"{name}: no Parquet files in {root / source.dir}")
        for path in paths[::every]:
            base = path.rsplit("/", 1)[1]
            fuzzy = str(root / source.fuzzy / base) if source.fuzzy else None
            files.append(SourceFile(name, path, str(root / source.decon / base), str(root / source.exact / base), fuzzy))
    return files


def _ids_in(path: str, wanted: set[str]) -> set[str]:
    """Return the wanted ids a sparse dedup attribute file lists; a missing file lists none."""
    if not StoragePath(path).exists():
        return set()
    with StoragePath(path).open("rb") as f:
        ids = pq.read_table(f, columns=["id"]).column("id").to_pylist()
    return {str(i) for i in ids} & wanted


def _contaminated(path: str, wanted: set[str]) -> dict[str, bool]:
    with StoragePath(path).open("rb") as f:
        table = pq.read_table(f, columns=["id", "contaminated"])
    marks = {
        str(i): bool(c)
        for i, c in zip(table.column("id").to_pylist(), table.column("contaminated").to_pylist(), strict=True)
        if str(i) in wanted
    }
    if missing := wanted - marks.keys():
        raise ValueError(f"{path}: {len(missing)} documents have no decontamination mark")
    return marks


def file_matches(file: SourceFile, matcher: Matcher) -> list[dict]:
    """Search one normalized file and attach the store's drop flags to each match."""
    found: list[tuple[str, Target]] = []
    with StoragePath(file.path).open("rb") as f:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=READ_BATCH_ROWS, columns=["id", "text"]):
            ids, texts = batch.column("id"), batch.column("text")
            # One document at a time: a batch of large repository documents can hold GBs of text.
            for row in range(batch.num_rows):
                text = texts[row].as_py()
                if text:
                    found += [(str(ids[row].as_py()), t) for t in matcher.found(text)]
    if not found:
        return []
    wanted = {doc_id for doc_id, _ in found}
    contaminated = _contaminated(file.decon, wanted)
    exact = _ids_in(file.exact, wanted)
    fuzzy = _ids_in(file.fuzzy, wanted) if file.fuzzy else set()
    return [
        {
            "source": file.source,
            "path": file.path,
            "doc_id": doc_id,
            "record_id": t.record_id,
            "field": t.field,
            "contaminated": contaminated[doc_id],
            "exact_dup": doc_id in exact,
            "fuzzy_dup": doc_id in fuzzy,
            "kept": not (contaminated[doc_id] or doc_id in exact or doc_id in fuzzy),
        }
        for doc_id, t in found
    ]


def targets_path(out_prefix: str) -> StoragePath:
    return StoragePath(out_prefix) / "targets.json"


def match_path(out_prefix: str, path: str) -> StoragePath:
    return StoragePath(out_prefix) / "matches" / f"{hashlib.sha256(path.encode()).hexdigest()[:20]}.parquet"


def _search_shard(out_prefix: str):
    def run(items: Iterator[dict], shard) -> Iterator[str]:
        with targets_path(out_prefix).open("r") as f:
            matcher = Matcher([Target(**t) for t in json.load(f)])
        for item in items:
            out = match_path(out_prefix, item["path"])
            # A finished file's output is written whole, so a rerun skips it.
            if out.exists():
                continue
            rows = file_matches(SourceFile(**item), matcher)
            with out.open("wb") as f:
                pq.write_table(pa.Table.from_pylist(rows, schema=MATCH_SCHEMA), f)
            yield str(out)

    return run


def record_summary(targets: list[Target], matches: list[dict]) -> list[dict]:
    """Per record: whether one document, and one kept document, holds every searched field."""
    fields_of: dict[str, set[str]] = defaultdict(set)
    for t in targets:
        fields_of[t.record_id].add(t.field)
    in_doc: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    kept_doc: dict[tuple[str, str, str], bool] = {}
    source_of: dict[tuple[str, str, str], str] = {}
    for m in matches:
        key = (m["record_id"], m["path"], m["doc_id"])
        in_doc[key].add(m["field"])
        kept_doc[key] = m["kept"]
        source_of[key] = m["source"]
    full: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key, fields in in_doc.items():
        if fields == fields_of[key[0]]:
            full[key[0]].append(key)
    rows = []
    for record_id in sorted(fields_of):
        docs = sorted(full.get(record_id, []))
        kept = [k for k in docs if kept_doc[k]]
        rows.append(
            {
                "record_id": record_id,
                "fields": sorted(fields_of[record_id]),
                "in_any_doc": bool(docs),
                "in_kept_doc": bool(kept),
                "kept_docs": len(kept),
                "kept_sources": sorted({source_of[k] for k in kept}),
                "sample_kept_docs": [f"{path}#{doc_id}" for _, path, doc_id in kept[:SAMPLE_DOCS]],
            }
        )
    return rows


def _share(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)" if total else "0"


def report_markdown(
    title: str, n_records: int, targets: list[Target], records: list[dict], matches: list[dict], scope: str
) -> str:
    searched = len(records)
    lines = [
        f"# {title}",
        "",
        f"Scope: {scope}. {n_records:,} records, {searched:,} with at least one searched field.",
        "",
        "| One document holds every searched field | Records (share of searched) |",
        "| :---- | :---- |",
        f"| Any document | {_share(sum(r['in_any_doc'] for r in records), searched)} |",
        f"| A document the store kept | {_share(sum(r['in_kept_doc'] for r in records), searched)} |",
        "",
        "| Field | Records searched | Found in a kept document |",
        "| :---- | :---- | :---- |",
    ]
    searched_by_field = Counter(t.field for t in targets)
    kept_by_field = Counter(f for f, _ in {(m["field"], m["record_id"]) for m in matches if m["kept"]})
    for field in sorted(searched_by_field):
        lines.append(
            f"| `{field}` | {searched_by_field[field]:,} | {_share(kept_by_field[field], searched_by_field[field])} |"
        )
    sources = Counter(s for r in records for s in r["kept_sources"])
    if sources:
        lines += ["", "| Source (kept documents holding a whole record) | Records |", "| :---- | :---- |"]
        lines += [f"| `{s}` | {n:,} |" for s, n in sources.most_common(TOP_SOURCES)]
    return "\n".join(lines) + "\n"


def hf_records(repo: str, revision: str, config: str | None, split: str) -> list[dict]:
    return list(load_dataset(repo, config, split=split, revision=revision))


def parquet_records(pattern: str) -> list[dict]:
    paths = sorted(str(p) for p in StoragePath(pattern).glob())
    if not paths:
        raise ValueError(f"no files match {pattern}")
    rows = []
    for path in paths:
        with StoragePath(path).open("rb") as f:
            rows += pq.read_table(f).to_pylist()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_prefix")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf", help="Hugging Face dataset repository")
    source.add_argument("--parquet", help="glob of Parquet files")
    parser.add_argument("--revision", help="pinned Hugging Face revision (commit SHA)")
    parser.add_argument("--config", help="Hugging Face config name")
    parser.add_argument("--split", help="Hugging Face split")
    parser.add_argument("--fields", nargs="+", required=True, help="string fields to search")
    parser.add_argument("--id-field", help="unique record id field (default: the row number)")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--sources", nargs="+", help="search only these store sources")
    parser.add_argument("--every", type=int, default=1, help="search every N-th file of each source")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--worker-ram", default="32g")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.hf:
        if not (args.revision and args.split):
            raise ValueError("--hf needs --revision and --split")
        rows = hf_records(args.hf, args.revision, args.config, args.split)
        title = " ".join(x for x in (f"{args.hf}@{args.revision[:10]}", args.config, args.split) if x)
    else:
        rows = parquet_records(args.parquet)
        title = args.parquet
    targets = targets_for(rows, args.id_field, args.fields, args.min_words)
    if not targets:
        raise ValueError(f"no field of any record has {args.min_words} or more words")
    targets_path(args.out_prefix).write_text(json.dumps([asdict(t) for t in targets]))
    files = store_files(store_sources(), args.sources, args.every)
    logger.info("%s: %d records, %d targets, %d files", title, len(rows), len(targets), len(files))

    ctx = ZephyrContext(
        name="store-copies", resources=ResourceConfig(cpu=1, ram=args.worker_ram), max_workers=args.max_workers
    )
    ctx.execute(Dataset.from_list([vars(f) for f in files]).map_shard(_search_shard(args.out_prefix)))

    matches = []
    for f in files:
        out = match_path(args.out_prefix, f.path)
        if not out.exists():
            raise ValueError(f"{f.path} was not searched: {out} is missing")
        with out.open("rb") as fh:
            matches += pq.read_table(fh).to_pylist()
    records = record_summary(targets, matches)
    with (StoragePath(args.out_prefix) / "records.parquet").open("wb") as fh:
        pq.write_table(pa.Table.from_pylist(records), fh)
    sources = sorted({f.source for f in files})
    scope = f"{len(files):,} files of {len(sources)} sources" + (
        f", every {args.every}th file" if args.every > 1 else ""
    )
    report = report_markdown(title, len(rows), targets, records, matches, scope)
    (StoragePath(args.out_prefix) / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
