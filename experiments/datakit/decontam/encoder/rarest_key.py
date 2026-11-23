# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1: choose the rarest 20-character key of each eval item, and measure its candidate volume.

``count``: for every item, up to ``WINDOWS_PER_ITEM`` evenly spaced windows of its item
stream (letters and digits) are candidate keys. Short question streams include the lettered
choices. A Zephyr job over reference shards counts in how many records each candidate window
occurs; each worker writes sparse counts.

``choose``: sum the counts; each item keeps its least frequent window (ties: the earliest) and
the next least frequent window that does not overlap it.

``measure``: a Zephyr job over measure shards counts the records with a candidate and the
candidate pairs, for the first window, key 1, and key 1 or 2; it also writes a sample of
candidate rows (record, item, key) for the classifier data.
"""

import argparse
import io
import json
import logging
import random
import time
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.tables import parquet_files, read_table, shard_name, write_table
from experiments.datakit.decontam.encoder.text_keys import (
    WINDOW,
    KeyStream,
    key_streams,
    record_hit_indices,
    window_hashes,
)

logger = logging.getLogger(__name__)
WINDOWS_PER_ITEM = 16
MAX_DOC_CHARS = 5_000_000
SAMPLE_ROWS_PER_SHARD = 400
MEASURE_SEED = 7
REFERENCE_SEED = 11


def item_streams(root: str, width: int) -> tuple[list[str], list[str], list[str]]:
    """One entry per item and key stream (``text_keys.key_streams``): item ids, stream kinds, streams."""
    ids, kinds, streams = [], [], []
    for path in parquet_files(root):
        for row in read_table(path, ["id", "question", "choices", "passage"]).to_pylist():
            for kind, stream in key_streams(row["question"], row["choices"] or [], row["passage"] or "", width).items():
                ids.append(row["id"])
                kinds.append(str(kind))
                streams.append(stream)
    return ids, kinds, streams


def candidate_windows(streams: list[str], width: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hashes of the candidate windows, the item of each, and its offset in the item stream."""
    hashes, owners, offsets = [], [], []
    for number, stream in enumerate(streams):
        if len(stream) < width:
            continue
        last = len(stream) - width
        starts = sorted({round(k * last / (WINDOWS_PER_ITEM - 1)) for k in range(WINDOWS_PER_ITEM)}) if last else [0]
        window = window_hashes(stream, width)
        hashes.append(window[starts])
        owners.append(np.full(len(starts), number, dtype=np.int32))
        offsets.append(np.array(starts, dtype=np.int32))
    return np.concatenate(hashes), np.concatenate(owners), np.concatenate(offsets)


def prepare(items: str, output: str, width: int) -> None:
    ids, kinds, streams = item_streams(items, width)
    hashes, owners, offsets = candidate_windows(streams, width)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        hashes=hashes,
        owners=owners,
        offsets=offsets,
        ids=np.array(ids, dtype=object),
        kinds=np.array(kinds, dtype=object),
    )
    StoragePath(prefix_join(output, "windows.npz")).write_bytes(buffer.getvalue())
    logger.info(
        "prepare: %d items, %d key streams, %d candidate windows, %d streams without a key",
        len(set(ids)),
        len(ids),
        len(hashes),
        len(ids) - len(np.unique(owners)),
    )


class Windows(NamedTuple):
    """The candidate windows of all key streams: hash, stream number, and offset in the stream; and the item id
    and stream kind of each stream."""

    hashes: np.ndarray
    owners: np.ndarray
    offsets: np.ndarray
    ids: np.ndarray
    kinds: np.ndarray


def load_windows(output: str) -> Windows:
    data = np.load(io.BytesIO(StoragePath(prefix_join(output, "windows.npz")).read_bytes()), allow_pickle=True)
    return Windows(data["hashes"], data["owners"], data["offsets"], data["ids"], data["kinds"])


class KeyTable:
    """Keys of one kind: the sorted unique hashes, and the items behind each hash."""

    def __init__(self, hashes: np.ndarray, items: np.ndarray) -> None:
        self.unique, inverse = np.unique(hashes, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self.items = items[order]
        self.starts = np.searchsorted(inverse[order], np.arange(len(self.unique) + 1))

    def items_of(self, found: np.ndarray) -> list:
        return [item for index in found.tolist() for item in self.items[self.starts[index] : self.starts[index + 1]]]


def make_count(output: str, width: int):
    def count(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        unique = np.unique(load_windows(output).hashes)
        counts = np.zeros(len(unique), dtype=np.uint32)
        records = 0
        started = time.monotonic()
        for path in paths:
            for record in load_file(InputFileSpec(path=path, columns=["text"])):
                text = str(record.get("text") or "")
                if len(text) > MAX_DOC_CHARS:
                    continue
                records += 1
                counts[record_hit_indices(text, unique, width)] += 1
        nonzero = np.nonzero(counts)[0]
        buffer = io.BytesIO()
        np.savez(buffer, hashes=unique[nonzero], counts=counts[nonzero])
        StoragePath(prefix_join(output, f"counts/part-{shard.shard_idx:05d}.npz")).write_bytes(buffer.getvalue())
        yield {"records": records, "nonzero": len(nonzero), "seconds": time.monotonic() - started}

    return count


KEY_COLUMNS = (
    "id",
    "kind",
    "first_hash",
    "first_count",
    "key1_hash",
    "key1_count",
    "key1_offset",
    "key2_hash",
    "key2_count",
)


def keys_table(windows: Windows, window_counts: np.ndarray, width: int = WINDOW) -> pa.Table:
    """One row per key stream: its first window, its rarest window (key 1), and the rarest window that does
    not overlap key 1 (key 2). Hash columns are uint64; 0 means "no such key", and its count is -1.

    A stream other than the question keeps its key only when its rarest window occurs in no reference
    record: an extra key that common text holds (a generic stem taken for a passage) would only add load.
    """
    hashes, owners, offsets, ids, kinds = windows
    columns: dict[str, list] = {name: [] for name in KEY_COLUMNS}
    order = np.lexsort((offsets, window_counts, owners))
    keyed: set[int] = set()
    start = 0
    while start < len(order):
        owner = int(owners[order[start]])
        end = start
        while end < len(order) and owners[order[end]] == owner:
            end += 1
        group = order[start:end]
        first = group[0]
        if str(kinds[owner]) != KeyStream.QUESTION and window_counts[first] > 0:
            start = end
            continue
        second = next((k for k in group[1:] if abs(int(offsets[k]) - int(offsets[first])) >= width), None)
        lowest = group[int(np.argmin(offsets[group]))]
        keyed.add(owner)
        values = (
            str(ids[owner]),
            str(kinds[owner]),
            int(hashes[lowest]),
            int(window_counts[lowest]),
            int(hashes[first]),
            int(window_counts[first]),
            int(offsets[first]),
            int(hashes[second]) if second is not None else 0,
            int(window_counts[second]) if second is not None else -1,
        )
        for name, value in zip(columns, values, strict=True):
            columns[name].append(value)
        start = end
    for owner in range(len(ids)):
        if owner not in keyed:
            for name, value in zip(columns, (str(ids[owner]), str(kinds[owner]), 0, -1, 0, -1, -1, 0, -1), strict=True):
                columns[name].append(value)
    return pa.table(
        {
            name: pa.array(np.array(values, dtype=np.uint64)) if name.endswith("_hash") else pa.array(values)
            for name, values in columns.items()
        }
    )


def choose(output: str, width: int = WINDOW) -> None:
    """Sum the reference counts of every candidate window and write each item's keys (``keys_table``)."""
    windows = load_windows(output)
    parts_hashes, parts_counts = [], []
    for path in parquet_files(prefix_join(output, "counts"), "*.npz"):
        part = np.load(io.BytesIO(StoragePath(path).read_bytes()))
        parts_hashes.append(part["hashes"])
        parts_counts.append(part["counts"].astype(np.int64))
    seen, inverse = np.unique(np.concatenate(parts_hashes), return_inverse=True)
    totals = np.bincount(inverse, weights=np.concatenate(parts_counts)).astype(np.int64)
    slot = np.minimum(np.searchsorted(seen, windows.hashes), len(seen) - 1)
    window_counts = np.where(seen[slot] == windows.hashes, totals[slot], 0)
    table = keys_table(windows, window_counts, width)
    write_table(table, prefix_join(output, "keys.parquet"))
    counts = np.array([c for c in table.column("key1_count").to_pylist() if c >= 0])
    firsts = np.array([c for c in table.column("first_count").to_pylist() if c >= 0])
    logger.info(
        "keys: %d items, %d keyed; key-1 frequency median %d, p99 %d, max %d; first-window median %d, p99 %d",
        table.num_rows,
        len(counts),
        np.median(counts),
        np.percentile(counts, 99),
        counts.max(),
        np.median(firsts),
        np.percentile(firsts, 99),
    )


def make_measure(output: str, width: int):
    def measure(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        keys = read_table(prefix_join(output, "keys.parquet")).to_pandas()
        keys = keys[keys["key1_hash"] != 0]
        # Each key row stands for an item and the stream kind it was taken from.
        entries = (keys["id"] + "\t" + keys["kind"]).to_numpy()
        tables = {}
        for name, column in (("first", "first_hash"), ("key1", "key1_hash"), ("key2", "key2_hash")):
            present = (keys[column] != 0).to_numpy()
            tables[name] = KeyTable(keys.loc[present, column].astype("uint64").to_numpy(), entries[present])
        rng = random.Random(shard.shard_idx)
        for path in paths:
            started = time.monotonic()
            stats = {"path": path, "records": 0, "chars": 0}
            for name in ("first", "key1", "either"):
                stats[f"{name}_records"] = 0
                stats[f"{name}_pairs"] = 0
            sample = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                text = str(record.get("text") or "")
                if len(text) > MAX_DOC_CHARS:
                    continue
                stats["records"] += 1
                stats["chars"] += len(text)
                found = {name: record_hit_indices(text, table.unique, width) for name, table in tables.items()}
                first = {entry.split("\t")[0] for entry in tables["first"].items_of(found["first"])}
                kind_of = {}
                for entry in tables["key1"].items_of(found["key1"]):
                    item, kind = entry.split("\t")
                    kind_of.setdefault(item, kind)
                items1 = set(kind_of)
                either = items1 | {entry.split("\t")[0] for entry in tables["key2"].items_of(found["key2"])}
                for name, items in (("first", first), ("key1", items1), ("either", either)):
                    stats[f"{name}_records"] += bool(items)
                    stats[f"{name}_pairs"] += len(items)
                for item in items1:
                    # Reservoir sample of stage-1 (key 1) candidate rows for the classifier data.
                    row = {"id": str(record["id"]), "item": str(item), "via": "key1", "kind": kind_of[item]}
                    seen = stats["key1_pairs"]
                    if len(sample) < SAMPLE_ROWS_PER_SHARD:
                        sample.append(row)
                    elif rng.random() < SAMPLE_ROWS_PER_SHARD / seen:
                        sample[rng.randrange(SAMPLE_ROWS_PER_SHARD)] = row
            stats["seconds"] = time.monotonic() - started
            name = shard_name(path)
            StoragePath(prefix_join(output, f"measure/{name}.json")).write_text(json.dumps(stats))
            if sample:
                for row in sample:
                    row["path"] = path
                write_table(pa.Table.from_pylist(sample), prefix_join(output, f"sample/{name}.parquet"))
            yield stats

    return measure


def shard_lists(
    sources_path: str, input_root: str, measure_seed: int, excluded: frozenset[str], per_source: int = 1
) -> tuple[list[str], list[str]]:
    """Measure shards (*per_source* per source) and disjoint reference shards (2 per source), none in *excluded*.

    The draws use one random stream each, in the order of the source list, so the list order fixes the sample.
    """
    with StoragePath(sources_path).open("r") as source:
        names = json.load(source)
    measure_rng, reference_rng = random.Random(measure_seed), random.Random(REFERENCE_SEED)
    measure, reference = [], []
    for name in names:
        shards = [
            shard for shard in parquet_files(prefix_join(input_root, f"{name}/outputs/main")) if shard not in excluded
        ]
        picked = measure_rng.sample(shards, min(per_source, len(shards)))
        measure += picked
        rest = [shard for shard in shards if shard not in picked]
        reference += reference_rng.sample(rest, min(2, len(rest)))
    return measure, reference


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "count", "choose", "measure"])
    parser.add_argument(
        "--sources", required=True, help="JSON list of normalized source names, such as sources.json here"
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ram", default="32g")
    parser.add_argument(
        "--width", type=int, default=WINDOW, help="key window width in stream characters; same for every step"
    )
    parser.add_argument("--measure-seed", type=int, default=MEASURE_SEED, help="seed of the measure-shard draw")
    parser.add_argument(
        "--exclude-shards",
        nargs="*",
        default=[],
        help="shards.json files of other stage-1 runs whose shards are not drawn; copy their keys.parquet to --output",
    )
    parser.add_argument("--measure-per-source", type=int, default=1, help="measure shards drawn per source")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.items, args.output, args.width)
        return
    if args.step == "choose":
        choose(args.output, args.width)
        return
    excluded = frozenset(
        shard
        for path in args.exclude_shards
        for drawn in [json.loads(StoragePath(path).read_text())]
        for shard in drawn["measure"] + drawn["reference"]
    )
    measure, reference = shard_lists(args.sources, args.input_root, args.measure_seed, excluded, args.measure_per_source)
    with StoragePath(prefix_join(args.output, "shards.json")).open("w") as sink:
        json.dump({"measure": measure, "reference": reference}, sink)
    files = reference if args.step == "count" else measure
    logger.info("%s: %d shards", args.step, len(files))
    resources = ResourceConfig(cpu=1, ram=args.ram)
    context = ZephyrContext(name=f"decon-encoder-{args.step}", resources=resources, max_workers=len(files))
    scan = make_count(args.output, args.width) if args.step == "count" else make_measure(args.output, args.width)
    outcome = context.execute(Dataset.from_list(files).map_shard(scan), map_task_resources=resources)
    logger.info("%s done: %d results", args.step, len(outcome.results))


if __name__ == "__main__":
    main()
