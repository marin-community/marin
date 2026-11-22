# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1: choose the rarest 20-character key of each eval item, and measure its candidate volume.

``count``: for every item, up to ``WINDOWS_PER_ITEM`` evenly spaced windows of its question
stream (letters and digits) are candidate keys. A Zephyr job over reference shards counts in
how many records each candidate window occurs; each worker writes sparse counts.

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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.text_keys import WINDOW, record_hit_indices, text_stream, window_hashes

logger = logging.getLogger(__name__)
WINDOWS_PER_ITEM = 16
MAX_DOC_CHARS = 5_000_000
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SAMPLE_ROWS_PER_SHARD = 400


def item_streams(root: str) -> tuple[list[str], list[str]]:
    """Item ids and the stream that their keys come from: the question, or the block for a short question."""
    ids, streams = [], []
    for path in sorted(str(p) for p in StoragePath(root + "/*.parquet").glob()):
        with StoragePath(path).open("rb") as source:
            table = pq.read_table(io.BytesIO(source.read()), columns=["id", "question", "choices"]).to_pylist()
        for row in table:
            stream = text_stream(row["question"])
            if len(stream) < WINDOW and row["choices"]:
                stream += "".join(LETTERS[i % 26] + text_stream(c) for i, c in enumerate(row["choices"]))
            ids.append(row["id"])
            streams.append(stream)
    return ids, streams


def candidate_windows(streams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hashes of the candidate windows, the item of each, and its offset in the item stream."""
    hashes, owners, offsets = [], [], []
    for number, stream in enumerate(streams):
        if len(stream) < WINDOW:
            continue
        last = len(stream) - WINDOW
        starts = sorted({round(k * last / (WINDOWS_PER_ITEM - 1)) for k in range(WINDOWS_PER_ITEM)}) if last else [0]
        window = window_hashes(stream)
        hashes.append(window[starts])
        owners.append(np.full(len(starts), number, dtype=np.int32))
        offsets.append(np.array(starts, dtype=np.int32))
    return np.concatenate(hashes), np.concatenate(owners), np.concatenate(offsets)


def prepare(items: str, output: str) -> None:
    ids, streams = item_streams(items)
    hashes, owners, offsets = candidate_windows(streams)
    buffer = io.BytesIO()
    np.savez(buffer, hashes=hashes, owners=owners, offsets=offsets, ids=np.array(ids, dtype=object))
    with StoragePath(prefix_join(output, "windows.npz")).open("wb") as sink:
        sink.write(buffer.getvalue())
    logger.info(
        "prepare: %d items, %d candidate windows, %d without a key",
        len(ids),
        len(hashes),
        len(ids) - len(np.unique(owners)),
    )


def load_windows(output: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with StoragePath(prefix_join(output, "windows.npz")).open("rb") as source:
        data = np.load(io.BytesIO(source.read()), allow_pickle=True)
    return data["hashes"], data["owners"], data["offsets"], data["ids"]


class KeyTable:
    """Keys of one kind: the sorted unique hashes, and the items behind each hash."""

    def __init__(self, hashes: np.ndarray, items: np.ndarray) -> None:
        self.unique, inverse = np.unique(hashes, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self.items = items[order]
        self.starts = np.searchsorted(inverse[order], np.arange(len(self.unique) + 1))

    def items_of(self, found: np.ndarray) -> list:
        return [item for index in found.tolist() for item in self.items[self.starts[index] : self.starts[index + 1]]]


def make_count(output: str):
    def count(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        hashes, _, _, _ = load_windows(output)
        unique = np.unique(hashes)
        counts = np.zeros(len(unique), dtype=np.uint32)
        records = 0
        started = time.monotonic()
        for path in paths:
            for record in load_file(InputFileSpec(path=path, columns=["text"])):
                text = str(record.get("text") or "")
                if len(text) > MAX_DOC_CHARS:
                    continue
                records += 1
                counts[record_hit_indices(text, unique)] += 1
        nonzero = np.nonzero(counts)[0]
        buffer = io.BytesIO()
        np.savez(buffer, hashes=unique[nonzero], counts=counts[nonzero])
        with StoragePath(prefix_join(output, f"counts/part-{shard.shard_idx:05d}.npz")).open("wb") as sink:
            sink.write(buffer.getvalue())
        yield {"records": records, "nonzero": len(nonzero), "seconds": time.monotonic() - started}

    return count


def choose(output: str) -> None:
    hashes, owners, offsets, ids = load_windows(output)
    parts_hashes, parts_counts = [], []
    for path in StoragePath(prefix_join(output, "counts/*.npz")).glob():
        with path.open("rb") as source:
            part = np.load(io.BytesIO(source.read()))
        parts_hashes.append(part["hashes"])
        parts_counts.append(part["counts"].astype(np.int64))
    seen, inverse = np.unique(np.concatenate(parts_hashes), return_inverse=True)
    totals = np.bincount(inverse, weights=np.concatenate(parts_counts)).astype(np.int64)
    slot = np.minimum(np.searchsorted(seen, hashes), len(seen) - 1)
    window_counts = np.where(seen[slot] == hashes, totals[slot], 0)
    # The reference frequency of every candidate window, for the classifier's rarity hint.
    frequency = pa.table({"hash": pa.array(seen, type=pa.uint64()), "count": pa.array(totals)})
    buffer = io.BytesIO()
    pq.write_table(frequency, buffer)
    with StoragePath(prefix_join(output, "window_counts.parquet")).open("wb") as sink:
        sink.write(buffer.getvalue())
    # One row per item. Hash columns are uint64; 0 means "no such key" (an item with no window, or no second key).
    columns: dict[str, list] = {
        name: []
        for name in (
            "id",
            "first_hash",
            "first_count",
            "key1_hash",
            "key1_count",
            "key1_offset",
            "key2_hash",
            "key2_count",
        )
    }
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
        second = next((k for k in group[1:] if abs(int(offsets[k]) - int(offsets[first])) >= WINDOW), None)
        lowest = group[int(np.argmin(offsets[group]))]
        keyed.add(owner)
        values = (
            str(ids[owner]),
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
            for name, value in zip(columns, (str(ids[owner]), 0, -1, 0, -1, -1, 0, -1), strict=True):
                columns[name].append(value)
    table = pa.table(
        {
            name: pa.array(np.array(values, dtype=np.uint64)) if name.endswith("_hash") else pa.array(values)
            for name, values in columns.items()
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    with StoragePath(prefix_join(output, "keys.parquet")).open("wb") as sink:
        sink.write(buffer.getvalue())
    counts = np.array([c for c in columns["key1_count"] if c >= 0])
    firsts = np.array([c for c in columns["first_count"] if c >= 0])
    logger.info(
        "keys: %d items, %d keyed; key-1 frequency median %d, p99 %d, max %d; first-window median %d, p99 %d",
        len(columns["id"]),
        len(counts),
        np.median(counts),
        np.percentile(counts, 99),
        counts.max(),
        np.median(firsts),
        np.percentile(firsts, 99),
    )


def make_measure(output: str):
    def measure(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        with StoragePath(prefix_join(output, "keys.parquet")).open("rb") as source:
            keys = pq.read_table(io.BytesIO(source.read())).to_pandas()
        keys = keys[keys["key1_hash"] != 0]
        tables = {}
        for name, column in (("first", "first_hash"), ("key1", "key1_hash"), ("key2", "key2_hash")):
            present = keys[column] != 0
            tables[name] = KeyTable(
                keys.loc[present, column].astype("uint64").to_numpy(), keys.loc[present, "id"].to_numpy()
            )
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
                found = {name: record_hit_indices(text, table.unique) for name, table in tables.items()}
                first = tables["first"].items_of(found["first"])
                items1 = set(tables["key1"].items_of(found["key1"]))
                either = items1 | set(tables["key2"].items_of(found["key2"]))
                for name, items in (("first", first), ("key1", items1), ("either", either)):
                    stats[f"{name}_records"] += bool(items)
                    stats[f"{name}_pairs"] += len(items)
                for item in either:
                    # Reservoir sample of candidate rows for the classifier data.
                    row = {"id": str(record["id"]), "item": str(item), "via": "key1" if item in items1 else "key2"}
                    seen = stats["either_pairs"]
                    if len(sample) < SAMPLE_ROWS_PER_SHARD:
                        sample.append(row)
                    elif rng.random() < SAMPLE_ROWS_PER_SHARD / seen:
                        sample[rng.randrange(SAMPLE_ROWS_PER_SHARD)] = row
            stats["seconds"] = time.monotonic() - started
            name = path.split("/normalized/")[1].replace("/", "__")
            with StoragePath(prefix_join(output, f"measure/{name}.json")).open("w") as sink:
                json.dump(stats, sink)
            if sample:
                for row in sample:
                    row["path"] = path
                buffer = io.BytesIO()
                pq.write_table(pa.Table.from_pylist(sample), buffer)
                with StoragePath(prefix_join(output, f"sample/{name}.parquet")).open("wb") as sink:
                    sink.write(buffer.getvalue())
            yield stats

    return measure


def shard_lists(sources_path: str, input_root: str) -> tuple[list[str], list[str]]:
    """Measure shards (1 per source) and disjoint reference shards (2 per source).

    The draws use one random stream each, in the order of the source list, so the list order fixes the sample.
    """
    with StoragePath(sources_path).open("r") as source:
        names = json.load(source)
    measure_rng, reference_rng = random.Random(7), random.Random(11)
    measure, reference = [], []
    for name in names:
        shards = sorted(str(p) for p in StoragePath(f"{input_root}/{name}/outputs/main/*.parquet").glob())
        picked = measure_rng.sample(shards, min(1, len(shards)))
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
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.items, args.output)
        return
    if args.step == "choose":
        choose(args.output)
        return
    measure, reference = shard_lists(args.sources, args.input_root)
    with StoragePath(prefix_join(args.output, "shards.json")).open("w") as sink:
        json.dump({"measure": measure, "reference": reference}, sink)
    files = reference if args.step == "count" else measure
    logger.info("%s: %d shards", args.step, len(files))
    resources = ResourceConfig(cpu=1, ram=args.ram)
    context = ZephyrContext(name=f"decon-encoder-{args.step}", resources=resources, max_workers=len(files))
    scan = make_count(args.output) if args.step == "count" else make_measure(args.output)
    outcome = context.execute(Dataset.from_list(files).map_shard(scan), map_task_resources=resources)
    logger.info("%s done: %d results", args.step, len(outcome.results))


if __name__ == "__main__":
    main()
