# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the stage-1 candidate volume of a keys build, and sample candidate rows for the classifier data.

A Zephyr job over measure shards (``--per-source`` per source, drawn apart from the reference shards that the
keys were counted on) counts the records with a candidate and the candidate pairs, for the first window of each
stream, key 1, and key 1 or 2. It writes ``measure/<shard>.json`` per shard and a reservoir sample of key-1
candidate rows (record, item, key) as ``sample/<shard>.parquet``, which ``pairs.py`` turns into pairs.

    python experiments/datakit/decontam/encoder/measure_keys.py --keys <keys step output> \
        --sources experiments/datakit/decontam/encoder/sources.json --input-root <normalized root> --output <dir>
"""

import argparse
import json
import logging
import random
import time
from collections.abc import Iterator

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon import KEYS_FILENAME, reference_shards
from marin.datakit.decon_keys import MAX_DOC_CHARS, WINDOW, KeyTable, record_window_hashes
from marin.datakit.decon_tables import parquet_files, read_table, write_table
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.config import REFERENCE_SEED, REFERENCE_SHARDS_PER_SOURCE
from experiments.datakit.decontam.encoder.pairs import shard_name

logger = logging.getLogger(__name__)
SAMPLE_ROWS_PER_SHARD = 400
MEASURE_SEED = 7


def make_measure(keys_path: str, output: str, width: int):
    def measure(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        keys = read_table(keys_path).to_pandas()
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
                hashes = record_window_hashes(text, width)
                found = {name: table.found(hashes) for name, table in tables.items()}
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


def measure_shards(sources_path: str, input_root: str, seed: int, per_source: int) -> list[str]:
    """*per_source* shards of every source in the source list, none of them a reference shard of the keys."""
    with StoragePath(sources_path).open("r") as source:
        dirs = [prefix_join(input_root, f"{name}/outputs/main") for name in json.load(source)]
    reference = set(reference_shards(dirs, REFERENCE_SHARDS_PER_SOURCE, REFERENCE_SEED))
    rng = random.Random(seed)
    picked: list[str] = []
    for directory in dirs:
        shards = [shard for shard in parquet_files(directory) if shard not in reference]
        picked += rng.sample(shards, min(per_source, len(shards)))
    return picked


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", required=True, help="output directory of the keys step (holds keys.parquet)")
    parser.add_argument("--sources", required=True, help="JSON list of normalized source names, such as sources.json")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ram", default="32g")
    parser.add_argument("--width", type=int, default=WINDOW, help="key window width; the keys build's width")
    parser.add_argument("--seed", type=int, default=MEASURE_SEED, help="seed of the measure-shard draw")
    parser.add_argument("--per-source", type=int, default=1, help="measure shards drawn per source")
    args = parser.parse_args()
    files = measure_shards(args.sources, args.input_root, args.seed, args.per_source)
    StoragePath(prefix_join(args.output, "shards.json")).write_text(json.dumps({"measure": files}))
    # pairs.py reads the keys of the sample from this output.
    keys_path = prefix_join(args.keys, KEYS_FILENAME)
    StoragePath(prefix_join(args.output, KEYS_FILENAME)).write_bytes(StoragePath(keys_path).read_bytes())
    logger.info("measure: %d shards", len(files))
    resources = ResourceConfig(cpu=1, ram=args.ram)
    context = ZephyrContext(name="decon-measure-keys", resources=resources, max_workers=len(files))
    scan = make_measure(keys_path, args.output, args.width)
    outcome = context.execute(Dataset.from_list(files).map_shard(scan), map_task_resources=resources)
    logger.info("measure done: %d shards", len(outcome.results))


if __name__ == "__main__":
    main()
