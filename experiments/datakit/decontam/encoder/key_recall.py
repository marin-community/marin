# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 recall on judged copies: does the excerpt of each judged copy hold its item's key?

Judged pairs name their item by question text; the item table maps that text to item ids, and the
stage-1 keys table gives each id its keys. Reports, per judges' label, the share of excerpts
that hold the first window, key 1, and key 2.
"""

import argparse
import json
import logging
from collections import Counter

from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.encoder.tables import parquet_files, read_frame
from experiments.datakit.decontam.encoder.text_keys import text_stream, window_hashes

logger = logging.getLogger(__name__)


def norm(text: str) -> str:
    return " ".join((text or "").split())


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, help="stage-1 output with keys.parquet")
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet)")
    parser.add_argument("--judged", required=True, help="judged pairs with question, right and judges columns")
    args = parser.parse_args()
    judged = read_frame(args.judged)
    wanted = set(judged["question"].map(norm))
    ids: dict[str, str] = {}
    for path in parquet_files(args.items):
        table = read_frame(path, ["id", "question"])
        for item_id, question in zip(table["id"], table["question"], strict=True):
            key = norm(question)
            if key in wanted and key not in ids:
                ids[key] = item_id
    keys = read_frame(prefix_join(args.stage1, "keys.parquet"))
    # Read hash columns whole: one row of mixed integer columns would turn the uint64 hashes into floats.
    row_of = {item_id: number for number, item_id in enumerate(keys["id"])}
    columns = {name: keys[name].to_numpy(dtype="uint64") for name in ("first_hash", "key1_hash", "key2_hash")}
    counts: Counter = Counter()
    for question, right, label in zip(judged["question"], judged["right"], judged["judges"], strict=True):
        item_id = ids.get(norm(question))
        if item_id is None or item_id not in row_of:
            counts[(label, "no item")] += 1
            continue
        windows = set(window_hashes(text_stream(right)).tolist())
        for name, values in columns.items():
            value = int(values[row_of[item_id]])
            counts[(label, name)] += bool(value and value in windows)
        counts[(label, "items")] += 1
    summary = {f"{label}|{name}": count for (label, name), count in counts.items()}
    with StoragePath(prefix_join(args.stage1, "key_recall.json")).open("w") as sink:
        json.dump(summary, sink, indent=1)
    labels = sorted({label for label, _ in counts})
    for label in labels:
        n = counts[(label, "items")]
        if n:
            logger.info(
                "%s: %d pairs (%d without an item); first %.3f key1 %.3f key2 %.3f",
                label,
                n,
                counts[(label, "no item")],
                counts[(label, "first_hash")] / n,
                counts[(label, "key1_hash")] / n,
                counts[(label, "key2_hash")] / n,
            )


if __name__ == "__main__":
    main()
