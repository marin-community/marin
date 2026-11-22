# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn the stage-1 sample of candidates into (left, right) pairs for the classifier.

``items``: join the sampled candidate rows with the item table (question, choices, answer).
``extract``: a Zephyr job over the sampled shards reads each candidate record and cuts the text
around the place where the item starts (or, for an edited copy, where its key matches). It also
records text checks that serve as features and silver labels: question in the record, answer
after it, and every choice after it.
"""

import argparse
import io
import logging
from collections import defaultdict
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

from experiments.datakit.decontam.encoder.text_keys import WINDOW, readings, text_stream

logger = logging.getLogger(__name__)
BEFORE = 1500
AFTER = 3000


def read_table(path: str) -> pa.Table:
    with StoragePath(path).open("rb") as source:
        return pq.read_table(io.BytesIO(source.read()))


def write_table(table: pa.Table, path: str) -> None:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    with StoragePath(path).open("wb") as sink:
        sink.write(buffer.getvalue())


def join_items(items_root: str, stage1: str, output: str) -> None:
    sample = pa.concat_tables([read_table(str(p)) for p in StoragePath(prefix_join(stage1, "sample/*.parquet")).glob()])
    wanted = set(sample.column("item").to_pylist())
    rows = []
    for path in StoragePath(items_root + "/*.parquet").glob():
        table = read_table(str(path)).select(["id", "eval", "question", "choices", "answer"]).to_pylist()
        rows += [row for row in table if row["id"] in wanted]
    keys = read_table(prefix_join(stage1, "keys.parquet")).select(["id", "key1_offset"]).to_pylist()
    offsets = {row["id"]: row["key1_offset"] for row in keys if row["id"] in wanted}
    for row in rows:
        row["key1_offset"] = offsets.get(row["id"], -1)
    items = pa.Table.from_pylist(rows)
    write_table(items, prefix_join(output, "sample_items.parquet"))
    write_table(sample, prefix_join(output, "sample_rows.parquet"))
    logger.info("items: %d sampled candidate rows, %d items found of %d", sample.num_rows, items.num_rows, len(wanted))


def _stream(text: str) -> str:
    return text_stream(text)


def _offset(text: str, position: int) -> int:
    """The smallest prefix length of *text* whose stream is longer than *position*."""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high) // 2
        if len(_stream(text[:middle])) > position:
            high = middle
        else:
            low = middle + 1
    return low


def _located(streams: list[tuple[str, str]], question: str, key: str, offset: int) -> tuple[str, str, int]:
    """The first record reading that holds the item, its stream, and the item's stream position (-1 if none)."""
    for text, stream in streams:
        at = stream.find(question) if question else -1
        if at < 0 and key:
            # The record holds the key (stage 1 found it); the item starts that far before it.
            hit = stream.find(key)
            at = max(0, hit - offset) if hit >= 0 else -1
        if at >= 0:
            return text, stream, at
    return *streams[0], -1


def make_extract(output: str):
    def extract(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        rows = read_table(prefix_join(output, "sample_rows.parquet")).to_pylist()
        items = {row["id"]: row for row in read_table(prefix_join(output, "sample_items.parquet")).to_pylist()}
        for path in paths:
            wanted: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                if row["path"] == path and row["item"] in items:
                    wanted[row["id"]].append(row["item"])
            pairs = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                record_id = str(record["id"])
                if record_id not in wanted:
                    continue
                record_text = str(record.get("text") or "")
                streams = [(reading, _stream(reading)) for reading in readings(record_text)]
                for item_id in wanted[record_id]:
                    item = items[item_id]
                    question = _stream(item["question"])
                    offset = item.get("key1_offset", -1)
                    key = question[offset : offset + WINDOW] if offset is not None and offset >= 0 else ""
                    text, stream, at = _located(streams, question, key, offset)
                    start = _offset(text, max(at, 0)) - 1 if at >= 0 else 0
                    right = text[max(0, start - BEFORE) : start + len(item["question"]) + AFTER]
                    after = stream[at + len(question) :] if at >= 0 and stream.startswith(question, at) else ""
                    answer = _stream(item["answer"])
                    pairs.append(
                        {
                            "path": path,
                            "record": record_id,
                            "item": item_id,
                            "eval": item["eval"],
                            "question": item["question"],
                            "choices": list(item["choices"] or []),
                            "answer": item["answer"] or "",
                            "right": right,
                            "record_chars": len(record_text),
                            "question_in_record": bool(question) and question in stream,
                            "answer_after": bool(after) and bool(answer) and answer in after,
                            "choices_after": (
                                bool(after)
                                and bool(item["choices"])
                                and all(_stream(c) in after for c in item["choices"] if _stream(c))
                            ),
                        }
                    )
            name = path.split("/normalized/")[1].replace("/", "__")
            if pairs:
                write_table(pa.Table.from_pylist(pairs), prefix_join(output, f"pairs/{name}.parquet"))
            yield {"path": path, "pairs": len(pairs)}

    return extract


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["items", "extract"])
    parser.add_argument("--stage1", required=True, help="stage-1 output with keys.parquet and sample/*.parquet")
    parser.add_argument("--items", required=True, help="the item table directory (items/*.parquet)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.step == "items":
        join_items(args.items, args.stage1, args.output)
        return
    paths = sorted(set(read_table(prefix_join(args.output, "sample_rows.parquet")).column("path").to_pylist()))
    resources = ResourceConfig(cpu=1, ram="16g")
    context = ZephyrContext(name="decon-encoder-pairs", resources=resources, max_workers=len(paths))
    outcome = context.execute(
        Dataset.from_list(paths).map_shard(make_extract(args.output)), map_task_resources=resources
    )
    logger.info("extract done: %d pairs", sum(result["pairs"] for result in outcome.results))


if __name__ == "__main__":
    main()
