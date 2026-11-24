# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn the stage-1 sample of candidates into (left, right) pairs for the classifier.

``items``: join the sampled candidate rows with the item table (question, choices, answer).
``extract``: a Zephyr job over the sampled shards reads each candidate record and cuts the text
around the place where the item starts (or, for an edited copy, where its key matches). It also
records three text checks for analysis: the whole question in the record, the answer after it,
and every choice after it.
"""

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterator

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon_keys import WINDOW, KeyStream, excerpt, key_streams, record_readings, text_stream
from marin.datakit.decon_tables import parquet_files, read_table, write_table
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

# Path segment under which the normalized pretraining shards live.
NORMALIZED = "/normalized/"


def shard_name(path: str) -> str:
    """A flat name for one normalized shard: its path below ``normalized/``, with ``/`` written as ``__``."""
    if NORMALIZED not in path:
        raise ValueError(f"{path} is not under a {NORMALIZED} prefix")
    return path.split(NORMALIZED, 1)[1].replace("/", "__")


def join_items(items_root: str, stage1: str, output: str) -> None:
    sample = pa.concat_tables([read_table(path) for path in parquet_files(prefix_join(stage1, "sample"))])
    wanted = set(sample.column("item").to_pylist())
    rows = []
    for path in parquet_files(items_root):
        table = read_table(path, ["id", "eval", "question", "choices", "answer", "passage"]).to_pylist()
        rows += [row for row in table if row["id"] in wanted]
    keys = read_table(prefix_join(stage1, "keys.parquet")).select(["id", "kind", "key1_offset"]).to_pylist()
    offsets = {(row["id"], row["kind"]): row["key1_offset"] for row in keys if row["id"] in wanted}
    # The key offset belongs to the stream kind that matched, so it is kept on each sample row.
    sample = sample.append_column(
        "key1_offset",
        pa.array(
            [
                offsets.get((item, kind), -1)
                for item, kind in zip(sample.column("item").to_pylist(), sample.column("kind").to_pylist(), strict=True)
            ],
            pa.int64(),
        ),
    )
    items = pa.Table.from_pylist(rows)
    write_table(items, prefix_join(output, "sample_items.parquet"))
    write_table(sample, prefix_join(output, "sample_rows.parquet"))
    logger.info("items: %d sampled candidate rows, %d items found of %d", sample.num_rows, items.num_rows, len(wanted))


def make_extract(output: str):
    def extract(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        rows = read_table(prefix_join(output, "sample_rows.parquet")).to_pylist()
        items = {row["id"]: row for row in read_table(prefix_join(output, "sample_items.parquet")).to_pylist()}
        for path in paths:
            wanted: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                if row["path"] == path and row["item"] in items:
                    wanted[row["id"]].append(row)
            pairs = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                record_id = str(record["id"])
                if record_id not in wanted:
                    continue
                record_text = str(record.get("text") or "")
                readings_of = record_readings(record_text)
                for candidate in wanted[record_id]:
                    item_id, kind, offset = candidate["item"], candidate["kind"], candidate["key1_offset"]
                    item = items[item_id]
                    question = text_stream(item["question"])
                    # The key window comes from the stream that stage 1 keyed (see decon_keys.key_streams).
                    keyed = key_streams(item["question"], list(item["choices"] or []), item["passage"] or "")
                    stream_of_kind = keyed.get(KeyStream(kind), "")
                    key = stream_of_kind[offset : offset + WINDOW] if offset >= 0 else ""
                    right, stream, at = excerpt(readings_of, len(item["question"]), question, key, offset)
                    after = stream[at + len(question) :] if at >= 0 and stream.startswith(question, at) else ""
                    answer = text_stream(item["answer"])
                    pairs.append(
                        {
                            "path": path,
                            "record": record_id,
                            "item": item_id,
                            "kind": kind,
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
                                and all(text_stream(c) in after for c in item["choices"] if text_stream(c))
                            ),
                        }
                    )
            name = shard_name(path)
            if pairs:
                write_table(pa.Table.from_pylist(pairs), prefix_join(output, f"pairs/{name}.parquet"))
            yield {"path": path, "pairs": len(pairs)}

    return extract


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["items", "extract"])
    parser.add_argument("--stage1", required=True, help="measure_keys.py output with sample/*.parquet")
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
