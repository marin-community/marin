# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a bounded TaskTrove prefix when one Parquet row group exceeds the sample budget."""

import argparse
import base64
import hashlib
import json
import random
import struct
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import BinaryIO

import requests
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile
from zstandard import ZstdDecompressor

from experiments.post_training.task_curation_partitions import assign_partitions
from experiments.post_training.tasktrove.taskbinary import read_task_binary

READ_BLOCK_BYTES = 64 * 1024
PAGE_HEADER_READ_BYTES = 8192


class TransferBudget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.transferred = 0
        self.ranges: list[dict[str, int]] = []
        self.lock = Lock()


class BudgetedHfFile(HfFileSystemFile):
    def __init__(self, fs: HfFileSystem, path: str, *, budget: TransferBudget, **kwargs):
        self.budget = budget
        super().__init__(fs, path, **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        # Protect the cumulative transfer budget when files share this counter.
        with self.budget.lock:
            requested = end - start
            if requested > self.budget.maximum - self.budget.transferred:
                raise OSError(f"Range {start}:{end} exceeds remaining transfer budget")
            headers = {"Range": f"bytes={start}-{end - 1}", **self.fs._api._build_hf_headers()}
            with requests.get(self.url(), headers=headers, timeout=60, stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise OSError("Server ignored the bounded range request; refusing its response body")
                chunks = []
                received = 0
                record = {"start": start, "end": end, "received": 0}
                self.budget.ranges.append(record)
                for chunk in response.iter_content(chunk_size=READ_BLOCK_BYTES):
                    received += len(chunk)
                    self.budget.transferred += len(chunk)
                    record["received"] = received
                    if received > requested:
                        raise OSError("Range response exceeds the requested bytes")
                    chunks.append(chunk)
                return b"".join(chunks)


class BudgetedHfFileSystem(HfFileSystem):
    def __init__(self, *, budget: TransferBudget):
        self.budget = budget
        super().__init__(block_size=READ_BLOCK_BYTES)

    def _open(self, path: str, mode: str = "rb", block_size: int | None = None, revision: str | None = None, **kwargs):
        if mode != "rb":
            raise ValueError("The prefix sampler only reads files")
        return BudgetedHfFile(
            self,
            path,
            budget=self.budget,
            mode=mode,
            block_size=block_size or READ_BLOCK_BYTES,
            revision=revision,
            **kwargs,
        )


class SnappyPrefixReader:
    """Decode raw Snappy incrementally, retaining the prefix needed for byte-array entries."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.decoded = bytearray()
        self.buffer = bytearray()
        self.offset = 0
        value = 0
        shift = 0
        while True:
            byte = self.read_compressed(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
        self.expected_length = value

    def read_compressed(self, count: int) -> bytes:
        while len(self.buffer) - self.offset < count:
            block = self.stream.read(max(READ_BLOCK_BYTES, count - len(self.buffer) + self.offset))
            if not block:
                raise ValueError("Truncated Snappy block")
            self.buffer.extend(block)
        result = bytes(self.buffer[self.offset : self.offset + count])
        self.offset += count
        return result

    def ensure_decoded(self, count: int) -> None:
        while len(self.decoded) < count:
            tag = self.read_compressed(1)[0]
            kind = tag & 3
            if kind == 0:
                size = tag >> 2
                length = size + 1 if size < 60 else int.from_bytes(self.read_compressed(size - 59), "little") + 1
                self.decoded.extend(self.read_compressed(length))
                continue
            if kind == 1:
                length = 4 + ((tag >> 2) & 7)
                offset = ((tag & 224) << 3) | self.read_compressed(1)[0]
            else:
                length = 1 + (tag >> 2)
                offset = int.from_bytes(self.read_compressed(2 if kind == 2 else 4), "little")
            if offset <= 0 or offset > len(self.decoded):
                raise ValueError("Invalid Snappy copy offset")
            # A copy may reference bytes written by that same copy.
            start = len(self.decoded) - offset
            previous = bytes(self.decoded[start : start + min(offset, length)])
            self.decoded.extend((previous * ((length + offset - 1) // offset))[:length])


def snappy_dictionary_prefix(stream: BinaryIO, count: int) -> list[bytes]:
    """Read only the leading length-prefixed dictionary values from a raw Snappy page."""
    decoder = SnappyPrefixReader(stream)
    cursor = 0
    values = []
    for _ in range(count):
        decoder.ensure_decoded(cursor + 4)
        length = struct.unpack_from("<I", decoder.decoded, cursor)[0]
        cursor += 4
        decoder.ensure_decoded(cursor + length)
        values.append(bytes(decoder.decoded[cursor : cursor + length]))
        cursor += length
    return values


def plain_dictionary_prefix(stream: BinaryIO, count: int) -> list[bytes]:
    """Read the requested prefix of decoded length-prefixed dictionary entries."""
    values = []
    for _ in range(count):
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError("Truncated dictionary entry length")
        length = struct.unpack("<I", header)[0]
        value = stream.read(length)
        if len(value) != length:
            raise ValueError("Truncated dictionary entry")
        values.append(value)
    return values


def column_prefix(parquet, column, stream: BinaryIO, count: int) -> list[bytes]:
    """Return a non-null byte-array prefix from V1 pages and supported dictionary codecs."""
    from fastparquet import core, parquet_thrift  # noqa: PLC0415 -- optional page-decoding dependency.
    from fastparquet.cencoding import NumpyIO, ThriftObject  # noqa: PLC0415

    metadata = column.meta_data
    if count > metadata.num_values:
        raise ValueError("The row group has fewer rows than the requested prefix")
    indexes: list[int | bytes] = []
    page_offset = metadata.data_page_offset
    while len(indexes) < count:
        stream.seek(page_offset)
        buffer = NumpyIO(stream.read(PAGE_HEADER_READ_BYTES))
        header = ThriftObject.from_buffer(buffer, "PageHeader")
        header_size = buffer.tell()
        if header.type != parquet_thrift.PageType.DATA_PAGE:
            raise ValueError("This bounded reader requires V1 data pages")
        stream.seek(page_offset + header_size)
        page = BytesIO(stream.read(header.compressed_page_size))
        definitions, repetitions, page_indexes = core.read_data_page(page, parquet.schema, header, metadata)
        if repetitions is not None or (definitions is not None and (definitions == 0).any()):
            raise ValueError("This bounded reader requires flat non-null columns")
        encoding = header.data_page_header.encoding
        if encoding not in {
            parquet_thrift.Encoding.RLE_DICTIONARY,
            parquet_thrift.Encoding.PLAIN_DICTIONARY,
            parquet_thrift.Encoding.PLAIN,
        }:
            raise ValueError("This bounded reader requires dictionary or plain byte-array encoding")
        if len(page_indexes) == 0:
            raise ValueError("An empty data page cannot advance the requested prefix")
        selected = page_indexes[: count - len(indexes)]
        if encoding == parquet_thrift.Encoding.PLAIN:
            indexes.extend(value.encode() if isinstance(value, str) else bytes(value) for value in selected)
        else:
            indexes.extend(int(index) for index in selected)
        page_offset += header_size + header.compressed_page_size
    dictionary_indexes = [index for index in indexes if isinstance(index, int)]
    if not dictionary_indexes:
        return [value for value in indexes if isinstance(value, bytes)]
    stream.seek(metadata.dictionary_page_offset)
    buffer = NumpyIO(stream.read(PAGE_HEADER_READ_BYTES))
    dictionary_header = ThriftObject.from_buffer(buffer, "PageHeader")
    if dictionary_header.type != parquet_thrift.PageType.DICTIONARY_PAGE:
        raise ValueError("The declared dictionary offset must point to a dictionary page")
    stream.seek(metadata.dictionary_page_offset + buffer.tell())
    required = max(dictionary_indexes) + 1
    if metadata.codec == parquet_thrift.CompressionCodec.SNAPPY:
        dictionary = snappy_dictionary_prefix(stream, required)
    elif metadata.codec == parquet_thrift.CompressionCodec.ZSTD:
        with ZstdDecompressor().stream_reader(stream, read_size=READ_BLOCK_BYTES, closefd=False) as decoded:
            dictionary = plain_dictionary_prefix(decoded, required)
    elif metadata.codec == parquet_thrift.CompressionCodec.UNCOMPRESSED:
        dictionary = plain_dictionary_prefix(stream, required)
    else:
        raise ValueError(f"Unsupported bounded dictionary codec: {metadata.codec}")
    return [dictionary[value] if isinstance(value, int) else value for value in indexes]


def sample_prefix(
    name: str,
    output: Path,
    count: int,
    seed: int,
    *,
    config: str,
    revision: str,
    max_transfer_bytes: int,
) -> dict:
    """Read leading rows by decoding only their dictionary entries and required index pages."""
    from fastparquet import ParquetFile  # noqa: PLC0415 -- optional page-decoding dependency.

    budget = TransferBudget(max_transfer_bytes)
    filesystem = BudgetedHfFileSystem(budget=budget)
    shard = f"{config}/tasks.parquet"
    path = f"datasets/open-thoughts/TaskTrove@{revision}/{shard}"
    with filesystem.open(path, block_size=READ_BLOCK_BYTES, cache_type="none") as stream:
        parquet = ParquetFile(stream)
        rows = []
        groups = []
        for group_index, group in enumerate(parquet.row_groups):
            remaining = count - len(rows)
            if remaining == 0:
                break
            size = min(remaining, group.num_rows)
            columns = {column.meta_data.path_in_schema[0]: column for column in group.columns}
            paths = column_prefix(parquet, columns["path"], stream, size)
            binaries = column_prefix(parquet, columns["task_binary"], stream, size)
            rows.extend((path, binary, group_index) for path, binary in zip(paths, binaries, strict=True))
            groups.append(group_index)
        source_rows = parquet.count()
    snapshots = []
    for index, (archive_path, binary, group_index) in enumerate(rows):
        files = read_task_binary(binary).files
        instruction = files["instruction.md"].decode()
        snapshots.append(
            {
                "path": archive_path.decode(),
                "instruction": instruction,
                "files": {path: base64.b64encode(data).decode() for path, data in files.items()},
                "archive_sha256": hashlib.sha256(binary).hexdigest(),
                "sample_index": index,
                "sample_row_group": group_index,
                "sample_group": hashlib.sha256(instruction.encode()).hexdigest(),
            }
        )
        if "tests/verifier_data.json" in files:
            snapshots[-1]["verifier_data"] = json.loads(files["tests/verifier_data.json"])
    if len(snapshots) != count:
        raise ValueError(f"Read {len(snapshots)} rows; requested {count}")
    assign_partitions(snapshots, random.Random(f"{seed}:{name}"), "sample_index")
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in snapshots)
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sample.jsonl").write_text(serialized)
    manifest = {
        "name": name,
        "dataset": "open-thoughts/TaskTrove",
        "revision": revision,
        "shard": shard,
        "source_rows": source_rows,
        "row_groups": groups,
        "sample_rows": len(snapshots),
        "seed": seed,
        "sampling": (
            "First N archived rows, page-wise bounded read across leading row groups; "
            "ordered prefix, not population-uniform"
        ),
        "grouping": "SHA256 of public instruction",
        "transfer_budget_bytes": budget.maximum,
        "transferred_parquet_bytes": budget.transferred,
        "http_ranges": budget.ranges,
        "partitions": {
            partition: sum(row["sample_partition"] == partition for row in snapshots)
            for partition in ("development", "holdout")
        },
        "snapshot_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "reader": "fastparquet V1 dictionary/plain data pages and bounded Snappy/Zstd/plain dictionary prefix",
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--max-transfer-bytes", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            sample_prefix(
                args.name,
                args.output,
                args.count,
                args.seed,
                config=args.config,
                revision=args.revision,
                max_transfer_bytes=args.max_transfer_bytes,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
