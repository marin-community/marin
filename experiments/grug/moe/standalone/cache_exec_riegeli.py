# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-011: unwrap riegeli-serialized executables from the JAX persistent cache.

XLA serializes ``ExecutableAndOptionsProto`` with riegeli (snappy-2, see
``xla/util/split_proto``): the huge ``serialized_executable`` field (the
GPU ``CompilationResultProto``) is written as its own string record. This tool
implements a minimal riegeli record reader (block/chunk framing + pure-python
snappy block decoding, hashes unchecked), extracts the largest record, walks it
as protobuf, and reports/extracts the embedded PTX and CUDA ELF payloads.

Usage (iris CPU pod):
    python -m experiments.grug.moe.standalone.cache_exec_riegeli \
        [--copy-to s3://bucket/prefix/] s3://.../jit_train_step-...-cache
"""

import pathlib
import struct
import sys

import fsspec

from experiments.grug.moe.standalone.cache_exec_carve import decompress
from experiments.grug.moe.standalone.cache_exec_walk import _parse_fields, _sniff, walk

BLOCK_SIZE = 1 << 16
BLOCK_HEADER = 24
CHUNK_HEADER = 40
EM_CUDA = 190


def flatten_blocks(buf: bytes) -> bytes:
    """Drop the 24-byte block header at every 64 KiB file boundary."""
    out = []
    for i in range(0, len(buf), BLOCK_SIZE):
        out.append(buf[i + BLOCK_HEADER : i + BLOCK_SIZE])
    return b"".join(out)


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not b & 0x80:
            return result, pos
        shift += 7


def snappy_raw_decompress(buf: bytes) -> bytes:
    """Pure-python snappy block-format decoder (no framing)."""
    expected, pos = read_varint(buf, 0)
    out = bytearray()
    n = len(buf)
    while pos < n:
        tag = buf[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            ln = tag >> 2
            if ln >= 60:
                extra = ln - 59
                ln = int.from_bytes(buf[pos : pos + extra], "little")
                pos += extra
            ln += 1
            out += buf[pos : pos + ln]
            pos += ln
            continue
        if kind == 1:  # copy with 1-byte offset
            ln = ((tag >> 2) & 0x7) + 4
            off = ((tag >> 5) << 8) | buf[pos]
            pos += 1
        elif kind == 2:  # copy with 2-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(buf[pos : pos + 2], "little")
            pos += 2
        else:  # copy with 4-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        start = len(out) - off
        if start < 0:
            raise ValueError(f"bad snappy copy offset at {pos}")
        for _ in range(ln):  # copies may overlap; byte-by-byte is correct
            out.append(out[start])
            start += 1
    if len(out) != expected:
        raise ValueError(f"snappy: got {len(out)} expected {expected}")
    return bytes(out)


def decode_stream(data: bytes) -> bytes:
    """Riegeli compressor stream: varint(uncompressed size) + snappy block data."""
    _, pos = read_varint(data, 0)
    return snappy_raw_decompress(data[pos:])


def iter_records(stream: bytes):
    """Yield (chunk_type, record_bytes) from a flattened riegeli chunk stream."""
    pos = 0
    n = len(stream)
    while pos + CHUNK_HEADER <= n:
        data_size = int.from_bytes(stream[pos + 8 : pos + 16], "little")
        chunk_type = stream[pos + 24]
        num_records = int.from_bytes(stream[pos + 25 : pos + 32], "little")
        data = stream[pos + CHUNK_HEADER : pos + CHUNK_HEADER + data_size]
        pos += CHUNK_HEADER + data_size
        ct = chr(chunk_type)
        if ct in ("s", "m", "p"):
            continue
        if ct != "r":
            raise ValueError(f"unhandled chunk type {ct!r} (transposed?)")
        compression = data[0]
        sizes_len, p = read_varint(data, 1)
        sizes_blob = data[p : p + sizes_len]
        values_blob = data[p + sizes_len :]
        if compression == 0:
            sizes_raw, values_raw = sizes_blob, values_blob
        elif compression == ord("s"):
            sizes_raw = decode_stream(sizes_blob)
            values_raw = decode_stream(values_blob)
        else:
            raise ValueError(f"unhandled compression {compression:#x}")
        sp = 0
        sizes = []
        for _ in range(num_records):
            sz, sp = read_varint(sizes_raw, sp)
            sizes.append(sz)
        vp = 0
        for sz in sizes:
            yield ct, values_raw[vp : vp + sz]
            vp += sz


def analyze(path: str, copy_to: str | None) -> None:
    with fsspec.open(path, "rb") as f:
        raw = f.read()
    entry = decompress(raw)[4:]  # strip 4-byte big-endian compile time
    stream = flatten_blocks(entry)
    print(f"ENTRY {path}")
    records = list(iter_records(stream))
    print(f"  records={len(records)} sizes={[len(r) for _, r in records]}")
    big = max(records, key=lambda t: len(t[1]))[1]
    print(f"  largest_record={len(big)} head={big[:16].hex()}")
    out: list = []
    if _parse_fields(big, 0, len(big)) is not None:
        walk(big, 0, len(big), "", out)
    else:
        print("  largest record is not a bare proto; trying offset resync")
        for start in range(1, 64):
            if _parse_fields(big, start, len(big)) is not None:
                print(f"  proto_start_offset={start}")
                walk(big, start, len(big), "", out)
                break
    seen = set()
    fields = []
    for p, off, ln, sniff in sorted(out, key=lambda t: -t[2]):
        if off in seen:
            continue
        seen.add(off)
        fields.append((p, off, ln, sniff))
    for p, off, ln, sniff in fields[:30]:
        print(f"  field {p} off={off} len={ln} :: {sniff}")
    if copy_to:
        name = pathlib.Path(path).name[:44]
        fs2, _ = fsspec.url_to_fs(copy_to)
        for p, off, ln, sniff in fields:
            if sniff.startswith("ELF") or sniff == "PTX":
                kind = "elf" if sniff.startswith("ELF") else "ptx"
                dst = f"{copy_to.rstrip('/')}/{name}-{off}.{kind}"
                with fs2.open(dst, "wb") as g:
                    g.write(big[off : off + ln])
                print(f"  copied {dst} ({ln} B, {sniff})")
        dst = f"{copy_to.rstrip('/')}/{name}-inner.pb"
        with fs2.open(dst, "wb") as g:
            g.write(big)
        print(f"  copied {dst} (full inner proto)")


def main() -> None:
    copy_to = None
    args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--copy-to":
            copy_to = next(it)
        else:
            args.append(a)
    for path in args:
        analyze(path, copy_to)
        print()


def _elf_machine(payload: bytes) -> int:
    (machine,) = struct.unpack_from("<H", payload, 18)
    return machine


if __name__ == "__main__":
    main()
