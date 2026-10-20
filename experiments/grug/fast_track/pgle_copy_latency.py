# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a PGLE profile so XLA's latency-hiding scheduler overlaps the large host-offload copies.

``jax.experimental.profiler.get_profiled_instructions_proto`` records every ``copy-start`` (a host<->device
memcpy) as an instruction *cost* and emits no start-to-done *latency*. The profile-guided latency estimator then
charges the whole transfer to the compute stream at ``copy-start`` and, lacking a latency entry, falls back to that
cost as the pair's latency, which the next big copy's own cost already covers. At the hero shape the two expert
momentum D2H copies (~105 ms each) end up back to back after Newton-Schulz with nothing under them.

This moves the measured time of every ``copy-start`` above ``--min-us`` from its cost to a
``copy-start.N -> copy-done.N`` latency and sets its cost to ``COPY_START_COST_US``, so the scheduler keeps that
much real work between the start and the done. Scheduling only: the program's math is unchanged.

Usage: python -m experiments.grug.fast_track.pgle_copy_latency IN.pb OUT.pb [--min-us 1000]
"""

import argparse
import re
import struct

COPY_START_COST_US = 1.0
"""Cost left on a rewritten ``copy-start``: the launch, not the transfer."""

_COPY_START = re.compile(r"copy-start(\.\d+)*")


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        result |= (byte & 0x7F) << shift
        i += 1
        shift += 7
        if byte < 0x80:
            return result, i


def _fields(buf: bytes) -> list[tuple[int, int, bytes | int | float]]:
    """``(field number, wire type, value)`` of a serialized message (wire types 0, 1, 2 and 5)."""
    out = []
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _read_varint(buf, i)
        elif wire == 1:
            value = struct.unpack("<d", buf[i : i + 8])[0]
            i += 8
        elif wire == 2:
            length, i = _read_varint(buf, i)
            value = buf[i : i + length]
            i += length
        elif wire == 5:
            value = struct.unpack("<f", buf[i : i + 4])[0]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append((number, wire, value))
    return out


def _string_field(number: int, text: str) -> bytes:
    data = text.encode()
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def _double_field(number: int, value: float) -> bytes:
    return _varint(number << 3 | 1) + struct.pack("<d", value)


def _message_field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _cost(name: str, cost_us: float) -> bytes:
    """``ProfiledInstructionsProto.InstructionCost``: name = 1, cost_us = 2."""
    return _string_field(1, name) + _double_field(2, cost_us)


def _latency(source: str, target: str, latency_us: float) -> bytes:
    """``ProfiledInstructionsProto.Latency``: source = 1, target = 2, latency_us = 3."""
    return _string_field(1, source) + _string_field(2, target) + _double_field(3, latency_us)


def copy_latency_profile(profile: bytes, min_us: float) -> tuple[bytes, list[tuple[str, float]]]:
    """The rewritten ``ProfiledInstructionsProto`` and the ``(copy-start name, latency_us)`` pairs it moved.

    Costs (field 1) and existing latencies (field 2) are kept in order; each ``copy-start`` cost above ``min_us``
    becomes ``COPY_START_COST_US`` plus a latency to its ``copy-done`` (same suffix)."""
    out = bytearray()
    moved = []
    for number, wire, value in _fields(profile):
        if wire != 2:
            raise ValueError(f"unexpected non-message top-level field {number}")
        if number != 1:
            out += _message_field(number, value)
            continue
        entry = {n: v for n, _, v in _fields(value)}
        name = entry[1].decode()
        cost_us = float(entry.get(2, 0.0))
        if _COPY_START.fullmatch(name) and cost_us > min_us:
            out += _message_field(1, _cost(name, COPY_START_COST_US))
            out += _message_field(2, _latency(name, name.replace("copy-start", "copy-done", 1), cost_us))
            moved.append((name, cost_us))
        else:
            out += _message_field(1, value)
    return bytes(out), moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source")
    parser.add_argument("dest")
    parser.add_argument("--min-us", type=float, default=1000.0, help="Rewrite copy-starts costing more than this.")
    args = parser.parse_args()
    with open(args.source, "rb") as f:
        profile, moved = copy_latency_profile(f.read(), args.min_us)
    with open(args.dest, "wb") as f:
        f.write(profile)
    for name, latency_us in moved:
        print(f"{name}: cost {latency_us:.0f} us -> latency")
    print(f"{len(moved)} copy-starts rewritten -> {args.dest}")


if __name__ == "__main__":
    main()
