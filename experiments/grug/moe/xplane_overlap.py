# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report GPU comm/compute overlap from an xprof xplane dump.

Answers "did the collective actually run concurrently with the GEMMs?" for a captured
profile: for every op on the GPU device timelines it reports total occupancy and the
fraction of that occupancy that is concurrent with ops on *other* streams, plus the top
concurrent partners. Reads directly from the run's xprof S3 root, so it runs as an iris
CPU job next to the training cluster (the submitting sandbox has no S3 credentials).

Usage: python -m experiments.grug.moe.xplane_overlap <xprof_s3_root> [num_hosts]
"""

import sys
from collections import defaultdict

import fsspec

TOP_OPS = 30
TOP_PARTNERS = 4
# NCCL device kernels carry every collective on this stack (SendRecv is the expert all-to-all).
COMM_PREFIX = "ncclDevKernel"


def read_varint(buf: memoryview, i: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buf[i]
        value |= (byte & 0x7F) << shift
        i += 1
        if not byte & 0x80:
            return value, i
        shift += 7


def iter_fields(buf: memoryview):
    """Yield (field_number, payload) for a protobuf message; payload is bytes or int."""
    i, end = 0, len(buf)
    while i < end:
        key, i = read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = read_varint(buf, i)
            yield field, value
        elif wire == 2:
            length, i = read_varint(buf, i)
            yield field, buf[i : i + length]
            i += length
        elif wire == 1:
            yield field, buf[i : i + 8]
            i += 8
        elif wire == 5:
            yield field, buf[i : i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def plane_name(plane: memoryview) -> str:
    for field, payload in iter_fields(plane):
        if field == 2:
            return bytes(payload).decode(errors="replace")
    return ""


def event_names(plane: memoryview) -> dict[int, str]:
    """XPlane.event_metadata: map<int64, XEventMetadata> -> {id: name}."""
    names: dict[int, str] = {}
    for field, payload in iter_fields(plane):
        if field != 4:
            continue
        key, value = None, None
        for entry_field, entry_payload in iter_fields(payload):
            if entry_field == 1:
                key = entry_payload
            elif entry_field == 2:
                value = entry_payload
        if key is None or value is None:
            continue
        for meta_field, meta_payload in iter_fields(value):
            if meta_field == 2:
                names[key] = bytes(meta_payload).decode(errors="replace")
    return names


def line_events(line: memoryview) -> tuple[str, list[tuple[int, int, int]]]:
    """Return (line name, [(metadata_id, start_ps, duration_ps)])."""
    name, timestamp_ns, events = "", 0, []
    for field, payload in iter_fields(line):
        if field == 2:
            name = bytes(payload).decode(errors="replace")
        elif field == 3:
            timestamp_ns = payload
        elif field == 4:
            metadata_id, offset_ps, duration_ps = 0, 0, 0
            for event_field, event_payload in iter_fields(payload):
                if event_field == 1:
                    metadata_id = event_payload
                elif event_field == 2:
                    offset_ps = event_payload
                elif event_field == 3:
                    duration_ps = event_payload
            events.append((metadata_id, offset_ps, duration_ps))
    return name, [(m, timestamp_ns * 1000 + o, d) for m, o, d in events]


def union_length(intervals: list[tuple[int, int]]) -> int:
    total, current_end = 0, -1
    for start, end in sorted(intervals):
        if start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def overlap_by_partner(
    target: list[tuple[int, int]], others: list[tuple[int, int, str]]
) -> tuple[int, dict[str, int]]:
    """Overlap of `target` intervals with `others` (start, end, op name) on other streams."""
    target = sorted(target)
    others = sorted(others)
    per_partner: dict[str, int] = defaultdict(int)
    covered: list[tuple[int, int]] = []
    j = 0
    active: list[tuple[int, int, str]] = []
    for start, end in target:
        while j < len(others) and others[j][0] < end:
            active.append(others[j])
            j += 1
        active = [o for o in active if o[1] > start]
        for other_start, other_end, name in active:
            low, high = max(start, other_start), min(end, other_end)
            if high > low:
                per_partner[name] += high - low
                covered.append((low, high))
    return union_length(covered), dict(per_partner)


def analyze(path: str, data: bytes) -> None:
    buf = memoryview(data)
    for field, payload in iter_fields(buf):
        if field != 1:
            continue
        name = plane_name(payload)
        if not name.startswith("/device:GPU"):
            continue
        names = event_names(payload)
        streams: dict[str, list[tuple[str, int, int]]] = {}
        for plane_field, plane_payload in iter_fields(payload):
            if plane_field != 3:
                continue
            line, events = line_events(plane_payload)
            if not events:
                continue
            streams[line] = [(names.get(m, f"?{m}"), s, s + d) for m, s, d in events]

        span = max(e for events in streams.values() for _, _, e in events) - min(
            s for events in streams.values() for _, s, _ in events
        )
        print(f"\n=== {path.rsplit('/', 1)[-1]} :: {name} (trace span {span / 1e9:.2f} ms)")
        for line, events in sorted(streams.items()):
            busy = union_length([(s, e) for _, s, e in events])
            comm = union_length([(s, e) for op, s, e in events if op.startswith(COMM_PREFIX)])
            print(f"  line {line!r}: {len(events)} events, busy {busy / 1e9:.2f} ms, collective {comm / 1e9:.2f} ms")

        # Exposed collective time: collective kernels with nothing else running anywhere else.
        comm_intervals = [(s, e) for events in streams.values() for op, s, e in events if op.startswith(COMM_PREFIX)]
        noncomm = sorted(
            (s, e, op) for events in streams.values() for op, s, e in events if not op.startswith(COMM_PREFIX)
        )
        comm_total = union_length(comm_intervals)
        comm_covered, _ = overlap_by_partner(sorted(comm_intervals), noncomm)
        print(
            f"  collectives: {comm_total / 1e9:.2f} ms total, {comm_covered / 1e9:.2f} ms concurrent with "
            f"non-collective work, {(comm_total - comm_covered) / 1e9:.2f} ms exposed "
            f"({100 * (comm_total - comm_covered) / max(span, 1):.1f}% of span)"
        )

        totals: dict[tuple[str, str], int] = defaultdict(int)
        op_intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for line, events in streams.items():
            for op, start, end in events:
                totals[(line, op)] += end - start
                op_intervals[(line, op)].append((start, end))

        others_by_stream: dict[str, list[tuple[int, int, str]]] = {}
        for line in streams:
            others_by_stream[line] = sorted(
                (start, end, other_op)
                for other_line, events in streams.items()
                if other_line != line
                for other_op, start, end in events
            )

        print(f"  {'stream':<14}{'op':<48} {'total ms':>9} {'overlap%':>9}  partners")
        for (line, op), total in sorted(totals.items(), key=lambda kv: -kv[1])[:TOP_OPS]:
            overlap, partners = overlap_by_partner(op_intervals[(line, op)], others_by_stream[line])
            top = ", ".join(
                f"{p}:{v / 1e9:.1f}ms" for p, v in sorted(partners.items(), key=lambda kv: -kv[1])[:TOP_PARTNERS]
            )
            stream_id = line.split("(", 1)[0].strip()
            print(f"  {stream_id:<14}{op[:48]:<48} {total / 1e9:>9.2f} {100 * overlap / max(total, 1):>8.1f}%  {top}")


def main() -> None:
    root = sys.argv[1].rstrip("/")
    num_hosts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fs = fsspec.filesystem(root.split("://", 1)[0])
    xplanes = sorted(p for p in fs.find(root) if p.endswith(".xplane.pb"))
    if not xplanes:
        raise FileNotFoundError(f"no .xplane.pb under {root}")
    print(f"found {len(xplanes)} xplane files; analyzing {num_hosts}")
    for path in xplanes[:num_hosts]:
        with fs.open(path, "rb") as handle:
            data = handle.read()
        print(f"\n#### {path} ({len(data)} bytes)")
        analyze(path, data)


if __name__ == "__main__":
    main()
