# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map GPU kernel events in an xprof dump back to the HLO instructions that launched them.

`xplane_overlap` says *how much* of a kernel is hidden; this says *which HLO op* it came from.
Given a kernel-name substring it groups that kernel's events by stream and by the `hlo_op` /
`hlo_module` / `program_id` stats XLA attaches to each event, so a kernel that appears both
inline on the compute stream and on an async collective stream can be split by instruction.

Usage: python -m experiments.grug.moe.xplane_op_detail <xprof_s3_root> <kernel-substring> [hosts]
"""

import sys
from collections import defaultdict

import fsspec

from experiments.grug.moe.xplane_overlap import event_names, iter_fields, plane_name

TOP_ROWS = 40


def stat_names(plane) -> dict[int, str]:
    """XPlane.stat_metadata: map<int64, XStatMetadata> -> {id: name}."""
    names: dict[int, str] = {}
    for field, payload in iter_fields(plane):
        if field != 5:
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


def read_stats(event, stats_by_id: dict[int, str]) -> dict[str, str]:
    """Decode an XEvent's stats into {stat name: value}, resolving ref values."""
    out: dict[str, str] = {}
    for field, payload in iter_fields(event):
        if field != 4:
            continue
        key, value = None, None
        for stat_field, stat_payload in iter_fields(payload):
            if stat_field == 1:
                key = stat_payload
            elif stat_field == 5:
                value = bytes(stat_payload).decode(errors="replace")
            elif stat_field == 7:
                value = stats_by_id.get(stat_payload, f"?ref{stat_payload}")
            elif stat_field in (3, 4):
                value = str(stat_payload)
        if key is not None and value is not None:
            out[stats_by_id.get(key, f"?{key}")] = value
    return out


def analyze(path: str, data: bytes, needle: str) -> None:
    for field, payload in iter_fields(memoryview(data)):
        if field != 1:
            continue
        name = plane_name(payload)
        if not name.startswith("/device:GPU"):
            continue
        events = event_names(payload)
        stats_by_id = stat_names(payload)
        wanted = {i for i, n in events.items() if needle in n}
        if not wanted:
            continue

        print(f"\n=== {path.rsplit('/', 1)[-1]} :: {name}")
        rows: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for plane_field, plane_payload in iter_fields(payload):
            if plane_field != 3:
                continue
            line, timestamp = "", 0
            pending = []
            for line_field, line_payload in iter_fields(plane_payload):
                if line_field == 2:
                    line = bytes(line_payload).decode(errors="replace")
                elif line_field == 3:
                    timestamp = line_payload
                elif line_field == 4:
                    pending.append(line_payload)
            if not line.startswith("Stream"):
                continue
            for event in pending:
                metadata_id, duration = 0, 0
                for event_field, event_payload in iter_fields(event):
                    if event_field == 1:
                        metadata_id = event_payload
                    elif event_field == 3:
                        duration = event_payload
                if metadata_id not in wanted:
                    continue
                stats = read_stats(event, stats_by_id)
                hlo = stats.get("hlo_op") or stats.get("tf_op") or "?"
                module = stats.get("hlo_module") or stats.get("program_id") or "?"
                rows[(line.split("(", 1)[0].strip(), hlo, module)].append(duration)
                if len(rows) == 1 and len(rows[(line.split("(", 1)[0].strip(), hlo, module)]) == 1:
                    print(f"  sample stats: {stats}")
            del timestamp

        print(f"  {'stream':<14}{'hlo op':<40}{'module':<28}{'count':>7}{'total ms':>10}{'mean us':>10}")
        for (stream, hlo, module), durations in sorted(rows.items(), key=lambda kv: -sum(kv[1]))[:TOP_ROWS]:
            total = sum(durations)
            print(
                f"  {stream:<14}{hlo[:40]:<40}{module[:28]:<28}{len(durations):>7}"
                f"{total / 1e9:>10.2f}{total / len(durations) / 1e6:>10.1f}"
            )


def main() -> None:
    root = sys.argv[1].rstrip("/")
    needle = sys.argv[2]
    num_hosts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    fs = fsspec.filesystem(root.split("://", 1)[0])
    xplanes = sorted(p for p in fs.find(root) if p.endswith(".xplane.pb"))
    if not xplanes:
        raise FileNotFoundError(f"no .xplane.pb under {root}")
    print(f"found {len(xplanes)} xplane files; analyzing {num_hosts} for kernels matching {needle!r}")
    for path in xplanes[:num_hosts]:
        with fs.open(path, "rb") as handle:
            data = handle.read()
        print(f"\n#### {path} ({len(data)} bytes)")
        analyze(path, data, needle)


if __name__ == "__main__":
    main()
