# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report which collectives XLA left synchronous, and what covers the async ones.

Reads an ``--xla_dump_to`` directory, picks the largest post-optimization HLO dump, and walks
each computation in schedule order. For every collective it prints whether the post-scheduling
``GpuConvertAsyncCollectivesToSync`` pass collapsed it (``is_sync":true``) and, for the pairs
that stayed async, how many real instructions sit between start and done — the slack the
scheduler actually found. Output goes to stdout so it can be harvested from job logs.

Usage: python -m experiments.grug.moe.schedule_report <xla_dump_dir> [name-substring]
"""

import pathlib
import re
import sys

NOP_OPCODES = ("parameter", "constant", "bitcast", "get-tuple-element", "tuple(")
# Instruction defs only need their name: shapes contain spaces and nested parens, so do not
# try to parse the opcode out of them -- classify off substrings in the line instead.
INSTRUCTION = re.compile(r"^\s*%?([\w.\-]+)\s*=\s")
COMPUTATION = re.compile(r"^\s*(ENTRY\s+)?%?([\w.\-]+)\s*[\({]")
COLLECTIVE = re.compile(r"all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute|async-start|async-done")
TOP_COVER = 5
OP_NAME = re.compile(r'op_name="([^"]*)"')


def leg_of(line: str) -> str:
    """The jaxpr scope tail (e.g. dispatch/all_to_all) plus fwd/bwd, from HLO metadata."""
    match = OP_NAME.search(line)
    if not match:
        return "?"
    name = match.group(1)
    return ("bwd " if "transpose(" in name else "fwd ") + "/".join(name.split("/")[-2:])


def pick_dump(directory: pathlib.Path) -> pathlib.Path:
    candidates = sorted(directory.glob("*after_optimizations*.txt"), key=lambda p: p.stat().st_size)
    if not candidates:
        candidates = sorted(directory.glob("*.txt"), key=lambda p: p.stat().st_size)
    if not candidates:
        raise FileNotFoundError(f"no HLO text dumps under {directory}")
    return candidates[-1]


def main() -> None:
    directory = pathlib.Path(sys.argv[1])
    needle = sys.argv[2] if len(sys.argv) > 2 else "all-to-all"
    dump = pick_dump(directory)
    print(f"dump: {dump} ({dump.stat().st_size} bytes)")
    lines = dump.read_text(errors="replace").splitlines()

    sync_total = sum(1 for line in lines if '"is_sync":true' in line or '"is_sync": true' in line)
    print(f"instructions tagged is_sync=true: {sync_total}")

    computation = "?"
    order: list[tuple[int, str, str, str]] = []  # (index, computation, name, line)
    for index, line in enumerate(lines):
        match = COMPUTATION.match(line)
        if match and "=" not in line.split("{")[0]:
            computation = match.group(2)
        instruction = INSTRUCTION.match(line)
        if instruction:
            order.append((index, computation, instruction.group(1).lstrip("%"), line))

    starts: dict[str, tuple[int, str, str]] = {}
    print(f"\n{'leg':<26}{'instruction':<28}{'state':<10}{'cover':>6}  covering ops")
    for position, (_, computation, name, line) in enumerate(order):
        # XLA writes native collectives with hyphens (all-to-all-start) and shard_map-wrapped
        # ones with underscores (all_to_all.112-start); match either spelling.
        flat = line.replace("_", "-")
        if needle.replace("_", "-") not in flat and not COLLECTIVE.search(flat):
            continue
        if "-start(" in line or "async-start" in line:
            starts[name] = (position, computation, line)
            continue
        done = re.search(r"-done\((?:%)?([\w.\-]+)\)", line)
        if done and done.group(1) in starts:
            start_position, start_computation, start_line = starts.pop(done.group(1))
            between = [
                (n, l) for _, c, n, l in order[start_position + 1 : position] if c == start_computation
            ]
            real = [(n, l) for n, l in between if not any(op in l for op in NOP_OPCODES)]
            cover = ", ".join(n for n, _ in real[:TOP_COVER]) or "-"
            # The GPU pass keeps collapsed pairs in the schedule and only tags the start, so
            # "async" here means the pair survived; is_sync marks the ones executed inline.
            state = "SYNC" if '"is_sync":true' in start_line else "async"
            print(
                f"{leg_of(start_line)[:26]:<26}{done.group(1)[:28]:<28}{state:<10}{len(real):>6}  {cover}"
            )
        elif needle.replace("_", "-") in flat and "-done" not in flat:
            state = "SYNC" if '"is_sync":true' in line else "plain"
            print(f"{leg_of(line)[:26]:<26}{name[:28]:<28}{state:<10}{'-':>6}")

    for name, (_, computation, _) in starts.items():
        print(f"{computation[:26]:<26}{name[:28]:<28}{'unmatched':<10}")


if __name__ == "__main__":
    main()
