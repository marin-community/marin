# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report which collectives XLA left synchronous, and what work covers the async ones.

Reads an ``--xla_dump_to`` directory, picks the largest post-optimization HLO dump, and walks
each computation in schedule order (a scheduled module is printed in its schedule sequence).
For every collective it prints the MoE leg it belongs to, whether the post-scheduling
``GpuConvertAsyncCollectivesToSync`` pass collapsed it, and which real instructions sit between
start and done -- the slack the scheduler actually found.

Two properties of GPU HLO dumps mislead naive parsers, and this module handles both:

* A collapsed collective still *looks* async. ``GpuConvertAsyncCollectivesToSync`` does not
  delete the pair; it tags the start with ``is_sync`` and re-emits start/done adjacently. The
  tag, not the shape of the text, says whether a collective runs inline on the compute stream.
* Instruction lines cannot be parsed as ``name = <shape> opcode(``, because tuple shapes contain
  spaces and nested parens. ``shard_map``-wrapped collectives are also spelled ``all_to_all.112``
  with underscores while native ones use hyphens, so name matching must normalize both.

Usage: python -m experiments.grug.moe.schedule_report <xla_dump_dir> [name-substring]
"""

import pathlib
import re
import sys
from collections import Counter

# What the async->sync pass treats as "not work" when deciding whether a pair is covered.
NOP_MARKERS = (" parameter(", " constant(", " bitcast(", " get-tuple-element(", " tuple(")
INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s")
COMPUTATION_NAME = re.compile(r"%?([\w.\-]+)\s*\(")
OP_NAME = re.compile(r'op_name="([^"]*)"')
IS_SYNC = '"is_sync":true'
# Async halves are named foo-start / foo-start.12 -- the disambiguating suffix follows the tag.
START_NAME = re.compile(r"-start(\.\d+)?$")
DONE_NAME = re.compile(r"-done(\.\d+)?$")
TOP_COVER = 6


def pick_dump(directory: pathlib.Path) -> pathlib.Path:
    """The largest post-optimization dump, which is the train step rather than a helper module."""
    # The backend-specific dump (…sm_100a_gpu_after_optimizations) is the scheduled one; prefer it.
    candidates = sorted(directory.glob("*gpu_after_optimizations*.txt"), key=lambda path: path.stat().st_size)
    if not candidates:
        candidates = sorted(directory.glob("*after_optimizations*.txt"), key=lambda path: path.stat().st_size)
    if not candidates:
        candidates = sorted(directory.glob("*.txt"), key=lambda path: path.stat().st_size)
    if not candidates:
        raise FileNotFoundError(f"no HLO text dumps under {directory}")
    return candidates[-1]


def leg_of(line: str) -> str:
    """The MoE leg (fwd/bwd plus the jaxpr scope tail) taken from HLO metadata."""
    match = OP_NAME.search(line)
    if not match:
        return "?"
    name = match.group(1)
    return ("bwd " if "transpose(" in name else "fwd ") + "/".join(name.split("/")[-2:])


def computations(lines: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Split a dump into ``(computation name, [(instruction name, line)])`` in schedule order.

    Computations are delimited by column: a header starts at column 0 and ends the line with an
    open brace, and a lone ``}`` at column 0 closes it. Counting braces instead does not survive a
    real dump -- ``backend_config``, ``replica_groups`` and metadata strings put unbalanced braces
    on instruction lines, and one bad line desynchronizes everything after it.
    """
    result: list[tuple[str, list[tuple[str, str]]]] = []
    name: str | None = None
    body: list[tuple[str, str]] = []
    for line in lines:
        indented = line[:1].isspace()
        if not indented and line.rstrip().endswith("{"):
            header = line.split("{")[0]
            match = COMPUTATION_NAME.search(header)
            name = match.group(1) if match else (header.strip() or "?")
            body = []
            continue
        if not indented and line.strip() == "}":
            if name is not None:
                result.append((name, body))
            name, body = None, []
            continue
        if name is None:
            continue
        instruction = INSTRUCTION.match(line)
        if instruction:
            body.append((instruction.group(1), line))
    return result


def is_nop(line: str) -> bool:
    return any(marker in line for marker in NOP_MARKERS)


def main() -> None:
    directory = pathlib.Path(sys.argv[1])
    needle = (sys.argv[2] if len(sys.argv) > 2 else "all-to-all").replace("_", "-")
    dump = pick_dump(directory)
    print(f"dump: {dump} ({dump.stat().st_size} bytes)")
    lines = dump.read_text(errors="replace").splitlines()
    print(f"instructions tagged is_sync: {sum(1 for line in lines if IS_SYNC in line)}")

    parsed = computations(lines)
    print(
        f"computations: {len(parsed)}, instructions: {sum(len(b) for b in (body for _, body in parsed))}, "
        f"largest: {max((len(body) for _, body in parsed), default=0)}, "
        f"raw start lines matching {needle!r}: "
        f"{sum(1 for line in lines if needle in line.replace('_', '-') and '-start' in line.replace('_', '-'))}"
    )

    census: Counter[tuple[str, str]] = Counter()
    cover_rows: list[tuple[str, str, str, int, str]] = []
    for computation, body in parsed:
        for index, (name, line) in enumerate(body):
            # Match on the defined name, not the line: a done line names its start as an operand.
            flat_name = name.replace("_", "-")
            if needle not in flat_name or not START_NAME.search(flat_name):
                continue
            state = "SYNC" if IS_SYNC in line else "async"
            leg = leg_of(line)
            census[(leg, state)] += 1
            # The done's operand carries the start's shape before the name, so look for a
            # reference to %<start> on any line that defines a matching done.
            done = next(
                (i for i, (other_name, other) in enumerate(body) if DONE_NAME.search(other_name.replace("_", "-")) and f"%{name}" in other),
                None,
            )
            if done is None:
                cover_rows.append((computation, leg, name, -1, "no matching done in this computation"))
                continue
            between = [(n, l) for n, l in body[index + 1 : done] if not is_nop(l)]
            names = ", ".join(n for n, _ in between[:TOP_COVER]) or "-"
            cover_rows.append((computation, leg, name, len(between), names))

    print(f"\n{'leg':<26}{'state':<8}{'count':>6}")
    for (leg, state), count in sorted(census.items()):
        print(f"{leg[:26]:<26}{state:<8}{count:>6}")

    print(f"\n{'computation':<24}{'leg':<26}{'start':<24}{'cover':>6}  covering instructions")
    for computation, leg, name, cover, names in cover_rows:
        print(f"{computation[:24]:<24}{leg[:26]:<26}{name[:24]:<24}{cover:>6}  {names}")


if __name__ == "__main__":
    main()
