# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward carry D2H stall check: per-layer exposure of the carry offload copy.

XLA assigns async compute ops (the carry dynamic-update-slice to host and the per-layer weight
dynamic-slice fusions) to 4 memcpy streams round-robin over the whole module. When the first weight
slice a layer needs lands on the carry D2H's stream, the compute stream waits ~3-4 ms per layer
(0.15-0.19 s/step). Any program change can move the assignment, so check every trace.

Usage: carry_stall.py <rows.pkl from tfop_dump.py>
"""

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anatomy_lib import intersect, length, merge

PS_PER_MS = 1e9
COLLECTIVE = re.compile(r"nccl|RaggedAllToAll")
MEMCPY = re.compile(r"memcpy|memset", re.I)


def main(path: str) -> None:
    data = pickle.load(open(path, "rb"))
    rows = sorted((r for r in data["rows"] if r[6] == "jit_train_step"), key=lambda r: r[2])
    launches = data["launches"]
    busy = merge(
        [(r[2], r[3]) for r in rows if r[0].startswith("Stream #17") and not MEMCPY.search(r[1])]
        + [(r[2], r[3]) for r in rows if COLLECTIVE.search(r[1])]
    )
    carry = [r for r in rows if "D2H" in r[1] and "dynamic-update-slice" in r[5]]
    exposed = sorted((r[3] - r[2]) - length(intersect([[r[2], r[3]]], busy)) for r in carry)
    steps = len(launches)
    stream_names = sorted({r[0][:10] for r in carry})
    print(f"carry D2H copies: {len(carry)} over {steps} steps on {stream_names}")
    print(f"exposed: {sum(exposed) / PS_PER_MS / steps:.1f} ms/step (healthy ~0-10 ms, stalled ~150-190 ms)")
    print(f"median per copy: {exposed[len(exposed) // 2] / PS_PER_MS:.2f} ms of {len(exposed)}")


if __name__ == "__main__":
    main(sys.argv[1])
