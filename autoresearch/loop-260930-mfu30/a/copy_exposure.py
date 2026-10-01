"""Exposed (not under compute or collectives) memcpy time per step, by copy class.

    uv run python copy_exposure.py <rows.pkl> [<rows.pkl> ...]
"""

import collections
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anatomy_lib import intersect, length, merge
from stream_check import label

COLLECTIVE = re.compile(r"nccl|RaggedAllToAll")
MEMCPY = re.compile(r"memcpy|memset", re.I)


def exposure(path: str) -> dict[str, float]:
    data = pickle.load(open(path, "rb"))
    rows = [r for r in data["rows"] if r[6] == "jit_train_step"]
    steps = len(data["launches"])
    busy = merge(
        [(r[2], r[3]) for r in rows if r[0].startswith("Stream #17") and not MEMCPY.search(r[1])]
        + [(r[2], r[3]) for r in rows if COLLECTIVE.search(r[1])]
    )
    out = collections.Counter()
    for stream, name, start, end, _tf_op, hlo_op, *_ in rows:
        if "Memcpy" not in name:
            continue
        exposed = (end - start) - length(intersect([[start, end]], busy))
        out[label(name, hlo_op or "")] += exposed / 1e12 / steps
    return out


def main(paths: list[str]) -> None:
    tables = [exposure(p) for p in paths]
    keys = sorted({k for t in tables for k in t}, key=lambda k: -max(t.get(k, 0) for t in tables))
    print("class".ljust(24) + "".join(Path(p).parent.name[-22:].rjust(24) for p in paths))
    for k in keys:
        print(k.ljust(24) + "".join(f"{t.get(k, 0):24.4f}" for t in tables))
    print("total".ljust(24) + "".join(f"{sum(t.values()):24.4f}" for t in tables))


if __name__ == "__main__":
    main(sys.argv[1:])
