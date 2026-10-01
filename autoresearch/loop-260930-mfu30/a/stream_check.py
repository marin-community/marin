"""Which memcpy streams carry the layer-carry host copies, step by step, and what else shares them.

XLA borrows its async streams from a pool on every execution, so the xprof stream that serves a given
execution-stream id can change from step to step. The check therefore groups copies per step. With
XLA_GPU_HOST_TRANSFER_STREAMS=1 the carry H2D (`wrapped_dynamic-slice*`, MemcpyH2D) and D2H
(`wrapped_dynamic-update-slice*`, MemcpyD2H) should share their stream with no D2D slice copy.

    uv run python stream_check.py <rows.pkl from tfop_dump.py>
"""

import bisect
import collections
import pickle
import sys


def label(name: str, hlo_op: str) -> str:
    if "dynamic-update-slice" in hlo_op and "D2H" in name:
        return "carry D2H"
    if "dynamic-slice" in hlo_op and "H2D" in name and "wrapped" in hlo_op:
        return "carry H2D"
    if hlo_op.startswith("copy-start"):
        return "opt-state " + name.split()[0]
    return "other " + name.split()[0]


def main(path: str) -> None:
    data = pickle.load(open(path, "rb"))
    launches = data["launches"]
    rows = [r for r in data["rows"] if r[6] == "jit_train_step" and "Memcpy" in r[1] and not r[0].startswith("Stream #17")]
    per = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for stream, name, start, _end, _tf_op, hlo_op, *_ in rows:
        step = max(bisect.bisect_right(launches, start) - 1, 0)
        per[step][stream[:10]][label(name, hlo_op or "")] += 1
    clean = True
    for step in sorted(per):
        print(f"step {step}:")
        for stream, counts in sorted(per[step].items()):
            has_carry = counts["carry D2H"] or counts["carry H2D"]
            shared = has_carry and any(k.startswith("other") for k in counts)
            clean &= not shared
            tag = " <- carry SHARED with slices" if shared else (" <- carry alone" if has_carry else "")
            print(f"  {stream}: {dict(counts.most_common())}{tag}")
    print("RESULT:", "carry copies never share a stream with D2D slice copies" if clean else "carry copies share streams")


if __name__ == "__main__":
    main(sys.argv[1])
