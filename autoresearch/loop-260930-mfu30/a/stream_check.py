"""Which memcpy streams carry the layer-carry host copies, and what else shares them.

With XLA_GPU_HOST_TRANSFER_STREAMS=1 the carry H2D (`wrapped_dynamic-slice*`, MemcpyH2D) and D2H
(`wrapped_dynamic-update-slice*`, MemcpyD2H) should each own a stream that no D2D weight-slice copy uses.

    uv run python stream_check.py <rows.pkl from tfop_dump.py>
"""

import collections
import pickle
import sys


def main(path: str) -> None:
    data = pickle.load(open(path, "rb"))
    rows = [r for r in data["rows"] if r[6] == "jit_train_step" and "Memcpy" in r[1]]
    steps = len(data["launches"])
    per_stream = collections.defaultdict(collections.Counter)
    for stream, name, start, end, _tf_op, hlo_op, *_ in rows:
        kind = name.split()[0][:9]
        if "dynamic-update-slice" in (hlo_op or "") and "D2H" in name:
            label = "carry D2H"
        elif "dynamic-slice" in (hlo_op or "") and "H2D" in name:
            label = "carry H2D"
        elif (hlo_op or "").startswith("copy-start"):
            label = f"opt-state {kind}"
        else:
            label = f"other {kind}"
        per_stream[stream[:12]][label] += 1
    carry_streams = {s for s, c in per_stream.items() if c["carry D2H"] or c["carry H2D"]}
    for stream in sorted(per_stream):
        counts = {k: round(v / steps, 1) for k, v in per_stream[stream].most_common()}
        print(f"{stream}: {counts}")
    shared = {s: dict(per_stream[s]) for s in carry_streams if any(k.startswith("other") for k in per_stream[s])}
    print("carry streams:", sorted(carry_streams), "| shared with other copies:" , "NONE" if not shared else shared)


if __name__ == "__main__":
    main(sys.argv[1])
